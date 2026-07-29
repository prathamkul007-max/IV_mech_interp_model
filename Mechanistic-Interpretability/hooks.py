"""Activation-capture infrastructure for IVModel, without modifying gpt2/model.py.

Approach: temporarily monkeypatch the module-level `gpt2.model.scaled_dot_product_attention`
function (CausalMultiHeadSelfAttention.forward calls it by global name, which Python resolves
at call time, so redirecting the module attribute redirects every call made during the scoped
forward pass) to capture the post-softmax attention pattern, combined with a forward hook on
each layer's `w_qkv` submodule to capture raw q/k/v. The original function is restored in a
`finally` block. This guarantees numerical identity with the real, unmodified model -- the
traced function's body is a literal copy of the original.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

import gpt2.model as gpt2_model_module
from gpt2.iv_model import IVModel


class ActivationCache(dict):
    """Keys look like 'blocks.{layer}.attn.hook_pattern' / '...hook_q' / '...hook_k' / '...hook_v',
    mirroring TransformerLens's cache naming for familiarity. Populated by manual hooks, not
    TransformerLens.
    """


def _traced_scaled_dot_product_attention_factory(cache: ActivationCache, counter: list[int]):
    """Returns a function that is a literal copy of gpt2.model.scaled_dot_product_attention's
    body, except it also records the post-softmax, pre-dropout attention pattern into `cache`,
    keyed by a call-order-derived layer index.

    Layer attribution assumption: IVModel.decoder_blocks is an nn.Sequential executed in strict
    order with exactly one causal_self_attention call per block per forward pass, so the n-th
    call within one run_with_cache() invocation deterministically corresponds to layer n. This
    breaks only under nested/re-entrant calls to the same model's forward within a single
    run_with_cache call, which this toolkit never does.
    """
    def traced(query, key, value, mask=None, dropout=None):
        layer = counter[0]
        counter[0] += 1

        d_k = query.size(-1)
        attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_probs.masked_fill_(mask == False, float('-inf'))  # noqa: E712

        attention_probs = F.softmax(attention_probs, dim=-1)
        cache[f'blocks.{layer}.attn.hook_pattern'] = attention_probs.detach().clone()

        if dropout is not None:
            if isinstance(dropout, float):
                dropout = torch.nn.Dropout(dropout)
            attention_probs = dropout(attention_probs)

        output = attention_probs @ value
        return output

    return traced


def _make_qkv_hook(cache: ActivationCache, layer_idx: int, attn_module):
    def hook(module, inputs, output):
        batch_size, seq_length, _ = output.size()
        d_model = attn_module.d_model
        num_heads = attn_module.num_heads
        d_k = attn_module.d_k

        q, k, v = output.split(d_model, dim=-1)
        q = q.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)

        cache[f'blocks.{layer_idx}.attn.hook_q'] = q.detach()
        cache[f'blocks.{layer_idx}.attn.hook_k'] = k.detach()
        cache[f'blocks.{layer_idx}.attn.hook_v'] = v.detach()

    return hook


def run_with_cache(model: IVModel, features: torch.Tensor) -> tuple[torch.Tensor, ActivationCache]:
    """Runs model(features) once, capturing per-layer attention patterns and q/k/v.

    Model should be in eval() mode -- if it's in training mode, dropout makes hook_pattern
    differ run-to-run from what a plain model(features) call would produce under the same
    (different) dropout mask, so a warning is emitted.
    """
    if model.training:
        warnings.warn(
            'model is in training mode; dropout will make hook_pattern non-deterministic '
            'relative to a plain forward pass. Call model.eval() first for reliable results.'
        )

    cache = ActivationCache()
    counter = [0]
    traced_fn = _traced_scaled_dot_product_attention_factory(cache, counter)

    handles = []
    for layer_idx, block in enumerate(model.decoder_blocks):
        attn_module = block.causal_self_attention
        handles.append(attn_module.w_qkv.register_forward_hook(_make_qkv_hook(cache, layer_idx, attn_module)))

    original_fn = gpt2_model_module.scaled_dot_product_attention
    gpt2_model_module.scaled_dot_product_attention = traced_fn
    try:
        with torch.no_grad():
            predictions = model(features)
    finally:
        gpt2_model_module.scaled_dot_product_attention = original_fn
        for handle in handles:
            handle.remove()

    return predictions, cache


def verify_hooks(model: IVModel, features: torch.Tensor, atol: float = 1e-5) -> dict:
    """Runs four correctness checks and returns a dict of booleans (plus 'all_passed'):
      1. non_invasive: hooked forward produces identical output to a plain forward pass.
      2. row_sum_to_one: each attention pattern row sums to 1 (softmax property).
      3. causal_mask_respected: no attention mass on future positions.
      4. reconstruction_matches: recomputing attn_probs @ v -> rl_projection reproduces the
         real causal_self_attention module's output, captured independently via a second hook.
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        pred_plain = model(features)

    recon_outputs: dict[int, torch.Tensor] = {}

    def make_recon_hook(layer_idx):
        def hook(module, inputs, output):
            recon_outputs[layer_idx] = output.detach()
        return hook

    recon_handles = [
        model.decoder_blocks[i].causal_self_attention.register_forward_hook(make_recon_hook(i))
        for i in range(model.config.num_layers)
    ]
    try:
        pred_hooked, cache = run_with_cache(model, features)
    finally:
        for handle in recon_handles:
            handle.remove()

    results = {'non_invasive': bool(torch.allclose(pred_plain, pred_hooked, atol=atol))}

    row_sum_ok = True
    causal_ok = True
    reconstruction_ok = True
    for i in range(model.config.num_layers):
        pattern = cache[f'blocks.{i}.attn.hook_pattern']
        v = cache[f'blocks.{i}.attn.hook_v']
        seq_len = pattern.size(-1)

        row_sums = pattern.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            row_sum_ok = False

        upper_triangular = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=pattern.device), diagonal=1)
        if pattern[..., upper_triangular].abs().max().item() > 1e-6:
            causal_ok = False

        attn_module = model.decoder_blocks[i].causal_self_attention
        reconstructed = pattern @ v  # (batch, heads, seq, d_k)
        batch_size = reconstructed.size(0)
        reconstructed = reconstructed.transpose(1, 2).contiguous().view(batch_size, seq_len, attn_module.d_model)
        reconstructed = attn_module.rl_projection(reconstructed)
        if not torch.allclose(reconstructed, recon_outputs[i], atol=atol):
            reconstruction_ok = False

    results['row_sum_to_one'] = row_sum_ok
    results['causal_mask_respected'] = causal_ok
    results['reconstruction_matches'] = reconstruction_ok
    results['all_passed'] = all(results.values())

    model.train(was_training)
    return results


if __name__ == '__main__':
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_iv_model import load_iv_checkpoint

    parser = argparse.ArgumentParser(description='Verify activation-capture hooks against a real IVModel checkpoint')
    parser.add_argument('--repo-id', type=str, required=True, help='e.g. Pratham007xo/iv-forecast-constituent-124m')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    model, scaler = load_iv_checkpoint(args.repo_id)
    dummy_features = torch.randn(args.batch_size, model.config.seq_length, model.config.num_features)
    results = verify_hooks(model, dummy_features)
    for key, value in results.items():
        print(f'{key}: {value}')
    if not results['all_passed']:
        raise SystemExit(1)
