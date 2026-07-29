"""QK circuit analysis for IVModel: visualizes query-key interaction strength per layer/head.

Adapted from a GPT-2/TransformerLens-based version: same einsum('bhqd,bhkd->bhqk', Q, K)
approach, but the `layer` param is fully generalized (any 0..config.num_layers-1, no hardcoded
GPT-2-small-specific layer index) and axis labels are relative timestep offsets instead of
decoded string tokens, since IVModel has no tokenizer/vocab.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks import run_with_cache
from sample_windows import timestep_labels


def extract_qk_patterns(model, features: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (Q, K) for the given layer, each of shape (batch, num_heads, seq_length, d_k)."""
    if not 0 <= layer < model.config.num_layers:
        raise ValueError(f'layer must be in [0, {model.config.num_layers - 1}], got {layer}')

    _, cache = run_with_cache(model, features)
    return cache[f'blocks.{layer}.attn.hook_q'], cache[f'blocks.{layer}.attn.hook_k']


def compute_qk_interactions(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Query-key interaction strength averaged over batch and heads: (seq_length, seq_length)."""
    interactions = torch.einsum('bhqd,bhkd->bhqk', Q, K)
    return interactions.mean(dim=(0, 1))


def plot_qk_interactions_static(model, features: torch.Tensor, layer: int, title: str | None = None, ax=None):
    Q, K = extract_qk_patterns(model, features, layer)
    interactions = compute_qk_interactions(Q, K).cpu().numpy()
    labels = timestep_labels(interactions.shape[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(interactions, xticklabels=labels, yticklabels=labels, cmap='coolwarm', center=0, ax=ax)
    ax.set_xlabel('key (day)')
    ax.set_ylabel('query (day)')
    ax.set_title(title or f'Layer {layer} QK interaction (avg over batch/heads)')
    return ax


def plot_qk_interactions_interactive(model, features: torch.Tensor, layer: int, title: str | None = None) -> go.Figure:
    Q, K = extract_qk_patterns(model, features, layer)
    interactions = compute_qk_interactions(Q, K).cpu().numpy()
    labels = timestep_labels(interactions.shape[0])

    fig = go.Figure(data=go.Heatmap(z=interactions, x=labels, y=labels, colorscale='RdBu', zmid=0))
    fig.update_layout(
        title=title or f'Layer {layer} QK interaction (avg over batch/heads)',
        xaxis_title='key (day)',
        yaxis_title='query (day)',
    )
    return fig
