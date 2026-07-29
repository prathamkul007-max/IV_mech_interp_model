"""Fixed-lag attention detection for IVModel: identifies attention heads that consistently
attend to a fixed earlier timestep (e.g. "yesterday", "5 trading days ago").

This replaces GPT-2's induction-head detection (which looks for literal token-repetition
patterns -- meaningless for continuous time-series data) with a numeric-time-series analog: for
each (layer, head), does attention mass concentrate at a fixed lag L relative to the query
position, averaged over a sample of windows? The underlying attention-variance-style heuristic
is kept from the original tool, but the computation is now well-defined (the original's
`act.mean(dim=-2)` reduction didn't correspond to any clean, interpretable quantity), and the
fake hardcoded-label precision/recall/F1 evaluation is dropped entirely -- there is no ground
truth for real IV data, so it's replaced with a principled uniform-causal-attention baseline
instead.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks import run_with_cache


def _lag_profile_from_pattern(head_pattern: np.ndarray) -> dict:
    """head_pattern: (N, seq_length, seq_length) attention pattern for one head, across N windows.

    For each lag L in [1, seq_length-1]:
      mean_attention(L) = mean over n and over all q >= L of pattern[n, q, q-L]
      baseline(L)        = mean over the same q range of 1/(q+1) (what uniform causal attention gives)
    """
    seq_length = head_pattern.shape[-1]
    lag_scores = np.zeros(seq_length - 1)
    lag_baselines = np.zeros(seq_length - 1)
    for lag in range(1, seq_length):
        q_idx = np.arange(lag, seq_length)
        k_idx = q_idx - lag
        values = head_pattern[:, q_idx, k_idx]  # (N, num_valid_q)
        lag_scores[lag - 1] = values.mean()
        lag_baselines[lag - 1] = np.mean(1.0 / (q_idx + 1))

    detected_lag = int(np.argmax(lag_scores)) + 1
    peak_score = float(lag_scores[detected_lag - 1])
    baseline = float(lag_baselines[detected_lag - 1])
    attention_variance = float(np.var(lag_scores))
    threshold = attention_variance * 0.5  # inherited constant from the original heuristic
    is_fixed_lag_head = bool(peak_score > threshold and peak_score > 2 * baseline)

    return {
        'detected_lag': detected_lag,
        'peak_score': peak_score,
        'threshold': threshold,
        'baseline': baseline,
        'is_fixed_lag_head': is_fixed_lag_head,
    }


def compute_lag_profile(model, features) -> pd.DataFrame:
    """Runs the model on a batch of windows `features` and returns a DataFrame with one row per
    (layer, head): detected_lag, peak_score, threshold, baseline, is_fixed_lag_head.
    """
    num_heads = model.config.num_heads
    num_layers = model.config.num_layers

    _, cache = run_with_cache(model, features)

    rows = []
    for layer in range(num_layers):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern'].cpu().numpy()  # (N, num_heads, seq, seq)
        for head in range(num_heads):
            profile = _lag_profile_from_pattern(pattern[:, head])
            rows.append({'layer': layer, 'head': head, **profile})

    return pd.DataFrame(rows)


def plot_lag_heatmap(df: pd.DataFrame):
    """Rows=layer, cols=head, cell color=detected_lag, annotated with peak_score."""
    num_layers = df['layer'].max() + 1
    num_heads = df['head'].max() + 1
    lag_grid = np.zeros((num_layers, num_heads))
    annot_grid = np.empty((num_layers, num_heads), dtype=object)
    for _, row in df.iterrows():
        lag_grid[int(row['layer']), int(row['head'])] = row['detected_lag']
        annot_grid[int(row['layer']), int(row['head'])] = f"{int(row['detected_lag'])}\n({row['peak_score']:.3f})"

    fig, ax = plt.subplots(figsize=(1.5 * num_heads, 1.2 * num_layers))
    sns.heatmap(
        lag_grid, annot=annot_grid, fmt='', cmap='viridis', ax=ax,
        xticklabels=[f'head {h}' for h in range(num_heads)],
        yticklabels=[f'layer {layer}' for layer in range(num_layers)],
    )
    ax.set_title('Detected fixed lag per (layer, head), with peak attention score')
    return fig


def plot_layer_scores(df: pd.DataFrame, layer: int):
    """Bar chart of peak_score vs threshold for one layer, highlighting flagged heads."""
    layer_df = df[df['layer'] == layer].sort_values('head')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:orange' if flagged else 'tab:blue' for flagged in layer_df['is_fixed_lag_head']]
    ax.bar(layer_df['head'], layer_df['peak_score'], color=colors, label='peak_score')
    ax.plot(layer_df['head'], layer_df['threshold'], color='black', marker='o', linestyle='--', label='threshold')
    ax.set_xlabel('head')
    ax.set_ylabel('score')
    ax.set_title(f'Layer {layer}: peak attention score vs threshold (orange = flagged as fixed-lag head)')
    ax.legend()
    return fig
