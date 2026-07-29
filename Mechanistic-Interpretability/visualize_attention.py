"""Attention-pattern heatmap visualization for IVModel.

Axis labels are relative timestep offsets (e.g. 't-31' ... 't') rather than decoded string
tokens -- IVModel has no tokenizer/vocab, its input is a window of numeric feature vectors, one
per trading day. See sample_windows.py / README.md for the timestep-label caveat.
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


def get_attention_pattern(model, features: torch.Tensor, layer: int, head: int, batch_idx: int = 0) -> torch.Tensor:
    """Returns the (seq_length, seq_length) attention pattern for one (layer, head, batch item)."""
    _, cache = run_with_cache(model, features)
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # (batch, num_heads, seq, seq)
    return pattern[batch_idx, head]


def plot_attention_heatmap_static(model, features: torch.Tensor, layer: int, head: int, batch_idx: int = 0, ax=None):
    """Static matplotlib/seaborn heatmap for one (layer, head, batch item)."""
    pattern = get_attention_pattern(model, features, layer, head, batch_idx).cpu().numpy()
    labels = timestep_labels(pattern.shape[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pattern, xticklabels=labels, yticklabels=labels, cmap='viridis', ax=ax, cbar=True)
    ax.set_xlabel('key (attended-to day)')
    ax.set_ylabel('query (current day)')
    ax.set_title(f'Layer {layer}, Head {head} attention pattern')
    return ax


def plot_attention_heatmap_interactive(model, features: torch.Tensor, layer: int, head: int, batch_idx: int = 0) -> go.Figure:
    """Interactive Plotly heatmap for one (layer, head, batch item)."""
    pattern = get_attention_pattern(model, features, layer, head, batch_idx).cpu().numpy()
    labels = timestep_labels(pattern.shape[0])

    fig = go.Figure(data=go.Heatmap(z=pattern, x=labels, y=labels, colorscale='Viridis'))
    fig.update_layout(
        title=f'Layer {layer}, Head {head} attention pattern',
        xaxis_title='key (attended-to day)',
        yaxis_title='query (current day)',
    )
    return fig


def plot_all_heads_grid(model, features: torch.Tensor, layer: int, batch_idx: int = 0):
    """Static grid of all heads' attention patterns for one layer, one batch item."""
    num_heads = model.config.num_heads
    _, cache = run_with_cache(model, features)
    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][batch_idx].cpu().numpy()  # (num_heads, seq, seq)
    labels = timestep_labels(pattern.shape[-1])

    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for head in range(num_heads):
        ax = axes[head // ncols][head % ncols]
        sns.heatmap(pattern[head], xticklabels=labels, yticklabels=labels, cmap='viridis', ax=ax, cbar=False)
        ax.set_title(f'Head {head}')
    for head in range(num_heads, nrows * ncols):
        axes[head // ncols][head % ncols].axis('off')
    fig.suptitle(f'Layer {layer}: all heads')
    fig.tight_layout()
    return fig
