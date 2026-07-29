"""Generates plain-English summaries from the actual computed analysis outputs
(fixed_lag_detection's DataFrame, hooks.verify_hooks's results dict). These are grounded in
whatever numbers a given run actually produces -- nothing here is a canned/hardcoded finding.
"""
from __future__ import annotations

import pandas as pd


def summarize_lag_profile(df: pd.DataFrame, model_name: str) -> str:
    flagged = df[df['is_fixed_lag_head']]
    lines = [f'### {model_name}: fixed-lag attention summary', '']
    lines.append(
        f'- {len(flagged)} of {len(df)} (layer, head) pairs are flagged as fixed-lag heads '
        f'(peak attention at one lag exceeds both the head\'s own variance-derived threshold '
        f'and 2x the uniform-causal-attention baseline at that lag).'
    )

    if len(flagged) == 0:
        lines.append(
            '- No heads show attention concentrated at a specific fixed lag beyond the uniform '
            'baseline -- attention in this model may be more content-based (driven by the actual '
            'feature values) than positional/lag-based.'
        )
        return '\n'.join(lines)

    top = df.sort_values('peak_score', ascending=False).iloc[0]
    lines.append(
        f'- Strongest signal: layer {int(top["layer"])}, head {int(top["head"])} consistently '
        f'attends {int(top["detected_lag"])} trading day(s) back '
        f'(peak attention {top["peak_score"]:.3f} vs a uniform baseline of {top["baseline"]:.3f} '
        f'at that lag -- {top["peak_score"] / max(top["baseline"], 1e-6):.1f}x the null).'
    )

    lag_counts = flagged['detected_lag'].value_counts().sort_index()
    common_lag = int(lag_counts.idxmax())
    lines.append(
        f'- Most common detected lag among flagged heads: {common_lag} day(s) back '
        f'({int(lag_counts.max())} of {len(flagged)} flagged heads).'
    )

    by_layer = flagged.groupby('layer').size().sort_values(ascending=False)
    layer_list = ', '.join(f'{int(layer)} ({int(count)} heads)' for layer, count in by_layer.items())
    lines.append(f'- Flagged heads by layer: {layer_list}.')

    return '\n'.join(lines)


def compare_lag_profiles(constituent_df: pd.DataFrame, index_df: pd.DataFrame) -> str:
    def flagged_lag_mode(df: pd.DataFrame) -> int | None:
        flagged = df[df['is_fixed_lag_head']]
        if len(flagged) == 0:
            return None
        return int(flagged['detected_lag'].value_counts().idxmax())

    c_lag = flagged_lag_mode(constituent_df)
    i_lag = flagged_lag_mode(index_df)
    c_flagged = int(constituent_df['is_fixed_lag_head'].sum())
    i_flagged = int(index_df['is_fixed_lag_head'].sum())

    lines = ['### Constituent-level vs. index-level: fixed-lag comparison', '']
    lines.append(f'- Constituent-level model: {c_flagged} flagged heads, most common lag = {c_lag}.')
    lines.append(f'- Index-level model: {i_flagged} flagged heads, most common lag = {i_lag}.')

    if c_lag is not None and i_lag is not None:
        if c_lag == i_lag:
            lines.append(
                f'- Both models converge on the same dominant lag ({c_lag} day(s)) despite being '
                f'trained on different data granularity (per-stock vs. equal-weighted index) -- '
                f'suggests this lag reflects a genuine property of how IV evolves day-to-day, not '
                f'an artifact of one dataset.'
            )
        else:
            lines.append(
                f'- The two models settled on different dominant lags ({c_lag} vs. {i_lag} days), '
                f'which could reflect the index series being smoother/less noisy (averaging over '
                f'many tickers) than any single stock\'s IV, changing what lag is most predictive.'
            )
    return '\n'.join(lines)


def summarize_hook_verification(results: dict, model_name: str) -> str:
    status = 'PASSED' if results.get('all_passed') else 'FAILED'
    lines = [f'### {model_name}: hook verification -- {status}', '']
    for key, value in results.items():
        if key == 'all_passed':
            continue
        lines.append(f'- {key}: {value}')
    if not results.get('all_passed'):
        lines.append(
            '\n**Warning**: one or more checks failed -- the captured attention patterns may not '
            'accurately reflect this model\'s real behavior. Do not trust the visualizations above '
            'until this is resolved.'
        )
    return '\n'.join(lines)
