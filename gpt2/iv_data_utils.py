"""Shared data-prep helpers for the options-IV forecasting scripts.

Used by both `scripts/prepare_options_iv.py` (per-ticker/constituent-level)
and `scripts/prepare_options_iv_index.py` (synthetic equal-weighted index-level).
"""

import json
from typing import Optional

import numpy as np
import pandas as pd

# the 7 implied-volatility buckets across the moneyness spectrum, used as the
# forecasting target; matched as an exact underscore-separated token so that
# e.g. "itm" does not spuriously match inside "ditm"
IV_BUCKET_TOKENS = ['ditm', 'itm', 'sitm', 'atm', 'sotm', 'otm', 'dotm']

# keywords used to detect which columns are usable numeric features
FEATURE_KEYWORDS = ['iv', 'hv', 'vix', 'oi', 'interest', 'contract', 'spread', 'expir']


def _tokenize(col: str) -> list[str]:
    return [tok for tok in col.lower().replace('-', '_').split('_') if tok]


def detect_ticker_column(df: pd.DataFrame) -> str:
    for candidate in ('ticker', 'symbol'):
        if candidate in df.columns:
            return candidate
    raise ValueError(f'Could not find a ticker/symbol column among: {df.columns.tolist()}')


def detect_date_column(df: pd.DataFrame) -> str:
    for candidate in ('date', 'quote_date', 'trade_date'):
        if candidate in df.columns:
            return candidate
    raise ValueError(f'Could not find a date column among: {df.columns.tolist()}')


def detect_target_columns(df: pd.DataFrame) -> list[str]:
    """Find the 7 IV-bucket columns (one per moneyness bucket)."""
    target_cols = []
    for col in df.columns:
        tokens = _tokenize(col)
        if 'iv' in tokens and any(bucket in tokens for bucket in IV_BUCKET_TOKENS):
            target_cols.append(col)
    if len(target_cols) != len(IV_BUCKET_TOKENS):
        raise ValueError(
            f'Expected to find {len(IV_BUCKET_TOKENS)} IV-bucket columns '
            f'({IV_BUCKET_TOKENS}), found {len(target_cols)}: {target_cols}. '
            f'Inspect df.columns and adjust detection logic in gpt2/iv_data_utils.py.'
        )
    return sorted(target_cols)


def detect_feature_columns(df: pd.DataFrame, ticker_col: str, date_col: str) -> list[str]:
    """Select all usable numeric feature columns (includes the IV-bucket target columns,
    since past IV values are also valid inputs for forecasting future IV)."""
    feature_cols = []
    for col in df.columns:
        if col in (ticker_col, date_col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        tokens = _tokenize(col)
        if any(keyword in tok for tok in tokens for keyword in FEATURE_KEYWORDS):
            feature_cols.append(col)
    return sorted(feature_cols)


def load_raw_dataframe(dataset_name: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    import datasets

    ds = datasets.load_dataset(dataset_name, split='train', cache_dir=cache_dir)
    df = ds.to_pandas()
    print(f'Loaded {len(df)} rows with columns: {df.columns.tolist()}')
    return df


def clean_and_prepare(
    df: pd.DataFrame,
    ticker_col: str,
    date_col: str,
    feature_cols: list[str],
    target_cols: list[str],
    seq_length: int,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])

    # forward-fill short gaps in feature columns within each ticker's history
    df[feature_cols] = df.groupby(ticker_col, group_keys=False)[feature_cols].ffill()

    # can't safely impute a target column, so drop rows still missing a target value
    df = df.dropna(subset=target_cols)
    df = df.dropna(subset=feature_cols)

    # drop tickers with too little history to form even one window
    ticker_counts = df.groupby(ticker_col)[date_col].transform('count')
    df = df[ticker_counts >= seq_length + 1]

    return df.reset_index(drop=True)


def fit_scaler(train_features: np.ndarray, feature_cols: list[str], target_cols: list[str]) -> dict:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    target_indices = [feature_cols.index(col) for col in target_cols]
    return {
        'feature_names': feature_cols,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'target_indices': target_indices,
    }


def apply_scaler(features: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler['mean'], dtype=np.float32)
    std = np.asarray(scaler['std'], dtype=np.float32)
    return ((features - mean) / std).astype(np.float32)


def save_scaler(scaler: dict, seq_length: int, path: str) -> None:
    scaler = dict(scaler)
    scaler['seq_length'] = seq_length
    with open(path, 'w') as f:
        json.dump(scaler, f, indent=2)


def make_windows(
    features: np.ndarray,
    target_indices: list[int],
    seq_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping (input, target) windows from a single contiguous,
    date-ordered feature matrix of shape (num_rows, num_features).

    input window:  rows [t     : t+seq_length]
    target window: rows [t + 1 : t+seq_length + 1], IV-bucket columns only
    """
    num_rows = features.shape[0]
    window_len = seq_length + 1
    if num_rows < window_len:
        return np.empty((0, seq_length, features.shape[1]), dtype=np.float32), \
            np.empty((0, seq_length, len(target_indices)), dtype=np.float32)

    starts = range(0, num_rows - window_len + 1, stride)
    inputs, targets = [], []
    for start in starts:
        window = features[start:start + window_len]
        inputs.append(window[:-1])
        targets.append(window[1:, target_indices])
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)
