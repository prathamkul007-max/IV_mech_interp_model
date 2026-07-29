#!/usr/bin/env python

"""
Build a synthetic, equal-weighted "S&P 500" implied-volatility index series
from the options-IV-SP500 dataset
(https://huggingface.co/datasets/gauss314/options-IV-SP500), and window it
for training the index-level IV-forecasting model (see gpt2/iv_model.py).

There are no market-cap weights in the source dataset, so each date's
synthetic index row is the equal-weighted (plain) mean across all tickers
with data on that date, for every numeric feature column. Unlike
`prepare_options_iv.py`, this collapses the ticker dimension entirely,
leaving a single date-ordered time series (~1 row per trading day) which is
then windowed the same way.
"""

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.iv_data_utils import (
    apply_scaler,
    clean_and_prepare,
    detect_date_column,
    detect_feature_columns,
    detect_target_columns,
    detect_ticker_column,
    fit_scaler,
    load_raw_dataframe,
    make_windows,
    save_scaler,
)


def prepare_options_iv_index(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_raw_dataframe(args.dataset_name, cache_dir=args.cache_dir)
    ticker_col = detect_ticker_column(df)
    date_col = detect_date_column(df)
    target_cols = detect_target_columns(df)
    feature_cols = detect_feature_columns(df, ticker_col, date_col)
    print(f'Using {len(feature_cols)} feature columns: {feature_cols}')
    print(f'Using {len(target_cols)} target columns: {target_cols}')

    # seq_length + 1 is not a meaningful per-ticker history requirement here
    # since we are about to collapse the ticker dimension away; use 1 so
    # clean_and_prepare doesn't drop tickers based on per-ticker row count
    df = clean_and_prepare(df, ticker_col, date_col, feature_cols, target_cols, seq_length=1)

    # collapse the ticker dimension: one synthetic "index" row per date,
    # equal-weighted (plain mean) across all tickers available that date
    index_df = df.groupby(date_col)[feature_cols].mean().sort_index()
    print(f'Synthetic index series has {len(index_df)} rows (one per trading day)')

    cutoff_date = index_df.index.to_series().quantile(1.0 - args.valid_frac, interpolation='nearest')
    print(f'Validation cutoff date: {cutoff_date}')

    train_features = index_df[index_df.index <= cutoff_date].to_numpy(dtype=np.float32)
    valid_features = index_df[index_df.index > cutoff_date].to_numpy(dtype=np.float32)

    scaler = fit_scaler(train_features, feature_cols, target_cols)
    save_scaler(scaler, args.seq_length, os.path.join(args.output_dir, 'scaler.json'))
    target_indices = scaler['target_indices']

    train_features_scaled = apply_scaler(train_features, scaler)
    valid_features_scaled = apply_scaler(valid_features, scaler)

    train_X, train_Y = make_windows(train_features_scaled, target_indices, args.seq_length, args.stride)
    valid_X, valid_Y = make_windows(valid_features_scaled, target_indices, args.seq_length, args.stride)

    print(f'Train windows: {train_X.shape}, valid windows: {valid_X.shape}')

    np.savez(os.path.join(args.output_dir, 'train.npz'), X=train_X, Y=train_Y)
    np.savez(os.path.join(args.output_dir, 'valid.npz'), X=valid_X, Y=valid_Y)
    print(f'Saved train.npz, valid.npz, scaler.json to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Build and window a synthetic equal-weighted S&P 500 IV index series',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()
    prepare_options_iv_index(args)


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--seed',
        help='Seed for random number generator',
        type=int,
        default=1061109567,
    )
    parser.add_argument(
        '--dataset-name',
        help='Name of the HuggingFace dataset to load',
        type=str,
        default='gauss314/options-IV-SP500',
    )
    parser.add_argument(
        '--seq-length',
        help='Length of the input window (number of days of history)',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--stride',
        help='Stride between the start of consecutive windows (recommend small, e.g. 1, since '
             'there are far fewer base rows than in the per-ticker dataset)',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--valid-frac',
        help='Fraction of the date range (by quantile of dates present) held out for validation, taken from the end',
        type=float,
        default=0.15,
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory',
        type=str,
        default='./options_iv_index_data',
    )
    parser.add_argument(
        '--cache-dir',
        help='Where to cache the downloaded dataset. If `None`, use the default cache directory of the datasets library',
        type=str,
    )


if __name__ == '__main__':
    main()
