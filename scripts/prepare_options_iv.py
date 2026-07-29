#!/usr/bin/env python

"""
Download and window the options-IV-SP500 dataset
(https://huggingface.co/datasets/gauss314/options-IV-SP500) for training
the per-stock IV-forecasting model (see gpt2/iv_model.py).

Each stock ticker's daily history is windowed into overlapping sequences of
`seq_length` days; the model is trained to predict, for every day in the
window, the next day's 7 implied-volatility buckets (DITM, ITM, sITM, ATM,
sOTM, OTM, DOTM).
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


def prepare_options_iv(args: argparse.Namespace) -> None:
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

    df = clean_and_prepare(df, ticker_col, date_col, feature_cols, target_cols, args.seq_length)

    # time-respecting split: a single global date cutoff for the whole dataset
    cutoff_date = df[date_col].quantile(1.0 - args.valid_frac, interpolation='nearest')
    print(f'Validation cutoff date: {cutoff_date}')

    train_features_list = []
    for _, ticker_df in df[df[date_col] <= cutoff_date].groupby(ticker_col):
        train_features_list.append(ticker_df[feature_cols].to_numpy(dtype=np.float32))
    train_features_concat = np.concatenate(train_features_list, axis=0)

    scaler = fit_scaler(train_features_concat, feature_cols, target_cols)
    save_scaler(scaler, args.seq_length, os.path.join(args.output_dir, 'scaler.json'))
    target_indices = scaler['target_indices']

    train_inputs, train_targets = [], []
    valid_inputs, valid_targets = [], []
    for _, ticker_df in df.groupby(ticker_col):
        train_rows = ticker_df[ticker_df[date_col] <= cutoff_date]
        valid_rows = ticker_df[ticker_df[date_col] > cutoff_date]

        if len(train_rows) >= args.seq_length + 1:
            features = apply_scaler(train_rows[feature_cols].to_numpy(dtype=np.float32), scaler)
            inputs, targets = make_windows(features, target_indices, args.seq_length, args.stride)
            train_inputs.append(inputs)
            train_targets.append(targets)

        if len(valid_rows) >= args.seq_length + 1:
            features = apply_scaler(valid_rows[feature_cols].to_numpy(dtype=np.float32), scaler)
            inputs, targets = make_windows(features, target_indices, args.seq_length, args.stride)
            valid_inputs.append(inputs)
            valid_targets.append(targets)

    train_X = np.concatenate(train_inputs, axis=0)
    train_Y = np.concatenate(train_targets, axis=0)
    valid_X = np.concatenate(valid_inputs, axis=0)
    valid_Y = np.concatenate(valid_targets, axis=0)

    print(f'Train windows: {train_X.shape}, valid windows: {valid_X.shape}')

    np.savez(os.path.join(args.output_dir, 'train.npz'), X=train_X, Y=train_Y)
    np.savez(os.path.join(args.output_dir, 'valid.npz'), X=valid_X, Y=valid_Y)
    print(f'Saved train.npz, valid.npz, scaler.json to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the options-IV-SP500 dataset for per-stock IV forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()
    prepare_options_iv(args)


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
        help='Stride between the start of consecutive windows for the same ticker',
        type=int,
        default=5,
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
        default='./options_iv_data',
    )
    parser.add_argument(
        '--cache-dir',
        help='Where to cache the downloaded dataset. If `None`, use the default cache directory of the datasets library',
        type=str,
    )


if __name__ == '__main__':
    main()
