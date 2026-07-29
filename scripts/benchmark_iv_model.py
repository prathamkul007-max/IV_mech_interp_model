#!/usr/bin/env python

"""
Benchmark a trained IV-forecasting checkpoint on a held-out (validation) set.

Reports, since this is a regression task with no single standard notion of
"accuracy":
  - directional accuracy: % of predictions that correctly call whether
    tomorrow's IV goes up or down relative to today, per IV bucket and
    overall.
  - tolerance-band accuracy: % of predictions within a relative tolerance
    (e.g. +/-10%) of the true value, per IV bucket and overall.
  - RMSE / MAE in real (unscaled) IV units, for reference.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.iv_dataset import IVDataset
from gpt2.iv_model import IVModel, IVModelConfig
from gpt2.run_pretrain_iv import load_scaler


def unscale(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std + mean


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    today_values: np.ndarray,
    tolerance: float,
    bucket_names: list[str],
) -> dict:
    """All arrays have shape (N, seq_length, num_targets), already unscaled."""
    abs_error = np.abs(predictions - targets)
    rel_error = abs_error / np.clip(np.abs(targets), 1e-6, None)

    actual_direction = np.sign(targets - today_values)
    predicted_direction = np.sign(predictions - today_values)
    moved = actual_direction != 0

    def per_bucket_and_overall(mask_fn):
        per_bucket = {}
        for i, name in enumerate(bucket_names):
            per_bucket[name] = mask_fn(i)
        overall = mask_fn(None)
        return per_bucket, overall

    def tolerance_pct(bucket_idx):
        vals = rel_error if bucket_idx is None else rel_error[..., bucket_idx]
        return float((vals <= tolerance).mean() * 100)

    def directional_pct(bucket_idx):
        if bucket_idx is None:
            mask = moved
            correct = actual_direction == predicted_direction
        else:
            mask = moved[..., bucket_idx]
            correct = actual_direction[..., bucket_idx] == predicted_direction[..., bucket_idx]
        if mask.sum() == 0:
            return float('nan')
        return float(correct[mask].mean() * 100)

    def rmse(bucket_idx):
        vals = abs_error if bucket_idx is None else abs_error[..., bucket_idx]
        return float(np.sqrt((vals ** 2).mean()))

    def mae(bucket_idx):
        vals = abs_error if bucket_idx is None else abs_error[..., bucket_idx]
        return float(vals.mean())

    tolerance_per_bucket, tolerance_overall = per_bucket_and_overall(tolerance_pct)
    directional_per_bucket, directional_overall = per_bucket_and_overall(directional_pct)
    rmse_per_bucket, rmse_overall = per_bucket_and_overall(rmse)
    mae_per_bucket, mae_overall = per_bucket_and_overall(mae)

    return {
        'tolerance': tolerance,
        'num_predictions': int(predictions.size // len(bucket_names)),
        'overall': {
            'tolerance_accuracy_pct': tolerance_overall,
            'directional_accuracy_pct': directional_overall,
            'rmse': rmse_overall,
            'mae': mae_overall,
        },
        'per_bucket': {
            name: {
                'tolerance_accuracy_pct': tolerance_per_bucket[name],
                'directional_accuracy_pct': directional_per_bucket[name],
                'rmse': rmse_per_bucket[name],
                'mae': mae_per_bucket[name],
            }
            for name in bucket_names
        },
    }


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = load_scaler(args.scaler_file)
    target_indices = scaler['target_indices']
    mean = np.asarray(scaler['mean'], dtype=np.float32)[target_indices]
    std = np.asarray(scaler['std'], dtype=np.float32)[target_indices]
    bucket_names = [scaler['feature_names'][i] for i in target_indices]

    state = torch.load(args.checkpoint, map_location=device)
    model_config = IVModelConfig(**state['config'])
    model = IVModel(model_config).to(device)
    model.load_state_dict(state['model'])
    model.eval()

    dataset = IVDataset(args.valid_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_predictions, all_targets, all_today = [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        predictions = model(inputs).cpu().numpy()
        today = inputs[..., target_indices].cpu().numpy()

        all_predictions.append(unscale(predictions, mean, std))
        all_targets.append(unscale(targets.numpy(), mean, std))
        all_today.append(unscale(today, mean, std))

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    today = np.concatenate(all_today, axis=0)

    return compute_metrics(predictions, targets, today, args.tolerance, bucket_names)


def print_report(name: str, results: dict) -> None:
    print(f'\n=== {name} ===')
    print(f"Validation predictions: {results['num_predictions']} (tolerance = +/-{results['tolerance'] * 100:.0f}%)")
    overall = results['overall']
    print(
        f"Overall: directional accuracy = {overall['directional_accuracy_pct']:.2f}% | "
        f"tolerance-band accuracy = {overall['tolerance_accuracy_pct']:.2f}% | "
        f"RMSE = {overall['rmse']:.4f} | MAE = {overall['mae']:.4f}"
    )
    print(f"{'bucket':<8} {'dir. acc %':>12} {'tol. acc %':>12} {'RMSE':>10} {'MAE':>10}")
    for bucket, metrics in results['per_bucket'].items():
        print(
            f"{bucket:<8} {metrics['directional_accuracy_pct']:>12.2f} "
            f"{metrics['tolerance_accuracy_pct']:>12.2f} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark a trained IV-forecasting checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pt checkpoint file')
    parser.add_argument('--valid-file', type=str, required=True, help='Path to the validation .npz file')
    parser.add_argument('--scaler-file', type=str, required=True, help='Path to the scaler.json file')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for running inference')
    parser.add_argument(
        '--tolerance', type=float, default=0.10,
        help='Relative tolerance for tolerance-band accuracy (0.10 = predictions within +/-10%% of the true value count as correct)',
    )
    parser.add_argument('--output-json', type=str, default=None, help='If set, write the full results as JSON to this path')
    parser.add_argument('--name', type=str, default=None, help='Label for the printed report (defaults to the checkpoint path)')
    args = parser.parse_args()

    results = benchmark(args)
    print_report(args.name or args.checkpoint, results)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved results to {args.output_json}')


if __name__ == '__main__':
    main()
