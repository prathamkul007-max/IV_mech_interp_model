#!/usr/bin/env python

"""
Load a trained IVModel checkpoint (and its feature scaler) from the
Hugging Face Hub, e.g. Pratham007xo/iv-forecast-constituent-124m or
Pratham007xo/iv-forecast-index-124m.

Note: the checkpoint dict may also contain a 'scaler' key holding AMP
GradScaler state (only if mixed-precision training was used) -- that is
unrelated to the feature-normalization scaler.json returned here.
"""

import argparse
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.iv_model import IVModel, IVModelConfig
from gpt2.model import get_device


def _find_checkpoint_filename(repo_id: str) -> str:
    files = list_repo_files(repo_id)
    checkpoints = [f for f in files if f.startswith('iv_model-') and f.endswith('.pt')]
    if not checkpoints:
        raise ValueError(f'No iv_model-*.pt checkpoint found in {repo_id} (files: {files})')
    return max(checkpoints, key=lambda f: int(f.removeprefix('iv_model-').removesuffix('.pt')))


def load_iv_checkpoint(repo_id: str, cache_dir: str | None = None, device: str = 'auto') -> tuple[IVModel, dict]:
    """Downloads the checkpoint + scaler.json for `repo_id` and returns (model, scaler_dict).

    `scaler_dict` is the feature-normalization scaler (feature_names/mean/std/target_indices/
    seq_length), not the AMP scaler that may also live inside the checkpoint's 'scaler' key.
    """
    checkpoint_filename = _find_checkpoint_filename(repo_id)
    ckpt_path = hf_hub_download(repo_id, filename=checkpoint_filename, cache_dir=cache_dir)
    scaler_path = hf_hub_download(repo_id, filename='scaler.json', cache_dir=cache_dir)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = IVModelConfig(**checkpoint['config'])
    model = IVModel(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(get_device(device))

    with open(scaler_path) as f:
        scaler = json.load(f)

    return model, scaler


def main():
    parser = argparse.ArgumentParser(
        description='Smoke-test loading an IVModel checkpoint from the Hugging Face Hub',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--repo-id', type=str, required=True, help='e.g. Pratham007xo/iv-forecast-constituent-124m')
    parser.add_argument('--cache-dir', type=str, default=None)
    args = parser.parse_args()

    model, scaler = load_iv_checkpoint(args.repo_id, cache_dir=args.cache_dir)
    print(f'Loaded model config: {model.config}')
    print(f'Num parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print(f'Num features in scaler: {len(scaler["feature_names"])}')
    print(f'Target indices: {scaler["target_indices"]}')
    print(f'Target bucket names: {[scaler["feature_names"][i] for i in scaler["target_indices"]]}')


if __name__ == '__main__':
    main()
