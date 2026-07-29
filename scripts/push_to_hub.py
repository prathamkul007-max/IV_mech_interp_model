#!/usr/bin/env python

"""
Push an already-trained IV-forecasting checkpoint (and its scaler) to the
Hugging Face Hub, without retraining. Useful when training already happened
in a previous session/run (e.g. --push-to-hub was not set at training time).

The access token is never hardcoded here: pass it via --hf-token, or set
the HF_TOKEN environment variable before running this script.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.hub_utils import push_model_to_hub


def main():
    parser = argparse.ArgumentParser(
        description='Push a trained IV-forecasting checkpoint to the Hugging Face Hub',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pt checkpoint file')
    parser.add_argument('--scaler-file', type=str, required=True, help='Path to the scaler.json file used for training')
    parser.add_argument('--hub-repo-id', type=str, required=True, help='Hugging Face Hub repo id, e.g. "username/my-model"')
    parser.add_argument('--hub-private', action='store_true', help='Create the repo as private')
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='Hugging Face Hub access token. If not provided, falls back to the HF_TOKEN environment variable',
    )
    args = parser.parse_args()

    push_model_to_hub(
        checkpoint_path=args.checkpoint,
        scaler_file=args.scaler_file,
        repo_id=args.hub_repo_id,
        token=args.hf_token,
        private=args.hub_private,
    )


if __name__ == '__main__':
    main()
