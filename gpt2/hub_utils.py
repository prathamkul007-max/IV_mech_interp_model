"""Helpers for pushing trained IV-forecasting checkpoints to the Hugging Face Hub.

The access token is never read from a hardcoded value in this codebase:
pass it explicitly via --hf-token, or set the HF_TOKEN environment variable.
"""

import os
from typing import Optional


def push_model_to_hub(
    checkpoint_path: str,
    scaler_file: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
) -> None:
    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError(
            'No Hugging Face token provided. Pass --hf-token or set the HF_TOKEN environment variable.'
        )

    create_repo(repo_id, token=token, private=private, exist_ok=True)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=os.path.basename(checkpoint_path),
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=scaler_file,
        path_in_repo='scaler.json',
        repo_id=repo_id,
    )
    print(f'Pushed {checkpoint_path} and {scaler_file} to https://huggingface.co/{repo_id}')
