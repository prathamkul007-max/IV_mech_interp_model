"""Loads real validation windows to use as model input for interpretability analyses.

v1 uses relative day-offset labels only ('t-31' ... 't') since train.npz/valid.npz (produced by
scripts/prepare_options_iv.py / prepare_options_iv_index.py) do not retain per-window date/ticker
metadata -- see README.md for the caveat and possible follow-up.
"""

import numpy as np


def load_sample_windows(npz_path: str, n: int = 8, seed: int = 0) -> np.ndarray:
    """Returns a random sample of n windows from X in `npz_path`, shape (n, seq_length, num_features)."""
    data = np.load(npz_path)
    X = data['X']
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[indices]


def timestep_labels(seq_length: int) -> list[str]:
    """Relative day-offset labels for a window of length seq_length, e.g. ['t-31', ..., 't-1', 't']."""
    return [f't-{seq_length - 1 - i}' if i < seq_length - 1 else 't' for i in range(seq_length)]
