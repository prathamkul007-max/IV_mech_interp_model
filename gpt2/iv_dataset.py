import numpy as np
import torch
from torch.utils.data import Dataset


class IVDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """A map-style dataset for training the IV-forecasting model.

    Expects an .npz file with two arrays:
      - X: (num_windows, seq_length, num_features) float32
      - Y: (num_windows, seq_length, num_targets) float32
    """
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.inputs = data['X']
        self.targets = data['Y']
        if len(self.inputs) != len(self.targets):
            raise ValueError('X and Y must have the same number of windows')

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inputs = torch.from_numpy(self.inputs[idx]).float()
        targets = torch.from_numpy(self.targets[idx]).float()
        return inputs, targets
