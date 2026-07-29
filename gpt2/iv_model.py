"""A GPT-2 style causal decoder repurposed for numeric time-series forecasting.

Consumes continuous per-day feature vectors instead of token ids, and predicts
continuous target values instead of a distribution over a vocabulary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from gpt2.model import GPTDecoderBlock, LayerNorm


@dataclass
class IVModelConfig:
    num_features: int
    num_targets: int
    seq_length: int = 32
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    eps: float = 1e-7
    activation: str = 'gelu'


class IVModel(nn.Module):
    def __init__(self, config: IVModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.d_model)
        self.positional_embedding = nn.Embedding(config.seq_length, config.d_model)
        self.pe_dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.Sequential(*[GPTDecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = LayerNorm(config.d_model, eps=config.eps)
        self.regression_head = nn.Linear(config.d_model, config.num_targets)

        self.post_init()

    def forward(self, features: Tensor) -> Tensor:
        batch_size, seq_length, _ = features.size()
        feature_embeddings = self.input_projection(features)
        pos = torch.arange(0, seq_length, dtype=torch.int64, device=features.device)
        pos_embeddings = self.positional_embedding(pos)
        x = self.pe_dropout(feature_embeddings + pos_embeddings)
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        predictions = self.regression_head(x)  # (batch_size, seq_length, num_targets)
        return predictions

    def post_init(self) -> None:
        self._init_model_weights()

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: self._init_weights(module, std=std))

        # as in GPT-2 paper, weights of residual layers at initialization are scaled
        # by a factor of 1/sqrt(N) where N is the number of residual layers,
        # in this case N is equal to 2 * num_layers
        scaling_factor = 1 / math.sqrt(2 * self.config.num_layers)
        for param_name, param in self.named_parameters():
            if param_name.endswith('rl_projection.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=std * scaling_factor)

    def _init_weights(self, module, std: float = 0.02):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
