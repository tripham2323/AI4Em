"""Stateless-per-window, unidirectional three-class LSTM classifier."""
from __future__ import annotations

import math

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Consume scaled [B,100,F] windows and return unnormalized [B,3] logits.

    Every call starts from PyTorch's zero hidden/cell state. No state is carried
    between overlapping windows or between videos. Internal LSTM dropout is
    always zero; the only dropout is in the classification head.
    """

    def __init__(self, feature_names: tuple[str, ...], *, hidden_size: int = 64,
                 num_layers: int = 1, head_dropout: float = .3):
        super().__init__()
        names = tuple(feature_names)
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise ValueError("Feature names must be nonempty unique strings in training order")
        for name, value in (("hidden_size", hidden_size), ("num_layers", num_layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(head_dropout, (int, float)) or not math.isfinite(head_dropout) or not 0 <= head_dropout < 1:
            raise ValueError("head_dropout must be finite and in [0,1)")
        self.feature_names = names
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_dropout = float(head_dropout)
        self.lstm = nn.LSTM(len(names), hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False, dropout=0.)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                  nn.Dropout(self.head_dropout), nn.Linear(32, 3))

    @property
    def architecture(self) -> dict:
        return {"type": "LSTMClassifier", "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "head_dropout": self.head_dropout,
                "head_hidden_size": 32, "bidirectional": False,
                "internal_dropout": 0., "sequence_length": 100, "num_classes": 3}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (100, len(self.feature_names)):
            raise ValueError(f"Expected tensor [B,100,{len(self.feature_names)}]")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input windows must contain finite floating-point scaled values")
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])
