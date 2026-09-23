"""The minimal baseline classifier.

One convolution, one pool, one dense layer, one output layer -- deliberately
the smallest thing that could work, so that any later architecture has a
measured baseline to beat rather than an assumed one.

The dense layer dominates the parameter count, and `pool` is the only knob
that controls it: the flattened width scales as 1/pool**2.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import N_CLASSES


class CNN(nn.Module):
    """conv -> pool -> dense -> output, on a [B, 1, H, W] image."""

    def __init__(self, height: int, width: int, n_filters: int = 16,
                 kernel_size: int = 3, pool: int = 4, hidden: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.pool = int(pool)

        pooled_h = self.height // self.pool
        pooled_w = self.width // self.pool
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(
                f"pool={self.pool} is larger than the {self.height}x{self.width} "
                f"input: nothing would survive pooling")
        flat = n_filters * pooled_h * pooled_w

        self.conv = nn.Conv2d(1, n_filters, kernel_size, padding="same")
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.pool)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flat, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"expected input of size {self.height}x{self.width}, got "
                f"{tuple(x.shape[-2:])}")
        x = self.relu(self.conv(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.dense(x))
        x = self.dropout(x)
        return self.head(x)

    def n_parameters(self) -> int:
        """Total trainable parameters; the dense layer is most of them."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.height}x{self.width}, "
                f"{self.n_parameters():,} parameters)")
