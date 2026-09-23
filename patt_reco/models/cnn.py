"""The baseline classifier, and its depth knob.

`n_blocks=1` is the minimal baseline: one convolution, one pool, one dense
layer, one output layer -- deliberately the smallest thing that could work,
so that any later architecture has a measured baseline to beat rather than an
assumed one.

`n_blocks=k` stacks k conv/pool blocks with the channel count doubling each
time. Depth is the only thing that changes: no batch normalisation, no
residual connections, no change to dropout, so a measured improvement is
attributable to depth alone.

The dense layer dominates the parameter count at shallow depths, and `pool`
is the knob that controls it -- each block divides both spatial dimensions by
`pool`, so the flattened width falls as 1/pool**(2k) while the channel count
only doubles.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import N_CLASSES


class CNN(nn.Module):
    """k x (conv -> ReLU -> pool) -> dense -> output, on a [B, 1, H, W] image."""

    def __init__(self, height: int, width: int, n_filters: int = 16,
                 kernel_size: int = 3, pool: int = 4, hidden: int = 64,
                 dropout: float = 0.0, n_blocks: int = 1):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
        self.height = int(height)
        self.width = int(width)
        self.pool = int(pool)
        self.n_blocks = int(n_blocks)

        pooled_h, pooled_w = self.height, self.width
        for _ in range(self.n_blocks):
            pooled_h //= self.pool
            pooled_w //= self.pool
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(
                f"{self.n_blocks} blocks of pool={self.pool} leave nothing of a "
                f"{self.height}x{self.width} input: reduce n_blocks or pool")

        # The first conv keeps the attribute name it had when there was only
        # one, so n_blocks=1 produces an identical state_dict and every
        # checkpoint written by the baseline still loads.
        self.conv = nn.Conv2d(1, n_filters, kernel_size, padding="same")
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(n_filters * 2 ** (i - 1), n_filters * 2 ** i,
                      kernel_size, padding="same")
            for i in range(1, self.n_blocks)
        ])
        channels = n_filters * 2 ** (self.n_blocks - 1)
        flat = channels * pooled_h * pooled_w

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
        x = self.maxpool(self.relu(self.conv(x)))
        for conv in self.extra_convs:
            x = self.maxpool(self.relu(conv(x)))
        x = self.flatten(x)
        x = self.relu(self.dense(x))
        x = self.dropout(x)
        return self.head(x)

    def n_parameters(self) -> int:
        """Total trainable parameters; the dense layer is most of them."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.height}x{self.width}, "
                f"{self.n_blocks} block{'s' if self.n_blocks > 1 else ''}, "
                f"{self.n_parameters():,} parameters)")
