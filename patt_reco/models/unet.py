"""U-Net for per-view semantic segmentation.

One view at a time through a **shared-weight** encoder-decoder. Views are not
stacked as channels: they have different geometry, so a filter that means one
thing in the 0-degree view means something else in the +60-degree view. Passing
them through the same weights separately is the correct siamese treatment and
also makes the model independent of how many views a detector has.

GroupNorm rather than BatchNorm, because the 4 GB budget forces small batches.
Bilinear upsampling rather than transposed convolution: same quality here, fewer
parameters, no checkerboard artefacts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import N_CLASSES


def _norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max(1, min(max_groups, channels // 4))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _norm(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Returns `sem_logits` of shape [B, V, C, H, W]."""

    def __init__(self, n_classes: int = N_CLASSES, in_channels: int = 1,
                 base_width: int = 32, depth: int = 4):
        super().__init__()
        self.n_classes = n_classes
        widths = [base_width * 2**i for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(widths[i], widths[i + 1]))
            for i in range(depth))
        self.ups = nn.ModuleList(
            DoubleConv(widths[depth - i] + widths[depth - i - 1], widths[depth - i - 1])
            for i in range(depth))
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward_view(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        out = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = up(torch.cat([out, skip], dim=1))
        return self.head(out)

    def forward(self, batch: dict) -> dict:
        views = batch["views"] if isinstance(batch, dict) else batch
        b, v = views.shape[:2]
        # fold views into the batch axis: that *is* the weight sharing. The
        # channels_last conversion happens here, not on the caller's [B,V,1,H,W]
        # input -- that layout is only defined for rank-4 tensors.
        flat = views.reshape(b * v, *views.shape[2:]).contiguous(
            memory_format=torch.channels_last)
        logits = self.forward_view(flat)
        return {"sem_logits": logits.reshape(b, v, self.n_classes, *logits.shape[-2:])}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
