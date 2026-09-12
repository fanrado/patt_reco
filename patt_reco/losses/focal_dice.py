"""Semantic-segmentation loss: focal + soft Dice, with two weightings.

Occupancy is 2-5%, so plain cross-entropy collapses to predicting `empty`
everywhere. Focal handles that.

The second weighting is the one PLAN.md §3.1 missed and §9.9 measured: a ring
covers ~7x the pixels of a track while being a third as common, so a
pixel-averaged loss quietly optimises for rings. Weighting by inverse
*pixels-per-object* rather than inverse class frequency corrects that -- it asks
the model to get each *object* right, not each pixel.

Ambiguous pixels (fed by several objects) are down-weighted by 1/n_contrib:
their label is a convention, not a fact, and they should not drive gradients.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import N_CLASSES


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               class_weight: torch.Tensor | None = None,
               pixel_weight: torch.Tensor | None = None) -> torch.Tensor:
    """logits [N, C, ...], target [N, ...] int64."""
    log_prob = F.log_softmax(logits, dim=1)
    target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    focal_term = (1.0 - target_log_prob.exp()).clamp(min=0).pow(gamma)
    loss = -focal_term * target_log_prob

    if class_weight is not None:
        loss = loss * class_weight.to(loss.device)[target]
    if pixel_weight is not None:
        loss = loss * pixel_weight
        return loss.sum() / pixel_weight.sum().clamp(min=1e-6)
    return loss.mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, n_classes: int = N_CLASSES,
                   pixel_weight: torch.Tensor | None = None, eps: float = 1.0,
                   ignore_empty: bool = True) -> torch.Tensor:
    """Dice over classes, computed on the whole batch rather than per image.

    Per-image Dice is unstable when a class is absent from an image, which is the
    common case here; batch-level Dice keeps the denominator populated.
    """
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, n_classes).permute(
        0, -1, *range(1, target.ndim)).to(probs.dtype)

    if pixel_weight is not None:
        weight = pixel_weight.unsqueeze(1)
        probs, one_hot = probs * weight, one_hot * weight

    dims = [0] + list(range(2, probs.ndim))
    intersection = (probs * one_hot).sum(dims)
    cardinality = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    if ignore_empty:
        dice = dice[1:]
    return 1.0 - dice.mean()


class SemanticLoss(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, focal_gamma: float = 2.0,
                 focal_weight: float = 1.0, dice_weight: float = 1.0,
                 class_weight: np.ndarray | None = None,
                 ambiguity_weighting: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ambiguity_weighting = ambiguity_weighting
        if class_weight is None:
            self.register_buffer("class_weight", None)
        else:
            self.register_buffer("class_weight",
                                 torch.as_tensor(np.asarray(class_weight), dtype=torch.float32))

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        logits = outputs["sem_logits"]                      # [B, V, C, H, W]
        target = batch["semantic"].to(logits.device)        # [B, V, H, W]
        b, v, c = logits.shape[:3]
        logits = logits.reshape(b * v, c, *logits.shape[-2:])
        target = target.reshape(b * v, *target.shape[-2:])

        pixel_weight = None
        if self.ambiguity_weighting and "n_contrib" in batch:
            contrib = batch["n_contrib"].to(logits.device).reshape(b * v, *target.shape[-2:])
            pixel_weight = 1.0 / contrib.clamp(min=1).to(logits.dtype)

        focal = focal_loss(logits, target, self.focal_gamma, self.class_weight, pixel_weight)
        dice = soft_dice_loss(logits, target, self.n_classes, pixel_weight)
        total = self.focal_weight * focal + self.dice_weight * dice
        return total, {"loss": float(total.detach()), "focal": float(focal.detach()),
                       "dice": float(dice.detach())}


def class_weights_from_dataset(reader, n_events: int = 200, n_classes: int = N_CLASSES,
                               scheme: str = "pixels_per_object") -> np.ndarray:
    """Estimate per-class loss weights by scanning a sample of a dataset.

    `pixels_per_object` weights by the inverse of how many pixels an object of
    that class typically occupies -- see the module docstring. `inverse_freq` is
    the conventional choice, kept for comparison. Weights are normalised to mean
    1 over the non-empty classes, and `empty` is pinned to 1.
    """
    # validate before scanning: a typo in the config must fail loudly, not fall
    # through to unit weights because the sample happened to be empty
    if scheme not in ("pixels_per_object", "inverse_freq", "none"):
        raise ValueError(f"unknown class-weight scheme {scheme!r}")
    if scheme == "none":
        return np.ones(n_classes, dtype=np.float32)

    pixels = np.zeros(n_classes, dtype=np.float64)
    objects = np.zeros(n_classes, dtype=np.float64)
    for i in range(min(n_events, len(reader))):
        record = reader[i]
        pixels += np.bincount(record.dense_semantic().reshape(-1), minlength=n_classes)
        objects += np.bincount(record.obj_cls, minlength=n_classes)

    weights = np.ones(n_classes, dtype=np.float64)
    present = np.arange(1, n_classes)[pixels[1:] > 0]
    if present.size == 0:
        return weights.astype(np.float32)

    if scheme == "pixels_per_object":
        per_object = pixels[present] / np.maximum(objects[present], 1.0)
        raw = 1.0 / per_object
    else:                                   # inverse_freq
        raw = 1.0 / pixels[present]

    weights[present] = raw / raw.mean()
    return weights.astype(np.float32)
