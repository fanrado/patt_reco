"""Classical baselines -- built before any network, and for two reasons.

They set the floor a learned model has to clear. And they are a generator check:
if a plain Hough transform reconstructs an isolated L0 track, fine; if it
reconstructs a *busy* L2 event, the generator is too easy and needs fixing
before any network work is worth doing (see PLAN.md §7).

No sklearn/skimage dependency -- the Hough transform and the clustering are
short enough to own, and owning them keeps the floor honest and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..config import EMPTY, N_CLASSES, TRACK
from ..dataset.preprocess import normalise

# 8-connectivity: diagonal neighbours belong to the same object, which matters
# for thin diagonal tracks -- 4-connectivity shatters them into dashes
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


@dataclass
class Prediction:
    """What every model in this project returns, classical or learned."""

    semantic: np.ndarray        # [V, H, W] uint8
    instance: np.ndarray        # [V, H, W] int16, -1 where predicted empty

    @property
    def foreground(self) -> np.ndarray:
        return self.semantic != EMPTY


class Baseline:
    name = "baseline"

    def predict(self, adc: np.ndarray) -> Prediction:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# foreground
# --------------------------------------------------------------------------- #


class ThresholdBaseline(Baseline):
    """n-sigma threshold, every surviving pixel called `track`.

    The floor for the foreground/background half of the task. Its mIoU is
    meaningless by construction -- it cannot tell classes apart -- which is
    exactly why foreground IoU is reported separately from mIoU everywhere.
    """

    name = "threshold"

    def __init__(self, n_sigma: float = 3.0, default_class: int = TRACK):
        # 3 sigma is the measured optimum on L2. Note it is *not* optimal on a
        # noise-free set: there the MAD estimate is zero, the preprocess floor
        # turns the cut into an absolute 3 ADC, and the optimum sits near 8.
        # See PLAN.md 9.9 -- and scan the threshold before quoting a floor.
        self.n_sigma = n_sigma
        self.default_class = default_class

    def mask(self, adc: np.ndarray) -> np.ndarray:
        return normalise(adc) > self.n_sigma

    def predict(self, adc: np.ndarray) -> Prediction:
        mask = self.mask(adc)
        semantic = np.where(mask, self.default_class, EMPTY).astype(np.uint8)
        instance = np.where(mask, 0, -1).astype(np.int16)
        return Prediction(semantic, instance)


# --------------------------------------------------------------------------- #
# lines
# --------------------------------------------------------------------------- #


def hough_lines(mask: np.ndarray, n_angles: int = 180, n_lines: int = 12,
                min_votes: int = 25, tolerance: float = 1.5) -> np.ndarray:
    """Return a boolean mask of pixels lying on the strongest straight lines.

    Standard (rho, theta) accumulator. Peaks are taken greedily with local
    suppression so one strong line does not claim several neighbouring bins.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros_like(mask, dtype=bool)

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_angles, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    diag = int(np.ceil(np.hypot(*mask.shape)))
    rho_offset = diag
    n_rho = 2 * diag + 1

    # [n_points, n_angles] -> accumulate into [n_rho, n_angles]
    rho = np.round(rows[:, None] * cos_t[None, :] + cols[:, None] * sin_t[None, :]).astype(np.int64)
    rho += rho_offset
    angle_index = np.broadcast_to(np.arange(n_angles), rho.shape)
    flat = rho.reshape(-1) * n_angles + angle_index.reshape(-1)
    accumulator = np.bincount(flat, minlength=n_rho * n_angles).reshape(n_rho, n_angles)

    out = np.zeros_like(mask, dtype=bool)
    work = accumulator.copy()
    for _ in range(n_lines):
        peak = int(work.argmax())
        votes = work.flat[peak]
        if votes < min_votes:
            break
        r_index, a_index = divmod(peak, n_angles)
        distance = np.abs(rows * cos_t[a_index] + cols * sin_t[a_index]
                          - (r_index - rho_offset))
        out[rows[distance <= tolerance], cols[distance <= tolerance]] = True
        # suppress this peak's neighbourhood so the next iteration finds a new line
        r0, r1 = max(0, r_index - 3), min(n_rho, r_index + 4)
        a0, a1 = max(0, a_index - 3), min(n_angles, a_index + 4)
        work[r0:r1, a0:a1] = 0
    return out


class HoughBaseline(ThresholdBaseline):
    """Threshold, then call anything on a strong straight line a `track`."""

    name = "hough"

    def __init__(self, n_sigma: float = 3.0, line_class: int = TRACK,
                 other_class: int = TRACK, **hough_kwargs):
        super().__init__(n_sigma=n_sigma, default_class=other_class)
        self.line_class = line_class
        self.hough_kwargs = hough_kwargs

    def predict(self, adc: np.ndarray) -> Prediction:
        mask = self.mask(adc)
        semantic = np.where(mask, self.default_class, EMPTY).astype(np.uint8)
        instance = np.full(mask.shape, -1, dtype=np.int16)
        next_id = 0
        for v in range(mask.shape[0]):
            lines = hough_lines(mask[v], **self.hough_kwargs)
            semantic[v][lines] = self.line_class

            # every foreground pixel gets an instance, not just the ones on a
            # line: pixels on a line are grouped by that line, the rest fall back
            # to connected components. Otherwise the semantic and instance
            # outputs disagree about which pixels are occupied.
            on_line, n_lines = ndimage.label(lines, structure=CONNECTIVITY_8)
            leftover = mask[v] & ~lines
            rest, n_rest = ndimage.label(leftover, structure=CONNECTIVITY_8)

            view_instance = np.full(mask.shape[1:], -1, dtype=np.int32)
            view_instance[on_line > 0] = on_line[on_line > 0] + next_id - 1
            view_instance[rest > 0] = rest[rest > 0] + next_id + n_lines - 1
            instance[v] = view_instance
            next_id += n_lines + n_rest
        return Prediction(semantic, instance)


# --------------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------------- #


class ConnectedComponents(ThresholdBaseline):
    """Threshold, then 8-connected components. The instance-segmentation floor.

    It cannot separate two objects that touch -- which is precisely what the
    busy-event benchmark is about, so the gap between this and a learned model
    is the thing worth measuring.
    """

    name = "connected"

    def predict(self, adc: np.ndarray) -> Prediction:
        mask = self.mask(adc)
        semantic = np.where(mask, self.default_class, EMPTY).astype(np.uint8)
        instance = np.full(mask.shape, -1, dtype=np.int16)
        next_id = 0
        for v in range(mask.shape[0]):
            labelled, n = ndimage.label(mask[v], structure=CONNECTIVITY_8)
            instance[v] = np.where(labelled > 0, labelled + next_id - 1, -1)
            next_id += n
        return Prediction(semantic, instance)


BASELINES = {
    "threshold": ThresholdBaseline,
    "hough": HoughBaseline,
    "connected": ConnectedComponents,
}
