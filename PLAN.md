# `patt_reco` — plan

**Owner:** fanrado
**Last marked:** 2026-09-23

## 1. Goal

Separate track-like from shower-like images with a deep learning classifier.
Binary classification on single-object greyscale images; the signal is shape.

## 2. Data

Two shape classes, generated in 3D and projected to a **single** 2D view:

- **track** (label 0) — one line, straight through to clearly curved, at an
  arbitrary 3D angle. Curvature is a plain geometric parameter (1/radius).
- **shower** (label 1) — a branching cascade grown breadth-first from an apex
  and deposited as transversely scattered points, so it reads as a spray
  rather than a fan of clean lines.

No detector, no detector response, no noise. Variation comes from geometry
alone. Per-point values carry no physical interpretation: a track's value is a
flat constant, and a shower's values are drawn from a range purely to vary
between images. Neither class can be identified from an intensity pattern.

Each class is described by its own standalone config — `configs/data/tracks.yaml`
and `configs/data/showers.yaml` — which can be read, edited and previewed
independently.

## 3. Storage

`scripts/make_dataset.py` turns one or more configs into `train.npz`,
`val.npz` and `test.npz`: `images` float32 `[N, H, W]`, `labels` uint8 `[N]`,
alongside a `meta.json` recording the source configs, class names, shuffle
seed and per-split counts.

`(config.seed, index)` fully determines an image, and each split draws from a
distinct index range, so splits never overlap and generation is bit-identical
however it is run.

## 4. Status

The generator is rebuilt and clean: geometry primitives, the rasteriser, the
npz dataset builder, a torch `Dataset` over a split, and greyscale display.
The classifier and training loop are **not built**.

## 5. Next

- a small CNN classifier;
- a simple training loop;
- accuracy and confusion metrics.

To be planned separately, once the generated images look right.
