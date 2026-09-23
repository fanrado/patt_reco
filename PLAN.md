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

## 4. Model and training

The baseline classifier is deliberately the smallest thing that could work:
one convolution, one max pool, one dense layer, one output layer. It is not
tuned. The point of building it first is to have a *measured* baseline that
later architectures can be compared against, rather than an assumed one.

- **Loss** — cross-entropy over `N_CLASSES` logits, not a single logit, so a
  third shape class stays a config change rather than a rewrite.
- **Data** — the npz splits, read through `ImageDataset`.
- **Optimisation** — AdamW, a cosine schedule over the whole run, gradient
  clipping. No mixed precision or accumulation; the model is tiny.
- **Metrics** — accuracy, per-class recall, the confusion matrix and ROC-AUC,
  all computed in-repo.

Every run writes a directory holding its config, its environment, a metrics
CSV and checkpoints. A checkpoint carries the model config and image size, so
it can be evaluated without the training config.

## 5. Status

The generator and the training half are both built and working end to end:
generate, preview, build splits, train, evaluate.

**The baseline already saturates the task.** On the 128x128 set it reaches
accuracy 1.0000 and ROC-AUC 1.0000 on the 2000-image test split, with a
perfectly diagonal confusion matrix, after a single epoch. That is a real
result, not a bug: as configured, the two classes are trivially separable.

## 6. Open questions

Both are for after the baseline runs, and the first depends on the second.

1. **Which architectural change is worth making?** To be decided against the
   measured baseline rather than guessed. This question is currently
   unanswerable: accuracy is saturated, so every candidate scores 1.0 and
   nothing can be ranked.
2. **How hard can the task be made before the model fails?** The knobs are
   smaller images, shorter tracks, a tighter shower opening angle and less
   transverse spread, and fewer training images. Near-perfect accuracy is a
   reason to move here, not a reason to stop — until the task discriminates
   between models, question 1 has no measurable answer.
