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

Both halves are built and work end to end: generate, preview, build splits,
train, evaluate, sweep.

**The minimal baseline saturates the original configs.** A 2-epoch smoke run
reaches test accuracy 1.0000 and ROC-AUC 1.0000 with confusion
`[[1000, 0], [0, 1000]]` on 2000 images. The consequence is the important
part: **accuracy on the easy set cannot rank models**, which is exactly what
the minimal baseline was built to enable. Any later architecture also scores
1.0, so nothing can be compared.

The easy configs (`tracks.yaml`, `showers.yaml`) are kept deliberately. They
are the wiring check: a change that breaks the pipeline shows up there
immediately as an accuracy below 1.0.

## 6. The difficulty axis

`scripts/sweep.py` owns one scalar `d` in [0, 1] and interpolates four
generator fields between the easy setting and a hard one:

| field | d=0 | d=1 |
|---|---|---|
| `shower.open_angle` | [0.20, 0.55] | [0.02, 0.06] |
| `shower.spread` | [0.01, 0.03] | [0.001, 0.004] |
| `shower.max_nodes` | 200 | 7 |
| `track.curvature` | [0.0, 2.2] | [0.5, 3.0] |

The track curvature *floor* is raised off zero on purpose: leaving perfectly
straight lines in the set gives the model a giveaway feature that has nothing
to do with the shower side.

### The operating point, and what it does not achieve

`tracks_hard.yaml` / `showers_hard.yaml` freeze d=1.0. **Measured baseline
there: test accuracy 0.9995, ROC-AUC 1.0000 on 2000 images.** That is the
reference number later architectures are compared against, and it is still
saturated — the 0.75-0.95 target band was not reached.

The sweep is flat, not merely shallow: accuracy never falls below 0.999 and
AUC never leaves 1.0000 anywhere in d=0..1, so there are no bracketing levels
to bisect. The cause is visible in the previews: **at d=1 the axis inverts
the giveaway rather than removing it.** A 7-node, barely-opening shower
renders as a nearly straight polyline while every hard track is a strong arc,
so curvature alone separates the classes perfectly — "straight = track" has
simply become "straight = shower".

Training-set size does move the number. At d=1.0, measured test accuracy by
training images per class: 100 -> 0.9035, 250 -> 0.9460, 500 -> 0.9965,
1000 -> 0.9985, 4000 -> 0.9995. So 100-250 per class lands in the band.

## 7. Open questions

1. **Repair the difficulty axis.** Either drop the `track.curvature` row so
   tracks keep spanning straight-to-curved while showers tighten, or cap its
   floor near zero. As it stands the axis trades one clean separator for
   another, which is why no level is hard. Re-run the sweep afterwards to
   confirm the curve actually bends.
2. **Which architectural change is worth making?** Still not answerable, but
   for a narrower reason than before: there is now a measurable number
   (0.9995 at the operating point), it is just too close to the ceiling to
   rank anything. It becomes answerable once question 1 lands, or immediately
   if the benchmark adopts n_train=100 per class, which measures 0.9035.
