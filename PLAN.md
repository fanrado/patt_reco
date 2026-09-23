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

The goal is to make the two classes resemble each other until accuracy leaves
the ceiling, so the benchmark can rank models. `scripts/sweep.py` owns one
scalar `d` in [0, 1] and interpolates generator fields between an easy setting
and a hard one.

### The first design was wrong, instructively

The original axis swept four fields, including `track.curvature` from
[0.0, 2.2] up to [0.5, 3.0]. The stated reason was to remove the
"straight line = track" giveaway. It did the opposite.

Raising the track floor while collapsing the shower's opening angle did not
overlap the classes — **it swapped them**. At d=1 every shower was a
near-straight kinked polyline and every track a visible arc, so a single
feature still separated them perfectly. Curvature simply changed which class
it identified.

**The symptom was a flat curve.** Accuracy never left 0.999 and AUC never left
1.0000 anywhere in d=0..1:

| d | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|
| accuracy | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 0.9990 | 0.9995 |

*(superseded — measured on the inverted axis)*

Worth recording: **no amount of finer sweeping could have helped.** The
instinct on missing a target band is to bisect between the levels that bracket
it, but a flat curve has no bracketing levels. Bisecting it yields nothing. A
flat curve means the axis is wrong, not that its resolution is too coarse.

### What found it

Two diagnostics, neither of them the accuracy number:

- **Previewing the two hard configs side by side.** Hard showers were all
  near-straight; hard tracks all curved. The inversion is obvious in the
  images and invisible in the metric.
- **The misclassified grid from the in-band probe.** 15 of the 16 worst errors
  were gently curved tracks read as showers — the errors concentrating on
  exactly the feature the axis had made decisive.

### The corrected axis

Only the shower moves. `track.curvature` is no longer swept, so tracks keep
their full straight-to-curved range and a tight shower resembles *some* tracks
and not others. Neither "straight" nor "curved" identifies a class.

| field | d=0 | d=1 |
|---|---|---|
| `shower.open_angle` | [0.20, 0.55] | [0.02, 0.06] |
| `shower.spread` | [0.01, 0.03] | [0.001, 0.004] |
| `shower.max_nodes` | 200 | 7 |

Corrected sweep at full scale (4000 train/class) — the curve still does not
bend, and the harness self-check passes with d=0 at 1.0000:

| d | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|
| accuracy | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 0.9990 | 1.0000 |

AUC is 1.0000 at every level.

### The operating point

At d=1.0, varying the training set (n_val 500, n_test 1000 per class):

| n_train / class | accuracy | AUC | wrong |
|---|---|---|---|
| **100** | **0.8325** | 0.9289 | 335 / 2000 |
| 250 | 0.9015 | 0.9689 | 197 / 2000 |
| 500 | 0.9690 | 0.9974 | — |
| 4000 | 1.0000 | 1.0000 | 0 |

**Chosen: d=1.0 with n_train=100 per class.** Not the 250 that sits
mid-band — at 250 the baseline lands on the 90% acceptance threshold, where
run-to-run noise alone would flip a pass/fail verdict and a better model would
have no headroom. 100 puts the baseline clearly below the bar and still inside
the band.

### What this benchmark measures

The fallback to a small training set raised a fair concern: that the benchmark
would rank models by sample efficiency rather than by their ability to resolve
ambiguous shapes. **It measures both, and the shapes really are ambiguous.**

The axis fix did real work independently of the training-set size. At matched
n_train=100 the corrected axis scores 0.8325 against the inverted axis's
0.9035, so it is materially harder on its own. The error *structure* is more
telling than the number:

- **Inverted axis:** errors one-directional and confident (0.72-0.83), gently
  curved tracks read as showers — the curvature giveaway failing.
- **Corrected axis:** errors run in both directions at confidence ~0.50 — the
  model is uncertain, not confidently wrong.

Bidirectional errors at chance confidence are the signature of genuine
ambiguity. The shapes overlap; the small training set is what stops the model
memorising its way past that overlap.

## 7. Open questions

1. **Training is bimodal at this operating point.** Four seeds gave 0.5000,
   0.9360, 0.9160, 0.5000 — half collapse to predicting one class, and seed 0
   is the configured default. The cause is the optimisation budget:
   `optim.epochs=8` was derived at 4000/class where an epoch is 125 steps
   (1000 total); at 250/class an epoch is 8 steps and at 100/class only 4, so
   the same 8 epochs buys 15-30x less optimisation than the measurement
   assumed. Until epochs or batch size is re-derived for this point, the
   benchmark's verdict is decided by the seed rather than by the model. This
   blocks everything below it.
2. **Which architectural change is worth making?** Now answerable in
   principle — there is a baseline at 0.8325 with roughly seven points of
   headroom — but not in practice until question 1 is fixed.
