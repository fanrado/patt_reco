# patt_reco

Synthetic 2D images of two shapes, for binary pattern recognition:

- **tracks** — a single line, straight or gently curved, at an arbitrary 3D angle;
- **showers** — a branching cascade deposited as a spray of scattered points.

One object per image, rendered to a single greyscale view. The two classes
differ by **shape alone**: track intensity is a flat constant, and shower point
values are drawn from a range purely for variation, so there is no brightness
pattern to learn instead of the geometry.

**There is no detector simulation, no detector response, and no noise.** No
drift, diffusion or attenuation; no wire planes, stereo views or digitisation;
no per-pixel noise of any kind. Variation comes from geometry.

## Layout

| Layer | Module | What it does |
|---|---|---|
| geometry | `geometry/base.py` | `Primitive` base class and vector helpers |
| | `geometry/track.py` | the line primitive (straight or circular arc) |
| | `geometry/shower.py` | the branching scattered-point primitive |
| | `geometry/volume.py` | the axis-aligned box objects are generated in |
| | `geometry/project.py` | orthographic single-view rasteriser |
| dataset | `dataset/generate.py` | one `(seed, index)` → one labelled image |
| | `dataset/build.py` | writes the `train`/`val`/`test` npz splits |
| | `dataset/torch_dataset.py` | `ImageDataset`, a torch `Dataset` over one split |
| viz | `viz/display.py` | `plot_image` and `plot_grid`, greyscale |
| models | `models/cnn.py` | `CNN`, the minimal baseline classifier |
| train | `train/config.py` | `TrainConfig` and its nested blocks |
| | `train/loop.py` | `Trainer`: the loop, and the overfit wiring check |
| | `train/metrics.py` | accuracy, confusion matrix, recall, ROC-AUC |
| | `train/tracking.py` | `Run`: run directory, metrics log, checkpoints |
| config | `config.py` | `SourceConfig` and its nested blocks |
| | `cliutil.py` | `--set` dotted-path config overrides |

Two scripts drive the difficulty study: `scripts/sweep.py` runs the whole
pipeline at a range of difficulties, and `scripts/plot_sweep.py` turns its
results into a figure.

## Install

```bash
pip install -e ".[dev]"
```

`torch` is only needed for `ImageDataset`; install it with `pip install -e ".[dl]"`.

## Usage

Build a labelled dataset from the two shape configs:

```bash
python scripts/make_dataset.py configs/data/tracks.yaml configs/data/showers.yaml \
    -o data/tracks_vs_showers
```

This writes `train.npz`, `val.npz` and `test.npz` — `images` as float32
`[N, H, W]`, `labels` as uint8 `[N]` — plus a `meta.json` recording both source
configs, the class names, the shuffle seed and the per-split counts.

Eyeball one source on its own, without building anything:

```bash
python scripts/preview.py configs/data/tracks.yaml -n 8
```

Each config is standalone, so the two can be previewed and edited
independently. `build_dataset` checks before generating that every source
renders at the same size and that no two claim the same label.

## Classes

| Label | Name |
|---|---|
| 0 | `track` |
| 1 | `shower` |

## Training

Check the wiring before spending time on a real run -- a single convolution
cannot fail to fit one batch for lack of capacity, so if this does not reach
near-zero loss the plumbing is wrong:

```bash
python scripts/train.py configs/train/smoke.yaml --overfit
```

Then a short run that exercises every code path (its accuracy is not
meaningful), and the real one:

```bash
python scripts/train.py configs/train/smoke.yaml
python scripts/train.py configs/train/base.yaml
```

Each run writes `runs/<timestamp>_<name>/` holding `config.json`,
`environment.json`, `metrics.csv` and the `best.pt` / `last.pt` checkpoints.
Training prints the evaluation command when it finishes:

```bash
python scripts/evaluate.py runs/<run>/best.pt --data data/tracks_vs_showers/test.npz
```

That writes `test_report.json` next to the checkpoint, and a
`misclassified.png` grid of the worst mistakes when there are any. A
checkpoint carries its own model config and image size, so evaluation needs
no training config.

### The baseline

One convolution, one max pool, one dense layer, one output layer. It is a
deliberate starting point, not a tuned architecture: the reason to build the
smallest thing that works first is to have something later variants can be
measured against. `pool` is the knob that controls the size, since the dense
layer holds almost all the parameters.

### Metrics

Accuracy, per-class recall, the confusion matrix (rows true, columns
predicted) and ROC-AUC, all computed in `train/metrics.py` with numpy -- no
scikit-learn.

## Difficulty

Any config field can be overridden from the command line, on both
`make_dataset.py` and `train.py`:

```bash
python scripts/make_dataset.py configs/data/tracks.yaml configs/data/showers.yaml \
    --set render.height=32 --set shower.max_nodes=7 -o data/small
python scripts/train.py configs/train/base.yaml --set optim.epochs=4
```

`--set` is repeatable and the value is parsed as YAML, so lists work
(`--set track.curvature='[0.5, 3.0]'`). On `make_dataset.py` one override
list applies to *every* source; because a `SourceConfig` always carries both
a `track` and a `shower` block, an override aimed at the other shape is
simply **inert rather than an error**. On `train.py` overrides are applied
before the run directory is created, so `config.json` records what was
actually trained.

`scripts/sweep.py` uses that to walk one difficulty scalar `d` in [0, 1],
interpolating four fields — `shower.open_angle`, `shower.spread`,
`shower.max_nodes` and `track.curvature` — from the easy setting to a hard
one, and running generate/train/evaluate at each level:

```bash
python scripts/sweep.py --levels 0,0.2,0.4,0.6,0.8,1.0 -o sweeps/overlap
python scripts/plot_sweep.py sweeps/overlap/sweep_results.json
```

### The two config pairs

| pair | role |
|---|---|
| `tracks.yaml` / `showers.yaml` | the easy sanity check: the baseline scores **1.0000** here, so anything less means the pipeline is broken |
| `tracks_hard.yaml` / `showers_hard.yaml` | the benchmark: the configuration models are compared on |

Measured baseline on the hard pair: **test accuracy 0.9995, ROC-AUC 1.0000**
on 2000 images. That is still close to the ceiling — see `PLAN.md` for why
the difficulty axis does not yet separate models, and what would fix it.

## Reproducibility

`(config.seed, index)` fully determines an image. Every draw — the shape
parameters, the deposition and the scatter — comes from one generator seeded
from that pair, so generation is order-independent and bit-identical however
it is run. Each split uses a distinct index range, so no image ever appears in
more than one split.

## Not built yet

Any architecture beyond the baseline, and any study of how much harder the
task can be made. The generator's knobs -- image size, track curvature,
shower opening angle and spread -- control how separable the two classes are,
and how far they can be pushed before the baseline stops solving the task has
not been measured.
