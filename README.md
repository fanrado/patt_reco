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
| config | `config.py` | `SourceConfig` and its nested blocks |

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

## Reproducibility

`(config.seed, index)` fully determines an image. Every draw — the shape
parameters, the deposition and the scatter — comes from one generator seeded
from that pair, so generation is order-independent and bit-identical however
it is run. Each split uses a distinct index range, so no image ever appears in
more than one split.

## Not built yet

The CNN classifier and its training loop are the next piece of work. This
repository currently generates and stores the data, and nothing more.
