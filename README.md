# patt_reco

An **experiment-independent pattern-recognition benchmark** for particle-physics
detectors: synthetic 3D events rendered into 2D stereo readout views, with exact
per-pixel semantic and instance truth.

The premise is in [PLAN.md](PLAN.md). The short version: the patterns experiments
actually reconstruct — lines, scattered lines, arcs, rings, branching cones,
blobs — are not experiment-specific, and neither is the difficulty, which comes
from projecting them into a few 2D views, from overlap, and from noise. So build
the task once, properly, and let experiments adapt instead of restarting.

## Status

**M1 complete** (the generator) and **M2 in place** (classical baselines, a
semantic U-Net, training, evaluation and the busy-event validation harness).
Instance segmentation (M3) and the transformer arm (M4) are not built; see
[PLAN.md §6](PLAN.md).

```
layer        module                     what it does
---------------------------------------------------------------------------
geometry     patt_reco/geometry/        3D primitives, event composition
detector     patt_reco/detector/        3D -> 2D projection, transport, response
noise        patt_reco/noise/           incoherent, coherent, 1/f, artefacts
dataset      patt_reco/dataset/         seeds, sparse storage, HDF5 shards, torch Dataset
models       patt_reco/models/          U-Net, registry, classical baselines
losses       patt_reco/losses/          focal + Dice, class and ambiguity weighting
train        patt_reco/train/           loop, config, run tracking
eval         patt_reco/eval/            semantic and differential metrics
viz          patt_reco/viz/             three-view event displays
```

## The four phases

One script per phase. Each prints the command for the next one.

```bash
# 1. data generation -- the whole ladder, with train/val/test splits
python scripts/generate.py --all -j 8

# 2. model training
python scripts/train.py configs/train/base.yaml --overfit   # wiring check first
python scripts/train.py configs/train/base.yaml

# 3. model testing -- held out, same distribution as training
python scripts/test.py runs/<run>/best.pt --data data/l2_noise/test --baselines

# 4. model validation -- busy events and out of distribution
python scripts/validate.py runs/<run>/best.pt --snr 0.5,1,2,4 --baselines
```

**Phase 3 and phase 4 are different questions.** Phase 3 asks whether the model
learned the training distribution. Phase 4 asks whether it learned the *task* --
crowded events it never trained on, shapes it never saw, and a detector outside
its training envelope. A model can do well at the first and fail the second, and
that gap is the whole point of the project.

Splits are separate **seeds**, not slices: since `(seed, index)` determines an
event, different seeds cannot overlap, so leakage is impossible by construction
rather than by careful bookkeeping.

Run `--overfit` before believing any training run. It drives a single batch to
near-zero loss; if that fails, nothing else about the run is worth reading.

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"           # generator only: numpy, scipy, h5py, matplotlib
pip install -e ".[dl,dev]"        # + torch, for phases 2-4
pytest -q                         # 120 tests, ~3 s
```

**On torch and CUDA:** install a build that matches your *driver*, not the newest
one. A CUDA 13 wheel silently falls back to CPU on a driver that only supports
CUDA 12.x, and the only sign is `torch.cuda.is_available() == False`. On driver
535 (CUDA 12.2) the working choice is

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Every phase script prints which device it is using and why.

## Use

```python
import patt_reco as pr

cfg = pr.load_yaml("configs/data/l2_noise.yaml")
rec = pr.generate_event(cfg, index=0)       # (seed, index) -> the event, always

rec.adc(noise_cfg=cfg.noise)                # [3, 128, 128] digitised image
rec.dense_semantic()                        # [3, 128, 128] uint8 class labels
rec.dense_instance()                        # [3, 128, 128] int16 object ids
rec.dense_n_contrib()                       # how many objects fed each pixel

pr.plot_event(rec, noise_cfg=cfg.noise)
```

Generate a dataset, preview it, build a sweep slice:

```bash
python scripts/make_dataset.py configs/data/l2_noise.yaml -j 8
python scripts/preview.py configs/data/l3_busy.yaml -n 4 -o previews/
python scripts/make_dataset.py configs/data/l3_busy.yaml \
       --set event.n_objects='[5,5]' --set name=sweep_m5 -o data/sweep_m5
```

## The difficulty ladder

| config | contents | purpose |
|---|---|---|
| `l0_single` | 1 object, no noise, pinned detector | wiring, overfit-one-batch |
| `l1_multi` | 2–5 objects, no noise, randomised detector | multi-object, first crossings |
| `l2_noise` | 2–5 objects + full noise | **the training set** |
| `l3_busy` | 10–40 objects, cosmics, full noise | **validation only** |
| `l4_ood_shape` | spirals, zigzags, double rings | **validation only** |
| `l4_domain_shift` | familiar shapes, detector outside the training envelope | **validation only** |

L3 and L4 are never trained on. That is the entire point.

## Four things worth knowing before you use the labels

**1. Labels live on the deposited charge, not on the ADC.** The response
convolution happens after the truth is fixed, so a bipolar (induction-like)
response puts real signal — including negative lobes — at pixels labelled empty.
Undoing that is part of the task, and it is what has to transfer when the
response changes.

**2. The overlap rule.** A pixel fed by several objects is labelled with its
*highest-charge* contributor. `n_contrib` counts how many objects touched it, and
the full contribution list is stored for exactly those pixels, so evaluation can
exclude, down-weight, or fully score the ambiguous ones. In L3 about 16% of hit
pixels are ambiguous — large enough that the choice moves the numbers, which is
why it is recorded rather than assumed.

**3. Detector parameters are randomised per event.** Pitch, drift velocity,
diffusion, attenuation, response shape and polarity, gain, and dead channels are
all resampled for every event. A model trained on this cannot learn one
detector's look because there isn't one.

**4. Noise is not stored.** Only the sparse deposited charge and the readout
state are; the ADC image is recomputed at load time and noise is realised from
the event's seed. Datasets stay at ~22 kB/event, every epoch sees a fresh noise
realisation for free, and re-rendering the same event through a different
response kernel — the domain-shift test — costs one argument:

```python
rec.adc(noise_cfg=cfg.noise, kernel=response_kernel(4.0, bipolar=False))
```

## Reproducibility

`(config.seed, event_index)` determines an event completely — readout sampling,
composition and rendering all draw from one generator seeded by that pair.
Generation is embarrassingly parallel and bit-identical regardless of worker
count (there is a test for it). Every dataset directory carries a `manifest.json`
with the full config, its hash, the git SHA and library versions.

The order of random draws is part of that contract: changing it changes every
dataset ever generated.

## Classes

| id | name | primitive | physics |
|---|---|---|---|
| 0 | empty | — | — |
| 1 | track | straight segment | Landau dE/dx, optional Bragg peak |
| 2 | scattered | Highland random walk | θ₀ ∝ 1/p |
| 3 | helix | constant-field helix | arc in the bending plane |
| 4 | ring | circle, arbitrary normal | projects to an ellipse |
| 5 | shower | recursive branching cone | energy-conserving cascade |
| 6 | blob | 3D Gaussian | vertex / nuclear interaction |
| 7 | cosmic | full-volume straight track | cos²θ zenith distribution |

Out-of-distribution shapes (`spiral`, `zigzag`, `double_ring`) are labelled with
their nearest in-distribution class — helix, track, ring — so the L4 question is
sharp: does the model degrade into the plausible class, or hallucinate?
