# `patt_reco` — An Experiment-Independent Pattern Recognition Benchmark & Model

**Status:** M0 + M1 built (generator, storage, event display, 77 tests). M2-M5 not started.
See [§9 Build log](#9-build-log--what-changed-on-contact-with-reality) for what the plan got wrong.
**Owner:** fanrado
**Written:** 2026-09-11 · **Last marked:** 2026-09-12

Section headings carry a status marker: ✅ built · 🟡 partly built · ⬜ not started.
Where a built section differs from what was planned, the heading says so and §9 explains why.

---

## 0. Motivation and scope

Neutrino (and more generally particle-physics) experiments repeatedly rebuild the same
reconstruction machinery: find the tracks, find the showers, separate the ones that cross,
ignore the noise. The underlying *geometry* is not experiment-specific — lines, scattered
lines, arcs/helices, rings, branching cones, blobs — and neither is the difficulty, which
comes from (a) projecting 3D objects into a small number of 2D readout views, (b) overlap and
crossings in those projections, and (c) detector noise.

This project builds a **synthetic, experiment-independent dataset generator** for that geometry
plus **reference deep-learning models** and a **frozen benchmark** on busy events. The deliverable
is not "a network that reconstructs experiment X" — it is a *shared substrate*: a common task
definition, a common metric suite, and baselines strong enough that a new experiment can adapt
rather than restart.

### Design principles

1. **Physics-shaped, not experiment-shaped.** Multiple scattering, Bragg peaks, diffusion and
   bipolar induction responses are in the generator because they change the *pattern*; nothing is
   tuned to a specific detector's geometry, pitch, or electronics.
2. **Domain randomization by default.** Every detector parameter (pitch, drift velocity, response
   kernel, diffusion, noise spectrum, view angles) is sampled per event, not fixed. A model that
   survives this is far more likely to transfer than one trained on a single detector's look.
3. **Truth is exact and cheap.** Because we render from analytic 3D objects, per-pixel class,
   per-pixel instance, and per-object parameters are free and unambiguous (up to the overlap
   rule in §1.5). No "truth matching" heuristics.
4. **Everything reproducible from a seed.** `(dataset_seed, event_index)` fully determines an
   event. Datasets are distributed as configs + seeds first, bytes second.
5. **Classical baselines are first-class.** Hough / RANSAC / DBSCAN are implemented and reported
   alongside the networks. If a CNN does not beat Hough on clean single tracks, that gets said.

### Non-goals

- Simulating any specific detector, or matching any experiment's data distribution.
- Full physics generators (GENIE/GEANT). We approximate *topology*, not cross-sections.
- 3D reconstruction from the 2D views (stereo matching). Out of scope for v1; the generator
  keeps the 3D truth so it can be added later.

### Hardware reality check

Target machine: **NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM**, Python 3.12. Every number in this plan
(image size, batch size, model capacity) is chosen to fit that. Explicit `small/` and `large/`
config variants are provided so the same code scales to a cluster later without a rewrite.

---

## 1. Task 1 — Fake dataset generation ✅

Pipeline, in four decoupled layers. Each layer is independently testable and independently
swappable — this is what lets an experiment plug in *its* response function later without
touching the geometry code.

```
  geometry (3D, continuous)  →  detector (3D → 2D views)  →  noise  →  storage/labels
     primitives.py                projection.py               incoherent.py    schema.py
     scattering.py                response.py                 coherent.py      io_hdf5.py
     shower.py                    digitize.py                 background.py    torch_dataset.py
     compose.py                   defects.py
```

### 1.1 Geometry layer — 3D objects ✅

> Built in `patt_reco/geometry/primitives.py`. All six primitives, plus the `spiral`,
> `zigzag` and `double_ring` OOD shapes in `geometry/ood.py` that §4.2 needs.

All primitives expose the same interface:

```python
class Primitive(Protocol):
    cls: ClassId
    params: dict          # analytic parameters, stored as truth
    def sample(rng, cfg) -> "Primitive": ...
    def deposit(rng) -> tuple[np.ndarray, np.ndarray]:
        """Return (points[N,3] in cm, dQ[N] in arbitrary charge units)."""
```

| Class | Primitive | Physics knobs that make it non-trivial |
|---|---|---|
| `track` | straight segment | constant dE/dx, Landau fluctuation per step, optional **Bragg peak** for a stopping particle |
| `scattered` | segment + Highland multiple-scattering random walk | θ₀ per step ∝ 1/p → gently curving, kinked tracks; the realistic "curve" |
| `helix` | constant-B helix, R = p_T/(0.3·B) | projects to an arc/circle in the bending plane, to a sine-like wiggle elsewhere |
| `ring` | circle of radius R with arbitrary plane normal | Cherenkov-ring-like; projects to an ellipse |
| `shower` | recursive branching cone | split probability, opening angle from a narrow distribution, energy split, stop at E_min → fractal blob |
| `blob` | 3D Gaussian ball | vertex / nuclear interaction / low-energy deposit |
| `vee` | two tracks from a shared vertex | decay topology; tests vertex handling |
| `delta` | short branch off a parent track at a random arc-length | tests instance separation at small scales |

Objects are emitted as **point clouds with charge weights**, not as rasterized images — the
rendering resolution is a detector parameter, not a geometry parameter.

### 1.2 Composition — building an event ✅

> Built in `patt_reco/geometry/compose.py`. Vertex groups, forced crossings and delta rays
> all present. Containment is by clipping at the readout window rather than by an explicit
> probability knob — an object that leaves simply stops depositing where it leaves.
> `vee` is a two-member vertex group rather than a primitive, so each prong gets its own
> instance id (§9.8).

`compose.py` places primitives in a fiducial volume with control over exactly the things we
later want to *measure sensitivity to*:

- `n_objects` sampled from a configurable distribution (Poisson, or fixed for controlled tests).
- **Vertex groups**: with probability `p_vertex`, k objects share an origin (interaction-like).
- **Forced crossings**: an explicit sampler that places two objects to cross at a target angle in
  a target view. Essential — random placement almost never produces the small-angle crossings
  that are the actual hard case.
- **Containment**: objects may exit the volume (truncated) with configurable probability.

### 1.3 Detector layer — 3D → 2D views ✅

> Built across `detector/readout.py`, `response.py`, `projection.py` and `digitize.py`.
> Defects live in `readout.py` rather than a separate `defects.py` — they are sampled with
> the rest of the per-event detector state and separating them bought nothing.

A `PlaneView` projects a 3D point onto (channel, tick):

```
channel = (y·cos θ + z·sin θ) / pitch        tick = x / (v_drift · Δt)
```

- **View sets**: default is three stereo views at θ ∈ {0°, +60°, −60°} — the geometry essentially
  every wire-plane detector has, without being any particular one. Orthographic XY/XZ/YZ is also
  supported for pixel-readout-like setups.
- **Diffusion**: transverse/longitudinal Gaussian smearing with σ ∝ √(drift distance).
- **Attenuation**: exp(−t_drift/τ) charge loss.
- **Response convolution**: the binned (channel, tick) charge image is convolved along the tick
  axis with a 1-D field+electronics response kernel — where a *bipolar* induction-like kernel
  makes the problem genuinely hard (positive and negative lobes, cancellation between nearby
  tracks). *(Built in `detector/response.py`; see §9 for why it does not reuse
  `signal_processing_toolkit` as originally intended.)*
- **Digitization**: gain, pedestal, ADC quantization, saturation clipping.
- **Defects**: dead channels, hot channels, per-channel gain spread, baseline wander.

All of the above are sampled per event within configured ranges (principle 2).

**Default resolution:** 3 views × 128 channels × 128 ticks, occupancy ~2–5%. A `large` config
uses 256×256 for later scaling.

### 1.4 Noise layer 🟡

> Built in `patt_reco/noise/`. Incoherent, coherent and 1/f are there, as are hit-like
> blips. **Sticky ADC codes are not implemented.** Cosmic background moved to
> `geometry/compose.py`, since cosmics are objects with instance ids, not a noise term (§9.8).

- **Incoherent**: per-(channel,tick) Gaussian / white noise.
- **Coherent**: noise shared across channel groups (e.g. blocks of 16/32/64). This is the
  realistic, hard-to-filter component and the most valuable thing to train against.
- **1/f**: shaped in the frequency domain per channel.
- **Artifacts**: sporadic hit-like blips, sticky ADC codes.
- **Background activity**: random cosmic-like straight tracks traversing the full volume. These
  are labeled as their own class *and* carry instance ids, so we can evaluate both "reject it"
  and "reconstruct it" framings.

Noise is parameterized by a target **SNR sweep** so §4 can plot performance vs. SNR.

### 1.5 Labels — the data contract ✅

> Built in `detector/projection.py` and `dataset/schema.py`. `semantic` is derived from
> `instance` through the object table rather than stored, so the two cannot disagree (§9.4).

Per view, per pixel:

| Field | dtype | Meaning |
|---|---|---|
| `adc` | float32 | digitized signal (clean; noise added at load time, see §1.6) |
| `semantic` | uint8 | 0 empty, 1 track, 2 scattered, 3 helix/arc, 4 ring, 5 shower, 6 blob, 7 cosmic-bg |
| `instance` | int16 | object id; −1 for empty/noise-only pixels |
| `n_contrib` | uint8 | number of distinct objects depositing in this pixel |

**Overlap rule (important, and stated explicitly because it biases every metric):** a pixel whose
charge comes from several objects is assigned the *highest-charge contributor* for `semantic` and
`instance`, and `n_contrib > 1` flags it. The exact multi-label truth is preserved in a sparse
side table `contrib[(pixel_index, object_id, charge)]`, so evaluation can (a) exclude ambiguous
pixels, (b) down-weight them, or (c) score against the full set. All three are reported.

Event-level `objects` table: `id, class, 3d_params, per_view_params, total_charge, n_pixels,
visible` (an object fully hidden behind another in a given view is marked not-visible in that
view and excluded from that view's efficiency denominator).

### 1.6 Storage and reproducibility ✅

> Built in `dataset/schema.py`, `io_hdf5.py` and `generate.py`. Measured sizes and rates are
> in §9.7; the frozen test set needed no extra bytes (§9.5).

- **Store the clean sparse image + the noise config + the event seed; realize noise at load time.**
  Datasets stay small (~12 kB/event → 100k events ≈ 1.2 GB), and every epoch sees a fresh noise
  realization — free, physically-correct augmentation.
- **One frozen, fully-realized test set** (noise baked in) per difficulty level, so published
  numbers are exactly reproducible.
- Format: HDF5 shards (~5k events each) with sparse COO arrays; a `manifest.json` per dataset
  recording the config, its hash, the git commit, and the library versions.
- Determinism: `rng = np.random.default_rng(np.random.SeedSequence(dataset_seed, event_index))`.
  Generation is embarrassingly parallel and bit-identical regardless of worker count.

### 1.7 Difficulty ladder ✅

| Level | Contents | Purpose | Size | Config | Status |
|---|---|---|---|---|---|
| **L0** | 1 object, no noise, fixed detector params | wiring/debugging, overfit tests | 20k | `l0_single.yaml` | ✅ |
| **L1** | 2–5 objects, no noise, randomized detector | multi-object, first crossings | 40k | `l1_multi.yaml` | ✅ |
| **L2** | 2–5 objects + full noise | the main training set | 60k | `l2_noise.yaml` | ✅ |
| **L3** | 10–**40** objects, cosmics, SNR sweep | **busy events** (§4) — *validation only* | 20k | `l3_busy.yaml` | ✅ |
| **L4a** | unseen shapes (spiral, zigzag, double-ring) | out-of-distribution geometry | 10k | `l4_ood_shape.yaml` | ✅ |
| **L4b** | familiar shapes, shifted pitch/response/diffusion | out-of-distribution detector | 10k | `l4_domain_shift.yaml` | ✅ |

L3/L4 are **never trained on**. That is the whole point of §4.

Two changes from the plan as written: L3 tops out at 40 objects rather than 60 and uses a smaller
object-size distribution (§9.6), and L4 is split in two so a failure can be attributed to unseen
geometry or to an unseen detector rather than to both at once (§9.8).

The configs exist and generate; **the datasets themselves have not been generated to full size**
— only up to 4000-event benchmark runs. Budget ~5 minutes on 8 cores and ~3.7 GB for the lot.

### 1.8 Visualization 🟡

`viz/event_display.py`: three-view display with selectable panels. Non-negotiable — nearly every
generator bug is obvious in an event display and invisible in a loss curve.

Built: `adc`, `charge`, `semantic`, `instance`, `n_contrib`, driven from `scripts/preview.py`.
The ADC panel scales to a percentile rather than the maximum, because one Bragg peak is routinely
10x brighter than everything else and washes the rest of the event out.

Not built: the `prediction` and `error map` panels, which need a model to exist first (M2).

---

## 2. Task 2 — Models ⬜

> Nothing in this section is built. `patt_reco/models/` and `patt_reco/losses/` do not exist
> yet. The build order in §2.5 still stands, and §2.2 is the gate: the classical baselines
> come first because they also test whether the generator is too easy.

### 2.1 Common interface

```python
model(batch: dict[str, Tensor]) -> dict[str, Tensor]
# in : views [B, V, 1, H, W]  (+ per-view geometry metadata)
# out: sem_logits [B, V, C, H, W], and optionally
#      offsets [B, V, 2, H, W], embed [B, V, D, H, W],
#      query_masks [B, V, Q, H, W], query_cls [B, Q, C+1], query_params [B, Q, P]
```

A registry (`models.build(cfg)`) keeps models swappable from config. Losses and metrics key off
which outputs are present, so adding a model never requires touching the training loop.

### 2.2 Classical baselines (build these first)

`models/baselines.py`: thresholding + connected components; Hough transform for lines; circle
Hough / RANSAC for arcs and rings; DBSCAN on hit coordinates. Cheap, no training, and they
establish the floor that every learned model must clear. They also double as a generator sanity
check: if Hough cannot find a clean isolated line in an L0 event, the *generator* is wrong.

### 2.3 CNN arm

- **Phase 1 — semantic segmentation.** 2D U-Net, 4 down/up levels, base width 32, GroupNorm,
  ~5–8 M params. Input is **one view at a time** through a shared-weight (siamese) encoder —
  views have different geometry and must not be stacked as RGB channels.
- **Cross-view fusion (ablation).** Add a light attention block at the bottleneck that lets the
  three views exchange information via their shared tick (drift-time) axis — the one coordinate
  all views agree on. This is the honest way to exploit stereo without doing full 3D matching.
- **Phase 2 — instance segmentation.** Two extra heads on the same backbone:
  - *offset head*: regress each pixel's 2D displacement to its object's centroid;
  - *embedding head*: D=8 per-pixel embedding trained with a discriminative (pull/push/reg) loss.

  Instances = mean-shift or DBSCAN in (shifted-coords ⊕ embedding) space. Cheap and robust.

### 2.4 Transformer arm

Two variants, in priority order:

1. **Query-based mask decoder (Mask2Former-lite).** Q = 32 learned queries attend (masked
   cross-attention) to the CNN feature map; each query emits a class, a binary mask, and a
   parameter vector. Trained with Hungarian matching. This is the elegant formulation for this
   problem: it produces instances, classes, *and* physics parameters (start point, direction,
   radius, length) in one shot, with no post-hoc clustering and no threshold tuning.
2. **Hit-token transformer.** Tokens = the ~500 non-empty pixels, not the 16384 grid cells, with
   learned positional encoding from (channel, tick, adc). At 2–5% occupancy this is *cheaper*
   than dense convolution, is naturally resolution-independent, and generalizes across detectors
   with different pitch — which makes it the most interesting arm from a
   "experiment-independent" standpoint and the likely research contribution.

The hit-token model is also the natural bridge to native 3D (tokens = voxels) if the project
later moves off 2D projections.

### 2.5 Recommendation

Build in this order: **baselines → U-Net semantic → U-Net + instance heads → query decoder →
hit-token transformer.** Each stage is a working system, and each provides the comparison point
for the next. Do not start with the transformer.

---

## 3. Task 3 — Training and testing 🟡

> Only §3.6 is partly built: the generator invariants are covered by 77 passing tests. There
> is no training loop, no losses and no metrics. §3.1 should be revisited before M2 —
> the measured class imbalance (§9.9) has a second component the section does not account for.

### 3.1 Losses

| Head | Loss | Note |
|---|---|---|
| semantic | focal (γ=2) + soft-Dice | occupancy is 2–5%; plain cross-entropy collapses to "all empty" |
| offsets | smooth-L1, masked to foreground | normalized by object size |
| embedding | discriminative: `L_var + L_dist + 0.001·L_reg` | δ_v=0.5, δ_d=1.5 |
| queries | Hungarian: CE(class) + focal+Dice(mask) + smooth-L1(params) | with a no-object class |

Multi-task weights: start fixed (tuned by a short sweep), optionally switch to learned
uncertainty weighting. Ambiguous pixels (`n_contrib > 1`) are down-weighted by `1/n_contrib` in
the semantic loss — they are genuinely ill-posed and should not dominate gradients.

### 3.2 Fitting 4 GB

- Mixed precision (AMP, fp16 with bf16 fallback), `channels_last`.
- 128×128, batch 16 per step, gradient accumulation ×4 → effective batch 64.
- Gradient checkpointing on the encoder if the query decoder pushes past VRAM.
- Estimated: ~40 min/epoch on 60k L2 events; a full Phase-1 run ≈ 6–10 h. Plan runs overnight.
- Guardrail: a `--profile-memory` flag that reports peak VRAM for a config before a long run.

### 3.3 Data pipeline and augmentation

- On-the-fly noise realization (§1.6) — the single most valuable augmentation, and free.
- **Event mixing**: overlay two clean L2 events (charges add, instance ids offset) to synthesize
  busier events during training. Lets the model see multiplicity 10 without ever training on L3.
- Geometric augmentation is applied **at the 3D stage**, not to the 2D images — rotating a
  rendered view is physically wrong once the response kernel is anisotropic in tick. In 2D only
  channel-flip and small tick-shifts are permitted.
- `num_workers=4`, persistent workers, pinned memory.

### 3.4 Curriculum

Start on L0+L1, mix in L2 over the first ~20% of training, then ramp event-mixing probability.
Ablate against flat L2-only training — curricula often help less than claimed and that result is
worth reporting either way.

### 3.5 Experiment management

- Config: YAML → frozen dataclasses. No hidden defaults in code.
- `runs/<timestamp>_<config-hash>/` containing: resolved config, git SHA, `pip freeze`, TensorBoard
  logs, `metrics.csv`, per-epoch event-display PNGs, checkpoints (weights + config + RNG state).
- Local-only by design (TensorBoard, not a cloud tracker) — keeps the project dependency-light
  and shareable.

### 3.6 Testing (pytest) 🟡

**Generator invariants:**
- same `(seed, index)` → bit-identical event;
- every labeled pixel has nonzero charge, and vice versa (pre-noise);
- charge conservation: Σ rendered charge = Σ deposited charge × attenuation, to 1e-6;
- projecting a known analytic line gives a line of the expected slope in each view;
- instance ids are contiguous and match the object table.

**Model/training:**
- output shape and dtype contracts for every registered model;
- **overfit-one-batch**: every model must drive a 4-event batch to ~zero loss in <200 steps.
  This single test catches most label/loss/masking bugs;
- checkpoint save→load→identical-forward;
- metrics validated against hand-computed toy cases (a 4×4 image with known IoU/PQ).

**CI:** run the fast subset (generator + shapes + metrics) on every commit; the overfit test
nightly.

**Built so far — 77 tests, 0.6 s:**

| file | tests | covers |
|---|---|---|
| `tests/test_geometry.py` | 28 | primitive shapes, Bragg peaks, scattering vs momentum, shower energy conservation, composition, forced crossings |
| `tests/test_detector.py` | 17 | projection against the analytic formula, the shared tick axis, charge conservation, response normalisation, label/object-table agreement |
| `tests/test_noise.py` | 11 | reproducibility, SNR-scale linearity, coherence structure, 1/f spectrum |
| `tests/test_dataset.py` | 15 | determinism from (seed, index), lossless shard round-trip across buffer boundaries, derived-label consistency, serial == parallel |

**Not built:** every model and training test (shape contracts, overfit-one-batch,
checkpoint round-trip, hand-computed metric cases), and the CI workflow itself.

### 3.7 Metrics ⬜

*Semantic:* per-class IoU, mIoU, pixel accuracy, confusion matrix — reported three ways per the
overlap rule (all pixels / ambiguous excluded / ambiguity-weighted).

*Instance:* **Panoptic Quality** (PQ = SQ × RQ), Adjusted Rand Index, and the HEP-native pair —
**purity** (fraction of a predicted cluster's hits belonging to its matched truth object) and
**efficiency** (fraction of a truth object's hits captured). Plus object-count error.

*Objects:* detection efficiency and parameter resolution (Δθ, Δstart, ΔR, ΔL).

*Calibration:* reliability diagram and ECE on the semantic head — a model that says "80%" should
be right 80% of the time if downstream physics is going to use it.

**Everything is reported differentially**, not as a single scalar: efficiency vs. object length,
vs. energy, vs. crossing angle, vs. SNR, vs. local occupancy. A single mIoU number hides exactly
the failure modes this project exists to expose.

---

## 4. Task 4 — Validation on busy events ⬜

> Nothing here is built, but the inputs are: `l3_busy.yaml` and both L4 configs generate, and
> `noise_scale` / `--set` already drive the SNR and multiplicity sweeps §4.2 needs. What is
> missing is a model to evaluate, the metric code, and the frozen published datasets.

This is the acceptance test, and it is deliberately harder than the training distribution.

### 4.1 The benchmark set (L3, frozen)

10–60 objects per event, mixed classes, shared vertices, full-volume cosmic tracks, coherent +
incoherent noise, with deliberate small-angle crossings injected. Frozen bytes, versioned,
published with the metric code.

### 4.2 Stress axes (each a separate held-out slice)

| Axis | Sweep | Question it answers |
|---|---|---|
| Multiplicity | 5 → 60 objects | does it extrapolate beyond training multiplicity? |
| SNR | 20 → 1 | where does it break relative to the classical baselines? |
| Crossing angle | 90° → 5° | the canonical hard case: merged or split instances? |
| Occupancy | 2% → 20% | saturation behavior |
| OOD shape (L4) | spiral, zigzag, double-ring | does it classify or silently hallucinate a known class? |
| Domain shift (L4) | ×2 pitch, different response kernel, ×3 diffusion | the transfer question — the entire premise of the project |

### 4.3 Failure taxonomy

Every failure is bucketed and counted, not just averaged into a metric: *merged instances*
(two objects → one), *split instances*, *class confusion* (which pairs), *noise promoted to
signal*, *low-energy misses*, *boundary errors at crossings*. Each bucket gets a gallery of
representative event displays in the report.

### 4.4 Acceptance criteria (first pass targets, to be revised after M2)

- mIoU ≥ 0.85 on L2; ≥ 0.70 on L3.
- PQ ≥ 0.70 on L3 at multiplicity ≤ 20; graceful (not cliff-edge) degradation to multiplicity 60.
- Beats the best classical baseline on L3 at every SNR, and is not *worse* than it on L0.
- Domain-shift (L4) mIoU drop < 15% relative to L2.

These are targets for steering, not promises. The honest outcome of M5 may be "the transformer
wins only above multiplicity 20" — that is a publishable result and should be reported as such.

### 4.5 The real deliverable

A frozen benchmark + metric code + baseline numbers that another group can run in an afternoon.
That is what makes this project *reduce* duplicated effort rather than add one more private
pattern-recognition framework to the pile.

---

## 5. Repository layout 🟡

`✅` exists on disk and is committed · `⬜` planned, not written.

```
patt_reco/                        38 files, 38 commits, 77 tests
├── ✅ PLAN.md                    ← this file
├── ✅ README.md
├── ✅ .gitignore
├── ✅ pyproject.toml             (numpy, scipy, h5py, matplotlib, pyyaml core;
│                                  torch + tensorboard + sklearn behind a `dl` extra)
├── configs/
│   ├── ✅ data/    l0_single  l1_multi  l2_noise  l3_busy  l4_ood_shape  l4_domain_shift
│   ├── ⬜ model/   baseline_hough  unet_sem  unet_inst  query_decoder  hit_tokens
│   └── ⬜ train/   base  curriculum
├── patt_reco/
│   ├── ✅ config.py              all generator knobs + the class enum   (not in the original plan)
│   ├── geometry/   ✅ volume.py  ✅ primitives.py  ✅ compose.py  ✅ ood.py
│   ├── detector/   ✅ readout.py  ✅ response.py  ✅ projection.py  ✅ digitize.py
│   ├── noise/      ✅ incoherent.py  ✅ coherent.py  ✅ __init__.py
│   ├── dataset/    ✅ schema.py  ✅ io_hdf5.py  ✅ generate.py  ⬜ torch_dataset.py  ⬜ augment.py
│   ├── ⬜ models/  registry  unet  heads  query_decoder  hit_tokens  baselines
│   ├── ⬜ losses/  focal_dice  discriminative  hungarian
│   ├── ⬜ train/   loop  curriculum  tracking
│   ├── ⬜ eval/    metrics_sem  metrics_inst  differential  report
│   ├── viz/        ✅ event_display.py
│   └── ⬜ cli.py                 (scripts/ covers this for now)
├── scripts/  ✅ make_dataset.py  ✅ preview.py  ⬜ train.py  ⬜ evaluate.py  ⬜ benchmark_busy.py
├── tests/    ✅ conftest  ✅ test_geometry  ✅ test_detector  ✅ test_noise  ✅ test_dataset
└── ⬜ notebooks/  01_generator_tour  02_baselines  03_results
```

Four files in the planned tree were never written because they had nothing to hold:
`geometry/scattering.py` and `geometry/shower.py` are ~40 lines each and live in `primitives.py`;
`detector/defects.py` folded into `readout.py`, where the rest of the per-event detector state is
sampled; `noise/background.py` became cosmic composition in `geometry/compose.py` (§9.8). One file
was added that the plan missed entirely: `config.py`, which turned out to be where the whole
design actually lives.

`patt_reco` is its own package rather than living inside `signal_processing_toolkit` — different
scope, different dependency weight (torch). It turned out not to depend on it at all; see §9.1.
It is also now its own git repository.

---

## 6. Milestones

| ID | Deliverable | Done when | Status |
|---|---|---|---|
| **M0** | Scaffolding | package installs, CI runs, empty tests pass | **done** |
| **M1** | Generator + event display | L0-L2 generate; all §3.6 generator invariants pass; notebook tour renders | **done** — L0-L4 all generate, 77 tests pass |
| **M2** | Classical baselines + semantic U-Net | overfit test passes; L2 mIoU reported; U-Net vs. Hough table exists | not started |
| **M3** | Instance segmentation | PQ/ARI/purity/efficiency on L2; failure gallery | not started |
| **M4** | Transformer arm | query decoder and hit-token model trained; head-to-head on L2 | not started |
| **M5** | Busy-event benchmark + report | L3/L4 frozen and published; all §4.2 sweeps plotted; write-up | not started |

M1 and M2 carry most of the risk and most of the value — a correct generator with a working
semantic baseline is already a usable artifact even if M4 never happens.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Sim-to-real gap** — fake data may not transfer | Domain randomization by construction (§0.2); L4 domain-shift slice measures it directly; documented adapter interface so an experiment substitutes its own response/noise without touching geometry |
| **Label ambiguity at overlaps** biases every metric | Explicit overlap rule + `n_contrib` + sparse contribution table; all metrics reported three ways (§3.7) |
| **Class imbalance** (2–5% occupancy) | Focal+Dice, foreground-weighted sampling, per-class IoU rather than pixel accuracy |
| **Instance clustering is hyperparameter-sensitive** | Report clustering hyperparameters as tuned on validation only; the query-decoder arm exists precisely because it avoids post-hoc clustering |
| **4 GB VRAM ceiling** | Sizes in §3.2 are chosen to fit; `--profile-memory` guardrail; `small`/`large` configs so a cluster needs no rewrite |
| **The generator is too easy** and everything scores 0.99 | Baselines-first (§2.2): if Hough solves L2, the generator needs harder crossings/lower SNR before any network work proceeds |
| **Scope creep into 3D reconstruction** | Explicit non-goal; 3D truth is retained so it can be a v2 |

---

## 8. Open decisions 🟡

Three of the five are now settled by the build; the marker says how.

1. **Class granularity** — ⬜ still open. Is `helix` vs. `scattered` a distinction worth a class,
   or should "curved track" be one class? Built as *separate* classes (ids 2 and 3), against the
   original lean, because the generator can always merge two labels at load time and cannot split
   one. The ablation is still owed.
2. **Cosmic background: class or nuisance?** — ✅ settled. Built as its own class (id 7) *and*
   carrying instance ids, so both the "reject it" and "reconstruct it" framings are measurable.
   An experiment wanting pure rejection collapses the label at load time.
3. **View count** — ⬜ still open. Three stereo views is the built default, but
   `view_angles_deg` is an ordinary config tuple of any length, so a 2-view run costs one line.
   Nothing downstream assumes three. Whether to carry 2-view as a *reported* configuration is
   still undecided.
4. **Benchmark distribution** — 🟡 half settled. Seeds-in-the-repo now works: the configs are
   committed and `(seed, index)` reproduces any event bit-for-bit, with a test for it. Whether to
   also release a frozen tarball (L3 is ~1.6 GB) is still open, and is an M5 decision.
5. **Parameter regression targets** — 🟡 provisionally answered. `detector/projection.py` stores a
   fixed 10-slot vector per object — start point (3), direction (3), length, radius, charge,
   energy — with NaN where a field does not apply. That covers the certain ones and is cheap to
   extend. The dE/dx profile is *not* stored. Still worth asking a collaborator before M4.


---

## 9. Build log — what changed on contact with reality

M0 and M1 are built. Seven things in the plan above turned out to be wrong or underspecified;
they are recorded here rather than silently corrected, because the reasons matter more than
the fixes.

### 9.1 `signal_processing_toolkit` is not reused, and should not be

The plan called for reusing the sibling toolkit for the response convolution and the noise
generators (§0, §1.3, §1.4, §5). It is not used, for a concrete reason: its generators
(`generate_gaussian_noise`, `generate_white_noise`) draw from **numpy's global RNG**. That
directly breaks design principle 5 — `(dataset_seed, event_index)` must reproduce an event
bit-for-bit regardless of how many workers are running — because the global RNG's state depends
on call order across processes.

Its API is also shaped for 1-D time series (`duration`, `sample_rate`), not for
`[channels, ticks]` planes, so every call would have needed reshaping anyway.

`patt_reco` therefore has no dependency on it. If the toolkit ever grows explicit-`Generator`
variants, the noise module is two functions away from using them.

### 9.2 Labels sit on the deposited charge, and that needed deciding, not assuming

§1.5 did not say where truth lives relative to the response convolution. It matters: with a
bipolar kernel the ADC image carries real signal — including negative lobes — several ticks away
from any deposited charge.

Truth is defined on the **deposited charge, pre-response**. That is the physical quantity
(ionisation), it is unambiguous, and it makes the response a nuisance transformation the model
must undo — which is exactly the thing that has to transfer when the response changes. The
consequence to keep in mind: a strict "every ADC pixel with signal is labelled" invariant is
false by construction, and any metric that assumes it is measuring the wrong thing.

### 9.3 Diffusion is a 3D jitter, applied once — not an image blur

Blurring the rendered image would have been the obvious implementation and is wrong twice over:
it mixes objects together (destroying instance labels) and it applies the same smearing to every
pixel regardless of drift distance. Instead each deposit is split into sub-deposits and jittered
in 3D **once**, before projection. Every plane then sees the same diffused electron cloud, so the
views stay mutually consistent, and each deposit keeps its object identity.

### 9.4 `semantic` is derived, not stored

Storing both label images invites them to disagree. Only `instance` is stored; `semantic` is
looked up through the object table at load time. That removes a whole class of label bug by
construction, and it is what the "labels agree with the object table" test actually verifies.

### 9.5 `freeze_noise` does not freeze bytes

The plan wanted "one frozen, fully-realised test set" (§1.6). That turned out to be unnecessary:
noise is already deterministic from `[seed, index, 1]`, so a frozen test set needs no extra
bytes. The flag now selects *which seed* is used at load time — derived (reproducible) for test
sets, fresh (augmentation) for training.

### 9.6 Busy events need smaller objects, not just more of them

§4.2 targets occupancy up to 20%. Keeping the L2 object-size distribution and stacking 40 objects
gave **55% occupancy** — the truth image is a solid sheet, not a detector event.

Real busy readouts are mostly *short* tracks and small deposits with a few long ones. `l3_busy`
therefore overrides the geometry block (tracks 2-25 cm instead of 5-60, rings 3-10 cm instead of
4-20, showers 30-300 MeV instead of 50-1000) and caps multiplicity at 40. Measured mean occupancy
is now **22%**, with 16% of hit pixels ambiguous. Cosmics keep their full length.

The general lesson for the stress sweeps: multiplicity and occupancy are not the same axis, and
the plan conflated them.

### 9.7 Measured numbers, replacing the plan's estimates

| quantity | plan said | measured |
|---|---|---|
| storage | ~12 kB/event | 4 kB (L0), 17 kB (L2), 82 kB (L3) |
| L2 generation | — | 6.0 ms/event (1 core), 1.4 ms/event (8 workers) |
| L3 generation | — | 22.5 ms/event (1 core), 5.6 ms/event (8 workers) |
| occupancy | 2-5% (L2), 20% (L3) | 5.0% (L2), 22.4% (L3) |
| full ladder | — | ~5 min on 8 cores, ~3.7 GB |

One performance finding worth keeping: writing events to HDF5 one at a time cost **10 ms/event**,
nearly twice the cost of generating them, purely from per-call chunk overhead across ~25
datasets. Buffering 256 events per write cut total generation time by 2.8x. Compression level was
irrelevant (gzip-4 and gzip-1 differ by 4%); the call count was everything.

### 9.8 Smaller structural deviations

- **Cosmics live in `geometry/compose.py`, not `noise/background.py`.** They are objects with
  instance ids and truth parameters, not a noise term. The plan filed them under noise.
- **`vee` is not a primitive.** It is a vertex group in the composer, so each prong gets its own
  instance id — which is what instance labels require. Same for delta rays.
- **L4 is two configs, not one.** `l4_ood_shape` (unseen geometry, nominal detector) and
  `l4_domain_shift` (familiar shapes, detector outside the training envelope) are separate so a
  failure can be attributed to one cause or the other.
- **OOD shapes are labelled with their nearest in-distribution class** (spiral→helix,
  zigzag→track, double_ring→ring), which sharpens the §4.2 question from "does it cope" to "does
  it degrade into the plausible class or hallucinate a confident wrong one".

### 9.9 Still open from M1

- The notebook tour (`notebooks/01_generator_tour.ipynb`) is not written; `scripts/preview.py`
  covers the same ground from the command line.
- No CI workflow yet (§3.6 calls for one).
- **Class imbalance, measured** (200 events each). Two separate imbalances, and §3.1 only
  anticipated the first:

  | | L2 | L3 |
  |---|---|---|
  | empty pixels | 94.6% | 78.1% |
  | rings: share of hit pixels vs share of objects | 30.6% vs 10.7% | 25.2% vs 10.0% |
  | tracks: share of hit pixels vs share of objects | 8.7% vs 21.5% | 9.1% vs 28.8% |
  | pixels per object, ring : track | 1797 : 253 (7.1x) | 844 : 105 (8.0x) |

  Foreground/background imbalance is handled by focal+Dice as planned. The second one is not:
  a ring carries 7x the pixels of a track, so a pixel-averaged loss optimises for rings and a
  pixel-averaged mIoU flatters them. M2 should report per-class IoU (already planned) *and*
  weight the semantic loss by inverse pixels-per-object, not by inverse class frequency.
