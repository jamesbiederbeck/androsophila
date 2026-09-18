# Inverse haltere-stimulation model — data generation, training, and an on-device data-collection tool

**Verdict: a small MLP can propose a haltere-afferent stimulation set that
reproduces a target wing-muscle motor spike pattern, validated against the
real forward simulation (not just against its own training labels) — but
only because the training data had to be built in three escalating rounds
after each one exposed a specific, diagnosable failure mode.** Also built:
an Android screen (`FlyMapActivity`) that stimulates named haltere clusters
via touch and exports real device-collected training cases, which is now
part of the same growing dataset.

None of this is a claim about real fly haltere biophysics. See
`flappy/circuit.py` and `flappy/haltere_cluster_sweep.py`'s own docstrings
for that distinction — it holds throughout everything below.

## The question

`flappy/circuit.py` already drives the wing-muscle motor pool by injecting
one scalar current into all 205 haltere afferents, scaled by the bird's own
fall speed. The ask here was the *inverse* problem: given a target
wing-muscle motor spike pattern, propose which haltere afferents to
stimulate to cause it — a decoding step needed before this can be driven by
a real multi-axis IMU signal instead of one scalar.

Confirmed direction early on (this matters — the natural-looking
`input(sensory) → hidden → output(motor)` shape is actually the *forward*
direction): the network is `input(67 wing-muscle motor spike counts) →
hidden(512) → output(205 haltere afferents)`, trained with binary
cross-entropy (independent per-cell sigmoids — "propose a set", not exact
currents). Deliberately not unique: many different stimulated subsets can
produce similar motor output, so this predicts *a* plausible cause, not
*the* cause.

## Round 1: broad random data, and why the first version was wrong twice

`flappy/inverse_data.py` generates training pairs by resetting the
connectome to a cold-start state, stimulating a random subset of haltere
afferents at a random current, and recording the resulting wing-muscle
spike counts.

Two things had to be fixed before this data was usable at all:

1. **A single ~50ms pulse from cold start mostly produces silence**, even
   with broad, high-amplitude subsets (9/10 initial trials fired zero
   wing-muscle spikes). This matches `flappy/circuit.py`'s own finding that
   the haltere→flight-motor reflex is a sharp-threshold, *sustained*-drive
   circuit — the working desktop config holds current on for ~150 game
   ticks, not one frame. Fixed by holding each trial's stimulation on for
   500ms.
2. **Trials weren't independent.** An early GPU timing test ran 5 trials
   back-to-back on a brain that was never reset between them; results
   (silent → 1265 → 1210 → 1308 spikes) tracked cumulative network ramp-up
   from prior trials, not each trial's own stimulation. Fixed with an
   explicit `reset()` (zeroing `v`, `g`, `refractory`, the delay queue, and
   re-seeding the initial active set) before every trial, implemented once
   per backend (`NativeBrain` vs `GPUBrain` have different internal state
   layouts) since both are used depending on whether a run is a real
   learning run (native/default) or a frozen-weight simulation (GPU, per
   this session's own standing preference).

6,000 broad-random trials (`dataset-train.npz`), `min_k` defaulting to 30%
of the haltere population to avoid an overwhelmingly-silent dataset.

**First trained model (`hidden=10000`) overfit outright**: train BCE 0.51,
val BCE 0.688 — *worse* than the marginal-frequency baseline (0.6485,
predicting each cell's overall stimulation rate and ignoring the input
entirely). ~2.7M params against ~5,100 training rows was too much capacity.
Dropping to `hidden=512` fixed it immediately: val BCE 0.628, beating the
baseline, train/val gap small. This is the architecture used throughout
everything below — the CPU-only numpy implementation trains it in ~5-7
seconds, so there was never a reason to add a GPU training framework.

## Round 2: the model disagreed with the real simulation

Before trusting the model at all, it was checked against three tools that
answer three genuinely different questions:

- **Static analysis** (pure synaptic weight sums, no dynamics): says
  `SNpp14` is the *only* haltere-type cluster with a direct 1-hop synapse
  onto the DLM/DVM power muscles.
- **Full network sweep** (`flappy/haltere_cluster_sweep.py`, real forward
  simulation): stimulated each named haltere connectome-type cluster
  *alone* across several currents. Result contradicted the static
  analysis — `SNpp14` (the "textbook"-wired cluster) never fires the
  wing-muscle pool at any tested current, while `SNpp12` and `SNpp23`
  (**zero** measured direct synapses onto any wing-muscle motor neuron)
  produced the strongest effects of any cluster, via some indirect (2+ hop)
  pathway. Direct wiring is a bad predictor of actual network effect here.
- **Inverse model round-trip**: fed the sweep's own measured responses back
  into the trained model and compared its proposed stimulation to the
  cluster that actually produced each response. Result: IoU 0.25-0.36 for
  the large `SApp` clusters (close to training-distribution scale), and
  **0.0-0.14 for every small cluster** — including `SNpp12` and `SNpp23`,
  the two the sweep had just identified as most important. The model had
  only ever seen broad random subsets (k ≥ ~62 of 205); small, structurally
  coherent clusters were entirely out of distribution.

Fixed by `flappy/inverse_data_clusters.py`: 2,500 more trials sampling
unions of 1-3 named clusters, then a random subset of that union (so both
whole-cluster and partial-cluster activation are covered), current range
widened to 4-14mV. Retraining on the combined set closed the gap
completely — **IoU 0.75-1.0 across every cluster in the sweep**, small ones
included.

## Round 3: real device data, and a subtler kind of wrong

`FlyMapActivity` (below) let real touch input drive the same trained model
in the field, exported as CSV. Two things surfaced:

1. **Round-trip testing against the live sweep again** (treating the GPU
   forward simulation as oracle throughout) showed the model had learned a
   different failure mode: for a *silent* target (no motor spikes — the
   common case for weak/small random stimulations), it collapsed to
   proposing **all 205 haltere cells**, every time, because
   `log1p(0) = 0` regardless of what didn't fire — a silent target carries
   no distinguishing information, so the model had picked one fixed
   high-confidence answer. Run back through the simulation, that "stimulate
   everyone" answer itself produces loud firing — self-inconsistent for
   the one input regime it couldn't actually solve.
2. **The raw device CSV couldn't be used as training labels directly.**
   Rapid finger-dragging across the touch pad produces releases as little
   as 9ms apart, and many consecutive rows show *identical* per-region
   rates despite naming different clusters — the readout was sampled
   mid-decay from the *previous* touch's residual network activity, not an
   isolated response to that touch. Using those values directly would have
   trained on mislabeled data.

`flappy/inverse_data_device.py` resolves both: it reads which clusters the
CSV shows were actually touched (13/13 — the whole haltere-cluster
population got exercised in one session), then **re-simulates each one
cleanly** from a reset state — one whole-cluster trial plus 8 random-subset
permutations per cluster, at the device's recorded current, held for the
validated 500ms sustain rather than whatever few-millisecond duration the
real touch happened to last. 117 clean samples from 13 clusters. The script
is rerunnable against any future CSV export; each run writes a new,
distinctly-named dataset file rather than overwriting previous ones, so the
training set only grows.

Retraining on everything (broad-random + clustered + device-seeded) fixed
the silence-collapse: **14 fresh round-trip trials, all 12 silent-target
cases correctly proposed an empty stimulation set** (down from "propose
everyone" every time), and the 2 nonzero cases still fired correctly
(IoU 0.22, 0.72).

One targeted follow-up: `SNpp12` — the 2-cell cluster with the largest
measured effect in the whole sweep — regressed to IoU 0.111 after that
retrain (previously 1.0). With only 2 cells, exhaustive coverage is cheap:
33 trials covering all 3 non-empty subsets across currents 4-14mV. This
surfaced a real finding along the way — one of the two cells (body-graph
index 164845) drives nearly the full effect alone (823-1224 spikes across
8-14mV); the other (145211) is much weaker and non-monotonic. `SNpp12`'s
effect is dominated by a single cell, not a synergistic pair. Retraining on
the fully-combined dataset fixed the regression (IoU 1.0 at 6/7 tested
currents).

## Every retrain is a full retrain

Each call to `flappy/inverse_train.py` creates a brand-new,
randomly-initialized `InverseMLP` and trains it for 60 epochs from scratch
on whatever `outputs/flappy_inverse/dataset*.npz` currently matches — never
incremental fine-tuning of previous weights. By the final round this was
four separate dataset files (broad-random, clustered, device-seeded,
`SNpp12`-targeted) merged purely by the glob pattern at train time; adding
a new one and rerunning is the entire update mechanism.

## The Android side: `FlyMapActivity`

Swipe left from `MainActivity` (fling gesture, not literally the Android
"Activity" boundary being the point — it needed its own screen orientation,
landscape, which does require a separate `Activity`). Runs its **own**
`Brain` instance (`FlyLabRunner`), deliberately independent of
`MainActivity`'s camera-driven one — this screen is a stimulate-and-observe
tool over the flight-motor circuit (mirroring `flappy/tune_server.py`'s
role on the desktop), not a continuation of the live-vision demo, so it
runs on a constant baseline field rather than real retina input.

Layout: three landscape columns.

```
[ left haltere pad ] [ fly motor-neuron sprite ] [ right haltere pad ]
```

- **Middle**: `FlySpriteView` draws a 12-region body map (head, thorax,
  abdomen, 2 wings, 6 legs — user-supplied SVG geometry, ported directly to
  Android `Path`/`Canvas` calls, no SVG library dependency) with a
  permanent gray outline per region and a fill that starts fully
  transparent and ramps to that region's assigned color as its
  motor-neuron group's spike rate rises. Region membership comes from
  `flappy/fly_regions.py`, matching connectome `vnc_motor` subclasses to
  body parts (`ad`→abdomen, `wm`+type→thorax/wings, `fl`/`ml`/`hl`+side→legs,
  `hm`→head): 631/708 `vnc_motor` cells covered, the rest (`nm`, `xm`, and
  `wm` types other than b1/b2/hg1) legitimately outside the sprite's scope.
- **Left/right**: `HalterePadView`, one row per haltere connectome-type
  cluster (from `flappy/haltere_cluster_sweep.py`'s grouping, exported
  through the manifest — see below). No named visual subregions were
  supplied for this side, so it's an even vertical split, swappable later
  without touching the stimulation/logging plumbing.
- Every press/release is logged (`StimulationLog`) and exportable as CSV to
  the app's external files dir for pulling onto the desktop.

**New native capability required**: the JNI layer (`brain_jni.cpp`,
`brain.h`, `Brain.java`) had no way to inject an external current at all —
only `nativeStep(luminance, durationMs, sugar, laminaBias)`. Added
`stimIndices`/`stimCurrent` parameters that add to the per-cell `drive`
array *after* the existing retina/lamina/sugar fill, mirroring
`doom/native.py`'s host-side `stimulation` kwarg exactly — the audited
`kernel.cpp` itself is untouched, same as the desktop's own mechanism.

**New manifest sections**, added to `doom/export_android.py`:
`motor_regions` (the 12 sprite regions, indices + hex + description) and
`haltere_clusters` (the same clusters the desktop sweep uses), so the app
reads real connectome structure rather than hardcoding it.

**Verified live on a Pixel 8 Pro**, not just compiled: pressing `SApp_L`
lit up head/thorax/wing_l/wing_r/abdomen and left legs gray, matching the
desktop's own connectivity findings for that cluster; CSV export pulled and
parsed correctly (`timestamp_ms,cluster_name,current_mv,duration_ms,
<region>_hz...`).

**Also removed** while touching this app: the non-Drosophila eye plugins
(`HoneybeeEye`, `DragonflyEye`, `JumpingSpiderEye`, plus their root-level
Python prototype files) — out of scope for this session's ask but requested
alongside it, and a clean, independent change (`BugRegistry` now lists only
`DrosophilaEye`; README and in-app caption text updated to match).

## Known gaps

- The touch pad has no named visual subregions (an even split of whatever
  clusters exist) — the sprite side has exact user-supplied geometry, this
  side doesn't yet.
- `nm` (notum muscle, 24 cells) and `xm` (6 cells) `vnc_motor` subclasses
  have no sprite region and are invisible to this tool entirely.
- The CSV schema is a 12-region aggregate (cluster name + per-region Hz),
  not the same per-cell shape as the desktop dataset (205-dim binary
  stimulation vector, 67-dim per-wm-cell spike counts) —
  `inverse_data_device.py` bridges this by re-simulating cleanly rather
  than converting the raw log, but a future direct on-device per-cell log
  would need a schema change, not just a converter.
- No real accelerometer/IMU mapping exists yet. This entire line of work
  answers "what sensory pattern would cause a given motor response,"
  useful for bootstrapping candidate stimulation sets — it is not sensory
  transduction physics and doesn't claim to model how a real haltere
  converts mechanical bending into afferent firing. That mapping, if it
  gets built, has to come from outside this dataset (literature-grounded
  campaniform-field identities, or an explicitly-labeled engineered
  convention) — flagged and deliberately not resolved here.

## Files

- `flappy/inverse_data.py` — broad-random stimulation → response pairs,
  cold-reset-per-trial, sustained duration.
- `flappy/inverse_data_clusters.py` — structural-cluster-union random
  subsets, wider current range.
- `flappy/inverse_data_device.py` — re-simulates clusters an Android
  session's CSV shows were touched; rerunnable against new exports.
- `flappy/inverse_train.py` — full-retrain-from-scratch numpy MLP
  (`hidden=512`), BCE loss, hand-rolled Adam, reports a marginal-frequency
  baseline every epoch.
- `flappy/inverse_infer.py` — loads a trained model; `propose` (target →
  stimulation) and `roundtrip` (draws a fresh real stimulation, proposes
  from its response, re-simulates the proposal, reports IoU/spike-count
  agreement — the oracle-validation tool used throughout this doc).
- `flappy/haltere_cluster_sweep.py` — full-network, per-named-cluster
  stimulation sweep; also exposes `haltere_clusters()`, reused by the
  device-data and Android export paths.
- `flappy/fly_regions.py` — the 12-region sprite body map → `vnc_motor`
  cell indices.
- `doom/export_android.py` — extended with `motor_regions` and
  `haltere_clusters` manifest sections.
- Android (`~/code/android/androsophila`): `Brain.java`/`brain_jni.cpp`/
  `brain.h` (stimulation plumbing), `MotorRegions.java`, `HalterePads.java`,
  `FlyLabRunner.java`, `FlySpriteView.java`, `HalterePadView.java`,
  `StimulationLog.java`, `FlyMapActivity.java`, swipe wiring in
  `MainActivity.java`.
