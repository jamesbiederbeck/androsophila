# GPU kernel backend — event-driven design, beats the CPU kernel on this hardware

This documents a CuPy GPU implementation of the fixed-step LIF model
(`doom/gpu.py`, `GPUBrain`), now in its **event-driven** form: it matches the
CPU kernels' (`kernel.cpp`, `doom/engine.py`) behavior of only touching
edges for neurons that actually spiked, rather than reading the whole graph
every substep. This replaces three earlier dense-every-substep designs
(SpMV, thread-per-row, warp-per-row), all bandwidth-bound by construction —
preserved in `doom/gpu_attempts/` for reference; see that directory's
`README.md` for their numbers.

**Verdict: beats `NativeBrain` on this hardware, measured on an identical
standalone harness — but is not wired into `doom/server.py`.** No
`--backend` flag exists; using it requires importing `doom.gpu.GPUBrain`
directly or running `doom/benchmark_gpu.py` / `doom/validate_gpu.py`.
`doom/requirements.txt` and the default server startup path are unchanged —
this remains an explicit scope decision, not a limitation of the result.

## What was built

`doom/gpu.py`'s `GPUBrain(Brain)` runs two hand-written CUDA kernels per
dt=0.1ms substep, launched back to back on the default stream (same-stream
ordering guarantees kernel 1 completes before kernel 2 starts):

1. **`lif_decay_spike_reset`** — dense elementwise pass over all 166,700
   neurons: decay, threshold, and on spike, immediate reset (matching the
   reference's same-tick-reset/delayed-delivery schedule) plus an atomic
   append of the spiking neuron's index into the *future* delay slot's
   index list (`atomicAdd` on that slot's counter, then a write of the
   index). This stays dense, unlike `kernel.cpp`'s lazy active-list skip,
   because it's uniform branch-free SIMD work and already cheap.
2. **`lif_deliver_scatter`** — event-driven: reads the *current* slot's
   queued spike count directly from device memory (no host sync) and
   grid-strides one warp per queued spiking neuron over the graph's native
   **pre-major** CSR (`connectome_sim/prepare.py`'s `ptr`/`post`/`weight` exactly as
   produced — no transpose needed, unlike the dense designs' post-major
   gather layout), scattering `atomicAdd(&g[post[e]], weight[e])` into each
   destination, guarded by that destination's (already-updated-this-substep)
   refractory state.

The delay ring buffer is an index list (`(slots, n)` int32, one `(slots,)`
int32 counter array) mirroring `doom/engine.py`'s `queue`/`queue_count`
exactly, replacing the dense designs' `(slots, n)` boolean array.

**Reset-ordering note**: the reference's schedule computes a spike's future
delivery *before* resetting that neuron, but the reset always lands after
any same-tick delivery to that same neuron regardless of order, because the
reset unconditionally zeroes `v`/`g` — so whichever runs first, the final
state is identical. This implementation resets immediately inside kernel 1
(setting `refractory` to non-zero at the same time), which makes kernel 2's
existing refractory guard naturally skip delivery to a neuron that just
reset, producing the same final `g=0` as the reference's original ordering.
Verified by Tier 1 exact match below, not just argued.

Requires `doom/requirements-gpu.txt`; unchanged from the prior designs
(`cupy-cuda12x==13.3.0` plus separately-wheeled `nvidia-*-cu12` runtime
libraries, preloaded by `doom/gpu.py`'s `_configure_cuda_env`). Tested
against `numpy==1.24.4`.

## Tier 1 — Brian2 oracle (exact, tiny graph)

`tests/test_doom_reference.py` parametrizes `backend` over `['dense',
'native', 'gpu']`. **The `gpu` backend passes exactly** at both cadences
(0.1ms, 10ms): integer spike-count equality and voltage/conductance traces
within `atol=.002 mV` against the independent Brian2 oracle, identical to
the three prior GPU designs and to `dense`/`native`. This certifies the new
event-driven delivery + immediate-reset-then-guard ordering is correct at
small scale — `pytest tests/test_doom_reference.py -q`: 6 passed.

## Tier 2 — real graph cross-check against `NativeBrain`

`doom/validate_gpu.py` ran `GPUBrain` and `NativeBrain` from identical
initial state on the real 166,700-neuron/25.58M-edge graph, fed an
identical synthetic luminance sequence:

- 10 ticks (`outputs/connectome_sim/gpu-benchmark-20260916/validation-event-driven-10tick.json`):
  1,129/166,700 neurons (0.68%) mismatched by tick 10, first mismatch at
  tick 3. Total spikes: 178,382 native vs. 178,333 gpu.
- 200 ticks (`.../validation-event-driven-200tick.json`): 4,829/166,700
  (2.9%) mismatched by tick 200, first mismatch at tick 3. Total spikes:
  3,569,182 native vs. 3,568,496 gpu (0.019% relative).

These numbers are essentially identical in magnitude and shape to the prior
dense warp-per-row kernel's run (4,777/166,700 at 200 ticks, first mismatch
tick 4) — strong evidence this is the same underlying phenomenon
(floating-point summation-order sensitivity at spike thresholds, cascading
through a densely recurrent 166k-neuron network), not a defect introduced
by the event-driven redesign or its reordered reset. See the prior version
of this doc (in git history) for the full case ruling out a coding bug —
duplicate-edge handling, call-boundary correctness, and a moderate synthetic
50-neuron cross-check all still apply unchanged, since the underlying
per-edge math didn't change, only which edges get touched and when.

**Run-to-run determinism (new for this design)**: `atomicAdd` on float32 `g`
makes the order two concurrent scatter writes land in nondeterministic
across GPU thread-scheduling runs, unlike every prior design's fixed
CSR-order summation. Three repeated 50-tick runs of `GPUBrain` alone, from
identical initial state and input, produced **zero spike-count differences
out of 166,700 neurons across all three runs**. This is a real measurement,
not a guarantee: at ~62 spikes/substep scattering across ~9,500 edges into
166,700 possible destinations, most destinations receive at most one
contribution per substep, and a single float32 add has no order to be
sensitive to. The collision rate (>1 contribution to the same neuron in the
same substep) is workload- and connectivity-dependent, so "not observed to
be nondeterministic on this workload" is the accurate claim, not
"deterministic."

## Honest throughput comparison

**The previously reported ~13-15 tics/sec `NativeBrain` figure was measured
through the full `doom.server`/`run_broadcast.sh` pipeline (ViZDoom
stepping, rendering, retina sampling, broadcast) — not the same harness as
`doom/benchmark_gpu.py`, which drives `GPUBrain` standalone with synthetic
luminance and no game.** Comparing those two directly would repeat exactly
the kind of apples-to-oranges mistake this doc's history already corrected
once (the earlier wrong "overhead-bound" cuSPARSE claim). To fix that, both
backends were benchmarked on the identical standalone harness/workload:

| backend | harness | tics/sec | median step wall |
|---|---|---|---|
| `NativeBrain` (`outputs/connectome_sim/gpu-benchmark-20260916/benchmark-native-standalone.json`) | standalone, synthetic luminance, no server/game | 18.347 | 53.5 ms |
| `GPUBrain`, event-driven (`.../benchmark-event-driven.json`) | standalone, synthetic luminance, no server/game | **65.094** | 14.9 ms |

**~3.5x faster on this identical harness.** This is the number that's
comparable; the server-pipeline ~13-15 tics/sec figure in
`docs/doom-performance-review.md` measures a different, larger workload
(game stepping + rendering + broadcast, not just the neural kernel) and
should not be read as "GPU is only ~4-5x faster than the full app" — that
was an earlier, less careful framing corrected here.

**One workload caveat, in the CPU's favor, not accounted for above**: the
benchmark's synthetic input is uniform-random luminance sampled every tick,
which drives nearly every retina neuron nonzero every tick and keeps
`kernel.cpp`'s lazy active-list large and busy. Realistic game frames (dark
scenes, static views) would let the CPU's lazy-exact kernel skip more work
than this benchmark exercises, while the GPU kernel's cost is
input-independent (same two kernel launches regardless of how many neurons
are actually near threshold). This caveat is checked directly in the next
section, not left as a guess.

## Organic-workload check: real gameplay, not synthetic noise

`doom/capture_organic_workload.py` drives a real closed-loop ViZDoom episode
(`combat_survival`, seed 41027) exactly as `doom/server.py`'s `run_loop`
does: `NativeBrain` makes real decisions via `NeuralControls`, and those
actions genuinely change what the game shows next. The per-tick captured
light array and substep count are saved
(`outputs/connectome_sim/gpu-benchmark-20260916/organic-capture-combat_survival.npz`,
86% of retina samples nonzero on average — this scenario is not a dark
level). `doom/replay_organic_workload.py` then replays that identical
sequence into fresh `NativeBrain` and `GPUBrain` instances from identical
initial state — same fairness principle as `doom/validate_gpu.py`, just
with correlated, closed-loop-derived input instead of i.i.d. uniform noise
(`outputs/connectome_sim/gpu-benchmark-20260916/organic-replay-report.json`):

| | tics/sec | median step wall |
|---|---|---|
| `NativeBrain` | 18.743 | 52.2 ms |
| `GPUBrain`, event-driven | 60.762 | 14.8 ms |

**~3.24x faster on real gameplay input** — close to, and consistent with,
the 3.5x measured on synthetic uniform noise, and in the direction the
caveat above predicted (real gameplay is somewhat less favorable to the GPU
than worst-case-for-the-CPU synthetic noise, but the difference here is
small, not dramatic: this scenario's frames are mostly bright, so the CPU's
lazy active-list doesn't get much chance to shrink either way). Spike
totals agree to 5.3e-5 relative (4,604,627 native vs. 4,604,384 gpu);
5,842/166,700 neurons (3.5%) show the same floating-point summation-order
divergence pattern measured in every prior validation run on this graph —
consistent magnitude, not a new defect introduced by testing against real
input. The 3.0-3.5x range across repeated runs and workloads (synthetic and
organic) is the honest speedup estimate for this design on this hardware,
not a single cherry-picked number.

## Why it's fast now: what actually costs time per substep

The event-driven design's delivery kernel now does genuinely little work:
~62 spikes/substep × ~153 out-edges ≈ 9,500 edges × 8 bytes ≈ 76KB/substep,
negligible against this card's ~192 GB/s peak bandwidth. The dense
elementwise pass over all 166,700 neurons (~32 bytes read+write per neuron)
is ~5.3MB/substep, ~28µs at peak bandwidth. Measured is ~52µs/substep
(14.9ms ÷ 286 substeps/tic) — the difference is two kernel launches plus a
per-substep host-side scalar write (`self._queue_count[slot] = 0`), not
edge bandwidth. **This design is now launch-overhead-bound, not
bandwidth-bound** — exactly the regime the prior version of this doc
predicted would be needed to beat the CPU, and the prior dense-every-substep
bandwidth-roofline analysis (25.58M edges/substep, ~2-3.3 tics/sec ceiling)
no longer applies to this design; it's retired here, not extended.

## Bottom line

- Built, exactly validated against the independent Brian2 oracle (Tier 1),
  and honestly benchmarked against `NativeBrain` on an identical standalone
  harness: **~3.5x faster** (65.1 vs. 18.3 tics/sec) on synthetic
  uniform-random input, and **~3.2x faster** (60.8 vs. 18.7 tics/sec) on a
  captured real closed-loop `combat_survival` gameplay sequence, both on
  this GTX 1060. The two numbers agree closely, so 3.0-3.5x is the honest
  range for this design on this hardware, not a synthetic-input artifact.
- Tier 2 real-graph cross-check against `NativeBrain` shows the same
  floating-point summation-order divergence pattern as every prior GPU
  design, at the same magnitude — not a new defect from this redesign.
  Run-to-run nondeterminism from `atomicAdd` was not observed at this
  workload's scale (0/166,700 across 3 repeated runs), but is not
  architecturally guaranteed and should be re-measured if the workload
  (spike rate, connectivity) changes materially.
- Nothing about `doom/server.py`'s default startup, `doom/requirements.txt`,
  or the existing `NativeBrain`/`Brain` kernels changed. Using the GPU
  backend at all requires `doom/requirements-gpu.txt` and direct use of
  `doom.gpu.GPUBrain`, `doom/benchmark_gpu.py`, or `doom/validate_gpu.py`.
- Real-gameplay measurement is now done (see above): the 3.0-3.5x range
  holds on both synthetic and organic input, so the remaining open step is
  deciding whether to wire a `--backend gpu` flag into `doom/server.py` —
  not attempted here, since that was scoped out of both rounds of this
  work, not because the number is still in question.

See also: `docs/doom-performance-review.md` (the CPU-side investigation this
work followed up on), `doom/gpu_attempts/README.md` (the three superseded
dense-every-substep designs and their numbers), `outputs/connectome_sim/gpu-benchmark-20260916/`
(all raw benchmark/validation JSON, including the three dense-design files
from the prior round, this round's `benchmark-event-driven.json`,
`benchmark-native-standalone.json`, `validation-event-driven-10tick.json`,
`validation-event-driven-200tick.json`, and the organic-workload
`organic-capture-combat_survival.npz`/`organic-replay-report.json`).
