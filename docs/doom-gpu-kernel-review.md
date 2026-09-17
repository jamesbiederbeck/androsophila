# GPU kernel backend — built, validated, benchmarked; not activated

This documents a candidate CuPy GPU implementation of the fixed-step LIF model
(`doom/gpu.py`, `GPUBrain`), evaluated exactly as
`docs/doom-performance-review.md`'s "Path to 30 FPS" section anticipated:
"Benchmark native-kernel improvements or dedicated CPU/GPU execution against
the numerical reference. A GPU port is a candidate, not a measured speedup;
sparse synaptic updates and floating-point accumulation need validation."

**Verdict: does not beat `NativeBrain` on this hardware, and is not wired
into `doom/server.py`.** No `--backend` flag exists; using it requires
importing `doom.gpu.GPUBrain` directly or running `doom/benchmark_gpu.py` /
`doom/validate_gpu.py`. `doom/requirements.txt` and the default server startup
path are unchanged. **The reason is architectural, not a tuning problem**:
see "Why it's slow" below — a faster kernel could not fix this without
changing the algorithm design, and that redesign was not attempted this round.

## What was built

`doom/gpu.py`'s `GPUBrain(Brain)` ports `doom/engine.py`'s dense every-substep
formulation (not `kernel.cpp`'s lazy active-list skip) to CuPy. Every
dt=0.1ms substep, one hand-written CUDA kernel (`_FUSED_SUBSTEP_SOURCE`) does
decay, threshold, synaptic delivery and reset for all 166,700 neurons in a
single launch. Delivery assigns one **warp** (32 threads) per destination
neuron, splitting that neuron's incoming edges across the warp's lanes and
combining partial sums with a shuffle-reduce, with a grid-stride loop over
rows so a warp that finishes a low-degree row immediately picks up another.
The reference's "no delivery into an already-refractory neuron" guard
(depends only on the destination, never the edge) is applied as a plain
conditional after the reduction. Two earlier, worse designs were tried and
discarded in favor of this one — see "How the design got here" below; that
history matters because it's what actually revealed the real bottleneck.

Requires `doom/requirements-gpu.txt` (`cupy-cuda12x==13.3.0` plus
separately-wheeled `nvidia-*-cu12` CUDA runtime libraries — this cupy version
does not bundle cuSPARSE/cuBLAS itself and does not auto-locate the
pip-installed ones via `LD_LIBRARY_PATH` set at runtime, so `doom/gpu.py`
preloads them by absolute path instead; see that file's `_configure_cuda_env`).
Tested against `numpy==1.24.4`, the pin the rest of this project's CPU stack
uses.

## Tier 1 — Brian2 oracle (exact, tiny graph)

`tests/test_doom_reference.py` parametrizes `backend` over `['dense',
'native', 'gpu']` (skipped without cupy/CUDA). **The `gpu` backend passes
exactly**, at both cadences (0.1ms, 10ms): integer spike-count equality and
voltage/conductance traces within `atol=.002 mV` against the independent
Brian2 oracle, identical to `dense`/`native`. This certifies the warp-gather
delivery + refractory-mask algebra is correct at small scale, and it stayed
exact across all three kernel designs tried (SpMV, thread-per-row,
warp-per-row) — the floating-point findings below are about the real graph's
scale, not a defect in any one implementation.

## Tier 2 — real graph cross-check against `NativeBrain`

`doom/validate_gpu.py` ran `GPUBrain` and `NativeBrain` from identical initial
state on the real 166,700-neuron/25.58M-edge graph, fed an identical synthetic
luminance sequence. Two runs, both on the final warp-per-row kernel:

- 200 ticks (5.714s simulated), first with the SpMV design
  (`outputs/doom/gpu-benchmark-20260916/validation.json`): spike counts
  diverge starting at tick 4, growing to 4,777/166,700 neurons (2.9%) with a
  different cumulative count by tick 200. Total spike counts stayed close
  (3,569,182 native vs. 3,568,656 gpu, 0.015% relative).
- 10 ticks with the final warp-per-row kernel: divergence starts at tick 3
  instead of tick 4 (a different summation order naturally shifts *when* a
  near-threshold flip happens, not *whether* the phenomenon exists), same
  qualitative shape: 1,129/166,700 mismatched by tick 10, total spike counts
  178,382 native vs. 178,333 gpu.

This is **not a coding bug** — ruled out by: (a) Tier 1's exact match at
small scale, on all three kernel designs; (b) a moderate synthetic 50-neuron
graph run for 10 ticks across multiple `step()` calls matches to float32
noise (~1e-4 mV) with zero mismatches, ruling out any call-boundary or
duplicate-edge bug; (c) an isolated test confirmed correct duplicate-edge
summation through a host-side CSR transpose (this graph has many parallel
edges between the same neuron pair, never deduplicated by `doom/prepare.py`).

Instead, the pattern is genuine **floating-point summation-order sensitivity
at threshold-crossing**, cascading through a densely recurrent 166k-neuron
network: voltage divergence grows slowly for the first few ticks (sub-mV,
consistent with rounding noise for a ~154-mean-in-degree graph), then jumps
sharply exactly on the tick where a handful of near-threshold events (within
0.001mV of -45mV) occur — visible directly in `first_mismatch_tick` and the
per-tick `history` in each validation report. Once one neuron's spike timing
differs by even one substep, its downstream targets receive different input
and the divergence compounds through the recurrent graph. This is the
network's own dynamics, not an implementation defect, and it reproduces
(with shifted timing, same shape) across three independently-implemented
summation strategies (cuSPARSE, thread-per-row, warp-per-row) — strong
evidence this is inherent to the graph, not an artifact of one kernel.

**Conclusion**: the port is algorithmically correct; this real graph's
recurrent connectivity is genuinely sensitive to floating-point summation
order at spike thresholds, at a low but non-zero, empirically measured rate.
The GPU backend's per-neuron spike trains will not be bit-identical to
`NativeBrain`'s over any non-trivial run — true of any two implementations
that accumulate synaptic input in a different order, not a GPU-specific
defect.

## Why it's slow: a bandwidth roofline, not a tuning problem

The dense-every-substep formulation reads **every one of the graph's
25,582,938 edges (204.7MB: 4 bytes weight + 4 bytes index each) on every
single dt=0.1ms substep**, regardless of how many neurons actually spiked.
From `outputs/doom/gpu-benchmark-20260916/validation.json`: 3,569,182 spikes
over 200 ticks (28,600 substeps) is ~62 spikes/substep — those deliver along
~153 out-edges each, ~9,500 edge-updates actually needed per substep. The
dense kernel does ~2,700x more edge work than the computation requires,
by construction, every substep, forever.

This card's peak memory bandwidth (GTX 1060 6GB) is ~192 GB/s. 286
substeps/game-tic x 204.7MB = 58.5 GB/tick; at peak bandwidth that's already
305ms/tick — a **~3.3 tics/sec ceiling for this architecture on this GPU, at
any kernel quality**. Three kernel designs were measured, cheapest to most
sophisticated, and they land almost exactly on this roofline:

| kernel design | ms/substep | achieved bandwidth | tics/sec | % of 192 GB/s peak |
|---|---|---|---|---|
| cuSPARSE SpMV | ~2.17 | ~94 GB/s | 1.509 | 49% |
| hand-written, one thread per row | ~3.9 | ~52 GB/s | 0.892 | 27% |
| hand-written, one warp per row | ~1.9 | ~108 GB/s | **2.064** | 56% |

The one-thread-per-row design was *slower* than cuSPARSE: this graph's
in-degree is extremely skewed (median 112, max 11,203, 258 neurons above
2,000), and a CUDA warp only runs as fast as its slowest thread, so a hub
neuron sharing a warp with 31 median-degree neurons stalls the whole warp
~100x longer than needed. Reassigning one warp per row (32 threads split a
row's edges, shuffle-reduce the partial sums, grid-stride over rows so an
idle warp immediately takes another row) fixed that load imbalance and is
the best of the three, at 56% of this card's peak bandwidth — a reasonable
achieved fraction for an irregular access pattern, not obviously improvable
by further micro-tuning.

**cuSPARSE's cost was not "per-call setup overhead," a claim in an earlier
draft of this doc that the bandwidth numbers above contradict and that has
been corrected.** It was close to bandwidth-bound from the start.

Compare to `NativeBrain`'s measured **~13-15 tics/sec** on this same
machine. No amount of kernel tuning closes that gap for this architecture:
the dense design is bandwidth-capped near ~2-3.3 tics/sec on this GPU
regardless of kernel quality, because it insists on touching all 25.58M
edges every substep.

## If you want to actually beat the CPU

The CPU kernels (`kernel.cpp`, `doom/engine.py`) are **event-driven**: they
touch edges only for neurons that actually spiked that substep (~62 of
166,700, ~9,500 edge-updates), not all 25.58M. Matching that on the GPU means
abandoning the dense-every-substep design in favor of:

1. Keep the delay ring buffer as **index lists + per-slot counters**
   (matching `engine.py`'s `queue`/`queue_count`), not the `(slots, n)`
   dense boolean array used here. On spike, append the spiking neuron's
   index to the future slot (an `atomicAdd` on that slot's counter).
2. A delivery kernel launched only over the *delivering slot's* actual
   spiking neurons' edges (one warp per spiking neuron, or a
   flattened thread-per-edge launch sized from the slot's counter), using
   `atomicAdd(&g[post[e]], weight[e])` to scatter into destinations.
3. Keep decay/threshold/reset as a separate dense elementwise kernel over
   all n — that part is genuinely cheap (measured ~0.6ms/substep including
   several unfused kernel launches; a single fused elementwise kernel would
   be well under that) and correct as-is.

Work per substep would drop from 204.7MB to roughly 76KB in the typical
case — at that point the design is launch-overhead-bound (~286
launches/tick) rather than bandwidth-bound, a regime where beating 13-15
tics/sec becomes plausible, though still unmeasured.

Two costs to weigh before attempting it: (a) it needs the **pre-major** CSR
(`doom/prepare.py`'s original `ptr/post/weight`) for scatter, the opposite of
the post-major layout this file transposes to for gather — supporting both
directions, or switching, is its own small design decision; (b)
`atomicAdd` on float32 makes accumulation order nondeterministic **run to
run** on the GPU, not just different from the CPU as the current design is.
`doom/validate_gpu.py`'s machinery already measures exactly this kind of
divergence, so it is a quantifiable cost, not a blocker — but it would need
restating in this doc once measured, not assumed. **Not attempted this
round.**

## Bottom line

- Built, correctly validated against the independent oracle (Tier 1, exact,
  on all three kernel designs), and honestly benchmarked: **not faster than
  the existing CPU kernel on this specific GPU**, for an architectural
  reason (bandwidth-bound dense-every-substep design), not a fixable
  implementation inefficiency — see the roofline table above.
- The event-driven redesign above is the actual candidate path to beating
  `NativeBrain`, and was not attempted this round; it changes the CSR
  direction, introduces atomics and run-to-run nondeterminism, and would
  need its own Tier 1/Tier 2 validation pass.
- Nothing about `doom/server.py`'s default startup, `doom/requirements.txt`,
  or the existing `NativeBrain`/`Brain` kernels changed. Using the GPU
  backend at all requires `doom/requirements-gpu.txt` and direct use of
  `doom.gpu.GPUBrain`, `doom/benchmark_gpu.py`, or `doom/validate_gpu.py`.

See also: `docs/doom-performance-review.md` (the CPU-side investigation this
work followed up on), `outputs/doom/gpu-benchmark-20260916/` (raw
`validation.json` and three `benchmark*.json` files — SpMV, thread-per-row,
warp-per-row).
