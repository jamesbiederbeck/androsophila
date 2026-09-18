"""Generates training data for an inverse model of the flight reflex circuit:
given the wing-muscle motor spike counts a stimulation produced, learn to
propose which haltere afferents were stimulated to cause them.

Frozen weights throughout -- this runs the existing forward simulation (see
flappy/circuit.py for why haltere afferents are the only sensory population
wired to make wing-muscle motor neurons spike at all) many times from a
reset state to generate (stimulation, response) pairs. It does not train the
connectome itself; see flappy/training.py for the actual learning run.

Each trial resets the brain to its just-loaded cold-start state, then holds
a random subset of haltere afferents at a random current for the full trial
duration. A single short pulse from cold start mostly produces zero
wing-muscle spikes regardless of how many afferents are driven or how hard --
confirmed empirically (9/10 trials silent at duration_ms=50 even with broad,
high-amplitude subsets) -- consistent with flappy/circuit.py's own finding
that this is a sharp-threshold reflex requiring *sustained* drive (their
working configuration holds haltere current on across ~150 game ticks
before DLM reliably fires). duration_ms defaults here to 500ms accordingly.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np
from connectome_sim.native import NativeBrain
from connectome_sim.physiology.common import annotations
from flappy.circuit import haltere_afferents

ROOT = Path(__file__).resolve().parents[1]


def wing_muscle_readouts(brain):
    """Broader than flappy.circuit.wing_motor_readouts (DLM power muscle
    only, 10 cells): every subclass=='wm' cell, ~67 total, matching the
    "50/67 wing-muscle motor neurons" figure in flappy/circuit.py's own
    docstring."""
    a = annotations(brain.ids)
    idx = np.flatnonzero(a.subclass == 'wm')
    if not len(idx): raise ValueError('No wing-muscle motor neurons found in this graph')
    return idx.astype(np.int32)


def reset(brain):
    """Return the brain to its just-loaded cold-start state so trials are
    independent samples. Without this, later trials in a batch ride on
    residual activity ramped up by earlier trials instead of each starting
    from the same baseline -- confirmed empirically: a run of 5 back-to-back
    trials on a never-reset brain went silent/1265/1210/1308 spikes, tracking
    accumulated ramp-up rather than that trial's own stimulation."""
    is_gpu = hasattr(brain, '_cp')
    brain.v[:] = -52
    brain.g[:] = 0
    brain.refractory[:] = 0
    if is_gpu:
        brain._queue[:] = 0
        brain._queue_count[:] = 0
        brain._counts[:] = 0
    else:
        brain.queue[:] = 0
        brain.queue_count[:] = 0
        brain.counts[:] = 0
        brain.previous_drive[:] = 0
        brain.active[:] = 0
        brain.active_flag[:] = 0
        initial = np.unique(np.r_[brain.retina, brain.lamina, brain.sugar])
        brain.active[:len(initial)] = initial
        brain.active_flag[initial] = 1
        brain.nactive[:] = len(initial)
    brain.luminance[:] = 0
    brain.cursor = 0
    brain.total_spikes = 0
    brain.sim_ms = 0


def run_trial(brain, luminance, haltere_idx, wm_idx, rng, duration_ms, min_k, max_k, min_amp, max_amp):
    reset(brain)
    k = int(rng.integers(min_k, max_k + 1))
    chosen = rng.choice(haltere_idx, size=k, replace=False)
    amplitude = float(rng.uniform(min_amp, max_amp))
    counts, wall = brain.step(luminance, duration_ms, stimulation=(chosen, amplitude))
    stim = np.zeros(len(haltere_idx), dtype=np.uint8)
    stim[np.searchsorted(haltere_idx, chosen)] = 1
    return stim, counts[wm_idx].astype(np.int32), amplitude, wall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--backend', choices=['native', 'gpu'], default='gpu',
                    help='Frozen-weight simulation, not a learning run -- defaults to gpu.')
    p.add_argument('--trials', type=int, default=4000)
    p.add_argument('--duration-ms', type=float, default=500.)
    p.add_argument('--min-k', type=int, default=None,
                    help='Default: 30%% of haltere afferents, to avoid mostly-silent trials.')
    p.add_argument('--max-k', type=int, default=None, help='Default: all haltere afferents.')
    p.add_argument('--min-amp', type=float, default=4.)
    p.add_argument('--max-amp', type=float, default=10.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    path = ROOT / 'outputs/doom' / args.dataset / 'graph.npz'
    if args.backend == 'gpu':
        from connectome_sim.gpu import GPUBrain
        brain = GPUBrain(path)
    else:
        brain = NativeBrain(path)

    haltere_idx = haltere_afferents(brain)
    wm_idx = wing_muscle_readouts(brain)
    min_k = args.min_k if args.min_k is not None else max(1, round(0.3 * len(haltere_idx)))
    max_k = args.max_k if args.max_k is not None else len(haltere_idx)
    luminance = np.zeros(len(brain.retina), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    stims = np.zeros((args.trials, len(haltere_idx)), dtype=np.uint8)
    responses = np.zeros((args.trials, len(wm_idx)), dtype=np.int32)
    amplitudes = np.zeros(args.trials, dtype=np.float32)

    start = time.time()
    for i in range(args.trials):
        stim, response, amplitude, wall = run_trial(
            brain, luminance, haltere_idx, wm_idx, rng, args.duration_ms, min_k, max_k, args.min_amp, args.max_amp)
        stims[i] = stim
        responses[i] = response
        amplitudes[i] = amplitude
        if (i + 1) % 200 == 0 or i == args.trials - 1:
            elapsed = time.time() - start
            nonzero = int((responses[:i + 1].sum(axis=1) > 0).sum())
            print(json.dumps({'progress': i + 1, 'of': args.trials, 'elapsed_s': round(elapsed, 1),
                               'nonzero_response_trials': nonzero}), flush=True)

    out = Path(args.out) if args.out else ROOT / 'outputs/flappy_inverse' / f'dataset-{uuid.uuid4()}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, stim=stims, response=responses, amplitude=amplitudes,
             haltere_ids=brain.ids[haltere_idx], wm_ids=brain.ids[wm_idx],
             duration_ms=args.duration_ms, backend=args.backend, seed=args.seed,
             min_k=min_k, max_k=max_k, min_amp=args.min_amp, max_amp=args.max_amp)
    print(json.dumps({'status': 'done', 'out': str(out), 'trials': args.trials}))


if __name__ == '__main__':
    main()
