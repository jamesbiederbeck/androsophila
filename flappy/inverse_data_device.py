"""Generates clean, controlled inverse-model training cases seeded by which
haltere clusters were actually touched on the Android haltere-lab screen
(FlyMapActivity's exported CSV, e.g. outputs/flappy_inverse/
device_haltere_stimulation_log.csv) -- not the raw on-device log values
themselves.

Why not reuse the device log's own per-touch readouts directly: rapid
finger-dragging across the pad produces touches as short as 9-30ms, and many
consecutive rows show *identical* per-region rates despite naming different
clusters -- that's residual activity from the previous touch still decaying
when the next one's readout was sampled, not an isolated response to that
touch alone (see the CSV: 20+ back-to-back rows during one drag sequence
share the same six-decimal rates). Reusing those directly would launder
confounded data as clean supervised labels.

Instead, for each cluster the log shows was touched at least once, this
re-simulates cleanly from a reset state, exactly like flappy/inverse_data.py
and flappy/haltere_cluster_sweep.py's other trials:
  1. one whole-cluster stimulation (every cell in that cluster, via
     flappy/haltere_cluster_sweep.py's haltere_clusters());
  2. --permutations random subsets of that same cluster -- partial-cluster
     activation the device touches never isolate (a pad row is on-cluster or
     nothing).

Duration is fixed at 500ms (inverse_data.py's own validated sustain window),
not whatever short duration a given on-device touch happened to last.

Output uses flappy/inverse_data.py's schema (stim/response/amplitude/
haltere_ids/wm_ids), so flappy/inverse_train.py's dataset glob picks this up
automatically. Re-run this against a fresh CSV export to add more
permutations for whichever clusters get touched next -- each run writes a
new, distinctly-named file rather than overwriting previous ones, so the
desktop training set only ever grows.
"""
import argparse
import csv
import json
import time
import uuid
from pathlib import Path
import numpy as np
from doom.native import NativeBrain
from flappy.circuit import haltere_afferents
from flappy.inverse_data import reset, wing_muscle_readouts
from flappy.haltere_cluster_sweep import haltere_clusters

ROOT = Path(__file__).resolve().parents[1]


def touched_clusters(csv_path):
    """{cluster_name: current_mv} for every distinct cluster in the device
    log, using the most recent current recorded for it."""
    out = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            out[row['cluster_name']] = float(row['current_mv'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=str(ROOT / 'outputs/flappy_inverse/device_haltere_stimulation_log.csv'))
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--backend', choices=['native', 'gpu'], default='gpu',
                    help='Frozen-weight simulation, not a learning run -- defaults to gpu.')
    p.add_argument('--duration-ms', type=float, default=500.)
    p.add_argument('--permutations', type=int, default=8, help='Random subset trials per touched cluster.')
    p.add_argument('--min-permutation-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    wanted = touched_clusters(args.csv)
    print(json.dumps({'touched_clusters': wanted}))

    path = ROOT / 'outputs/doom' / args.dataset / 'graph.npz'
    if args.backend == 'gpu':
        from doom.gpu import GPUBrain
        brain = GPUBrain(path)
    else:
        brain = NativeBrain(path)

    haltere_idx = haltere_afferents(brain)
    wm_idx = wing_muscle_readouts(brain)
    clusters = haltere_clusters(brain)
    luminance = np.zeros(len(brain.retina), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    missing = sorted(set(wanted) - set(clusters))
    if missing:
        raise ValueError(f'CSV names clusters not found in this graph: {missing}')

    trials = []   # (indices, amplitude)
    for name, current in wanted.items():
        pool = clusters[name]
        trials.append((pool, current))
        min_k = max(1, int(round(len(pool) * args.min_permutation_frac)))
        for _ in range(args.permutations):
            k = int(rng.integers(min_k, len(pool) + 1)) if len(pool) > 1 else 1
            chosen = rng.choice(pool, size=k, replace=False)
            trials.append((chosen, current))

    n = len(trials)
    stims = np.zeros((n, len(haltere_idx)), dtype=np.uint8)
    responses = np.zeros((n, len(wm_idx)), dtype=np.int32)
    amplitudes = np.zeros(n, dtype=np.float32)

    start = time.time()
    for i, (indices, current) in enumerate(trials):
        reset(brain)
        counts, wall = brain.step(luminance, args.duration_ms, stimulation=(indices, current))
        stim = np.zeros(len(haltere_idx), dtype=np.uint8)
        stim[np.searchsorted(haltere_idx, indices)] = 1
        stims[i] = stim
        responses[i] = counts[wm_idx]
        amplitudes[i] = current
        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - start
            nonzero = int((responses[:i + 1].sum(axis=1) > 0).sum())
            print(json.dumps({'progress': i + 1, 'of': n, 'elapsed_s': round(elapsed, 1),
                               'nonzero_response_trials': nonzero}), flush=True)

    out = Path(args.out) if args.out else ROOT / 'outputs/flappy_inverse' / f'dataset-device-{uuid.uuid4()}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, stim=stims, response=responses, amplitude=amplitudes,
              haltere_ids=brain.ids[haltere_idx], wm_ids=brain.ids[wm_idx],
              duration_ms=args.duration_ms, backend=args.backend, seed=args.seed,
              source_csv=str(args.csv))
    print(json.dumps({'status': 'done', 'out': str(out), 'trials': n,
                       'clusters': len(wanted), 'permutations_per_cluster': args.permutations}))


if __name__ == '__main__':
    main()
