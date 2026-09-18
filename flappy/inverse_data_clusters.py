"""Augments flappy/inverse_data.py's training set with stimulations drawn
from structurally-defined haltere clusters (flappy/haltere_cluster_sweep.py)
instead of uniform random subsets of the whole population.

Why this exists: the inverse model trained only on flappy/inverse_data.py's
broad random subsets (k >= 30% of 205) failed to recover small, structurally
distinct clusters when tested against flappy/haltere_cluster_sweep.py's own
measured results -- IoU 0.0-0.14 for clusters of 2-8 cells (including SNpp12
and SNpp23, the two groups the sweep found have the largest actual network
effect), versus 0.25-0.36 for the ~75-cell SApp groups, which were at least
close to the training distribution's scale. This generates the missing
regime: single clusters, unions of a few clusters, and random subsets
*within* a cluster (not just fully on/off), at a wider current range -- the
sweep needed up to 12mV for some clusters to fire at all, beyond
inverse_data.py's default 4-10mV range.

Output uses the same schema as flappy/inverse_data.py (stim/response/
amplitude/haltere_ids/wm_ids), so flappy/inverse_train.py's dataset glob
picks up both without any changes.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np
from connectome_sim.native import NativeBrain
from flappy.circuit import haltere_afferents
from flappy.inverse_data import reset, wing_muscle_readouts
from flappy.haltere_cluster_sweep import haltere_clusters

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--backend', choices=['native', 'gpu'], default='gpu',
                    help='Frozen-weight simulation, not a learning run -- defaults to gpu.')
    p.add_argument('--trials', type=int, default=2500)
    p.add_argument('--duration-ms', type=float, default=500.)
    p.add_argument('--min-amp', type=float, default=4.)
    p.add_argument('--max-amp', type=float, default=14.)
    p.add_argument('--max-clusters', type=int, default=3, help='Max number of clusters unioned per trial before subsampling.')
    p.add_argument('--seed', type=int, default=2)
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
    clusters = haltere_clusters(brain)
    names = list(clusters)
    luminance = np.zeros(len(brain.retina), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    stims = np.zeros((args.trials, len(haltere_idx)), dtype=np.uint8)
    responses = np.zeros((args.trials, len(wm_idx)), dtype=np.int32)
    amplitudes = np.zeros(args.trials, dtype=np.float32)

    start = time.time()
    for i in range(args.trials):
        n_clusters = int(rng.integers(1, args.max_clusters + 1))
        chosen_names = rng.choice(names, size=n_clusters, replace=False)
        pool = np.unique(np.concatenate([clusters[n] for n in chosen_names]))
        # A random subset of the union, not always all of it -- covers
        # partial-cluster activation too, not just whole-cluster on/off.
        k = int(rng.integers(1, len(pool) + 1))
        chosen = rng.choice(pool, size=k, replace=False)
        amplitude = float(rng.uniform(args.min_amp, args.max_amp))
        reset(brain)
        counts, wall = brain.step(luminance, args.duration_ms, stimulation=(chosen, amplitude))
        stim = np.zeros(len(haltere_idx), dtype=np.uint8)
        stim[np.searchsorted(haltere_idx, chosen)] = 1
        stims[i] = stim
        responses[i] = counts[wm_idx]
        amplitudes[i] = amplitude
        if (i + 1) % 200 == 0 or i == args.trials - 1:
            elapsed = time.time() - start
            nonzero = int((responses[:i + 1].sum(axis=1) > 0).sum())
            print(json.dumps({'progress': i + 1, 'of': args.trials, 'elapsed_s': round(elapsed, 1),
                               'nonzero_response_trials': nonzero}), flush=True)

    out = Path(args.out) if args.out else ROOT / 'outputs/flappy_inverse' / f'dataset-clustered-{uuid.uuid4()}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, stim=stims, response=responses, amplitude=amplitudes,
             haltere_ids=brain.ids[haltere_idx], wm_ids=brain.ids[wm_idx],
             duration_ms=args.duration_ms, backend=args.backend, seed=args.seed)
    print(json.dumps({'status': 'done', 'out': str(out), 'trials': args.trials}))


if __name__ == '__main__':
    main()
