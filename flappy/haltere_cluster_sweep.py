"""Characterizes each haltere afferent connectome-type cluster by its
*measured* motor effect, not its assumed biological identity: stimulates
each cluster alone (never mixed with another cluster) at several sustained
currents and records the resulting wing-muscle motor spike vector.

This is the honest substitute for a real haltere biomechanics model (which
this repo does not have and the connectome cannot supply -- see
flappy/circuit.py and flappy/inverse_data.py's docstrings): instead of
claiming any cluster's *physical* tuning (which physical acceleration axis
it senses), we only claim what we can actually measure here -- what motor
pattern results when that cluster alone is driven. Any IMU-axis assignment
built on top of this table is an engineered convention chosen for
behavioral/effect distinctiveness, not a biological claim.

Clusters are the connectome type/instance groups found by inspecting direct
(1-hop) haltere->wing-muscle-motor-neuron connectivity: SApp splits cleanly
by side (SApp_L/SApp_R); the SNpp##/SNxx## clusters are small (2-8 cells)
and structurally distinct (some reach only steering muscles, one -- SNpp14
-- is the only direct line to the DLM/DVM power muscles, several have zero
direct 1-hop connectivity at all). See this run's own printed connectivity
table for the source of these group boundaries.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from connectome_sim.native import NativeBrain
from flappy.circuit import haltere_afferents
from flappy.inverse_data import reset, wing_muscle_readouts

ROOT = Path(__file__).resolve().parents[1]


def haltere_clusters(brain):
    """Group haltere afferents by (type, instance) -- the same grouping
    used to find the functional connectivity split. Returns
    {group_name: np.ndarray of brain-index positions}."""
    a = feather.read_table(ROOT / 'connectome_data/malecns_v1/annotations.feather').to_pandas().set_index('bodyId').loc[brain.ids]
    haltere_idx = haltere_afferents(brain)
    sub = a.iloc[haltere_idx]
    names = (sub['type'].fillna('none') + '_' + sub['instance'].fillna('')).values
    groups = {}
    for name, idx in zip(names, haltere_idx):
        groups.setdefault(str(name), []).append(int(idx))
    return {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--backend', choices=['native', 'gpu'], default='gpu',
                    help='Frozen-weight simulation, not a learning run -- defaults to gpu.')
    p.add_argument('--duration-ms', type=float, default=500.)
    p.add_argument('--currents', type=float, nargs='+', default=[4., 6., 8., 10., 12.])
    p.add_argument('--out', default=str(ROOT / 'outputs/flappy_inverse/cluster_sweep.npz'))
    args = p.parse_args()

    path = ROOT / 'outputs/doom' / args.dataset / 'graph.npz'
    if args.backend == 'gpu':
        from connectome_sim.gpu import GPUBrain
        brain = GPUBrain(path)
    else:
        brain = NativeBrain(path)

    wm_idx = wing_muscle_readouts(brain)
    clusters = haltere_clusters(brain)
    luminance = np.zeros(len(brain.retina), dtype=np.float32)

    group_names = sorted(clusters, key=lambda k: -len(clusters[k]))
    responses = np.zeros((len(group_names), len(args.currents), len(wm_idx)), dtype=np.int32)

    for gi, name in enumerate(group_names):
        idx = clusters[name]
        for ci, current in enumerate(args.currents):
            reset(brain)
            counts, wall = brain.step(luminance, args.duration_ms, stimulation=(idx, float(current)))
            responses[gi, ci] = counts[wm_idx]
            total = int(counts[wm_idx].sum())
            print(json.dumps({'group': name, 'n_cells': int(len(idx)), 'current': current,
                               'total_wm_spikes': total, 'wall_s': round(wall, 3)}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, responses=responses, group_names=np.array(group_names),
             group_sizes=np.array([len(clusters[g]) for g in group_names]),
             currents=np.array(args.currents), wm_ids=brain.ids[wm_idx],
             duration_ms=args.duration_ms, backend=args.backend)
    print(json.dumps({'status': 'done', 'out': str(out)}))


if __name__ == '__main__':
    main()
