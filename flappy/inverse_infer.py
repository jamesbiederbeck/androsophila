"""Inference for the inverse flight-circuit model trained by
flappy.inverse_train: given a target wing-muscle motor spike pattern,
propose which haltere afferents to stimulate to produce something like it.

Not exact -- see flappy/inverse_data.py's docstring: many different
stimulated subsets can produce a similar motor response, so this predicts
a plausible one, not the one, and its own validation numbers (66% cell-wise
accuracy against the specific subsets sampled during training, vs ~50% for
guessing each cell's marginal rate) say it is a genuinely-better-than-chance
proposer, not a solved inverse problem.

--roundtrip runs the actual empirical check that matters: draws a fresh
random stimulation from the real forward sim (never seen during training),
records its motor response, asks the model to propose a stimulation for
that response, then runs the *proposed* stimulation back through the same
forward sim to see whether it actually reproduces comparable motor spiking
-- not just whether it matches the training label.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from connectome_sim.native import NativeBrain
from flappy.circuit import haltere_afferents
from flappy.inverse_data import reset, wing_muscle_readouts

ROOT = Path(__file__).resolve().parents[1]


def load_model(path):
    d = np.load(path)
    return {k: d[k] for k in ['W1', 'b1', 'W2', 'b2', 'response_log_mean', 'response_log_std', 'haltere_ids', 'wm_ids']}


def propose(model, response_counts):
    response_counts = np.asarray(response_counts, dtype=np.float32)
    if response_counts.shape != model['wm_ids'].shape:
        raise ValueError(f"Expected {len(model['wm_ids'])} wing-muscle motor spike counts, got {response_counts.shape}")
    x = (np.log1p(response_counts) - model['response_log_mean']) / model['response_log_std']
    h = np.maximum(x @ model['W1'] + model['b1'], 0)
    logits = h @ model['W2'] + model['b2']
    return 1 / (1 + np.exp(-logits))


def roundtrip(model_path, backend, duration_ms, min_k, max_k, min_amp, max_amp, threshold, seed):
    model = load_model(model_path)
    path = ROOT / 'outputs/doom/malecns_v1/graph.npz'
    if backend == 'gpu':
        from connectome_sim.gpu import GPUBrain
        brain = GPUBrain(path)
    else:
        brain = NativeBrain(path)

    haltere_idx = haltere_afferents(brain)
    wm_idx = wing_muscle_readouts(brain)
    if not np.array_equal(brain.ids[haltere_idx], model['haltere_ids']) or not np.array_equal(brain.ids[wm_idx], model['wm_ids']):
        raise ValueError('Model cell indexing does not match this graph -- retrain or regenerate the dataset first')
    luminance = np.zeros(len(brain.retina), dtype=np.float32)
    rng = np.random.default_rng(seed)

    reset(brain)
    k1 = int(rng.integers(min_k, max_k + 1))
    stim1_idx = rng.choice(haltere_idx, size=k1, replace=False)
    amp1 = float(rng.uniform(min_amp, max_amp))
    response1, _ = brain.step(luminance, duration_ms, stimulation=(stim1_idx, amp1))
    response1 = response1[wm_idx]

    prob = propose(model, response1)
    proposed_mask = prob > threshold
    stim2_idx = haltere_idx[proposed_mask]
    if len(stim2_idx) == 0:
        response2 = np.zeros(len(wm_idx), dtype=np.int32)
    else:
        reset(brain)
        response2, _ = brain.step(luminance, duration_ms, stimulation=(stim2_idx, 8.0))
        response2 = response2[wm_idx]

    stim1_mask = np.isin(haltere_idx, stim1_idx)
    intersection = int((stim1_mask & proposed_mask).sum())
    union = int((stim1_mask | proposed_mask).sum())
    print(json.dumps({
        'original_stimulation': {'k': k1, 'amplitude': round(amp1, 2), 'wm_spikes': int(response1.sum())},
        'proposed_stimulation': {'k': int(proposed_mask.sum()), 'amplitude_used': 8.0, 'wm_spikes': int(response2.sum())},
        'overlap_iou': round(intersection / union, 3) if union else None,
        'both_fired': bool(response1.sum() > 0 and response2.sum() > 0),
        'both_silent': bool(response1.sum() == 0 and response2.sum() == 0),
    }, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    infer_p = sub.add_parser('propose', help='Propose a stimulation for a given motor spike-count target.')
    infer_p.add_argument('--model', default=str(ROOT / 'outputs/flappy_inverse/inverse_model_h512.npz'))
    infer_p.add_argument('--target', required=True, help='JSON array of wing-muscle motor spike counts, in wm_ids order.')
    infer_p.add_argument('--threshold', type=float, default=0.5)

    ids_p = sub.add_parser('show-ids', help='Print the ordered wm_ids/haltere_ids a model expects.')
    ids_p.add_argument('--model', default=str(ROOT / 'outputs/flappy_inverse/inverse_model_h512.npz'))

    rt_p = sub.add_parser('roundtrip', help='Empirically test the model against the real forward simulation.')
    rt_p.add_argument('--model', default=str(ROOT / 'outputs/flappy_inverse/inverse_model_h512.npz'))
    rt_p.add_argument('--backend', choices=['native', 'gpu'], default='gpu')
    rt_p.add_argument('--duration-ms', type=float, default=500.)
    rt_p.add_argument('--min-k', type=int, default=None)
    rt_p.add_argument('--max-k', type=int, default=None)
    rt_p.add_argument('--min-amp', type=float, default=4.)
    rt_p.add_argument('--max-amp', type=float, default=10.)
    rt_p.add_argument('--threshold', type=float, default=0.5)
    rt_p.add_argument('--seed', type=int, default=0)

    args = p.parse_args()
    if args.command == 'show-ids':
        model = load_model(args.model)
        print(json.dumps({'wm_ids': model['wm_ids'].tolist(), 'haltere_ids': model['haltere_ids'].tolist()}))
    elif args.command == 'propose':
        model = load_model(args.model)
        prob = propose(model, json.loads(args.target))
        chosen = np.flatnonzero(prob > args.threshold)
        print(json.dumps({'proposed_haltere_ids': model['haltere_ids'][chosen].tolist(),
                           'count': int(len(chosen)), 'of': int(len(model['haltere_ids'])),
                           'probabilities': [round(float(x), 3) for x in prob]}))
    else:
        min_k = args.min_k if args.min_k is not None else 60
        max_k = args.max_k if args.max_k is not None else 205
        roundtrip(args.model, args.backend, args.duration_ms, min_k, max_k, args.min_amp, args.max_amp, args.threshold, args.seed)


if __name__ == '__main__':
    main()
