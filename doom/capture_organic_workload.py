"""Capture a real closed-loop ViZDoom luminance sequence, for replay into
multiple GPUBrain/NativeBrain backends from identical initial state.

Unlike doom/validate_gpu.py's/doom/benchmark_gpu.py's synthetic
np.random.default_rng uniform luminance, this drives an actual ViZDoom
episode with NativeBrain making real decisions via NeuralControls, exactly
as doom/server.py's run_loop does (frame -> retinal_samples -> brain.step ->
controls.decode -> game.act). The captured per-tick light array and substep
count are then replayable into any backend for a fair, identical-input
comparison against a workload with the temporal/spatial correlation
structure and closed-loop feedback that synthetic i.i.d. noise lacks.

The recording brain (always NativeBrain, so this file has no GPU
dependency) is discarded after capture -- only its light/steps sequence is
saved. Whatever backend later replays this sequence does not feed its own
actions back into the game; that would make two backends' inputs diverge
after the first differing spike, which is the exact reason this script
exists (capture once, replay identically) rather than driving each backend
through its own live closed loop.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from connectome_sim.native import NativeBrain
from connectome_sim.engine import NeuralControls
from doom.game import Game, retinal_samples

ROOT = Path(__file__).resolve().parents[1]

def capture(ticks, dataset, scenario, seed, decoder):
    manifest = json.loads((ROOT / 'outputs/doom' / dataset / 'manifest.json').read_text())
    brain = NativeBrain(ROOT / 'outputs/doom' / dataset / 'graph.npz')
    controls = NeuralControls(manifest['readouts'], mode=decoder)
    game = Game(seed=seed, scenario=scenario, spectator=False)
    lights, steps_list = [], []
    cursor = 0
    try:
        for tick in range(1, ticks + 1):
            if game.observation()['finished']: game.new_episode()
            frame = game.pixels()
            light = retinal_samples(frame, brain.uv)
            target = int(round(tick * 10000 / 35)); steps = target - cursor; cursor = target
            counts, _ = brain.step(light, steps * .1, sugar=False)
            action = controls.decode(counts, steps * .1 / 1000)
            game.act(action)
            lights.append(light.copy()); steps_list.append(steps)
    finally:
        game.close()
    return np.stack(lights), np.asarray(steps_list, dtype=np.int64), int(brain.total_spikes)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=300)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--scenario', default='combat_survival', choices=['combat_survival', 'defend_the_center'])
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--decoder', default='bci', choices=['biological', 'bci'])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    lights, steps, total_spikes = capture(args.ticks, args.dataset, args.scenario, args.seed, args.decoder)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lights=lights, steps=steps,
                         nonzero_luminance_fraction=float((lights > 0.02).mean()))
    print(json.dumps({
        'ticks': args.ticks, 'scenario': args.scenario, 'seed': args.seed, 'decoder': args.decoder,
        'out': str(out), 'recording_backend_total_spikes': total_spikes,
        'nonzero_luminance_fraction': float((lights > 0.02).mean()),
    }, indent=2))

if __name__ == '__main__':
    main()
