"""Standalone Flappy Bird driver for the connectome simulation, no server/UI.

Drives the same fixed synaptic weights (outputs/doom/<dataset>/graph.npz) used
by doom/server.py through an unrelated sensory context -- no claim this is
biologically meaningful for Flappy Bird specifically, same honesty stance as
the rest of doom/. Mirrors the shape of doom/benchmark_gpu.py: a local script
for building and validating a harness, not the full HTTP broadcast/checkpoint/
audit-archive machinery doom/server.py has -- that's a separate, later step if
this is worth wiring up live.
"""
import argparse
import json
import time
from pathlib import Path
from doom.native import NativeBrain
from flappy.controls import FlapControls
from flappy.game import Game, FPS
from vision.retina import BilinearLuminance

ROOT = Path(__file__).resolve().parents[1]


def run(ticks, dataset, seed, backend):
    manifest = json.loads((ROOT / 'outputs/doom' / dataset / 'manifest.json').read_text())
    path = ROOT / 'outputs/doom' / dataset / 'graph.npz'
    if backend == 'gpu':
        from doom.gpu import GPUBrain
        brain = GPUBrain(path)
    else:
        brain = NativeBrain(path)
    controls = FlapControls(manifest['readouts'])
    game = Game(seed=seed)
    retina = BilinearLuminance()
    duration_ms = 1000 / FPS
    best_score, episodes, total_spikes = 0, 0, 0
    start = time.perf_counter()
    for _ in range(ticks):
        if game.observation()['finished']:
            episodes += 1
            best_score = max(best_score, game.observation()['score'])
            game.new_episode()
        frame = game.pixels()
        light = retina.sample(frame, brain.uv)
        counts, _ = brain.step(light, duration_ms, sugar=False)
        action = controls.decode(counts, duration_ms / 1000)
        total_spikes += int(counts.sum())
        game.act(action['flap'])
    best_score = max(best_score, game.observation()['score'])
    wall = time.perf_counter() - start
    game.close()
    return {
        'dataset': dataset, 'ticks': ticks, 'seed': seed, 'backend': backend,
        'neurons': int(brain.n), 'episodes_completed': episodes, 'best_score': int(best_score),
        'total_spikes': total_spikes, 'wall_seconds': round(wall, 3),
        'ticks_per_second': round(ticks / wall, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=200)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--backend', choices=['native', 'gpu'], default='native')
    args = p.parse_args()
    report = run(args.ticks, args.dataset, args.seed, args.backend)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
