"""Standalone Flappy Bird learning-run driver: dopamine-gated KC->MBON11
plasticity enabled (doom_learning_v6's rule and VisualMemoryBrain, reused
unmodified), driven by crashes into the ground or a pipe as the aversive
event instead of Doom's health loss -- see flappy/training.py for why that's
a deliberate divergence from doom/training.py's convention.

Weights genuinely change over a run (see the final memory() delta below);
there is no established path from KC->MBON11 to the DLM wing motor neurons
flappy.circuit.wing_motor_readouts reads for the flap decision, so this does
not and cannot demonstrate improved flap timing or survival -- same
"mechanism enabled, not validated" stance as doom/training.py and the v6
README. This is a local script, not a server -- no checkpointing or
audit-archive machinery, matching flappy/play.py's scope.
"""
import argparse
import json
import time
from pathlib import Path
from doom_learning_v6.calibration import calibrated_brain
from flappy.circuit import haltere_afferents, haltere_current_for_velocity, wing_motor_readouts
from flappy.controls import FlapControls
from flappy.game import Game, FPS
from flappy.training import FlapTraining

ROOT = Path(__file__).resolve().parents[1]


def run(ticks, seed, eta, learning, haltere_m, haltere_n, no_pipes):
    brain = calibrated_brain(eta=eta)
    training = FlapTraining(brain, enabled=learning)
    haltere = haltere_afferents(brain)
    controls = FlapControls(wing_motor_readouts(brain))
    game = Game(seed=seed, no_pipes=no_pipes)
    duration_ms = 1000 / FPS
    memory_before = training.telemetry()
    episode_lengths, flaps = [], 0
    start = time.perf_counter()
    for _ in range(ticks):
        before = game.observation()
        if before['finished']:
            episode_lengths.append(before['tick'])
            game.new_episode()
            training.new_round()
            before = game.observation()
        frame = game.pixels()
        current = haltere_current_for_velocity(before['y_velocity'], m=haltere_m, n=haltere_n)
        haltere_stim = (haltere, current) if current > 0 else None
        counts, _ = training.step(frame, duration_ms, haltere_stimulation=haltere_stim)
        action = controls.decode(counts, duration_ms / 1000)
        flaps += int(action['flap'])
        game.act(action['flap'])
        training.observe(game.observation()['finished'])
    episode_lengths.append(game.observation()['tick'])
    wall = time.perf_counter() - start
    memory_after = training.telemetry()
    game.close()
    return {
        'ticks': ticks, 'seed': seed, 'eta': eta, 'learning': learning,
        'haltere_m': haltere_m, 'haltere_n': haltere_n, 'no_pipes': no_pipes,
        'episodes_completed': len(episode_lengths), 'episode_lengths': episode_lengths,
        'flaps': flaps, 'wall_seconds': round(wall, 3),
        'ticks_per_second': round(ticks / wall, 3) if wall > 0 else None,
        'memory_before': memory_before, 'memory_after': memory_after,
        'weights_changed': memory_after['changed_edges'] > 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ticks', type=int, default=600)
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--eta', type=float, default=.001)
    p.add_argument('--learning', dest='learning', action='store_true', default=True)
    p.add_argument('--no-learning', dest='learning', action='store_false',
                    help='Freeze weights (control run): traces still decay, nothing is written')
    p.add_argument('--haltere-m', type=float, default=1.75)
    p.add_argument('--haltere-n', type=float, default=2.)
    p.add_argument('--no-pipes', action='store_true')
    args = p.parse_args()
    report = run(args.ticks, args.seed, args.eta, args.learning,
                 args.haltere_m, args.haltere_n, args.no_pipes)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
