"""Live reinforcement adapter for Flappy Bird, mirroring doom/training.py's
DamageTraining pattern: a crash schedules the next-step +4 mV-equivalent
PPL101 current for 200 ms (2,000 x 0.1 ms steps) -- the same chosen
sensitivity-test value doom_learning_v4 introduced and doom/training.py
already uses; not re-derived here, not a measured quantity.

Deliberate divergence from doom/training.py: Doom distinguishes non-fatal
damage (schedules a pulse) from terminal damage (does not, and a following
new_round() cancels any pulse still pending). Flappy Bird has no non-fatal
failure -- every crash ends the episode -- so the crash itself has to be the
trigger, and new_round() here does *not* cancel a pulse in progress: it is
allowed to continue into the next episode's early ticks, consistent with
this repo's existing "neural/weight state persists across game resets"
convention (doom-neuroscience-review.md).

Only images and the crash signal enter the learning rule. Flap decoding
reads DLM (flappy.circuit.wing_motor_readouts); KC->MBON11 plasticity has no
engineered path to those cells (checked directly: DLM's strongest direct
presynaptic partners are unnamed VNC interneurons and a couple of descending
neurons, none of them MBON-related). Weights genuinely change here -- see
telemetry()'s memory() report -- but there is no basis to claim flap timing
or survival improves as a result. Same "mechanism enabled, not validated"
stance as doom/training.py and the v6 README.
"""
import math
import numpy as np


class FlapTraining:
    def __init__(self, brain, enabled=True):
        self.brain = brain
        self.enabled = bool(enabled)
        self.brain.weights_frozen = not self.enabled
        self.until = 0
        self.events = 0
        self.delivered_steps = 0
        self.last_steps = 0

    def new_round(self):
        # Unlike doom/training.py's DamageTraining.new_round(), this does not
        # cancel a pending/active pulse -- see module docstring.
        self.last_steps = 0

    def step(self, frame, duration_ms, haltere_stimulation=None):
        if not math.isfinite(duration_ms) or duration_ms <= 0: raise ValueError('Positive duration required')
        b = self.brain
        steps = round(duration_ms / b.dt)
        if steps < 1: raise ValueError('Duration too short')
        active = min(steps, max(0, self.until - b.cursor))
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.
        # Split at the exact pulse boundary, same technique as
        # doom/training.py's DamageTraining.step -- the haltere current
        # (unrelated to the dopamine pulse) is present in both segments.
        for n, dan_pulse in [(active, (b.circuit['dan'], 4.)), (steps - active, None)]:
            if n:
                pulses = [] if haltere_stimulation is None else [haltere_stimulation]
                if dan_pulse is not None: pulses.append(dan_pulse)
                c, t = b.rgb_step(frame, n * b.dt, learning=self.enabled, stimulation=pulses or None)
                counts += c
                wall += t
        b.counts[:] = counts
        self.last_steps = active
        self.delivered_steps += active
        return counts, wall

    def observe(self, finished):
        if finished:
            self.events += 1
            self.until = self.brain.cursor + 2000
        return finished

    def telemetry(self):
        b = self.brain
        m = b.memory()
        ratios = b.weight[b.circuit['edges']] / b.baseline_plastic
        if not np.isfinite(ratios).all(): raise RuntimeError('Nonfinite memory efficacy')
        return {**m, 'enabled': self.enabled, 'validated': False,
                'events': self.events, 'delivered_ms': round(self.delivered_steps * b.dt, 3),
                'stimulus_active': self.last_steps > 0,
                'maximum_efficacy': float(ratios.max()), 'minimum_efficacy': float(ratios.min()),
                'mean_absolute_change': float(np.abs(ratios - 1).mean())}
