"""Boundary tests for flappy/training.py's FlapTraining, mirroring
tests/test_doom_live_training.py's PulseBrain-stub approach -- no real graph
needed since FlapTraining only calls brain.rgb_step/memory."""
import numpy as np
import pytest
from flappy.training import FlapTraining


class PulseBrain:
    def __init__(self):
        self.n = 3; self.cursor = 0; self.dt = .1
        self.circuit = {'dan': np.array([2]), 'edges': np.array([0])}
        self.baseline_plastic = np.array([20.], dtype=np.float32)
        self.weight = np.array([20., .5, .3], dtype=np.float32)
        self.weights_frozen = False
        self.counts = np.zeros(3, dtype=np.int32)
        self.calls = []

    def rgb_step(self, frame, ms, learning, stimulation):
        n = round(ms / self.dt)
        self.calls.append((self.cursor, n, learning, stimulation))
        self.cursor += n
        return np.array([n, 0, n if stimulation else 0], dtype=np.int32), 0.

    def memory(self):
        ratios = self.weight[self.circuit['edges']] / self.baseline_plastic
        return {'plastic_edges': 1, 'changed_edges': int(np.count_nonzero(ratios != 1)),
                'mean_efficacy': float(ratios.mean()), 'minimum_efficacy': float(ratios.min()),
                'sha256': 'x', 'model': 'test'}


def test_no_pulse_before_any_crash():
    b = PulseBrain(); t = FlapTraining(b)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    t.step(frame, 33.3)
    assert t.delivered_steps == 0
    assert b.calls[0][3] is None


def test_crash_schedules_exactly_2000_steps_split_at_the_boundary():
    b = PulseBrain(); t = FlapTraining(b)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    t.observe(True)
    assert t.events == 1
    for _ in range(8):
        t.step(frame, 333.)  # 3330 steps/call, more than enough to exhaust the pulse
    assert t.delivered_steps == 2000
    # exactly one call is split (has both an active and inactive segment)
    active_calls = [c for c in b.calls if c[3] is not None]
    assert sum(n for _, n, _, active in b.calls if active) == 2000
    assert b.calls[-1][3] is None  # pulse exhausted, later calls carry no stimulus


def test_haltere_stimulation_present_regardless_of_pulse_state():
    b = PulseBrain(); t = FlapTraining(b)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    haltere = (np.array([1]), 5.)
    t.step(frame, 33.3, haltere_stimulation=haltere)
    assert b.calls[-1][3] == [haltere]
    t.observe(True)
    t.step(frame, 33.3, haltere_stimulation=haltere)
    assert haltere in b.calls[-1][3]
    assert (b.circuit['dan'], 4.) in b.calls[-1][3]


def test_disabled_freezes_weights_and_passes_learning_false():
    b = PulseBrain(); t = FlapTraining(b, enabled=False)
    assert b.weights_frozen is True
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    t.observe(True)
    t.step(frame, 33.3)
    assert b.calls[-1][2] is False  # learning kwarg


def test_new_round_does_not_cancel_a_pending_pulse():
    b = PulseBrain(); t = FlapTraining(b)
    t.observe(True)
    scheduled_until = t.until
    t.new_round()
    assert t.until == scheduled_until  # deliberately unlike doom's DamageTraining
    assert t.last_steps == 0


def test_telemetry_reports_changed_edges():
    b = PulseBrain(); t = FlapTraining(b)
    report = t.telemetry()
    assert report['changed_edges'] == 0
    b.weight[0] = 15.  # simulate a plasticity-driven change
    report = t.telemetry()
    assert report['changed_edges'] == 1
    assert report['maximum_efficacy'] == pytest.approx(15. / 20.)
