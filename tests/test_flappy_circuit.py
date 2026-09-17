"""Unit coverage for flappy/circuit.py's cell-identity lookups, independent of
the real MaleCNS graph."""
import types
import numpy as np
import pandas as pd
import pytest
from flappy import circuit as circuit_module
from flappy.circuit import haltere_afferents, haltere_current_for_velocity, wing_motor_readouts


def synthetic_brain(monkeypatch, types_by_index, subclass_by_index):
    n = len(types_by_index)
    brain = types.SimpleNamespace(ids=np.arange(n, dtype=np.int64), n=n)
    frame = pd.DataFrame({'type': types_by_index, 'subclass': subclass_by_index,
                           'somaSide': ['L'] * n}, index=brain.ids)
    monkeypatch.setattr(circuit_module, 'annotations', lambda ids: frame.loc[ids])
    return brain


def test_wing_motor_readouts_finds_both_dlm_subtypes(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['DLMn a, b', 'DLMn c-f', 'other'], ['wm', 'wm', None])
    readouts = wing_motor_readouts(brain)
    assert {r['index'] for r in readouts} == {0, 1}
    assert all(r['type'] == 'DLMn' for r in readouts)
    assert {r['connectome_type'] for r in readouts} == {'DLMn a, b', 'DLMn c-f'}


def test_wing_motor_readouts_raises_when_absent(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['other'], [None])
    with pytest.raises(ValueError, match='DLM'):
        wing_motor_readouts(brain)


def test_haltere_afferents_finds_haltere_subclass(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['SApp', 'DLMn a, b', 'other'], ['haltere', 'wm', None])
    idx = haltere_afferents(brain)
    np.testing.assert_array_equal(idx, [0])
    assert idx.dtype == np.int32


def test_haltere_afferents_raises_when_absent(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['other'], [None])
    with pytest.raises(ValueError, match='haltere'):
        haltere_afferents(brain)


def test_haltere_current_zero_while_rising_or_level():
    assert haltere_current_for_velocity(-5., m=10., n=1., max_velocity=10.) == 0.
    assert haltere_current_for_velocity(0., m=10., n=1., max_velocity=10.) == 0.


def test_haltere_current_defaults_to_1_75_dy_squared():
    assert haltere_current_for_velocity(5.) == 1.75 * 25.
    assert haltere_current_for_velocity(0.) == 0.


def test_haltere_current_applies_power_law():
    assert haltere_current_for_velocity(5., m=2., n=1., max_velocity=10.) == 10.
    assert haltere_current_for_velocity(3., m=1., n=2., max_velocity=10.) == 9.
    assert haltere_current_for_velocity(4., m=2., n=0.5, max_velocity=10.) == 4.


def test_haltere_current_clips_above_max_velocity():
    assert haltere_current_for_velocity(50., m=1., n=1., max_velocity=10.) == 10.


def test_haltere_current_rejects_invalid_max_velocity():
    with pytest.raises(ValueError):
        haltere_current_for_velocity(5., m=10., n=1., max_velocity=0.)
