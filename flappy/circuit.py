"""Cell identities that give the Flappy Bird harness a genuine wing-motor
output channel and a way to drive it.

Confirmed empirically, not assumed: under plain visual drive (default lamina
bias, no external stimulation), no VNC motor neuron of any kind spikes across
600 ticks (20s) of real gameplay, including all ten reconstructed DLM
(dorsal longitudinal, flight power-muscle) motor neurons -- despite
descending neurons firing normally in the same run. The retained haltere
mechanosensory afferents (subclass 'haltere', 205 cells) synapse directly
onto 6/10 DLM cells and 50/67 wing-muscle motor neurons in one hop, and all
of them within two hops -- the same haltere-to-flight-motor reflex arc
documented in flies (Fayyazuddin & Dickinson 1996, J Neurosci;
Trimarchi & Schneiderman 1995, J Exp Biol). Driving those afferents with an
artificial current exploits a real, short reflex circuit to get a motor
output; it is not a model of voluntary flight-initiation decision-making,
same "engineered joystick, not biology" honesty stance as the sugar/PPL101
current channels elsewhere in this repo.

Also confirmed empirically: the direct (1-2 hop) haltere->DLM synaptic path
is weak and the broader haltere->wing-muscle-pool->DLM path is net
*inhibitory* by signed weight -- a first-order (linear, non-spiking) estimate
would predict this stimulation barely works. It works anyway, because the
recurrent spiking network is not in that linear regime: broad haltere
stimulation recruits far more total network activity (roughly 3x spike
count) and DLM firing turns on as a sharp threshold, not a graded response
to a *constant* current -- 0/150 ticks flap at 5 mV-equivalent current,
~97-99% of ticks flap at 7-10 mV. There is no tested constant current that
gives an intermediate flap rate.

haltere_current_for_velocity gates that current on the game's own falling
speed instead of holding it constant, closing an actual feedback loop:
falling faster raises haltere drive, which raises DLM firing probability,
which raises flap probability, which slows the fall. This still doesn't
model what halteres actually sense (angular velocity from wingbeat-driven
oscillation, not linear descent speed) -- it is a proportional-control
mapping (a tunable power law, A(dy) = m * dy^n) chosen because the sharp DLM
threshold needs *some* graded input to turn into graded behavior, not a
claim about haltere physiology. See flappy/tune_server.py for a live UI
that adjusts m and n against the running simulation.
"""
import math
import numpy as np
from doom_learning.common import annotations
from flappy_bird_gymnasium.envs.constants import PLAYER_MAX_VEL_Y

DLM_TYPES = ['DLMn a, b', 'DLMn c-f']


def wing_motor_readouts(brain):
    a = annotations(brain.ids)
    idx = np.flatnonzero(a.type.isin(DLM_TYPES))
    if not len(idx): raise ValueError('No DLM motor neurons found in this graph')
    return [{'index': int(i), 'id': str(brain.ids[i]), 'type': 'DLMn',
             'connectome_type': str(a.type.iloc[i]), 'side': str(a.somaSide.iloc[i])} for i in idx]


def haltere_afferents(brain):
    a = annotations(brain.ids)
    idx = np.flatnonzero(a.subclass == 'haltere')
    if not len(idx): raise ValueError('No haltere afferents found in this graph')
    return idx.astype(np.int32)


def haltere_current_for_velocity(y_velocity, m=1.75, n=2., max_velocity=PLAYER_MAX_VEL_Y):
    """A(dy) = m * dy^n, rectified to 0 while level or rising (y_velocity<=0)
    and clipped to the game's own terminal fall speed before exponentiating
    (undefined for negative bases at fractional n, unbounded otherwise).
    Default m=1.75, n=2 chosen via interactive search in
    flappy/tune_server.py, superseding the earlier linear m=1, n=1 baseline
    that just matched a constant-current test; neither is derived from
    anything but this harness's own runs."""
    if not math.isfinite(max_velocity) or max_velocity <= 0: raise ValueError('Positive max_velocity required')
    if not all(math.isfinite(x) for x in (y_velocity, m, n)): raise ValueError('Finite velocity, m and n required')
    if y_velocity <= 0: return 0.
    return m * min(float(y_velocity), max_velocity) ** n
