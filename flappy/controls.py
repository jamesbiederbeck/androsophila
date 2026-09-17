"""Flappy Bird's binary action decoded from spikes.

There is no biological "flap" readout -- flies do not play Flappy Bird. This
reuses the same boolean-trigger pattern doom/engine.py's NeuralControls
already uses for its own discrete action (attack, off the MN9 readout): flap
fires exactly when the chosen readout type spikes at all this tick. Same
"joystick gains, not biology" honesty as the rest of this project's motor
mapping -- an arbitrary but explicit and auditable engineering choice, not a
claimed correspondence. Uses the same manifest['readouts'] list and neuron
indices doom/server.py already loads; no new data.
"""
import math
import numpy as np


class FlapControls:
    def __init__(self, readouts, flap_readout_type='MN9'):
        self.readouts = readouts
        self.flap_readout_type = flap_readout_type
        self.rates = np.zeros(len(readouts))

    def decode(self, counts, seconds):
        if seconds <= 0: raise ValueError('Positive time required')
        raw = np.asarray([counts[r['index']] / seconds for r in self.readouts])
        self.rates = self.rates * math.exp(-seconds / .1) + raw * (1 - math.exp(-seconds / .1))
        flap = any(counts[r['index']] > 0 for r in self.readouts if r['type'] == self.flap_readout_type)
        return {'flap': bool(flap), 'readouts': [
            {**r, 'spikes': int(counts[r['index']]), 'rate_hz': round(float(rate), 3)}
            for r, rate in zip(self.readouts, self.rates)]}
