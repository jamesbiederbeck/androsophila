"""The 12-region body-map used by the Android app's motor-neuron sprite
(head/thorax/abdomen/wings/legs), and which vnc_motor connectome cells each
region actually reads out.

The region ids, hex colors and descriptions come from the sprite SVG and its
color->region->motor-neuron-target table (user-supplied). The cell-index
membership is derived here from the connectome's own subclass/type/instance
annotations -- not asserted, checked against the sprite's implied coverage:
708 total vnc_motor cells, 631 assigned to a region, the rest (nm, xm, wm
types other than b1/b2/hg1, and a few leg cells with no parseable L/R suffix)
legitimately outside what the sprite depicts.

Fill semantics (Android side, not computed here): each region's outline is
always drawn; its fill starts fully transparent and its color/opacity ramps
toward REGIONS[id]['hex'] as that region's motor cells' spike rate rises --
never the reverse (a spiking region gets more opaque, not differently
colored per intensity level).
"""
import pyarrow.feather as feather
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (region_id, hex, description) exactly as supplied.
REGION_INFO = {
    'head':   ('#FF0000', 'mouth_proboscis_antennal_mechanosensors'),
    'thorax': ('#00FF00', 'haltere_steering_indirect_flight_muscles'),
    'abdomen': ('#0000FF', 'abdominal_flexors_extensors'),
    'wing_l': ('#FFFF00', 'left_direct_flight_muscles (b1, b2, hg1)'),
    'wing_r': ('#FF00FF', 'right_direct_flight_muscles (b1, b2, hg1)'),
    'leg_l1': ('#00FFFF', 'left_prothoracic_coxa_femur_tibia'),
    'leg_l2': ('#FFA500', 'left_mesothoracic_coxa_femur_tibia'),
    'leg_l3': ('#800080', 'left_metathoracic_coxa_femur_tibia'),
    'leg_r1': ('#008080', 'right_prothoracic_coxa_femur_tibia'),
    'leg_r2': ('#A52A2A', 'right_mesothoracic_coxa_femur_tibia'),
    'leg_r3': ('#FFC0CB', 'right_metathoracic_coxa_femur_tibia'),
}

_POWER_TYPES = ['DLMn a, b', 'DLMn c-f', 'DVMn 1a-c', 'DVMn 2a, b', 'DVMn 3a, b']
_STEER_TYPES = ['b1 MN', 'b2 MN', 'hg1 MN']
_LEG_SUBCLASS = {'leg_l1': ('fl', 'L'), 'leg_l2': ('ml', 'L'), 'leg_l3': ('hl', 'L'),
                 'leg_r1': ('fl', 'R'), 'leg_r2': ('ml', 'R'), 'leg_r3': ('hl', 'R')}


def _side_of(instance):
    if not isinstance(instance, str): return None
    if instance.endswith('_L'): return 'L'
    if instance.endswith('_R'): return 'R'
    return None


def compute_regions(ids):
    """ids: brain.ids (bodyId per graph index, ids[i] = bodyId of neuron i).
    Returns {region_id: np.ndarray of graph-index positions}."""
    a = feather.read_table(ROOT / 'connectome_data/malecns_v1/annotations.feather').to_pandas().set_index('bodyId').loc[ids]
    motor = a[a.superclass == 'vnc_motor'].copy()
    motor['side'] = motor['instance'].map(_side_of)
    pos = {bid: i for i, bid in enumerate(ids)}

    regions = {}
    regions['head'] = motor[motor.subclass == 'hm'].index.map(pos.get).dropna().astype(int).tolist()
    regions['abdomen'] = motor[motor.subclass == 'ad'].index.map(pos.get).dropna().astype(int).tolist()
    wm = motor[motor.subclass == 'wm']
    regions['thorax'] = wm[wm.type.isin(_POWER_TYPES)].index.map(pos.get).dropna().astype(int).tolist()
    regions['wing_l'] = wm[wm.type.isin(_STEER_TYPES) & (wm.side == 'L')].index.map(pos.get).dropna().astype(int).tolist()
    regions['wing_r'] = wm[wm.type.isin(_STEER_TYPES) & (wm.side == 'R')].index.map(pos.get).dropna().astype(int).tolist()
    for region_id, (sub, side) in _LEG_SUBCLASS.items():
        sel = motor[(motor.subclass == sub) & (motor.side == side)]
        regions[region_id] = sel.index.map(pos.get).dropna().astype(int).tolist()
    return regions


if __name__ == '__main__':
    import numpy as np
    g = np.load(ROOT / 'outputs/doom/malecns_v1/graph.npz')
    regions = compute_regions(g['ids'])
    for name, idx in regions.items():
        print(name, len(idx))
