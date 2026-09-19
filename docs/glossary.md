# Glossary

Terms used across the connectome-driven simulation (`doom/`, `doom_learning*/`, `flappy/`)
and where they come from in the code.

## Connectome / data terms

- **UV** — each simulated photoreceptor's 2D coordinate in `[0,1]²`, stored per-cell in
  the `.npz` graph (`connectome_sim/prepare.py`). Derived from real reconstructed receptor
  positions via connectome contact analysis, so it's genuinely *retinotopic* (tied to
  actual eye geometry) — but the screen mapping that samples `uv` against a game frame
  (`retinal_samples` in `doom/game.py`) is an explicit convention (left/right
  overlapping halves across `[0,1]²`), not a calibrated field-of-view. "UV" here is the
  standard graphics-programming term for 2D texture/surface coordinates, unrelated to
  ultraviolet light.
- **`ids`** — the real connectome body IDs (MaleCNS dataset) for every simulated
  neuron, indexed the same way as all other per-neuron arrays (`weight`, `uv`,
  `retina`, etc.).
- **`annotations(ids)`** (`doom_learning/common.py`) — looks up each neuron's
  connectome metadata by `bodyId`: `type`, `instance`, `class`, `subclass`,
  `superclass`, `somaSide`, `somaNeuromere`.
  - **`type`** — the specific named cell type (e.g. `DLMn a, b`, `MBON11`, `PPL101`).
  - **`class` / `subclass` / `superclass`** — coarser groupings the connectome dataset
    assigns (e.g. `subclass='haltere'`, `subclass='wm'` for wing muscle motor,
    `superclass='cb_motor'` vs `superclass='vnc_motor'` for central-brain vs.
    ventral-nerve-cord motor neurons).
  - **`somaSide` / `somaNeuromere`** — left/right laterality and which body segment
    the soma sits in.

## Visual pathway

- **Retina** — the set of neuron indices flagged as photoreceptors in the graph
  (`retina` array); driven by `retinal_samples`, a Naka-Rushton-style saturating
  luminance transform of the sampled pixel at that neuron's `uv`.
- **Lamina** — the next visual processing layer downstream of the retina
  (`connectome_sim/prepare.py` flags cell types `L1, L2, L3, L5`). Real lamina neurons are
  graded (non-spiking); this repo's LIF proxy can't represent that, so it's
  approximated as a constant tonic "bias" current (`lamina_bias`, default 12
  mV-equivalent) injected into those cells whenever the sim is stepped — a declared
  simplification, not validated fly vision.

## Reward / plasticity circuit (mushroom body)

- **KC — Kenyon cell** — the mushroom body's principal intrinsic neuron type
  (subtypes like `KCg-d`, `KCab`, matched here by a `startswith('KC')` prefix match).
  Carries sparse, high-dimensional sensory (largely visual, in this graph)
  representations into the mushroom body.
- **MBON — Mushroom Body Output Neuron** — output cells of the mushroom body; this
  project's calibrated circuit specifically uses **MBON11**. KC→MBON synapses are the
  site of plasticity: the "memory" is literally the weight on these edges.
- **DAN — Dopaminergic Neuron** — carries the reinforcement ("teaching") signal; this
  project specifically uses **PPL101** (a named DAN type) as the reinforcement input.
  Modeled here as an ordinary fast excitatory spiking cell, *not* biological dopamine
  receptor/concentration dynamics — see `docs/doom-neuroscience-review.md`.
- **The plasticity rule** — KC→MBON11 synapses depress when a KC and PPL101 (DAN) are
  coactive within a short window, implementing a dopamine-gated associative learning
  rule (Hebbian/three-factor style). `doom/training.py`'s `DamageTraining` and
  `flappy/training.py`'s `FlapTraining` deliver a 200 ms, +4 mV-equivalent PPL101
  pulse as the "aversive" teaching signal after damage/a crash.
- **`circuit_spec`** (`doom_learning/circuit.py`) — the parameterized dict
  (`kc_prefix`, `mbon_types`, `dan_types`) selecting which KC/MBON/DAN cell types the
  plasticity rule applies to; defaults to `{'kc_prefix':'KC','mbon_types':['MBON11'],
  'dan_types':['PPL101']}`, matching the specific circuit the v6 calibration
  (`doom_learning_v6/calibration.py`) was tuned against.

## Motor / readout circuit

- **DN — Descending Neuron** — a cell carrying commands from the brain down to the
  ventral nerve cord, where motor neurons live. The `DNa*` / `DNp*` / `DNpe*` naming
  is the connectome's, grouping them by tract, not by function. This repo reads out
  from six DN/motor types in two *separate* fixed decoders (`NeuralControls` in
  `doom/engine.py`), and the distinction between them matters:
  - **`mode='biological'`** — DNa02, DNp09, MDN, MN9: cells picked because published
    work associates them with a motor role. A comparison arm only.
  - **`mode='bci'`** — DNp20, DNpe017: cells picked *because they responded to the
    screen*, after looking at the responses. An engineered brain-computer interface,
    not evidence of native fly motor meaning.
  Both are engineered game mappings. Neither establishes that the fly is "deciding"
  anything, and the gains below are joystick constants, not biology.
- **DNa02** — a descending neuron associated with steering. Used in the biological
  comparison decoder as the turn channel: `clip(0.06 × (right_Hz − left_Hz), ±6)`
  degrees per tic. The one readout here with a direct external citation
  ([DNa02 steering study](https://elifesciences.org/articles/102230)) — but that
  evidence supports *studying* DNa02, not this repo's screen geometry or gain.
- **DNp09** and **MDN — Moonwalker Descending Neuron** — paired against each other as
  the forward/backward channel in the biological comparison decoder:
  `clip(0.3 × (DNp09_Hz − MDN_Hz), ±20)`. MDN's name is the published one for a
  backward-walking command neuron; this repo cites no study of its own for either
  cell, so treat the forward/backward polarity as this harness's convention.
- **MN9** — a central-brain motor neuron (`superclass='cb_motor'`) used in Doom as the
  "attack" readout (any spike presses attack for one tic). Despite the name suggesting
  "motor neuron," it is *not* a wing motor neuron — a naming trap the Flappy Bird work
  specifically avoided; see **DLM** for the real flight motor cells.
- **DNp20** and **DNpe017** — the four cells the public BCI decoder actually reads
  (DNp20 R=10059, L=10162; DNpe017 L=10527, R=555871). DNp20 right-minus-left drives
  turn at `clip(0.12 × ΔHz, ±6)`; DNpe017 summed rate drives forward at
  `clip(0.4 × Hz, 0..20)`, and any DNpe017 spike presses attack. These cells and gains
  were chosen *after observing which cells responded to the screen*. There is no fly
  "shooting neuron" — the attack mapping is an interface decision, nothing more.
  The decoder sees no score, object coordinates, pixels or target detector.
- **DLM — Dorsal Longitudinal Muscle motor neuron** (`type` in `['DLMn a, b',
  'DLMn c-f']`) — the real wing/flight power-muscle motor neurons (`subclass='wm'`,
  `superclass='vnc_motor'`, soma in the T2 thoracic neuromere). Flappy Bird's flap
  decision reads out from these, instead of MN9.
- **Haltere afferents** (`subclass='haltere'`) — mechanosensory neurons from the
  haltere (a fly's reduced hindwing, used for gyroscopic flight stabilization in real
  flies). Synapse onto the DLM/wing-motor pool within 1-2 hops — a real reflex arc,
  exploited here as the stimulation channel that drives flapping, gated proportional
  to the bird's fall velocity (`flappy/circuit.py`).
- **`sugar`** — a separate stimulation channel (cell type `LB3c`) used in Doom as a
  score-contingent reward current; unrelated to the DAN/PPL101 teaching signal, and
  not used by the Flappy learning harness.
