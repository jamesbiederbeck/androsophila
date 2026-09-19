# 3D spectator view — previous reconstruction

**Superseded:** the live camera now uses the native Doom renderer. See
[the native observer architecture and build instructions](../deploy/doomfly/native-observer/README.md).
The implementation below documents the earlier illustrative arena for provenance.

This is an observer illustration of the ongoing Doom experiment, not a fly body
or flight simulator. The original first-person RGB remains the only visual input
to the modeled sensory neurons. No camera controls reach the host.

## Source and timing

ViZDoom 1.3.0 exports player POSITION_X/Y/Z, ANGLE and PITCH plus GameState objects
and sectors. The host samples these before the action, at the same engine tick as
the raw RGB input. `spectator.tick == game.tick - 1` because the existing game
counters describe the completed action. The optional spectator packet travels in
the same immutable broadcast frame. It is not passed to `training.step`,
`controls.decode`, or reinforcement. The integration timestep, graph, action
mapping, Doom rules, and learning parameters are unchanged.

The current renderer supports the square, one-sector combat-survival arena.
Coordinates are converted from Doom units to Three.js units at 1/32, with Doom Y
mapped to negative Three.js Z. Heading zero faces +X. Walls use engine endpoints
and height; the ceiling is omitted and walls viewed from outside disappear for a
cutaway view. Materials and low-poly actor markers are illustrations, not Doom
textures or full monster animations. Object markers do not infer health or AI.

The original procedural fly has six legs, segmented abdomen, compound-eye-like
facets, two translucent wings, antennae, halteres and a small gun. The avatar is
raised 40 Doom units above the player origin. Wings oscillate for visual effect
in wall time; they do not represent wing motor neuron activity. Gun appearance
does not infer muzzle flashes, hits, or successful attacks from the fire button.
No fly flight mechanics or new action is added.

Only interpolation between received poses is permitted; no extrapolation.
Large gaps, round resets, and run changes snap to the actual pose. Signal loss
holds the last received pose and stops wings/recording. Free camera motion is
independent of the avatar, including on round changes. Orbit follows the avatar.

## Recording

The browser draws the local 3D view to a 1280×720 canvas and captures it at 30 fps.
The header and footer burned into clips identify a spectator illustration and
show actual received counters (post-action), while poses remain pre-action.
This video rate is unrelated to neural or game speed. Recordings have no audio,
are never uploaded, and are bounded at three minutes / roughly 100 MB. Recording
stops on tab hiding, signal loss, or leaving the view. The download uses a browser-
supported WebM or MP4 codec. Unsupported browsers can use their screen recorder.

Three.js loads only when 3D mode is selected. One scene survives telemetry
updates. Resize, key/pointer listeners, animation loops, GPU resources, recording
tracks and object URLs are cleaned up when leaving. The 3D mode never opens a
second simulation connection or makes an action request.

## Validation / rollout

Three paired legacy-versus-observer engine trials, 700 ticks each at seeds 41027,
41028 and 41029, produced identical raw RGB, game observations and rewards for
identical prescribed actions, including death/reset. These are transport/engine
checks, not evidence of learning or biological fidelity. The retained local
report is `outputs/connectome_sim/spectator-3d-v1/engine-equivalence.json`.

Runtime provenance hashes the observer source as well as model source. A planned
update must stop/checkpoint the old worker, back up the completed generation,
then migrate only the explicitly reviewed observer source hashes in a copied
generation. Preserve byte-identical brain/decoder files and old/new identity
records. Resume with the same study, learned/fast state and fixed decoder; the
interrupted round is censored and a fresh arena is disclosed as usual. Never
weaken the general checkpoint identity check.

References:
- https://vizdoom.farama.org/main/api/python/doomGame/
- https://vizdoom.farama.org/1.2.3/api/python/gameState/
- https://threejs.org/docs/pages/OrbitControls.html
- https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream
