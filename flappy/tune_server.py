"""Tunable live web UI for the Flappy Bird haltere-stimulation harness.

Two sliders (m, n) live-adjust the power-law activation A(dy) = m * dy^n
that maps the bird's fall speed to haltere current, applied every tick
without restarting the simulation -- see flappy/circuit.py for why haltere
current is what drives the wing motor neurons at all. Pipes are disabled
(Game(no_pipes=True)) so the only failure mode is the ground/ceiling; the
search target here is stable flight, not score.

Streams coordinates, not frames: /state is a small JSON object (bird
y/velocity/rotation, haltere current, flap, spike counts) and the page
draws its own canvas sprite client-side, unlike doom/server.py and
flappy/server.py's base64 JPEG frame.

Frozen weights throughout -- still a simulation, not a learning run. Only
GET is exposed (including /params, which is a bounded, in-memory numeric
knob -- not a filesystem, shell, credential, or model-mutating endpoint).
"""
import argparse
import json
import signal
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from doom.native import NativeBrain
from flappy.circuit import haltere_afferents, haltere_current_for_velocity, wing_motor_readouts
from flappy.controls import FlapControls
from flappy.game import Game, FPS
from vision.retina import BilinearLuminance

ROOT = Path(__file__).resolve().parents[1]
latest = {'status': 'starting', 'generated_at_ms': 0}
# Plain dict item assignment/read is atomic under the GIL, same convention
# doom/server.py and flappy/server.py already rely on for `latest`.
params = {'m': 1.0, 'n': 1.0}
PARAM_BOUNDS = {'m': (0., 10.), 'n': (0.1, 4.)}
stop = threading.Event()


def run_loop(args):
    global latest
    try:
        path = ROOT / 'outputs/doom' / args.dataset / 'graph.npz'
        if args.backend == 'gpu':
            from doom.gpu import GPUBrain, GPU_BUILD
            brain = GPUBrain(path); build = GPU_BUILD
        else:
            from doom.native import BUILD
            brain = NativeBrain(path); build = BUILD
        haltere = haltere_afferents(brain)
        dlm = wing_motor_readouts(brain)
        controls = FlapControls(dlm)
        game = Game(seed=args.seed, no_pipes=True)
        geometry = game.geometry()
        retina = BilinearLuminance()
        run_id = str(uuid.uuid4())
        duration_ms = 1000 / FPS
        total_spikes, best_survival = 0, 0
        seq = 0
        start = time.monotonic()
        print(json.dumps({'status': 'running', 'run_id': run_id, 'port': args.port}), flush=True)
        while not stop.is_set():
            obs = game.observation()
            if obs['finished']:
                best_survival = max(best_survival, obs['tick'])
                game.new_episode()
                obs = game.observation()
            frame = game.pixels()
            light = retina.sample(frame, brain.uv)
            m, n = params['m'], params['n']
            current = haltere_current_for_velocity(obs['y_velocity'], m=m, n=n)
            stimulation = (haltere, current) if current > 0 else None
            counts, neural_wall = brain.step(light, duration_ms, sugar=False, stimulation=stimulation)
            action = controls.decode(counts, duration_ms / 1000)
            game.act(action['flap'])
            total_spikes += int(counts.sum())
            seq += 1
            wall_seconds = time.monotonic() - start
            obs = game.observation()
            latest = {
                'schema': 1, 'status': 'running', 'run_id': run_id, 'sequence': seq,
                'generated_at_ms': int(time.time() * 1000), 'backend': args.backend,
                'geometry': geometry,
                'bird': {'y': obs['y'], 'rotation': obs['rotation'], 'y_velocity': obs['y_velocity']},
                'flap': action['flap'], 'haltere_current': round(current, 3),
                'params': {'m': m, 'n': n},
                'game': {'episode': obs['episode'], 'tick': obs['tick'],
                         'best_survival_ticks': max(best_survival, obs['tick'])},
                'dlm_spikes': int(sum(int(counts[r['index']]) for r in dlm)),
                'clocks': {
                    'wall_seconds': round(wall_seconds, 3),
                    'neural_seconds': round(brain.sim_ms / 1000, 4),
                    'speed': round((brain.sim_ms / 1000) / wall_seconds, 4) if wall_seconds > 0 else 0,
                    'brain_step_ms': round(neural_wall * 1000, 3),
                },
                'total_spikes': total_spikes,
            }
    except Exception:
        latest = {'status': 'error', 'generated_at_ms': int(time.time() * 1000),
                   'message': 'The simulation stopped. No live data is available.'}
        import traceback; traceback.print_exc()
    finally:
        try: game.close()
        except Exception: pass


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Flappy haltere tuner</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;gap:24px;padding:20px;flex-wrap:wrap}
canvas{background:#4ec0e0;border:1px solid #333;border-radius:4px}
.panel{min-width:260px}
label{display:block;margin-top:14px;font-size:13px;color:#aaa}
input[type=range]{width:100%}
.val{font-family:monospace;color:#7fd}
.stat{font-family:monospace;font-size:13px;margin:2px 0}
.dead{color:#f66}
.alive{color:#7f7}
.flap{color:#ff7}
h1{font-size:16px;margin:0 0 4px}
.formula{font-family:monospace;color:#7fd;font-size:14px}
</style></head>
<body>
<canvas id="c" width="288" height="512"></canvas>
<div class="panel">
<h1>Haltere activation tuner</h1>
<div class="formula" id="formula">A(&Delta;y) = m &middot; &Delta;y&#8319;</div>
<label>m (gain) = <span class="val" id="mval"></span>
  <input type="range" id="m" min="0" max="10" step="0.05" value="1">
</label>
<label>n (exponent) = <span class="val" id="nval"></span>
  <input type="range" id="n" min="0.1" max="4" step="0.05" value="1">
</label>
<hr style="border-color:#333;margin-top:18px">
<div class="stat" id="status">connecting...</div>
<div class="stat">tick: <span id="tick"></span> &nbsp; episode: <span id="episode"></span></div>
<div class="stat">y_velocity: <span id="vy"></span></div>
<div class="stat">haltere current: <span id="cur"></span></div>
<div class="stat">flap: <span id="flapind"></span></div>
<div class="stat">DLM spikes/tick: <span id="dlm"></span></div>
<div class="stat">survived this episode: <span id="survived"></span> ticks</div>
<div class="stat">best survival: <span id="best"></span> ticks</div>
<div class="stat">sim speed: <span id="speed"></span>x realtime</div>
</div>
<script>
const $ = id => document.getElementById(id);
const ctx = $('c').getContext('2d');
let pending = null;

function pushParams() {
  const m = $('m').value, n = $('n').value;
  $('mval').textContent = m; $('nval').textContent = n;
  $('formula').innerHTML = 'A(&Delta;y) = ' + m + ' &middot; &Delta;y<sup>' + n + '</sup>';
  clearTimeout(pending);
  pending = setTimeout(() => fetch(`/params?m=${m}&n=${n}`), 80);
}
$('m').oninput = pushParams; $('n').oninput = pushParams;

function draw(s) {
  const g = s.geometry, b = s.bird;
  ctx.clearRect(0, 0, g.screen_width, g.screen_height);
  ctx.fillStyle = '#4ec0e0'; ctx.fillRect(0, 0, g.screen_width, g.screen_height);
  ctx.fillStyle = '#ded895'; ctx.fillRect(0, g.ground_y, g.screen_width, g.screen_height - g.ground_y);
  ctx.save();
  ctx.translate(g.player_x + g.player_width / 2, b.y + g.player_height / 2);
  ctx.rotate(-b.rotation * Math.PI / 180);
  ctx.fillStyle = s.flap ? '#ffd23f' : '#f2a413';
  ctx.beginPath();
  ctx.ellipse(0, 0, g.player_width / 2, g.player_height / 2, 0, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
}

async function poll() {
  try {
    const r = await fetch('/state'); const s = await r.json();
    if (s.status !== 'running') { $('status').textContent = s.status; return; }
    $('status').innerHTML = '<span class="alive">running</span>';
    $('tick').textContent = s.game.tick;
    $('episode').textContent = s.game.episode;
    $('vy').textContent = s.bird.y_velocity.toFixed(2);
    $('cur').textContent = s.haltere_current.toFixed(3) + ' mV-eq';
    $('flapind').innerHTML = s.flap ? '<span class="flap">FLAP</span>' : '&mdash;';
    $('dlm').textContent = s.dlm_spikes;
    $('survived').textContent = s.game.tick;
    $('best').textContent = s.game.best_survival_ticks;
    $('speed').textContent = s.clocks.speed;
    draw(s);
  } catch (e) { $('status').textContent = 'disconnected'; }
}
setInterval(poll, 150);
poll();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
        elif parsed.path == '/state':
            self._send(200, json.dumps(latest, separators=(',', ':')).encode(), 'application/json')
        elif parsed.path == '/params':
            query = urllib.parse.parse_qs(parsed.query)
            for key in ('m', 'n'):
                if key not in query: continue
                try:
                    value = float(query[key][0])
                except ValueError:
                    self.send_error(400, 'Invalid numeric parameter'); return
                lo, hi = PARAM_BOUNDS[key]
                if not (lo <= value <= hi):
                    self.send_error(400, f'{key} out of bounds [{lo}, {hi}]'); return
                params[key] = value
            self._send(200, json.dumps(params).encode(), 'application/json')
        else:
            self.send_error(404)

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass

    def log_message(self, *args): pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8771)
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--backend', choices=['native', 'gpu'], default='native')
    args = p.parse_args()

    def shutdown_signal(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    worker = threading.Thread(target=run_loop, args=(args,), daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); server.shutdown()


if __name__ == '__main__':
    main()
