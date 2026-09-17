"""Read-only live broadcast for the Flappy Bird harness, mirroring doom/server.py's
/state + /health shape closely enough that doom/stream_mjpeg.py works against it
unmodified (just point --origin at this server's port).

Deliberately smaller than doom/server.py: no checkpointing, audit archive, reward/
sugar reinforcement, observer/spectator camera, or multiple neural models/conditions
-- those are Doom-specific machinery this harness doesn't have yet. Only GET /state
and /health are exposed. No filesystem, shell, credentials, remote controls, or
model-mutating endpoint.

Frozen weights throughout -- this is a simulation, not a learning run. See
flappy/circuit.py for why haltere stimulation is needed for the wing motor
neurons to spike at all, and why it's scaled by the bird's own fall speed
rather than held constant.
"""
import argparse
import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from doom.native import NativeBrain
from doom.server import encoded_frame
from flappy.circuit import haltere_afferents, haltere_current_for_velocity, wing_motor_readouts
from flappy.controls import FlapControls
from flappy.game import Game, FPS
from vision.retina import BilinearLuminance

ROOT = Path(__file__).resolve().parents[1]
latest = {'status': 'starting', 'generated_at_ms': 0}
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
        controls = FlapControls(wing_motor_readouts(brain))
        game = Game(seed=args.seed)
        retina = BilinearLuminance()
        run_id = str(uuid.uuid4())
        duration_ms = 1000 / FPS
        total_spikes, best_score = 0, 0
        seq = 0
        start = time.monotonic()
        print(json.dumps({'status': 'running', 'run_id': run_id, 'port': args.port}), flush=True)
        while not stop.is_set():
            obs = game.observation()
            if obs['finished']:
                best_score = max(best_score, obs['score'])
                game.new_episode()
                obs = game.observation()
            frame = game.pixels()
            light = retina.sample(frame, brain.uv)
            current = haltere_current_for_velocity(obs['y_velocity'], gain=args.haltere_gain) if args.haltere_gain else 0.
            stimulation = (haltere, current) if current > 0 else None
            counts, neural_wall = brain.step(light, duration_ms, sugar=False, stimulation=stimulation)
            action = controls.decode(counts, duration_ms / 1000)
            game.act(action['flap'])
            total_spikes += int(counts.sum())
            seq += 1
            wall_seconds = time.monotonic() - start
            obs = game.observation()
            best_score = max(best_score, obs['score'])
            latest = {
                'schema': 1, 'status': 'running', 'run_id': run_id, 'sequence': seq,
                'generated_at_ms': int(time.time() * 1000), 'backend': args.backend,
                'model_revision': build['model_revision'],
                'frame': encoded_frame(frame), 'flap': action['flap'],
                'game': obs | {'best_score': best_score},
                'haltere_current': round(current, 3),
                'clocks': {
                    'wall_seconds': round(wall_seconds, 3),
                    'neural_seconds': round(brain.sim_ms / 1000, 4),
                    'speed': round((brain.sim_ms / 1000) / wall_seconds, 4) if wall_seconds > 0 else 0,
                    'brain_step_ms': round(neural_wall * 1000, 3),
                },
                'total_spikes': total_spikes,
                'readouts': action['readouts'],
            }
    except Exception as e:
        latest = {'status': 'error', 'generated_at_ms': int(time.time() * 1000),
                   'message': 'The simulation stopped. No live data is available.'}
        import traceback; traceback.print_exc()
    finally:
        try: game.close()
        except Exception: pass


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/state':
            body = json.dumps(latest, separators=(',', ':')).encode()
        elif self.path == '/health':
            body = json.dumps({k: latest.get(k) for k in ['status', 'run_id', 'sequence', 'generated_at_ms']},
                               separators=(',', ':')).encode()
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass

    def log_message(self, *args): pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8768)
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--seed', type=int, default=41027)
    p.add_argument('--backend', choices=['native', 'gpu'], default='native')
    p.add_argument('--haltere-gain', type=float, default=10.,
                    help='mV-equivalent current at terminal fall speed, scaled down for slower falls; 0 disables it')
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
