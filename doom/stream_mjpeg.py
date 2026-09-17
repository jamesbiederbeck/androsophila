"""Read-only MJPEG re-broadcast of doom/server.py's existing /state frames,
for opening directly in VLC or any MJPEG-capable viewer (Media > Open Network
Stream) instead of the doom-ui web viewer.

Polls the already-running broadcaster's /state endpoint (the same JSON the
doom-ui web viewer fetches) and re-serves just the 'frame' JPEG as a
multipart/x-mixed-replace stream. Does not touch doom/server.py, does not
add any new capability to the broadcaster -- only re-packages frames that
are already public over /state. No filesystem, shell, credentials, or
model-mutating access; strictly a read-only reformatter.
"""
import argparse
import base64
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = 'doomflyframe'

def fetch_jpeg(origin):
    with urllib.request.urlopen(f'{origin}/state', timeout=4) as resp:
        data = json.loads(resp.read())
    frame = data.get('frame')
    if not frame or not frame.startswith('data:image/jpeg;base64,'):
        return None, data.get('generated_at_ms')
    return base64.b64decode(frame.split(',', 1)[1]), data.get('generated_at_ms')

def make_handler(origin, poll_hz):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/stream.mjpg'):
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.end_headers()
            last_ts = None
            try:
                while True:
                    try:
                        jpeg, ts = fetch_jpeg(origin)
                    except Exception:
                        time.sleep(1 / poll_hz); continue
                    if jpeg is not None and ts != last_ts:
                        last_ts = ts
                        self.wfile.write(f'--{BOUNDARY}\r\n'.encode())
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    time.sleep(1 / poll_hz)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args): pass
    return Handler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', default='http://127.0.0.1:8766', help='doom/server.py broadcaster origin')
    p.add_argument('--bind', default='0.0.0.0', help='interface to serve the MJPEG stream on')
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--fps', type=float, default=8.0, help='poll rate; matches doom/broadcast.py DISPLAY_FPS by default')
    args = p.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(args.origin, args.fps))
    print(f'MJPEG stream at http://<this-machine-lan-ip>:{args.port}/stream.mjpg (source: {args.origin})')
    server.serve_forever()

if __name__ == '__main__':
    main()
