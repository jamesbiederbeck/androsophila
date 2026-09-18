"""Replay a captured organic workload (doom/capture_organic_workload.py) into
NativeBrain and GPUBrain from identical initial state, on the identical
per-tick light/steps sequence, reporting throughput and spike-count
agreement -- the same comparison doom/validate_gpu.py and
doom/benchmark_gpu.py make with synthetic uniform luminance, but against a
real captured closed-loop gameplay sequence.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from connectome_sim.native import NativeBrain
from connectome_sim.gpu import GPUBrain

ROOT = Path(__file__).resolve().parents[1]

def run_backend(cls, path, lights, steps):
    brain = cls(path)
    total = np.zeros(brain.n, dtype=np.int64)
    step_wall_ms = []
    start = time.perf_counter()
    for t in range(len(steps)):
        counts, wall = brain.step(lights[t], steps[t] * .1)
        total += counts
        step_wall_ms.append(wall * 1000)
    total_wall = time.perf_counter() - start
    return brain, total, np.asarray(step_wall_ms), total_wall

def run(capture_path, dataset):
    cap = np.load(capture_path)
    lights, steps = cap['lights'], cap['steps']
    ticks = len(steps)
    path = ROOT / 'outputs/doom' / dataset / 'graph.npz'
    native, native_total, native_step_ms, native_wall = run_backend(NativeBrain, path, lights, steps)
    gpu, gpu_total, gpu_step_ms, gpu_wall = run_backend(GPUBrain, path, lights, steps)
    mismatch = native_total != gpu_total
    return {
        'ticks': int(ticks), 'neurons': int(native.n),
        'workload': 'captured real ViZDoom closed-loop gameplay, replayed identically into both backends',
        'capture_file': str(capture_path),
        'nonzero_luminance_fraction': float(cap['nonzero_luminance_fraction']) if 'nonzero_luminance_fraction' in cap else None,
        'native': {
            'tics_per_second': round(ticks / native_wall, 3),
            'total_wall_seconds': round(native_wall, 4),
            'step_wall_ms_median': float(np.median(native_step_ms)),
            'spike_count_total': int(native_total.sum()),
        },
        'gpu_event_driven': {
            'tics_per_second': round(ticks / gpu_wall, 3),
            'total_wall_seconds': round(gpu_wall, 4),
            'step_wall_ms_median': float(np.median(gpu_step_ms)),
            'spike_count_total': int(gpu_total.sum()),
        },
        'speedup_gpu_over_native': round((ticks / gpu_wall) / (ticks / native_wall), 3),
        'spike_count_mismatch_neurons': int(mismatch.sum()),
        'spike_count_relative_diff': round(abs(int(native_total.sum()) - int(gpu_total.sum())) / int(native_total.sum()), 6),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True)
    p.add_argument('--dataset', default='malecns_v1')
    p.add_argument('--out')
    args = p.parse_args()
    report = run(Path(args.capture), args.dataset)
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
