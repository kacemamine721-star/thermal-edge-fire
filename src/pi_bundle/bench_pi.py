#!/usr/bin/env python3
"""Benchmark the fire-detection U-Net on ANY machine (Raspberry Pi, ARM VM, laptop).
Usage:  python3 bench_pi.py --model model_int8.onnx --threads 1
Power:  python3 bench_pi.py --model model_int8.onnx --threads 1 --loop-seconds 120
        (read the USB power meter during the loop and during idle; see README.md)"""
import argparse, json, platform, subprocess, time
import numpy as np
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument('--model', default='model_int8.onnx')
p.add_argument('--threads', type=int, default=1)
p.add_argument('--runs', type=int, default=100)
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--loop-seconds', type=float, default=0.0)
a = p.parse_args()


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith(('Model', 'model name', 'Hardware')):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'n/a'


x = np.load('sample_input.npy').astype(np.float32)
so = ort.SessionOptions()
so.intra_op_num_threads = a.threads
so.inter_op_num_threads = 1
sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
name = sess.get_inputs()[0].name
for _ in range(a.warmup):
    sess.run(None, {name: x})
times = []
for _ in range(a.runs):
    t0 = time.perf_counter()
    sess.run(None, {name: x})
    times.append((time.perf_counter() - t0) * 1e3)
t = np.array(times)
report = {
    'cpu': cpu_model(), 'machine': platform.machine(), 'onnxruntime': ort.__version__,
    'model': a.model, 'threads': a.threads, 'input_shape': list(x.shape),
    'latency_ms_mean': round(float(t.mean()), 2), 'p50': round(float(np.percentile(t, 50)), 2),
    'p95': round(float(np.percentile(t, 95)), 2), 'p99': round(float(np.percentile(t, 99)), 2),
    'cpu_governor': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
    'cpu_freq_mhz': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
    'temp': sh('vcgencmd measure_temp'), 'throttled': sh('vcgencmd get_throttled'),
}
print(json.dumps(report, indent=2))

if a.loop_seconds > 0:
    print(f'Looping for {a.loop_seconds:.0f} s: note the power meter reading now ...')
    n, t_start = 0, time.perf_counter()
    while time.perf_counter() - t_start < a.loop_seconds:
        sess.run(None, {name: x})
        n += 1
    elapsed = time.perf_counter() - t_start
    print(json.dumps({'loop_inferences': n, 'loop_seconds': round(elapsed, 1),
                      'seconds_per_inference': round(elapsed / n, 4)}, indent=2))
    print('energy per inference (Wh) = (P_load_W - P_idle_W) * seconds_per_inference / 3600')
