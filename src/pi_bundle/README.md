# Raspberry Pi benchmark bundle (primary model: pereira3c, bands ['c7', 'c6', 'c2'])

Target: Raspberry Pi 4 Model B / Compute Module 4 (Cortex-A72, the CPU class used by Chatar et al.);
KITSUNE itself used a CM3+ (Cortex-A53), so also run it there if you can get one.

## Run
```
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv venv && . venv/bin/activate
pip install onnxruntime numpy
python3 bench_pi.py --model model_fp32.onnx --threads 1
python3 bench_pi.py --model model_int8.onnx --threads 1
python3 bench_pi.py --model model_int8.onnx --threads 4
```
Record: CPU governor (use `performance` for the final numbers), temperature and `throttled` (must be 0x0).
Use a heatsink/fan so the CPU does not throttle.

## Energy per inference (USB power meter between supply and Pi)
1. Idle: note P_idle (W) after 2 minutes with nothing running.
2. `python3 bench_pi.py --model model_int8.onnx --threads 1 --loop-seconds 120` and note P_load (W) during the loop.
3. energy_per_inference_Wh = (P_load - P_idle) * seconds_per_inference / 3600.
Compare with KITSUNE's reported 0.10 Wh per inference (MiniVGGNet on a CM3+, classification only).

## Emulation (no board)
`docker run --platform linux/arm64 ...` with QEMU only checks that the ARM build runs; its timings are not valid.

Input shape: [1, 3, 256, 256] float32 (already preprocessed). Model MACs/frame: 2.32 G.
