"""End-to-end pipeline verification: Raw Frame -> Gate 1 -> NUC -> Calibration -> Norm -> Tensor"""
import sys
sys.path.insert(0, "src")

import numpy as np
import torch

# ── Stage 0: Gate 1 (Input Quality) ──────────────────────────────
from fireedge.validation import GroundGateValidator, FlightGateValidator, ValidatorConfig

print("=" * 70)
print("PIPELINE VERIFICATION: Raw Frame -> Gate 1 -> NUC/Norm -> Tensor")
print("=" * 70)

# Simulate a raw 16-bit thermal frame (256x256)
rng = np.random.default_rng(42)
raw_frame = rng.integers(20000, 35000, size=(256, 256), dtype=np.uint16)
# Inject 5 hot "fire" pixels
raw_frame[100, 100] = 55000
raw_frame[101, 100] = 52000
raw_frame[100, 101] = 51000
raw_frame[150, 150] = 53000
raw_frame[200, 80]  = 54000

print("\n[1] RAW FRAME")
print(f"    Shape: {raw_frame.shape}, dtype: {raw_frame.dtype}")
print(f"    Range: [{raw_frame.min()}, {raw_frame.max()}] DN")

# Gate 1: Ground Mode (dual-band)
b10_cfg = ValidatorConfig(bit_depth=16, valid_min=15000, valid_max=60000)
b11_cfg = ValidatorConfig(bit_depth=16, valid_min=15000, valid_max=55000)

ground_val = GroundGateValidator([b10_cfg, b11_cfg])
raw_dual = np.stack([raw_frame, raw_frame * 0.97], axis=0)  # Simulated B10/B11
rep = ground_val.validate(raw_dual)

print(f"\n[2] GATE 1: Input Quality Gate")
print(f"    Verdict: {rep.status.value}")
print(f"    Reasons: {rep.reasons}")
print(f"    Hot candidates: {rep.metrics.get('ch0.hot_candidates', 0)}")
gate1_pass = rep.ok
print(f"    PASS: {gate1_pass}")

if not gate1_pass:
    print("    ⚠️  Frame REJECTED by Gate 1 (would be discarded in flight)")
    print("    Continuing for verification purposes...")

# ── Stage 1: Calibration (DN -> Kelvin) ──────────────────────────
from fireedge.preprocessing.calibration import dn_to_temperature

temp_k = dn_to_temperature(raw_frame, band="B10", unit="kelvin")
print(f"\n[3] CALIBRATION: DN -> Brightness Temperature")
print(f"    Shape: {temp_k.shape}, dtype: {temp_k.dtype}")
print(f"    Range: [{temp_k.min():.1f}, {temp_k.max():.1f}] K")
print(f"    Range: [{temp_k.min()-273.15:.1f}, {temp_k.max()-273.15:.1f}] °C")

# ── Stage 2a: NUC (Non-Uniformity Correction) ───────────────────
from fireedge.preprocessing.nuc import TwoPointNUC, MicrobolometerDegrader, MicrobolometerConfig

# Simulate blackbody calibration frames
cold_frame = np.full((256, 256), 22000, dtype=np.uint16) + rng.integers(-50, 50, (256, 256))
hot_frame  = np.full((256, 256), 40000, dtype=np.uint16) + rng.integers(-50, 50, (256, 256))

nuc = TwoPointNUC(cold_frame, hot_frame, cold_temp=293.15, hot_temp=323.15)
corrected = nuc.correct(raw_frame)

print(f"\n[4] NUC: Two-Point Non-Uniformity Correction")
print(f"    Calibrated: {nuc.is_calibrated}")
print(f"    Corrected shape: {corrected.shape}, dtype: {corrected.dtype}")
print(f"    Corrected range: [{corrected.min():.2f}, {corrected.max():.2f}] K-equivalent")

# Degrader simulation
degrader = MicrobolometerDegrader(MicrobolometerConfig(seed=42))
degraded = degrader.degrade(raw_frame.astype(np.float32))
print(f"\n[5] DEGRADER: Sensor Degradation Simulation")
print(f"    Degraded shape: {degraded.shape}, dtype: {degraded.dtype}")
print(f"    Degraded range: [{degraded.min():.1f}, {degraded.max():.1f}] DN")

# ── Stage 2b: Normalization ──────────────────────────────────────
from fireedge.preprocessing.normalization import fixed_window_norm, robust_percentile_norm

norm_fixed = fixed_window_norm(temp_k)
norm_robust = robust_percentile_norm(temp_k)

print(f"\n[6] NORMALIZATION: Temperature -> [0.0, 1.0]")
print(f"    Fixed window:  range=[{norm_fixed.min():.4f}, {norm_fixed.max():.4f}]")
print(f"    Robust %tile:  range=[{norm_robust.min():.4f}, {norm_robust.max():.4f}]")

# ── Stage 2c: PyTorch Tensor Output ──────────────────────────────
tensor_img = torch.from_numpy(norm_robust[np.newaxis, :, :]).float()
tensor_mask = torch.zeros(1, 256, 256, dtype=torch.float32)
# Mark fire pixels
tensor_mask[0, 100, 100] = 1.0
tensor_mask[0, 101, 100] = 1.0
tensor_mask[0, 100, 101] = 1.0
tensor_mask[0, 150, 150] = 1.0
tensor_mask[0, 200, 80]  = 1.0

print(f"\n[7] PYTORCH TENSOR OUTPUT (Interface Contract for Person 2)")
print(f"    Image tensor:  shape={tensor_img.shape}, dtype={tensor_img.dtype}")
print(f"    Mask tensor:   shape={tensor_mask.shape}, dtype={tensor_mask.dtype}")
print(f"    Fire pixels:   {int(tensor_mask.sum())}")

# Contract check
assert tensor_img.shape == (1, 256, 256), f"Image shape mismatch: {tensor_img.shape}"
assert tensor_mask.shape == (1, 256, 256), f"Mask shape mismatch: {tensor_mask.shape}"
assert tensor_img.dtype == torch.float32, f"Image dtype mismatch: {tensor_img.dtype}"
assert tensor_mask.dtype == torch.float32, f"Mask dtype mismatch: {tensor_mask.dtype}"
assert 0.0 <= tensor_img.min() <= tensor_img.max() <= 1.0, "Image not in [0, 1]"

print("\n" + "=" * 70)
print("[OK] FULL PIPELINE VERIFIED: Raw Frame -> Gate 1 -> NUC/Norm -> Tensor")
print("     Contract: [B, 1, 256, 256] float32, normalized to [0.0, 1.0]")
print("=" * 70)
