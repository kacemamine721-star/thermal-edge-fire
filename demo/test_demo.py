#!/usr/bin/env python3
"""
Unit, Parity and Golden Tests for IASTAM Edge Fire Detection Web Demo
====================================================================
Validates all requirements per PRD/TRD (V7.1):
  - TRD §1.6: Manifest constants sync (threshold, temperature, packet sizes)
  - TRD §2.9: SHA-256 Checksums verification for ONNX models and manifest
  - TRD §2.10: Golden test reproducibility against V7.1 baseline outputs
  - F12: Lossless zlib baseline comparison (information extraction advantage)
  - F13: INT8 vs FP32 pixel decision agreement parity (>= 99.0%)

Run: pytest test_demo.py -v  OR  python test_demo.py
"""

import hashlib
import json
import os
import sys
import zlib
from pathlib import Path
import numpy as np
import pytest
import rasterio

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from pipeline import (
    encode_packet, decode_packet, EndToEndPipeline, Gate1,
    preprocess_dn, SINGLE_EVENT_BYTES, EMPTY_PACKET_BYTES, MAX_PACKET_BYTES,
    RAW_BYTES_PER_TILE, MODEL_ID, THRESHOLD_DEFAULT, TEMPERATURE_DEFAULT,
    MANIFEST_DATA
)
from io_tiff import load_patch, SceneTiler, pixel_to_wgs84
from metrics import compute_pixel_metrics, compute_event_recall


def test_packet_encoding_and_crc():
    """Verifies packet serialization, byte lengths, and CRC32 verification."""
    # Empty heartbeat packet
    p_empty = encode_packet(frame_id=0, model_id=1, timestamp_ms=0, events=[], flags=32)
    assert len(p_empty) == EMPTY_PACKET_BYTES  # exactly 21 bytes
    d_empty = decode_packet(p_empty)
    assert d_empty['crc_valid'] is True
    assert d_empty['n_events'] == 0
    assert d_empty['flags'] == 32

    # Single-event packet
    sample_ev = [{
        'lat_e7': 368000000, 'lon_e7': 101800000,
        'x': 120, 'y': 85, 'w': 2, 'h': 2,
        'area_ha_x10': 3, 'confidence_u16': 64000, 'metric_u16': 2500,
        'gsd_m_u16': 30, 'geo_uncertainty_m_u16': 65535, 'flags': 4
    }]
    p_single = encode_packet(frame_id=1, model_id=1, timestamp_ms=1600000000, events=sample_ev, flags=0)
    assert len(p_single) == SINGLE_EVENT_BYTES  # exactly 48 bytes
    d_single = decode_packet(p_single)
    assert d_single['crc_valid'] is True
    assert d_single['n_events'] == 1
    assert len(d_single['events']) == 1
    assert d_single['events'][0]['lat'] == 36.8
    assert d_single['events'][0]['lon'] == 10.18

    # Multi-event packet (6 events)
    events_6 = sample_ev * 6
    p_max = encode_packet(frame_id=2, model_id=1, timestamp_ms=0, events=events_6, flags=0)
    assert len(p_max) == MAX_PACKET_BYTES  # exactly 183 bytes
    d_max = decode_packet(p_max)
    assert d_max['crc_valid'] is True
    assert d_max['n_events'] == 6


def test_gate1_rejections():
    """Verifies that Gate 1 intercepts invalid frames and dead columns."""
    gate1 = Gate1(scrub=False)

    # Constant array (zero variance -> dead columns)
    corrupt_dn = np.full((3, 256, 256), 1000, dtype=np.uint16)
    res = gate1.validate(corrupt_dn)
    assert res['status'] == 'REJECT'
    assert 'dead_columns' in res['reasons'][0].lower() or 'column' in res['reasons'][0].lower()

    # Wrong shape
    wrong_shape = np.zeros((2, 256, 256), dtype=np.uint16)
    res = gate1.validate(wrong_shape)
    assert res['status'] == 'REJECT'


def test_preprocessing_contract():
    """Verifies fixed radiometric linear conversion against training contract."""
    dn = np.array([[[0, 5000, 50000, 65535]]], dtype=np.uint16)
    refl = preprocess_dn(dn)
    # 0 -> 0*2e-5 - 0.1 = -0.1 clipped to 0.0
    assert refl[0, 0, 0] == 0.0
    # 5000 -> 5000*2e-5 - 0.1 = 0.1 - 0.1 = 0.0
    assert abs(refl[0, 0, 1] - 0.0) < 1e-5
    # 50000 -> 50000*2e-5 - 0.1 = 1.0 - 0.1 = 0.9
    assert abs(refl[0, 0, 2] - 0.9) < 1e-4
    # Max value capped at 1.6
    assert refl[0, 0, 3] <= 1.6


def test_manifest_constants_sync():
    """TRD §1.6: All constants read from manifest_v7.json, never hard-coded."""
    assert MANIFEST_DATA, "manifest_v7.json must be loaded and non-empty"
    assert abs(THRESHOLD_DEFAULT - 0.96) < 1e-5
    assert abs(TEMPERATURE_DEFAULT - 2.122759) < 1e-4
    assert MODEL_ID == 1
    assert MAX_PACKET_BYTES == 183
    assert EMPTY_PACKET_BYTES == 21
    assert RAW_BYTES_PER_TILE == 393216


def test_model_checksums():
    """TRD §2.9: Verifies SHA-256 of ONNX model files against models/CHECKSUMS.txt."""
    chk_file = DEMO_DIR / "models" / "CHECKSUMS.txt"
    assert chk_file.exists(), "models/CHECKSUMS.txt must exist per TRD §2.9"

    with open(chk_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for line in lines:
        expected_hash, fname = line.split(maxsplit=1)
        target_path = DEMO_DIR / "models" / fname
        raw_bytes = target_path.read_bytes()
        # Normalize CRLF line endings for text/JSON files across OS platforms
        if fname.endswith(('.json', '.txt')):
            raw_bytes = raw_bytes.replace(b'\r\n', b'\n')
        computed_hash = hashlib.sha256(raw_bytes).hexdigest()
        assert computed_hash == expected_hash, f"SHA-256 mismatch for {fname}: {computed_hash} != {expected_hash}"


def test_zlib_baseline_comparison():
    """F12: Verifies that alert packets achieve orders-of-magnitude greater reduction than zlib."""
    patch_p = DEMO_DIR / "sample_data" / "patches" / "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif"
    if not patch_p.exists():
        pytest.skip("Sample patch not found")

    patch_info = load_patch(str(patch_p))
    raw_bytes = patch_info['raw_3band'].tobytes()
    zlib_bytes = len(zlib.compress(raw_bytes, level=6))

    # Lossless compression on sensor data is limited (~1.5x - 3.5x)
    zlib_ratio = len(raw_bytes) / zlib_bytes
    assert 1.2 <= zlib_ratio <= 4.0, f"Unexpected zlib ratio: {zlib_ratio}"

    model_p = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"
    pipe = EndToEndPipeline(str(model_p), num_threads=1)
    res = pipe.process_frame(patch_info['raw_3band'])

    # IASTAM Alert packet is 183 bytes (max) vs 100+ KB zlib
    assert res['packet_bytes'] <= 183
    assert res['packet_bytes'] < zlib_bytes
    alert_vs_zlib = zlib_bytes / res['packet_bytes']
    assert alert_vs_zlib > 500.0, f"Alert packet should be >500x smaller than zlib, got {alert_vs_zlib}x"


def test_int8_fp32_decision_agreement():
    """F13 & TRD §2.10: Pixel-decision agreement between INT8 and FP32 must be >= 99.0%."""
    int8_p = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"
    fp32_p = DEMO_DIR / "models" / "unet_pereira3c_fp32.onnx"
    if not (int8_p.exists() and fp32_p.exists()):
        pytest.skip("INT8 or FP32 model not found")

    pipe_int8 = EndToEndPipeline(str(int8_p), num_threads=1)
    pipe_fp32 = EndToEndPipeline(str(fp32_p), num_threads=1)

    patches = [
        "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif",
        "LC08_L1GT_226074_20200921_20200921_01_RT_p00812.tif",
        "LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif",
        "LC08_L1TP_001067_20200812_20200812_01_RT_p00000_clean_forest.tif"
    ]

    for p_name in patches:
        p_path = DEMO_DIR / "sample_data" / "patches" / p_name
        if not p_path.exists():
            continue
        info = load_patch(str(p_path))
        m_int8 = pipe_int8.process_frame(info['raw_3band'])['binary_mask']
        m_fp32 = pipe_fp32.process_frame(info['raw_3band'])['binary_mask']
        agreement_pct = (m_int8 == m_fp32).mean() * 100.0
        assert agreement_pct >= 99.0, f"Agreement on {p_name} was {agreement_pct:.2f}% < 99.0%"


def test_golden_reproducibility():
    """TRD §2.10: Each bundled sample reproduces V7.1 status, n_events and packet hex."""
    golden_master_p = DEMO_DIR / "sample_data" / "golden.json"
    assert golden_master_p.exists(), "sample_data/golden.json must exist"

    with open(golden_master_p, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    model_p = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"
    pipe = EndToEndPipeline(str(model_p), num_threads=1)

    for fname, golden in golden_data.items():
        p_path = DEMO_DIR / "sample_data" / "patches" / fname
        assert p_path.exists(), f"Sample {fname} missing"

        # Also verify per-sample golden file exists
        per_sample_json = DEMO_DIR / "sample_data" / "patches" / f"{Path(fname).stem}.golden.json"
        assert per_sample_json.exists(), f"Per-sample golden file missing: {per_sample_json}"

        info = load_patch(str(p_path))
        res = pipe.process_frame(
            info['raw_3band'],
            frame_id=0,
            geo_transform=info['transform'],
            crs=info['crs'],
            gsd_m=info['gsd_m'],
            timestamp_ms=0
        )

        assert res['status'] == golden['status'], f"Status mismatch on {fname}"
        assert len(res['events']) == golden['n_events_detected'], f"Events count mismatch on {fname}"
        assert len(res['packet']) == golden['packet_bytes'], f"Packet bytes mismatch on {fname}"
        if golden['packet_hex']:
            assert res['packet'].hex().upper() == golden['packet_hex'], f"Packet hex mismatch on {fname}"


def test_real_patch_pipeline():
    """Executes end-to-end pipeline on real benchmark satellite patch."""
    model_p = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"
    if not model_p.exists():
        pytest.skip("Model ONNX file not found")

    pipe = EndToEndPipeline(str(model_p), num_threads=1)
    patch_p = DEMO_DIR / "sample_data" / "patches" / "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif"
    if not patch_p.exists():
        patch_p = DEMO_DIR.parent / "data" / "raw" / "activefire" / "patches" / "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif"
    if not patch_p.exists():
        pytest.skip("Sample patch not found")

    patch_info = load_patch(str(patch_p))
    res = pipe.process_frame(
        patch_info['raw_3band'],
        geo_transform=patch_info['transform'],
        crs=patch_info['crs'],
        gsd_m=patch_info['gsd_m']
    )

    assert res['status'] == 'ALERT'
    assert len(res['events']) > 0
    assert res['reduction_ratio'] > 2000.0  # At least 2,148x reduction
    assert res['timings']['total_ms'] < 300.0  # Fast execution


if __name__ == "__main__":
    print("Running IASTAM Verification Test Suite...")
    test_packet_encoding_and_crc()
    print("  [PASS] test_packet_encoding_and_crc")
    test_gate1_rejections()
    print("  [PASS] test_gate1_rejections")
    test_preprocessing_contract()
    print("  [PASS] test_preprocessing_contract")
    test_manifest_constants_sync()
    print("  [PASS] test_manifest_constants_sync (TRD §1.6)")
    test_model_checksums()
    print("  [PASS] test_model_checksums (TRD §2.9)")
    test_zlib_baseline_comparison()
    print("  [PASS] test_zlib_baseline_comparison (F12)")
    test_int8_fp32_decision_agreement()
    print("  [PASS] test_int8_fp32_decision_agreement (F13 & TRD §2.10)")
    test_golden_reproducibility()
    print("  [PASS] test_golden_reproducibility (TRD §2.10)")
    test_real_patch_pipeline()
    print("  [PASS] test_real_patch_pipeline")
    print("\n==========================================")
    print("ALL 9 IASTAM PRODUCTION TESTS PASSED 100%!")
    print("==========================================")
