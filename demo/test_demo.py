#!/usr/bin/env python3
"""
Unit, parity and golden tests for the fire-detection web demo
=============================================================
  - Manifest constants (threshold, temperature, packet sizes) are read from the model manifest
  - SHA-256 checksums of the ONNX models and the manifest (models/CHECKSUMS.txt)
  - Gate 1 checks, packet encode/decode, dropped-event counter, event metrics
  - Golden reproducibility: the bundled samples reproduce status, event count and packet hex
  - Lossless zlib baseline comparison
  - INT8 vs FP32 pixel agreement (>= 99.0%)

Run: pytest test_demo.py -v   or   python test_demo.py
"""

import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from pipeline import (  # noqa: E402
    encode_packet, decode_packet, EndToEndPipeline, Gate1,
    preprocess_dn, SINGLE_EVENT_BYTES, EMPTY_PACKET_BYTES, MAX_PACKET_BYTES, MAX_EVENTS,
    RAW_BYTES_PER_TILE, MODEL_ID, THRESHOLD_DEFAULT, TEMPERATURE_DEFAULT,
    MANIFEST_DATA, FLAG_INPUT_REPAIRED, FLAG_HEARTBEAT, FLAG_TIME_UNAVAILABLE,
)
from io_tiff import load_patch  # noqa: E402
from metrics import compute_pixel_metrics, compute_event_recall  # noqa: E402

INT8_MODEL = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"
FP32_MODEL = DEMO_DIR / "models" / "unet_pereira3c_fp32.onnx"
PATCH_DIR = DEMO_DIR / "sample_data" / "patches"


def _sample_event(**kw):
    ev = {
        'lat_e7': 368000000, 'lon_e7': 101800000,
        'x': 120, 'y': 85, 'w': 2, 'h': 2,
        'area_ha_x10': 3, 'confidence_u16': 64000, 'metric_u16': 2500,
        'gsd_m_u16': 30, 'geo_uncertainty_m_u16': 65535, 'flags': 6,
    }
    ev.update(kw)
    return ev


def _frame(seed=0):
    """Synthetic raw frame with realistic DN values (reflectance 0.05-0.30) in all three bands."""
    rng = np.random.default_rng(seed)
    return rng.integers(7500, 20000, size=(3, 256, 256)).astype(np.uint16)


def test_packet_encoding_and_crc():
    """Packet serialisation, byte lengths and CRC32."""
    p_empty = encode_packet(frame_id=0, model_id=1, timestamp_ms=0, events=[], flags=32)
    assert len(p_empty) == EMPTY_PACKET_BYTES            # 21 bytes
    d_empty = decode_packet(p_empty)
    assert d_empty['crc_valid'] is True and d_empty['length_valid'] is True
    assert d_empty['n_events'] == 0 and d_empty['flags'] == 32

    p_single = encode_packet(frame_id=1, model_id=1, timestamp_ms=1600000000, events=[_sample_event()], flags=0)
    assert len(p_single) == SINGLE_EVENT_BYTES           # 48 bytes
    d_single = decode_packet(p_single)
    assert d_single['crc_valid'] is True
    assert d_single['n_events'] == 1 and len(d_single['events']) == 1
    assert d_single['events'][0]['lat'] == 36.8 and d_single['events'][0]['lon'] == 10.18
    assert d_single['events_dropped'] == 0
    assert d_single['events'][0]['flags'] == 6

    p_max = encode_packet(frame_id=2, model_id=1, timestamp_ms=0, events=[_sample_event()] * 6, flags=0)
    assert len(p_max) == MAX_PACKET_BYTES                # 183 bytes
    d_max = decode_packet(p_max)
    assert d_max['crc_valid'] is True and d_max['n_events'] == 6 and d_max['events_dropped'] == 0


def test_dropped_event_counter():
    """More than 6 events: only 6 are sent and the number dropped is recorded in the last event's flags nibble."""
    events = [_sample_event(confidence_u16=65000 - 100 * i) for i in range(9)]
    pkt = encode_packet(frame_id=3, model_id=1, timestamp_ms=0, events=events, flags=0)
    assert len(pkt) == MAX_PACKET_BYTES
    d = decode_packet(pkt)
    assert d['n_events'] == MAX_EVENTS and d['events_dropped'] == 3
    assert d['events'][0]['flags'] == 6 and d['events'][-1]['flags'] == 6      # flag bits are untouched
    assert [e['confidence_u16'] for e in d['events']] == [65000 - 100 * i for i in range(6)]
    # saturates at 15
    d2 = decode_packet(encode_packet(0, 1, 0, [_sample_event()] * (MAX_EVENTS + 40)))
    assert d2['events_dropped'] == 15


def test_corrupted_packet_is_reported():
    pkt = bytearray(encode_packet(0, 1, 0, [_sample_event()]))
    pkt[20] ^= 0xFF
    d = decode_packet(bytes(pkt))
    assert d['crc_valid'] is False
    short = encode_packet(0, 1, 0, [_sample_event()])[:-30] + b"\x00" * 4
    assert decode_packet(short)['length_valid'] is False
    with pytest.raises(ValueError):
        decode_packet(b"\x00" * 30)


def test_gate1_rejections():
    """Gate 1 rejects invalid frames for the right reasons."""
    gate1 = Gate1(scrub=False)

    res = gate1.validate(np.full((3, 256, 256), 1000, dtype=np.uint16))      # constant bands
    assert res['status'] == 'REJECT' and 'constant' in res['reasons'][0]

    res = gate1.validate(_frame()[:2])                                        # two channels instead of three
    assert res['status'] == 'REJECT' and 'bad shape' in res['reasons'][0]

    res = gate1.validate(np.zeros((3, 256, 256), dtype=np.uint16))            # no valid pixels
    assert res['status'] == 'REJECT' and 'valid area' in res['reasons'][0]

    bad = _frame().copy()
    bad[0, :, 40:120] = 0                                                      # 80 dead columns (31%) in one band
    res = gate1.validate(bad)
    assert res['status'] == 'REJECT' and 'dead columns' in res['reasons'][0]

    sat = _frame().copy()
    sat[1, :, :] = 65535
    assert gate1.validate(sat)['status'] == 'REJECT'                           # band saturated on 100% of pixels


def test_gate1_dead_columns_flagged_and_repaired():
    frame = _frame(1)
    frame[0, :, 10:13] = 0                                                     # 3 dead columns (1.2%) in band C7
    res = Gate1().validate(frame)
    assert res['status'] == 'OK' and res['n_dead_columns'] == 3 and res['n_dead_columns_repaired'] == 0
    assert any('not repaired' in r for r in res['reasons'])
    assert np.array_equal(res['dn'], frame)                                    # default: flag only

    res = Gate1(repair_dead_columns=True).validate(frame)
    assert res['status'] == 'OK' and res['n_dead_columns_repaired'] == 3 and res['flags'] & FLAG_INPUT_REPAIRED
    assert (res['dn'][0, :, 10:13] > 0).all()
    lo = np.minimum(frame[0, :, 9], frame[0, :, 13]).astype(int)
    hi = np.maximum(frame[0, :, 9], frame[0, :, 13]).astype(int)
    assert ((res['dn'][0, :, 11].astype(int) >= lo) & (res['dn'][0, :, 11].astype(int) <= hi)).all()
    assert np.array_equal(res['dn'][1:], frame[1:])                            # other bands untouched
    assert np.array_equal(frame[0, :, 10:13], np.zeros((256, 3), np.uint16))   # input array not modified

    edge = _frame(2)
    edge[2, :, 0] = 0                                                          # dead column at the frame edge
    fixed = Gate1(repair_dead_columns=True).validate(edge)['dn']
    assert np.array_equal(fixed[2, :, 0], edge[2, :, 1])

    clean = _frame(3)                                                          # no side effect on clean data
    out = Gate1(repair_dead_columns=True).validate(clean)
    assert out['flags'] == 0 and np.array_equal(out['dn'], clean)

    padded = _frame(4)
    padded[:, :, 250:] = 0                                                     # zero-padded border: all bands zero
    pad_res = Gate1(repair_dead_columns=True).validate(padded)
    assert pad_res['status'] == 'OK' and pad_res['n_dead_columns'] == 0


def test_gate1_transients_all_bands():
    frame = _frame(5)
    frame[1, 100, 100] = 65535                                                 # isolated saturated pixel in band C6
    res = Gate1(scrub=False).validate(frame)
    assert res['n_transients'] == 1 and res['dn'][1, 100, 100] == 65535
    res = Gate1(scrub=True).validate(frame)
    assert res['n_transients'] == 1 and res['dn'][1, 100, 100] < 65535 and res['flags'] & 4
    blob = _frame(6)
    blob[0, 50:54, 50:54] = 65535                                              # saturated 4x4 core is not a transient
    assert Gate1(scrub=True).validate(blob)['n_transients'] == 0


def test_preprocessing_contract():
    """Fixed radiometric conversion DN -> reflectance."""
    dn = np.array([[[0, 5000, 50000, 65535]]], dtype=np.uint16)
    refl = preprocess_dn(dn)
    assert refl[0, 0, 0] == 0.0                         # 0 -> -0.1 clipped to 0
    assert abs(refl[0, 0, 1] - 0.0) < 1e-5              # 5000 -> 0.0
    assert abs(refl[0, 0, 2] - 0.9) < 1e-4              # 50000 -> 0.9
    assert refl[0, 0, 3] <= 1.6


def test_manifest_constants_sync():
    """Constants come from the manifest, not from literals."""
    assert MANIFEST_DATA, "models/manifest.json (or manifest_v7.json) must be loaded and non-empty"
    assert abs(THRESHOLD_DEFAULT - 0.96) < 1e-5
    assert abs(TEMPERATURE_DEFAULT - 2.122759) < 1e-4
    assert MODEL_ID == 1
    assert MAX_PACKET_BYTES == 183
    assert EMPTY_PACKET_BYTES == 21
    assert RAW_BYTES_PER_TILE == 393216


def test_model_checksums():
    """SHA-256 of the model files against models/CHECKSUMS.txt."""
    chk_file = DEMO_DIR / "models" / "CHECKSUMS.txt"
    assert chk_file.exists(), "models/CHECKSUMS.txt must exist"
    with open(chk_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    for line in lines:
        expected_hash, fname = line.split(maxsplit=1)
        raw_bytes = (DEMO_DIR / "models" / fname).read_bytes()
        if fname.endswith(('.json', '.txt')):               # normalise CRLF so Windows and Linux agree
            raw_bytes = raw_bytes.replace(b'\r\n', b'\n')
        computed = hashlib.sha256(raw_bytes).hexdigest()
        assert computed == expected_hash, f"SHA-256 mismatch for {fname}: {computed} != {expected_hash}"


def test_event_metrics_match_notebook_definition():
    pred = np.zeros((64, 64), np.uint8)
    gt = np.zeros((64, 64), np.uint8)
    gt[10, 10] = 1                                       # found: prediction 3 px away (diagonal counts)
    pred[13, 13] = 1
    gt[40, 40] = 1                                       # missed
    pred[60, 5] = 1                                      # false event
    ev = compute_event_recall(pred, gt)
    assert ev['n_gt_events'] == 2 and ev['n_matched_events'] == 1 and ev['event_recall'] == 0.5
    assert ev['false_events'] == 1 and ev['n_pred_events'] == 2
    assert compute_event_recall(pred, np.zeros_like(gt))['event_recall'] is None
    m = compute_pixel_metrics(pred, gt)
    assert m['has_reference_fire'] is True and m['tp'] == 0 and m['fp'] == 2 and m['fn'] == 2
    assert compute_pixel_metrics(np.zeros_like(pred), np.zeros_like(gt))['has_reference_fire'] is False


def test_pipeline_plumbing_and_heartbeat():
    """Runs the whole pipeline on a synthetic frame (needs the INT8 model; a fire-free frame gives silence)."""
    if not INT8_MODEL.exists():
        pytest.skip("INT8 model not found")
    pipe = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=True)
    first = pipe.process_frame(_frame(7))
    assert first['status'] in ('HEARTBEAT_NO_FIRE', 'ALERT')              # first frame of the instance
    if first['status'] == 'HEARTBEAT_NO_FIRE':
        assert first['packet_bytes'] == EMPTY_PACKET_BYTES
        assert first['packet_json']['flags'] & FLAG_HEARTBEAT
        assert first['packet_json']['flags'] & FLAG_TIME_UNAVAILABLE
    silent = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=False)
    rej = silent.process_frame(np.zeros((3, 256, 256), np.uint16))
    assert rej['status'] == 'GATE1_REJECT' and rej['packet_json']['flags'] & FLAG_TIME_UNAVAILABLE


def test_zlib_baseline_comparison():
    """The alert packet is orders of magnitude smaller than the losslessly compressed frame."""
    patch_p = PATCH_DIR / "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif"
    if not patch_p.exists() or not INT8_MODEL.exists():
        pytest.skip("Sample patch or model not found")
    patch_info = load_patch(str(patch_p))
    raw_bytes = patch_info['raw_3band'].tobytes()
    zlib_bytes = len(zlib.compress(raw_bytes, level=6))
    zlib_ratio = len(raw_bytes) / zlib_bytes
    assert zlib_ratio >= 1.0, f"zlib made the frame larger: {zlib_ratio}"

    pipe = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=False)
    res = pipe.process_frame(patch_info['raw_3band'])
    assert res['packet_bytes'] <= MAX_PACKET_BYTES and res['packet_bytes'] < zlib_bytes
    assert zlib_bytes / res['packet_bytes'] > 500.0


def test_int8_fp32_decision_agreement():
    """Pixel-mask agreement between INT8 and FP32 on the bundled samples must be >= 99.0%."""
    if not (INT8_MODEL.exists() and FP32_MODEL.exists()):
        pytest.skip("INT8 or FP32 model not found")
    pipe_int8 = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=False)
    pipe_fp32 = EndToEndPipeline(str(FP32_MODEL), num_threads=1, heartbeat_enabled=False)
    patches = [
        "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif",
        "LC08_L1GT_226074_20200921_20200921_01_RT_p00812.tif",
        "LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif",
        "LC08_L1TP_001067_20200812_20200812_01_RT_p00000_clean_forest.tif",
    ]
    for p_name in patches:
        p_path = PATCH_DIR / p_name
        if not p_path.exists():
            continue
        info = load_patch(str(p_path))
        m_int8 = pipe_int8.process_frame(info['raw_3band'])['binary_mask']
        m_fp32 = pipe_fp32.process_frame(info['raw_3band'])['binary_mask']
        agreement_pct = (m_int8 == m_fp32).mean() * 100.0
        assert agreement_pct >= 99.0, f"Agreement on {p_name} was {agreement_pct:.2f}% < 99.0%"


def test_golden_reproducibility():
    """Each bundled sample reproduces the stored status, event count and packet hex."""
    golden_master_p = DEMO_DIR / "sample_data" / "golden.json"
    if not INT8_MODEL.exists():
        pytest.skip("INT8 model not found")
    assert golden_master_p.exists(), "sample_data/golden.json must exist (run generate_golden.py)"
    with open(golden_master_p, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    pipe = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=False)
    for fname, golden in golden_data.items():
        p_path = PATCH_DIR / fname
        assert p_path.exists(), f"Sample {fname} missing"
        assert (PATCH_DIR / f"{Path(fname).stem}.golden.json").exists(), "Per-sample golden file missing"
        info = load_patch(str(p_path))
        res = pipe.process_frame(info['raw_3band'], frame_id=0, geo_transform=info['transform'],
                                 crs=info['crs'], gsd_m=info['gsd_m'], timestamp_ms=0)
        assert res['status'] == golden['status'], f"Status mismatch on {fname}"
        assert len(res['events']) == golden['n_events_detected'], f"Events count mismatch on {fname}"
        assert len(res['packet']) == golden['packet_bytes'], f"Packet bytes mismatch on {fname}"
        if golden['packet_hex']:
            assert res['packet'].hex().upper() == golden['packet_hex'], f"Packet hex mismatch on {fname}"


def test_real_patch_pipeline():
    """End-to-end run on a real sample patch."""
    if not INT8_MODEL.exists():
        pytest.skip("Model ONNX file not found")
    patch_p = PATCH_DIR / "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif"
    if not patch_p.exists():
        patch_p = DEMO_DIR.parent / "data" / "raw" / "activefire" / "patches" / patch_p.name
    if not patch_p.exists():
        pytest.skip("Sample patch not found")
    pipe = EndToEndPipeline(str(INT8_MODEL), num_threads=1, heartbeat_enabled=False)
    info = load_patch(str(patch_p))
    res = pipe.process_frame(info['raw_3band'], geo_transform=info['transform'], crs=info['crs'], gsd_m=info['gsd_m'])
    assert res['status'] == 'ALERT' and len(res['events']) > 0
    assert res['reduction_ratio'] > 2000.0
    assert res['timings']['total_ms'] < 300.0


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = skipped = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except pytest.skip.Exception as exc:
            print(f"  [SKIP] {name}: {exc}")
            skipped += 1
        except Exception as exc:                     # noqa: BLE001
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return failed


if __name__ == "__main__":
    print("Running the demo verification tests...")
    sys.exit(1 if _run_all() else 0)