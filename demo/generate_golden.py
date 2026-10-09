#!/usr/bin/env python3
"""
Generate Golden Test Files for IASTAM Edge Fire-Detection Pipeline
Per TRD §1.7 and §2.10 (reproducibility testing against V7.1 baseline)
"""
import json
import sys
import zlib
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from pipeline import EndToEndPipeline, THRESHOLD_DEFAULT, TEMPERATURE_DEFAULT, MODEL_ID
from io_tiff import load_patch

PATCHES_DIR = DEMO_DIR / "sample_data" / "patches"
MODEL_PATH = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"

samples = [
    "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif",
    "LC08_L1GT_226074_20200921_20200921_01_RT_p00812.tif",
    "LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif",
    "LC08_L1TP_001067_20200812_20200812_01_RT_p00000_clean_forest.tif",
    "LC08_L1TP_001067_20200812_20200812_01_RT_p00441.tif"
]

pipe = EndToEndPipeline(str(MODEL_PATH), num_threads=1, threshold=THRESHOLD_DEFAULT, temperature=TEMPERATURE_DEFAULT)

golden_master = {}

for s in samples:
    p = PATCHES_DIR / s
    if not p.exists():
        print(f"Skipping missing sample: {s}")
        continue
    
    info = load_patch(str(p))
    res = pipe.process_frame(
        info["raw_3band"],
        frame_id=0,
        geo_transform=info["transform"],
        crs=info["crs"],
        gsd_m=info["gsd_m"],
        timestamp_ms=0
    )
    
    raw_bytes = info["raw_3band"].tobytes()
    zlib_bytes = len(zlib.compress(raw_bytes, level=6))
    
    record = {
        "filename": s,
        "model": "unet_pereira3c_int8.onnx",
        "threshold": THRESHOLD_DEFAULT,
        "temperature": TEMPERATURE_DEFAULT,
        "status": res["status"],
        "n_events_detected": len(res["events"]),
        "n_events_transmitted": min(len(res["events"]), 6),
        "packet_bytes": len(res["packet"]),
        "packet_hex": res["packet"].hex().upper() if res["packet"] else "",
        "crc32": res["packet_json"].get("crc32", "NONE") if res["packet_json"] else "NONE",
        "raw_bytes": len(raw_bytes),
        "zlib_compressed_bytes": zlib_bytes,
        "reduction_ratio": round(len(raw_bytes) / len(res["packet"]), 2) if len(res["packet"]) > 0 else "INFINITY (SILENCE)"
    }
    
    # Per-sample golden.json
    per_sample_path = PATCHES_DIR / f"{p.stem}.golden.json"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {per_sample_path.name}: status={record['status']}, events={record['n_events_detected']}, pkt_len={record['packet_bytes']}")
    
    golden_master[s] = record

# Master golden.json in sample_data
master_path = DEMO_DIR / "sample_data" / "golden.json"
with open(master_path, "w", encoding="utf-8") as f:
    json.dump(golden_master, f, indent=2)
print(f"Wrote master golden file: {master_path}")
