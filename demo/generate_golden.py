#!/usr/bin/env python3
"""
Generate the golden regression files for the demo pipeline
==========================================================
Runs the bundled sample patches through the INT8 pipeline (1 thread, scrub off, heartbeat off, frame_id 0,
timestamp 0) and stores status, event count, packet bytes and packet hex.

This is a self-consistency regression for the DEMO pipeline (the same code must keep producing the same packets).
It is not a comparison against the notebook. Re-run it, and commit the new files, whenever the packet layout, the event
extraction or the model files change.
"""
import json
import sys
import zlib
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from pipeline import EndToEndPipeline, THRESHOLD_DEFAULT, TEMPERATURE_DEFAULT  # noqa: E402
from io_tiff import load_patch  # noqa: E402

PATCHES_DIR = DEMO_DIR / "sample_data" / "patches"
MODEL_PATH = DEMO_DIR / "models" / "unet_pereira3c_int8.onnx"

samples = [
    "LC08_L1GT_226074_20200921_20200921_01_RT_p00811.tif",
    "LC08_L1GT_226074_20200921_20200921_01_RT_p00812.tif",
    "LC08_L1GT_117060_20200825_20200825_01_RT_p00625.tif",
    "LC08_L1TP_001067_20200812_20200812_01_RT_p00000_clean_forest.tif",
    "LC08_L1TP_001067_20200812_20200812_01_RT_p00441.tif",
]

# heartbeat_enabled=False: the result of a sample must not depend on its position in the list
# (a heartbeat is only sent for the first frame of a pipeline instance and every 30th frame).
pipe = EndToEndPipeline(str(MODEL_PATH), num_threads=1, threshold=THRESHOLD_DEFAULT,
                        temperature=TEMPERATURE_DEFAULT, heartbeat_enabled=False)

golden_master = {}

for s in samples:
    p = PATCHES_DIR / s
    if not p.exists():
        print(f"Skipping missing sample: {s}")
        continue

    info = load_patch(str(p))
    res = pipe.process_frame(
        info["raw_3band"], frame_id=0,
        geo_transform=info["transform"], crs=info["crs"], gsd_m=info["gsd_m"], timestamp_ms=0)

    raw_bytes = info["raw_3band"].tobytes()
    zlib_bytes = len(zlib.compress(raw_bytes, level=6))
    pkt_len = len(res["packet"])

    record = {
        "filename": s,
        "model": "unet_pereira3c_int8.onnx",
        "threshold": THRESHOLD_DEFAULT,
        "temperature": TEMPERATURE_DEFAULT,
        "heartbeat_enabled": False,
        "status": res["status"],
        "n_events_detected": len(res["events"]),
        "n_events_transmitted": min(len(res["events"]), 6),
        "n_events_dropped": res["n_events_dropped"],
        "packet_bytes": pkt_len,
        "packet_hex": res["packet"].hex().upper() if res["packet"] else "",
        "crc32": res["packet_json"].get("crc32", "NONE") if res["packet_json"] else "NONE",
        "raw_bytes": len(raw_bytes),
        "zlib_compressed_bytes": zlib_bytes,
        "reduction_ratio": round(len(raw_bytes) / pkt_len, 2) if pkt_len > 0 else "INFINITY (SILENCE)",
    }

    per_sample_path = PATCHES_DIR / f"{p.stem}.golden.json"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {per_sample_path.name}: status={record['status']}, events={record['n_events_detected']}, "
          f"pkt_len={record['packet_bytes']}")

    golden_master[s] = record

master_path = DEMO_DIR / "sample_data" / "golden.json"
with open(master_path, "w", encoding="utf-8") as f:
    json.dump(golden_master, f, indent=2)
print(f"Wrote master golden file: {master_path}")