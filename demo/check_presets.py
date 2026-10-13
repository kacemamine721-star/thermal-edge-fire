#!/usr/bin/env python3
"""
Diagnostic only (changes nothing): shows, for every preset, which mask files exist and how many fire pixels each holds,
and what find_ground_truth_mask() returns. Run from the demo folder:  python check_presets.py
Send the output back if a preset count does not match its label.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import rasterio
import app

print(f"MASK_DIR = {app.MASK_DIR}\n")
for name, f in app.PRESET_SAMPLES.items():
    stem = Path(f).stem
    print("=" * 100)
    print(name)
    print(f"  patch file : {f}")
    gt = app.find_ground_truth_mask(f)
    print(f"  find_ground_truth_mask -> {'None' if gt is None else int(gt.sum())} fire pixels")

    direct = [p for d in (app.MASK_DIR, Path("sample_data/masks")) for p in d.glob(f"{stem}*.tif")] if app.MASK_DIR.exists() else []
    for p in sorted({q.resolve() for q in direct}):
        with rasterio.open(p) as ds:
            print(f"  direct mask : {p.name:90s} {int((ds.read(1) > 0).sum())} px")

    parts = stem.rsplit("_p", 1)
    if len(parts) == 2 and app.MASK_DIR.exists():
        scene, pid = parts[0], "p" + parts[1]
        arrs = []
        for p in sorted(app.MASK_DIR.glob(f"{scene}*{pid}*.tif")):
            if p.stem == stem:
                continue
            with rasterio.open(p) as ds:
                a = (ds.read(1) > 0).astype(np.uint8)
            arrs.append(a)
            print(f"  algo mask   : {p.name:90s} {int(a.sum())} px")
        if arrs:
            s = np.sum(arrs, axis=0)
            print(f"  -> {len(arrs)} mask file(s): union {int((s >= 1).sum())} px | >=2 files agree {int((s >= 2).sum())} px | all agree {int((s == len(arrs)).sum())} px")
        else:
            print("  (no algorithm mask files found for this patch)")
