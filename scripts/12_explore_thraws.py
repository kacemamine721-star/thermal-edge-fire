"""Data Exploration: ESA-PhiLab THRawS Level-0 Raw Telemetry.

Performs:
1. Inspection of Level-0 raw instrument telemetry frames.
2. Raw ADC count histogram and noise floor analysis.
3. Quantifies raw detector striping (readout non-uniformity) before ground correction.
4. Stress-tests the FrameValidator against real raw spacecraft sensor artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap
from fireedge.config import get_reports_dir, resolve_path
from fireedge.io import describe_thraws, read_raw_granule
from fireedge.validation import FrameValidator, ValidatorConfig


def main():
    parser = argparse.ArgumentParser(description="Explore THRawS Level-0 raw data")
    parser.add_argument("--data-root", type=str, default="data/raw/thraws", help="Root folder of THRawS")
    args = parser.parse_args()

    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_path(args.data_root)

    print("=== THRawS Level-0 Raw Realism Exploration ===")
    print(f"Scanning directory: {data_root.resolve()}")

    files = list(data_root.glob("*.tif")) + list(data_root.glob("*.raw")) + list(data_root.glob("*.SAFE"))
    if not files:
        print(f"[!] No THRawS files found in {data_root}. Run scripts/02_get_thraws.py first.")
        return

    print(f"Found {len(files)} THRawS records.")
    target = files[0]
    meta = describe_thraws(target)
    print(f"Inspecting file: {target.name}")
    print(f"Metadata: {meta}")

    try:
        raw_frame = read_raw_granule(target, channel=0)
    except Exception as e:
        print(f"Reading raw granule failed: {e}")
        return

    # 1. Raw ADC counts histogram
    plt.figure(figsize=(8, 4))
    flat = raw_frame[raw_frame > 0].ravel()
    plt.hist(flat, bins=50, color="indigo", alpha=0.7, edgecolor="black")
    plt.title("Level-0 Raw Instrument ADC Count Histogram")
    plt.xlabel("Raw Digital Number (DN)")
    plt.ylabel("Pixel Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    hist_fig = figs_dir / "thraws_l0_hist.png"
    plt.tight_layout()
    plt.savefig(hist_fig, dpi=200)
    plt.close()
    print(f"[Saved] Histogram: {hist_fig}")

    # 2. Striping profile along detector columns
    col_prof = raw_frame.mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(col_prof, color="darkred", lw=1.2)
    plt.title("Raw Detector Column Profile (Non-Uniformity / Striping)")
    plt.xlabel("Detector Column Index")
    plt.ylabel("Mean Column DN")
    plt.grid(True, linestyle="--", alpha=0.5)

    stripe_fig = figs_dir / "thraws_striping_profile.png"
    plt.tight_layout()
    plt.savefig(stripe_fig, dpi=200)
    plt.close()
    print(f"[Saved] Striping profile: {stripe_fig}")

    # 3. Test with FrameValidator
    print("\nRunning FrameValidator on raw Level-0 frame...")
    cfg = ValidatorConfig(bit_depth=16, noise_floor=1.0)
    validator = FrameValidator(cfg)
    rep = validator.validate(raw_frame)

    print(f"Validator Verdict on raw L0: {rep.status.value}")
    print(f"Flags/Reasons: {rep.reasons}")
    print(f"Metrics: {rep.metrics}")

    print("\n[Done] THRawS exploration complete.")


if __name__ == "__main__":
    main()
