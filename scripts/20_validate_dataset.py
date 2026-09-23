"""Stage 1 Validation Quality Gate Pipeline: Calibration & Batch Validation.

Implements the quality gate in response to J. McDonald's feedback (§6.3):
1. Calibrates physical thresholds from clean non-fire frames.
2. Validates patches, categorizing each into PASS, PASS_WITH_FLAGS, or REJECT.
3. Evaluates False Rejection Rate on Fire Patches (protecting detection recall).
4. Measures cross-channel confirmation of fire anomalies across B10 & B11.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import _bootstrap
from fireedge.config import get_reports_dir, load_config, resolve_path
from fireedge.io import index_patches, read_image, read_mask
from fireedge.validation import (
    FrameValidator,
    Status,
    ValidatorConfig,
    calibrate_from_frames,
)


def main():
    parser = argparse.ArgumentParser(description="Calibrate and execute Validation Quality Gate")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of ActiveFire")
    parser.add_argument("--calib-count", type=int, default=100, help="Clean frames for calibration")
    parser.add_argument("--val-count", type=int, default=1000, help="Total frames to validate")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg["data"]["activefire_root"])
    interim_dir = resolve_path(cfg["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: Validation Quality Gate Pipeline ===")
    df_index = index_patches(data_root)
    if df_index.empty:
        print(f"[!] No ActiveFire patches found at {data_root}.")
        print("Run scripts/00_get_activefire.py or run synthetic tests via pytest tests/.")
        return

    # Split into clean (no-fire) patches and fire patches
    clean_patches = df_index[~df_index["has_mask"]]
    fire_patches = df_index[df_index["has_mask"]]

    print(f"Available clean (no fire) patches: {len(clean_patches)}")
    print(f"Available active-fire patches: {len(fire_patches)}")

    # 1. Calibration Phase (on clean frames)
    print("\n--- Phase 1: Calibrating Quality Gate Thresholds ---")
    calib_sample = clean_patches.sample(n=min(args.calib_count, len(clean_patches)), random_state=42)
    
    # We calibrate specifically for thermal channels: ch0=B10 (idx 8), ch1=B11 (idx 9)
    thermal_indices = cfg["activefire"]["thermal_channels"]  # [8, 9]
    b10_frames = []
    b11_frames = []

    for _, row in calib_sample.iterrows():
        img = read_image(row["image_path"], channels=thermal_indices)
        b10_frames.append(img[0])
        b11_frames.append(img[1])

    base_cfg = ValidatorConfig(bit_depth=cfg["validation"]["bit_depth"])
    cfg_b10 = calibrate_from_frames(b10_frames, base=base_cfg)
    cfg_b11 = calibrate_from_frames(b11_frames, base=base_cfg)

    print(f"B10 Calibrated Range: [{cfg_b10.valid_min:.1f}, {cfg_b10.valid_max:.1f}] DN | Max Grad: {cfg_b10.max_gradient:.1f}")
    print(f"B11 Calibrated Range: [{cfg_b11.valid_min:.1f}, {cfg_b11.valid_max:.1f}] DN | Max Grad: {cfg_b11.max_gradient:.1f}")

    calib_save_path = interim_dir / "calibrated_validator_config.yaml"
    calib_dict = {
        "channel_0_B10": {k: v for k, v in cfg_b10.__dict__.items() if not k.startswith("_")},
        "channel_1_B11": {k: v for k, v in cfg_b11.__dict__.items() if not k.startswith("_")},
    }
    with open(calib_save_path, "w") as f:
        yaml.safe_dump(calib_dict, f, default_flow_style=False)
    print(f"[Saved] Calibrated config: {calib_save_path}")

    # 2. Validation Run
    print("\n--- Phase 2: Batch Validation & Recall Protection Audit ---")
    validator = FrameValidator([cfg_b10, cfg_b11])

    # Sample balanced set of fire and non-fire patches for validation
    val_sample = df_index.sample(n=min(args.val_count, len(df_index)), random_state=42)
    val_records = []

    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample)):
        img = read_image(row["image_path"], channels=thermal_indices)
        report = validator.validate(img)

        val_records.append(
            {
                "stem": row["stem"],
                "has_fire": row["has_mask"],
                "status": report.status.value,
                "reasons": ";".join(report.reasons),
                "ch0_hot_pixels": report.metrics.get("ch0.hot_pixels", 0),
                "ch1_hot_pixels": report.metrics.get("ch1.hot_pixels", 0),
                "ch0_hot_confirmed_real": report.metrics.get("ch0.hot_confirmed_real", 0),
                "ch1_hot_confirmed_real": report.metrics.get("ch1.hot_confirmed_real", 0),
                "ch0_noise": report.metrics.get("ch0.noise", 0.0),
                "ch1_noise": report.metrics.get("ch1.noise", 0.0),
            }
        )

    df_results = pd.DataFrame(val_records)
    results_path = interim_dir / "validation_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"[Saved] Validation audit results: {results_path}")

    # 3. Recall-Protection Metric Analysis
    total_val = len(df_results)
    pass_cnt = (df_results["status"] == Status.PASS.value).sum()
    flag_cnt = (df_results["status"] == Status.PASS_WITH_FLAGS.value).sum()
    reject_cnt = (df_results["status"] == Status.REJECT.value).sum()

    print("\nOverall Status Distribution:")
    print(f"  PASS:            {pass_cnt} ({pass_cnt / total_val * 100:.2f}%)")
    print(f"  PASS_WITH_FLAGS: {flag_cnt} ({flag_cnt / total_val * 100:.2f}%)")
    print(f"  REJECT:          {reject_cnt} ({reject_cnt / total_val * 100:.2f}%)")

    # Critical Reviewer Metric: False Reject on Fire
    val_fire = df_results[df_results["has_fire"]]
    if len(val_fire) > 0:
        fire_rejects = (val_fire["status"] == Status.REJECT.value).sum()
        false_reject_rate = (fire_rejects / len(val_fire)) * 100.0
        print(f"\n[CRITICAL QUALITY GATE METRIC]")
        print(f"False Reject Rate on Fire Patches: {false_reject_rate:.3f}% ({fire_rejects}/{len(val_fire)})")
        if false_reject_rate < 1.0:
            print("  -> Quality Gate PASSED industry recall preservation standard (< 1.0% rejected)")
        else:
            print("  -> WARNING: Rejection rate exceeds 1%. Check rejection reasons and loosen bounds.")

    # 4. Generate Figures
    # Status Breakdown Plot
    plt.figure(figsize=(7, 5))
    counts = [pass_cnt, flag_cnt, reject_cnt]
    labels = ["PASS", "PASS_WITH_FLAGS", "REJECT"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    plt.bar(labels, counts, color=colors, alpha=0.8, edgecolor="black")
    plt.title("Quality Gate Verdict Distribution")
    plt.ylabel("Number of Frames")
    for i, v in enumerate(counts):
        plt.text(i, v + 5, f"{v} ({v / total_val * 100:.1f}%)", ha="center", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    status_fig = figs_dir / "validation_status_breakdown.png"
    plt.tight_layout()
    plt.savefig(status_fig, dpi=200)
    plt.close()
    print(f"[Saved] Figure: {status_fig}")

    print("\n[Done] Validation pipeline complete.")


if __name__ == "__main__":
    main()
