"""Stage 1 Validation Quality Gate Pipeline: Calibration & Batch Validation.

Implements the quality gate in response to J. McDonald's feedback (§6.3):
1. Calibrates physical thresholds from clean non-fire frames.
2. Validates patches, categorizing each into PASS, PASS_WITH_FLAGS, or REJECT.
3. Evaluates False Rejection Rate on Fire Patches (protecting detection recall).
4. Measures cross-channel confirmation of fire anomalies across B10 & B11.

Supports two operational modes:
  --mode ground  : Gate 1 Mode A — Dual-band B10/B11 cross-confirmation (GroundGateValidator)
  --mode flight  : Gate 1 Mode B — Single-channel LWIR PSF spatial halo (FlightGateValidator)

Supports regional filtering:
  --region all         : Validate all 217 scenes
  --region africa      : Validate only African scenes (North Africa + Sub-Saharan)
  --region north_africa: Validate only North Africa & Mediterranean scenes
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
    FlightGateValidator,
    FrameValidator,
    GroundGateValidator,
    Status,
    ValidatorConfig,
    calibrate_from_frames,
)

REGION_FILTERS = {
    "all": None,
    "africa": ["NORTH_AFRICA_MED", "SUB_SAHARAN_AFRICA"],
    "north_africa": ["NORTH_AFRICA_MED"],
    "sub_saharan": ["SUB_SAHARAN_AFRICA"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate and execute Validation Quality Gate (Ground or Flight mode)"
    )
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of ActiveFire")
    parser.add_argument("--calib-count", type=int, default=100, help="Clean frames for calibration")
    parser.add_argument("--val-count", type=int, default=1000, help="Total frames to validate (0 = all)")
    parser.add_argument(
        "--mode",
        choices=["ground", "flight"],
        default="ground",
        help="Validation mode: 'ground' (dual-band B10/B11) or 'flight' (single-band LWIR PSF halo)",
    )
    parser.add_argument(
        "--region",
        choices=list(REGION_FILTERS.keys()),
        default="all",
        help="Regional filter: 'all', 'africa', 'north_africa', or 'sub_saharan'",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg["data"]["activefire_root"])
    interim_dir = resolve_path(cfg["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "Ground (B10+B11 Cross-Confirm)" if args.mode == "ground" else "Flight (Single-Channel PSF Halo)"
    print(f"=== Stage 1: Validation Quality Gate Pipeline ===")
    print(f"Mode:   {mode_label}")
    print(f"Region: {args.region}")

    # Index all patches with geocoding
    df_index = index_patches(data_root)
    if df_index.empty:
        print(f"[!] No ActiveFire patches found at {data_root}.")
        print("Run scripts/00_get_activefire.py or run synthetic tests via pytest tests/.")
        return

    # Apply regional filter
    region_vals = REGION_FILTERS[args.region]
    if region_vals is not None:
        df_filtered = df_index[df_index["region"].isin(region_vals)]
        print(f"Filtered to {len(df_filtered)} patches in region(s): {region_vals}")
    else:
        df_filtered = df_index
        print(f"Using all {len(df_filtered)} patches")

    if df_filtered.empty:
        print("[!] No patches match the region filter.")
        return

    # Show region breakdown
    print("\nRegion distribution:")
    for region, count in df_filtered["region"].value_counts().items():
        print(f"  {region}: {count}")

    # Split into clean (no-fire) patches and fire patches
    clean_patches = df_filtered[~df_filtered["has_mask"]]
    fire_patches = df_filtered[df_filtered["has_mask"]]

    print(f"\nAvailable clean (no fire) patches: {len(clean_patches)}")
    print(f"Available active-fire patches: {len(fire_patches)}")

    # 1. Calibration Phase
    print("\n--- Phase 1: Calibrating Quality Gate Thresholds ---")
    # For calibration, use the filtered pool (fire pixels < 0.02% per patch, negligible)
    calib_pool = clean_patches if len(clean_patches) >= args.calib_count else df_filtered
    calib_sample = calib_pool.sample(n=min(args.calib_count, len(calib_pool)), random_state=42)
    print(f"Calibration pool: {len(calib_pool)} frames (fire contamination negligible at <0.02%)")

    # Thermal channel indices
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

    # Save calibrated config
    suffix = f"_{args.mode}_{args.region}"
    calib_save_path = interim_dir / f"calibrated_validator_config{suffix}.yaml"
    calib_dict = {
        "mode": args.mode,
        "region": args.region,
        "channel_0_B10": {k: v for k, v in cfg_b10.__dict__.items() if not k.startswith("_")},
        "channel_1_B11": {k: v for k, v in cfg_b11.__dict__.items() if not k.startswith("_")},
    }
    with open(calib_save_path, "w") as f:
        yaml.safe_dump(calib_dict, f, default_flow_style=False)
    print(f"[Saved] Calibrated config: {calib_save_path}")

    # 2. Validation Run
    print(f"\n--- Phase 2: Batch Validation & Recall Protection Audit ({args.mode.upper()} MODE) ---")

    # Create validator based on mode
    if args.mode == "ground":
        validator = GroundGateValidator([cfg_b10, cfg_b11])
    else:
        # Flight mode: single-channel LWIR (B10 only — the primary thermal band)
        validator = FlightGateValidator(cfg_b10)

    # Sample or use all
    if args.val_count > 0 and args.val_count < len(df_filtered):
        val_sample = df_filtered.sample(n=args.val_count, random_state=42)
    else:
        val_sample = df_filtered

    val_records = []
    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample), desc=f"Validating ({args.mode})"):
        if args.mode == "ground":
            img = read_image(row["image_path"], channels=thermal_indices)
        else:
            # Flight mode uses only B10 (single channel)
            img = read_image(row["image_path"], channels=[thermal_indices[0]])

        report = validator.validate(img)

        record = {
            "stem": row["stem"],
            "has_fire": row["has_mask"],
            "region": row.get("region", "UNKNOWN"),
            "center_lat": row.get("center_lat"),
            "center_lon": row.get("center_lon"),
            "status": report.status.value,
            "reasons": ";".join(report.reasons),
            "mode": args.mode,
        }

        if args.mode == "ground":
            record.update({
                "ch0_hot_pixels": report.metrics.get("ch0.hot_pixels", 0),
                "ch1_hot_pixels": report.metrics.get("ch1.hot_pixels", 0),
                "ch0_hot_confirmed_real": report.metrics.get("ch0.hot_confirmed_real", 0),
                "ch1_hot_confirmed_real": report.metrics.get("ch1.hot_confirmed_real", 0),
                "ch0_noise": report.metrics.get("ch0.noise", 0.0),
                "ch1_noise": report.metrics.get("ch1.noise", 0.0),
            })
        else:
            record.update({
                "ch0_hot_candidates": report.metrics.get("ch0.hot_candidates", 0),
                "ch0_seu_transients": report.metrics.get("ch0.seu_transients", 0),
                "ch0_hot_confirmed_optical": report.metrics.get("ch0.hot_confirmed_optical", 0),
                "ch0_noise": report.metrics.get("ch0.noise", 0.0),
            })

        val_records.append(record)

    df_results = pd.DataFrame(val_records)
    results_path = interim_dir / f"validation_results{suffix}.csv"
    df_results.to_csv(results_path, index=False)
    print(f"[Saved] Validation audit results: {results_path}")

    # 3. Recall-Protection Metric Analysis
    total_val = len(df_results)
    pass_cnt = (df_results["status"] == Status.PASS.value).sum()
    flag_cnt = (df_results["status"] == Status.PASS_WITH_FLAGS.value).sum()
    reject_cnt = (df_results["status"] == Status.REJECT.value).sum()

    print(f"\nOverall Status Distribution ({args.mode.upper()} mode, region={args.region}):")
    print(f"  PASS:            {pass_cnt} ({pass_cnt / total_val * 100:.2f}%)")
    print(f"  PASS_WITH_FLAGS: {flag_cnt} ({flag_cnt / total_val * 100:.2f}%)")
    print(f"  REJECT:          {reject_cnt} ({reject_cnt / total_val * 100:.2f}%)")

    # Regional breakdown
    if len(df_results["region"].unique()) > 1:
        print("\nPer-Region Breakdown:")
        for region in sorted(df_results["region"].unique()):
            region_df = df_results[df_results["region"] == region]
            n_region = len(region_df)
            rp = (region_df["status"] == Status.PASS.value).sum()
            rf = (region_df["status"] == Status.PASS_WITH_FLAGS.value).sum()
            rr = (region_df["status"] == Status.REJECT.value).sum()
            print(f"  {region} (n={n_region}):  PASS={rp}  FLAG={rf}  REJECT={rr}")

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

    # Mode-specific metrics
    if args.mode == "flight":
        total_seu = df_results["ch0_seu_transients"].sum()
        total_optical = df_results["ch0_hot_confirmed_optical"].sum()
        print(f"\n[PSF HALO DISCRIMINATION]")
        print(f"  Total SEU/Cosmic Ray Transients Detected: {total_seu}")
        print(f"  Total Optically-Confirmed Hotspots:       {total_optical}")
    else:
        total_confirmed_b10 = df_results["ch0_hot_confirmed_real"].sum()
        total_confirmed_b11 = df_results["ch1_hot_confirmed_real"].sum()
        print(f"\n[DUAL-BAND CROSS-CONFIRMATION]")
        print(f"  B10 Anomalies Confirmed by B11: {total_confirmed_b10}")
        print(f"  B11 Anomalies Confirmed by B10: {total_confirmed_b11}")

    # 4. Generate Figures
    # Status Breakdown Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overall status bar chart
    ax1 = axes[0]
    counts = [pass_cnt, flag_cnt, reject_cnt]
    labels = ["PASS", "FLAGS", "REJECT"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax1.bar(labels, counts, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_title(f"Quality Gate Verdicts — {args.mode.upper()} Mode\n(Region: {args.region}, N={total_val})")
    ax1.set_ylabel("Number of Frames")
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, f"{v}\n({v / total_val * 100:.1f}%)", ha="center", fontweight="bold", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Right: Per-region stacked bar
    ax2 = axes[1]
    regions = sorted(df_results["region"].unique())
    pass_by_region = []
    flag_by_region = []
    reject_by_region = []
    for r in regions:
        rdf = df_results[df_results["region"] == r]
        pass_by_region.append((rdf["status"] == Status.PASS.value).sum())
        flag_by_region.append((rdf["status"] == Status.PASS_WITH_FLAGS.value).sum())
        reject_by_region.append((rdf["status"] == Status.REJECT.value).sum())

    x = np.arange(len(regions))
    w = 0.5
    ax2.bar(x, pass_by_region, w, label="PASS", color="#2ca02c", alpha=0.8)
    ax2.bar(x, flag_by_region, w, bottom=pass_by_region, label="FLAGS", color="#ff7f0e", alpha=0.8)
    bottom2 = [p + f for p, f in zip(pass_by_region, flag_by_region)]
    ax2.bar(x, reject_by_region, w, bottom=bottom2, label="REJECT", color="#d62728", alpha=0.8)
    ax2.set_xticks(x)
    short_labels = [r.replace("_", "\n") for r in regions]
    ax2.set_xticklabels(short_labels, fontsize=8)
    ax2.set_title(f"Per-Region Breakdown — {args.mode.upper()} Mode")
    ax2.set_ylabel("Number of Frames")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    status_fig = figs_dir / f"validation_status_{args.mode}_{args.region}.png"
    plt.tight_layout()
    plt.savefig(status_fig, dpi=200)
    plt.close()
    print(f"\n[Saved] Figure: {status_fig}")

    # If flight mode, generate PSF halo discrimination figure
    if args.mode == "flight" and (total_seu > 0 or total_optical > 0):
        fig2, ax3 = plt.subplots(figsize=(7, 5))
        disc_labels = ["SEU / Cosmic Ray\n(Rejected)", "Optical Hotspot\n(Confirmed)"]
        disc_vals = [total_seu, total_optical]
        disc_colors = ["#d62728", "#2ca02c"]
        ax3.bar(disc_labels, disc_vals, color=disc_colors, alpha=0.85, edgecolor="black")
        ax3.set_title(f"PSF Halo Discrimination — Flight Mode\n(Region: {args.region}, N={total_val})")
        ax3.set_ylabel("Total Detections")
        for i, v in enumerate(disc_vals):
            ax3.text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold", fontsize=11)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        psf_fig = figs_dir / f"psf_halo_discrimination_{args.region}.png"
        plt.tight_layout()
        plt.savefig(psf_fig, dpi=200)
        plt.close()
        print(f"[Saved] Figure: {psf_fig}")

    print(f"\n[Done] Validation pipeline complete ({args.mode.upper()} mode, region={args.region}).")


if __name__ == "__main__":
    main()
