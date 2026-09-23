"""Data Exploration: Pereira et al. ActiveFire Landsat-8 Dataset.

Performs:
1. Inventory and structural integrity verification (patch sizes, channels, dtypes).
2. Per-band dynamic range, quantile, and noise analysis.
3. Cross-band correlation matrix.
4. Active-fire label density and geographic/temporal distribution.
5. Thermal LWIR (B10, B11) vs SWIR (B6, B7) separability analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import _bootstrap
from fireedge.config import get_reports_dir, load_config, resolve_path
from fireedge.exploration import (
    band_separability_report,
    collect_band_stats,
    correlation_matrix,
)
from fireedge.io import index_patches, read_image, read_mask


def main():
    parser = argparse.ArgumentParser(description="Explore Pereira ActiveFire dataset")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of ActiveFire")
    parser.add_argument("--max-samples", type=int, default=500, help="Max patches for stats sampling")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg["data"]["activefire_root"])
    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ActiveFire Exploration ===")
    print(f"Scanning directory: {data_root.resolve()}")

    df_index = index_patches(data_root)
    if df_index.empty:
        print(f"[!] No patches found in {data_root}. Run scripts/00_get_activefire.py first.")
        return

    # 1. Inventory Summary
    n_total = len(df_index)
    n_with_fire = int(df_index["has_mask"].sum())
    n_no_fire = n_total - n_with_fire
    fire_patch_pct = (n_with_fire / n_total) * 100.0

    print(f"Total patches indexed: {n_total}")
    print(f"Patches with fire mask: {n_with_fire} ({fire_patch_pct:.1f}%)")
    print(f"Patches with zero fire: {n_no_fire} ({100.0 - fire_patch_pct:.1f}%)")

    inv_path = reports_dir / "activefire_inventory.csv"
    df_index.to_csv(inv_path, index=False)
    print(f"[Saved] Inventory catalog: {inv_path}")

    # 2. Sample patches for band statistics
    sample_df = df_index.sample(n=min(args.max_samples, n_total), random_state=42)
    band_names = cfg["activefire"]["bands"]["names"]
    thermal_channels = cfg["activefire"]["thermal_channels"]
    swir_channels = cfg["activefire"]["swir_channels"]

    frames = []
    fire_pixels_by_band = {b: [] for b in ["B6", "B7", "B10", "B11"]}
    bg_pixels_by_band = {b: [] for b in ["B6", "B7", "B10", "B11"]}
    fire_pixel_fractions = []

    print(f"\nAnalyzing {len(sample_df)} sample patches...")
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        img = read_image(row["image_path"])
        frames.append(img)

        mask = read_mask(row["mask_path"], shape=img.shape[1:])
        fire_mask = mask > 0
        bg_mask = ~fire_mask

        fire_frac = float(fire_mask.mean())
        if row["has_mask"]:
            fire_pixel_fractions.append(fire_frac)

        # Extract values for separability
        # Channel 5=B6 (SWIR1), 6=B7 (SWIR2), 8=B10 (LWIR1), 9=B11 (LWIR2)
        channel_map = {"B6": 5, "B7": 6, "B10": 8, "B11": 9}
        for b_name, c_idx in channel_map.items():
            if c_idx < img.shape[0]:
                c_data = img[c_idx]
                f_pts = c_data[fire_mask & (c_data > 0)]
                b_pts = c_data[bg_mask & (c_data > 0)]
                if f_pts.size > 0:
                    fire_pixels_by_band[b_name].append(f_pts)
                if b_pts.size > 0:
                    # sample background
                    bg_pixels_by_band[b_name].append(np.random.choice(b_pts, size=min(100, b_pts.size), replace=False))

    # Compute band statistics
    print("\nComputing per-band distribution statistics...")
    df_stats = collect_band_stats(frames, band_names=band_names)
    stats_path = reports_dir / "activefire_band_stats.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f"[Saved] Band stats: {stats_path}")

    # Plot band histograms
    plt.figure(figsize=(12, 6))
    plt.bar(df_stats["band"], df_stats["mean"], yerr=df_stats["std"], capsize=5, color="teal", alpha=0.7)
    plt.title("Landsat-8 Band Mean Values (DN) with Standard Deviation")
    plt.xlabel("Spectral Band")
    plt.ylabel("Digital Number (16-bit)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    hist_fig = figs_dir / "activefire_band_means.png"
    plt.tight_layout()
    plt.savefig(hist_fig, dpi=200)
    plt.close()
    print(f"[Saved] Figure: {hist_fig}")

    # Correlation Matrix
    print("\nComputing cross-band correlation matrix...")
    corr = correlation_matrix(frames)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=band_names[: corr.shape[0]],
        yticklabels=band_names[: corr.shape[0]],
    )
    plt.title("Landsat-8 Cross-Band Pearson Correlation")
    corr_fig = figs_dir / "activefire_correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(corr_fig, dpi=200)
    plt.close()
    print(f"[Saved] Figure: {corr_fig}")

    # Thermal Separability Analysis
    print("\nEvaluating Fire vs Background Separability (LWIR vs SWIR)...")
    sep_input = {}
    for b_name in ["B10", "B11", "B6", "B7"]:
        f_arr = np.concatenate(fire_pixels_by_band[b_name]) if fire_pixels_by_band[b_name] else np.array([])
        b_arr = np.concatenate(bg_pixels_by_band[b_name]) if bg_pixels_by_band[b_name] else np.array([])
        sep_input[b_name] = (f_arr, b_arr)

    df_sep = band_separability_report(sep_input)
    sep_path = reports_dir / "activefire_separability.csv"
    df_sep.to_csv(sep_path, index=False)
    print(f"[Saved] Separability report:\n{df_sep.to_string(index=False)}")

    # Fire pixel fraction distribution plot
    if fire_pixel_fractions:
        plt.figure(figsize=(8, 5))
        plt.hist(fire_pixel_fractions, bins=40, color="crimson", edgecolor="black", alpha=0.8)
        plt.title("Distribution of Fire Pixel Fractions in Masked Patches")
        plt.xlabel("Fire Fraction per 256x256 Patch")
        plt.ylabel("Number of Patches")
        plt.yscale("log")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        label_fig = figs_dir / "activefire_label_distribution.png"
        plt.tight_layout()
        plt.savefig(label_fig, dpi=200)
        plt.close()
        print(f"[Saved] Figure: {label_fig}")

    print("\n[Done] ActiveFire exploration complete.")


if __name__ == "__main__":
    main()
