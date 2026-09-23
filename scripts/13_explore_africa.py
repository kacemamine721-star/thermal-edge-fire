"""Dedicated African Wildfire Data Exploration.

Analyzes the 44 African patches across:
1. Sub-Saharan Savannas & Agricultural Burns (39 scenes)
2. North Africa & Mediterranean Basin (5 scenes - Tunisia/Maghreb context)

Generates:
- reports/africa_inventory.csv
- reports/africa_band_stats.csv
- reports/africa_separability.csv
- reports/figures/africa_thermal_contrast.png
"""
from __future__ import annotations

import os
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
from fireedge.exploration import band_separability_report, collect_band_stats
from fireedge.io import read_image, read_mask


def main():
    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    inv_path = reports_dir / "africa_inventory.csv"
    if not inv_path.exists():
        base_inv = reports_dir / "activefire_inventory.csv"
        df = pd.read_csv(base_inv)
        df_africa = df[df["region"].isin(["NORTH_AFRICA_MED", "SUB_SAHARAN_AFRICA"])].copy()
        df_africa.to_csv(inv_path, index=False)
    else:
        df_africa = pd.read_csv(inv_path)

    print(f"=== Africa Wildfire Data Exploration ===")
    print(f"Total African patches: {len(df_africa)}")
    print(df_africa["region"].value_counts().to_string())

    # Band indices: B6=5, B7=6, B10=8, B11=9
    channel_map = {"B6": 5, "B7": 6, "B10": 8, "B11": 9}

    fire_pixels = {b: [] for b in channel_map}
    bg_pixels = {b: [] for b in channel_map}
    med_bg_b10 = []
    subsahara_bg_b10 = []

    for _, row in tqdm(df_africa.iterrows(), total=len(df_africa), desc="Parsing African scenes"):
        img = read_image(row["image_path"])
        mask = read_mask(row["mask_path"], shape=img.shape[1:])
        fire_mask = mask > 0
        bg_mask = ~fire_mask & (img[8] > 0)

        for b_name, c_idx in channel_map.items():
            if c_idx < img.shape[0]:
                c_data = img[c_idx]
                f_pts = c_data[fire_mask & (c_data > 0)]
                b_pts = c_data[bg_mask]
                if f_pts.size > 0:
                    fire_pixels[b_name].append(f_pts)
                if b_pts.size > 0:
                    bg_pixels[b_name].append(b_pts)

        # Record B10 background for regional comparison
        b10_bg = img[8][bg_mask]
        if b10_bg.size > 0:
            if row["region"] == "NORTH_AFRICA_MED":
                med_bg_b10.extend(b10_bg[::100])  # subsample
            else:
                subsahara_bg_b10.extend(b10_bg[::100])

    # 1. Separability Analysis for Africa
    print("\nEvaluating Thermal Separability on African Fires...")
    channel_samples = {}
    for b in channel_map:
        if fire_pixels[b] and bg_pixels[b]:
            f_cat = np.concatenate(fire_pixels[b])
            b_cat = np.concatenate(bg_pixels[b])
            channel_samples[b] = (f_cat, b_cat)

    sep_df = band_separability_report(channel_samples)
    sep_path = reports_dir / "africa_separability.csv"
    sep_df.to_csv(sep_path, index=False)
    print(f"[Saved] Africa Separability:\n{sep_df.to_string()}")

    # 2. Band Statistics for Africa
    band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"]
    sample_frames = [read_image(p) for p in df_africa["image_path"][:20]]
    stats_df = collect_band_stats(sample_frames, band_names=band_names)
    stats_path = reports_dir / "africa_band_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[Saved] Africa Band Stats: {stats_path}")

    # 3. Figure: Regional Thermal Background Contrast
    plt.figure(figsize=(9, 5))
    if len(med_bg_b10) > 0 and len(subsahara_bg_b10) > 0:
        sns.kdeplot(med_bg_b10, color="crimson", label="North Africa / Mediterranean Background (High LST)", fill=True, alpha=0.3)
        sns.kdeplot(subsahara_bg_b10, color="forestgreen", label="Sub-Saharan Background (Vegetation/Savanna)", fill=True, alpha=0.3)
        plt.title("Band 10 (LWIR) Land Surface Temperature: North Africa vs. Sub-Sahara", fontsize=11, fontweight="bold")
        plt.xlabel("Digital Number (DN) ~ Brightness Temperature")
        plt.ylabel("Density")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        fig_path = figs_dir / "africa_thermal_contrast.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Regional contrast plot: {fig_path}")

    print("\n[Done] African Data Exploration successfully completed.")


if __name__ == "__main__":
    main()
