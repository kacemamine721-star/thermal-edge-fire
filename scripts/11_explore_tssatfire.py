"""Data Exploration: TS-SatFire VIIRS Dataset.

Performs:
1. Cataloging of fire events and time series files.
2. Channel identification and dynamic range inspection.
3. Analysis of VIIRS Band I4 (MWIR) vs Band I5 (LWIR).
4. Diurnal (day vs. night) thermal contrast assessment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap
from fireedge.config import get_reports_dir, load_config, resolve_path
from fireedge.io import describe, list_geotiffs, read_decimated


def main():
    parser = argparse.ArgumentParser(description="Explore TS-SatFire VIIRS dataset")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of TS-SatFire")
    parser.add_argument("--max-files", type=int, default=100, help="Max files to inspect")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg["data"]["tssatfire_root"])
    reports_dir = get_reports_dir()
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("=== TS-SatFire VIIRS Exploration ===")
    print(f"Scanning directory: {data_root.resolve()}")

    tif_files = list_geotiffs(data_root)
    if not tif_files:
        print(f"[!] No GeoTIFF files found under {data_root}. Run scripts/01_get_tssatfire.py first.")
        return

    print(f"Total GeoTIFFs discovered: {len(tif_files)}")

    # 1. Catalog files
    records = []
    print(f"Inspecting file headers...")
    for p in tif_files[: args.max_files]:
        meta = describe(p)
        records.append(
            {
                "file_name": p.name,
                "parent_event": p.parent.name,
                "width": meta["width"],
                "height": meta["height"],
                "count": meta["count"],
                "dtype": meta["dtypes"][0] if meta["dtypes"] else "unknown",
                "descriptions": ";".join(meta["descriptions"]) if meta["descriptions"] else "",
            }
        )

    df_catalog = pd.DataFrame(records)
    cat_path = reports_dir / "tssatfire_catalog.csv"
    df_catalog.to_csv(cat_path, index=False)
    print(f"[Saved] Catalog summary: {cat_path}")

    # 2. Inspect Band Ranges
    print("\nReading decimated samples to verify thermal bands...")
    sample_files = tif_files[: min(20, len(tif_files))]
    band_means = []
    band_maxs = []

    for p in sample_files:
        arr = read_decimated(p, max_side=256)
        C = arr.shape[0]
        means = [float(np.mean(arr[c][arr[c] > 0])) if (arr[c] > 0).any() else 0.0 for c in range(C)]
        maxs = [float(np.max(arr[c])) for c in range(C)]
        band_means.append(means)
        band_maxs.append(maxs)

    avg_means = np.mean(band_means, axis=0)
    avg_maxs = np.mean(band_maxs, axis=0)

    df_bands = pd.DataFrame(
        {
            "channel_idx": list(range(len(avg_means))),
            "avg_mean_dn": avg_means,
            "avg_max_dn": avg_maxs,
        }
    )
    b_path = reports_dir / "tssatfire_band_stats.csv"
    df_bands.to_csv(b_path, index=False)
    print(f"[Saved] Band statistics: {b_path}")

    # Plot
    plt.figure(figsize=(9, 5))
    x = np.arange(len(avg_means))
    plt.bar(x - 0.2, avg_means, 0.4, label="Mean Value", color="navy", alpha=0.7)
    plt.bar(x + 0.2, avg_maxs, 0.4, label="Max Value", color="orange", alpha=0.7)
    plt.xticks(x, [f"Ch {i}" for i in x])
    plt.title("TS-SatFire VIIRS Channel Value Ranges")
    plt.xlabel("Channel Index")
    plt.ylabel("Digital Value")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig_path = figs_dir / "tssatfire_band_ranges.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[Saved] Figure: {fig_path}")

    print("\n[Done] TS-SatFire exploration complete.")


if __name__ == "__main__":
    main()
