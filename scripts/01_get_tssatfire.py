"""Acquire and structure TS-SatFire VIIRS dataset (Nature Scientific Data 2025).

Usage:
    python scripts/01_get_tssatfire.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

import _bootstrap
from fireedge.config import load_config, resolve_path

KAGGLE_DATASET = "z789456sx/ts-satfire"
REPO_URL = "https://github.com/zhaoyutim/TS-SatFire.git"


def clone_ts_satfire_repo(dest_dir: Path) -> bool:
    """Clone official TS-SatFire repo for ROI files and baseline benchmarks."""
    ext_dir = dest_dir.parent.parent / "external" / "TS-SatFire"
    ext_dir.parent.mkdir(parents=True, exist_ok=True)
    if (ext_dir / ".git").exists():
        print(f"TS-SatFire repository already cloned at {ext_dir}")
        return True
    try:
        print(f"Cloning TS-SatFire repository from {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, str(ext_dir)], check=True)
        print("Clone successful.")
        return True
    except Exception as e:
        print(f"Git clone error: {e}")
        return False


def download_from_kaggle(dest_dir: Path) -> bool:
    """Download TS-SatFire dataset using Kaggle CLI."""
    try:
        print(f"Attempting download of {KAGGLE_DATASET} via Kaggle API...")
        res = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest_dir), "--unzip"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("Kaggle download and unzipping complete.")
            return True
        else:
            print(f"Kaggle CLI returned non-zero code:\n{res.stderr}")
            return False
    except FileNotFoundError:
        print("Kaggle CLI is not installed or not in PATH.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch and catalog TS-SatFire dataset")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory")
    args = parser.parse_args()

    cfg = load_config()
    dest_dir = Path(args.dest) if args.dest else resolve_path(cfg["data"]["tssatfire_root"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=== TS-SatFire Acquisition Pipeline ===")
    print(f"Target directory: {dest_dir.resolve()}\n")

    # 1. Clone reference repository
    clone_ts_satfire_repo(dest_dir)

    # 2. Check existing GeoTIFFs
    existing_tifs = list(dest_dir.rglob("*.tif"))
    if existing_tifs:
        print(f"Found {len(existing_tifs)} existing GeoTIFF files in {dest_dir}.")
        return

    # 3. Attempt Kaggle download
    success = download_from_kaggle(dest_dir)

    if not success:
        print("\n" + "=" * 70)
        print("KAGGLE MANUAL DOWNLOAD REQUIRED:")
        print(f"1. Download the dataset from: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"2. Extract the GeoTIFF files into: {dest_dir.resolve()}")
        print("3. Run 'python scripts/11_explore_tssatfire.py' to catalog events and band distributions.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
