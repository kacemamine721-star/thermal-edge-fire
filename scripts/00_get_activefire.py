"""Download and unpack Pereira et al. ActiveFire dataset.

Usage:
    python scripts/00_get_activefire.py --release subset
    python scripts/00_get_activefire.py --release manual
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add project root src to path
import _bootstrap
from fireedge.config import get_data_dir, load_config, resolve_path

# Official verified Google Drive download IDs directly from Pereira et al. repository
GDRIVE_IDS = {
    # Small test subset: contains sample patches and masks in 256x256 format
    "subset": "1gwQdhXrxCybcO16vem09DfW5fPadAA_p",
    # 9,044 manually annotated patches
    "manual_images": "1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg",
    "manual_masks": "1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3",
    "manual_annotations": "1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf",
}


def download_with_gdown(file_id: str, dest_zip: Path) -> bool:
    """Attempt download using gdown library."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from Google Drive via gdown: {url}")
        res = gdown.download(url=url, output=str(dest_zip), quiet=False)
        return res is not None and dest_zip.exists() and dest_zip.stat().st_size > 1000
    except Exception as e:
        print(f"gdown error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Acquire Pereira et al. ActiveFire dataset")
    parser.add_argument(
        "--release",
        choices=["subset", "manual"],
        default="subset",
        help="Dataset release tier to fetch (default: subset)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory (default: data/raw/activefire)",
    )
    args = parser.parse_args()

    cfg = load_config()
    dest_dir = Path(args.dest) if args.dest else resolve_path(cfg["data"]["activefire_root"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ActiveFire Acquisition Pipeline ===")
    print(f"Target release: {args.release}")
    print(f"Target directory: {dest_dir.resolve()}\n")

    # Check if data already exists
    existing_tifs = list(dest_dir.rglob("*.tif"))
    if existing_tifs:
        print(f"Found {len(existing_tifs)} existing GeoTIFF files in {dest_dir}. Ready for indexing.")
        return

    if args.release == "subset":
        file_id = GDRIVE_IDS["subset"]
        zip_path = dest_dir / "samples.zip"
        success = download_with_gdown(file_id, zip_path)
        if success and zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            print("Extraction complete.")
            # Reorganize if extracted into nested folder
            sample_dirs = list(dest_dir.glob("**/patches"))
            print(f"Discovered extracted patch directories: {sample_dirs}")
            return
    elif args.release == "manual":
        # Download images and masks
        for key in ["manual_images", "manual_masks"]:
            file_id = GDRIVE_IDS[key]
            zip_path = dest_dir / f"{key}.zip"
            success = download_with_gdown(file_id, zip_path)
            if success and zip_path.exists():
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_dir)
                print(f"Extracted {key}.")

    # Check final state
    tifs = list(dest_dir.rglob("*.tif"))
    if tifs:
        print(f"\n[Success] Found {len(tifs)} GeoTIFF files in {dest_dir}.")
    else:
        print("\n" + "=" * 70)
        print("GOOGLE DRIVE ACCESS NOTE:")
        print("Google Drive API occasionally blocks automated downloads without OAuth.")
        print(f"Direct download link: https://drive.google.com/file/d/{GDRIVE_IDS['subset']}/view?usp=sharing")
        print(f"Place 'samples.zip' or extract into: {dest_dir.resolve()}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
