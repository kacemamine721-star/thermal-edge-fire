"""Acquire sample authentic Level-0 raw Sentinel-2 granules from ESA-PhiLab PyRawS / THRawS.

Usage:
    python scripts/02_get_thraws.py
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap
from fireedge.config import get_project_root, resolve_path

PYRAWS_REPO = "https://github.com/ESA-PhiLab/PyRawS.git"


def clone_pyraws(dest_dir: Path) -> bool:
    """Clone ESA-PhiLab PyRawS repository into external/."""
    ext_dir = dest_dir.parent.parent / "external" / "PyRawS"
    ext_dir.parent.mkdir(parents=True, exist_ok=True)
    if (ext_dir / ".git").exists():
        print(f"PyRawS repo already cloned at {ext_dir}")
        return True
    try:
        print(f"Cloning PyRawS repository from {PYRAWS_REPO}...")
        subprocess.run(["git", "clone", PYRAWS_REPO, str(ext_dir)], check=True)
        return True
    except Exception as e:
        print(f"Failed to clone PyRawS: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Acquire sample THRawS Level-0 dataset")
    parser.add_argument("--dest", type=str, default="data/raw/thraws", help="Destination folder")
    args = parser.parse_args()

    dest_dir = resolve_path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=== ESA-PhiLab THRawS / PyRawS Acquisition ===")
    print(f"Target directory: {dest_dir.resolve()}\n")

    clone_pyraws(dest_dir)

    print("\n" + "=" * 70)
    print("THRawS Level-0 Raw Realism Database:")
    print("ESA Phi-Lab provides authentic Level-0 granules for high-temperature events.")
    print("Documentation and download links: https://github.com/ESA-PhiLab/PyRawS/blob/main/quickstart/DB.md")
    print(f"Granules downloaded should be placed into: {dest_dir.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
