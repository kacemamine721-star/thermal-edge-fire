"""Count ActiveFire training patches in North Africa / the Mediterranean.

The full imagery archive is too large to inspect locally, but its companion
metadata archive contains one Landsat MTL file per source scene.  This script
reads those coordinates, joins them to the reference ActiveFire patch index,
and writes reproducible regional inventories without extracting imagery.
"""
from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "interim" / "africa_metadata_extracted"
PATCH_INDEX = (
    ROOT
    / "data"
    / "external"
    / "activefire"
    / "src"
    / "train"
    / "voting"
    / "unet_16f_2conv_762"
    / "dataset"
    / "images_masks.csv"
)
REPORTS = ROOT / "reports"

# Tunisia/Maghreb plus the Mediterranean basin, matching the project scope.
LAT_MIN, LAT_MAX = 27.0, 46.0
LON_MIN, LON_MAX = -18.0, 36.0
PATCH_SCENE = re.compile(r"LC08_[A-Z0-9]+_(\d{3})(\d{3})_")


def mtl_number(text: str, key: str) -> float:
    match = re.search(rf"^\s*{key}\s*=\s*([-0-9.]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {key}")
    return float(match.group(1))


def read_scenes() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for archive in sorted(METADATA_DIR.glob("z*.zip")):
        with zipfile.ZipFile(archive) as zf:
            mtl_name = next(name for name in zf.namelist() if name.endswith("_MTL.txt"))
            text = zf.read(mtl_name).decode("utf-8", errors="replace")

        latitudes = [
            mtl_number(text, key)
            for key in (
                "CORNER_UL_LAT_PRODUCT", "CORNER_UR_LAT_PRODUCT",
                "CORNER_LL_LAT_PRODUCT", "CORNER_LR_LAT_PRODUCT",
            )
        ]
        longitudes = [
            mtl_number(text, key)
            for key in (
                "CORNER_UL_LON_PRODUCT", "CORNER_UR_LON_PRODUCT",
                "CORNER_LL_LON_PRODUCT", "CORNER_LR_LON_PRODUCT",
            )
        ]
        product = re.search(r'LANDSAT_PRODUCT_ID\s*=\s*"([^"]+)"', text)
        path = int(re.search(r"WRS_PATH\s*=\s*(\d+)", text).group(1))
        row = int(re.search(r"WRS_ROW\s*=\s*(\d+)", text).group(1))
        records.append({
            "wrs_path": path, "wrs_row": row,
            "scene_id": product.group(1) if product else archive.stem,
            "min_lat": min(latitudes), "max_lat": max(latitudes),
            "min_lon": min(longitudes), "max_lon": max(longitudes),
            "center_lat": sum(latitudes) / len(latitudes),
            "center_lon": sum(longitudes) / len(longitudes),
        })
    return pd.DataFrame(records)


def read_patch_index() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with PATCH_INDEX.open(newline="") as handle:
        for values in csv.reader(handle):
            if not values:
                continue
            match = PATCH_SCENE.search(values[0])
            if match:
                records.append({
                    "patch_name": values[0],
                    "mask_name": values[1] if len(values) > 1 else None,
                    "wrs_path": int(match.group(1)),
                    "wrs_row": int(match.group(2)),
                })
    return pd.DataFrame(records)


def main() -> None:
    scenes = read_scenes()
    patches = read_patch_index()
    merged = patches.merge(scenes, on=["wrs_path", "wrs_row"], how="inner")

    # Strict uses the scene centre; inclusive retains scenes whose footprint
    # intersects the target bounds, useful near the regional boundary.
    strict = merged[
        merged.center_lat.between(LAT_MIN, LAT_MAX)
        & merged.center_lon.between(LON_MIN, LON_MAX)
    ].copy()
    inclusive = merged[
        (merged.max_lat >= LAT_MIN) & (merged.min_lat <= LAT_MAX)
        & (merged.max_lon >= LON_MIN) & (merged.min_lon <= LON_MAX)
    ].copy()

    REPORTS.mkdir(exist_ok=True)
    scenes.to_csv(REPORTS / "africa_metadata_scene_inventory.csv", index=False)
    strict.to_csv(REPORTS / "north_africa_med_patch_inventory.csv", index=False)
    inclusive.to_csv(REPORTS / "north_africa_med_patch_inventory_inclusive.csv", index=False)

    def summary(label: str, frame: pd.DataFrame) -> str:
        scene_count = frame[["wrs_path", "wrs_row"]].drop_duplicates().shape[0]
        return f"{label}: {len(frame):,} patches from {scene_count} scenes"

    print(f"Metadata scenes: {len(scenes):,}")
    print(f"Reference indexed patches: {len(patches):,}")
    print(f"Patches matched to metadata: {len(merged):,}")
    print(summary("Strict centroid region", strict))
    print(summary("Footprint-intersection region", inclusive))


if __name__ == "__main__":
    main()
