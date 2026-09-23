"""I/O loader for Pereira et al. ActiveFire Landsat-8 dataset."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio

# Suffixes used by the various ground truth / baseline algorithms in the dataset
ALGO_SUFFIXES = ("_voting", "_intersection", "_kumar", "_murphy", "_schroeder")

# Regex to parse Landsat-8 scene identifier from patch name
# e.g.: LC08_L1TP_227068_20200812_20200821_01_T1_patch_001
SCENE_REGEX = re.compile(
    r"(?P<scene>LC08_[A-Z0-9]+_(?P<path>\d{3})(?P<row>\d{3})_(?P<date>\d{8})_\w+)"
)


def mask_key(stem: str) -> str:
    """Normalize patch stem by stripping algorithm and version tags.
    
    Example:
        'LC08_..._Murphy_p00625' -> 'LC08_..._p00625'
        'LC08_..._patch_001_voting' -> 'LC08_..._patch_001'
    """
    s = re.sub(r"_(Murphy|Schroeder|Kumar_Roy|GOLI_v2|GOLI|voting|intersection|v1|v2)_", "_", stem, flags=re.IGNORECASE)
    s = re.sub(r"_(Murphy|Schroeder|Kumar_Roy|GOLI_v2|GOLI|voting|intersection|v1|v2)$", "", s, flags=re.IGNORECASE)
    for suffix in ALGO_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def parse_scene(stem: str) -> Dict[str, Optional[str]]:
    """Extract Landsat metadata (scene ID, path, row, date) from filename stem."""
    match = SCENE_REGEX.search(stem)
    if match:
        d = match.groupdict()
        return {
            "scene_id": d["scene"],
            "path": d["path"],
            "row": d["row"],
            "date": d["date"],
        }
    return {"scene_id": None, "path": None, "row": None, "date": None}


def classify_region(lat: Optional[float], lon: Optional[float]) -> str:
    """Categorize geographic coordinate into targeted operational regions.
    
    Regions:
      - NORTH_AFRICA_MED: Tunisia, Maghreb, and Mediterranean Basin (high summer LST, dry maquis/pine)
      - SUB_SAHARAN_AFRICA: Tropical savannas and seasonal agricultural burns
      - GLOBAL_REFERENCE: Other worldwide benchmark scenes
    """
    if lat is None or lon is None:
        return "GLOBAL_REFERENCE"
    # North Africa & Mediterranean: 27°N to 46°N, 18°W to 36°E
    if 27.0 <= lat <= 46.0 and -18.0 <= lon <= 36.0:
        return "NORTH_AFRICA_MED"
    # Sub-Saharan Africa: 35°S to 27°N, 18°W to 52°E
    if -35.0 <= lat < 27.0 and -18.0 <= lon <= 52.0:
        return "SUB_SAHARAN_AFRICA"
    return "GLOBAL_REFERENCE"


def index_patches(
    root: Union[str, Path], algorithm: str = "voting", extract_geo: bool = True
) -> pd.DataFrame:
    """Index all image patches and pair them with their corresponding masks and regions."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"ActiveFire root directory not found: {root_path}")

    # Search for image patches
    img_dirs = [
        root_path / "patches",
        root_path / "images" / "patches",
        root_path / "images",
        root_path,
    ]
    img_dir = next((d for d in img_dirs if d.is_dir() and any(d.glob("*.tif"))), None)

    if img_dir is None:
        return pd.DataFrame(
            columns=[
                "stem",
                "image_path",
                "mask_path",
                "has_mask",
                "scene_id",
                "path",
                "row",
                "date",
                "center_lat",
                "center_lon",
                "region",
            ]
        )

    # Build mask lookup table recursively under masks/
    mask_lookup: Dict[str, Path] = {}
    masks_base = root_path / "masks"
    if masks_base.is_dir():
        for m_file in masks_base.rglob("*.tif"):
            key = mask_key(m_file.stem)
            # Prioritize matching requested algorithm if present
            if key not in mask_lookup or (algorithm and algorithm.lower() in str(m_file).lower()):
                mask_lookup[key] = m_file

    records: List[Dict] = []
    for img_path in sorted(img_dir.glob("*.tif")):
        if "masks" in str(img_path):
            continue

        stem = img_path.stem
        norm_key = mask_key(stem)
        matched_mask = mask_lookup.get(norm_key)

        scene_meta = parse_scene(stem)

        center_lat, center_lon = None, None
        if extract_geo:
            try:
                with rasterio.open(str(img_path)) as src:
                    if src.crs:
                        bounds = src.bounds
                        cx = (bounds.left + bounds.right) / 2.0
                        cy = (bounds.bottom + bounds.top) / 2.0
                        crs_str = str(src.crs)
                        match = re.search(r"zone\s*(\d+)\s*([NS])?", crs_str, re.IGNORECASE)
                        if match:
                            import pyproj
                            zone = int(match.group(1))
                            is_south = bool(match.group(2) and match.group(2).upper() == "S")
                            proj = pyproj.Proj(proj="utm", zone=zone, south=is_south, ellps="WGS84")
                            lon, lat = proj(cx, cy, inverse=True)
                            center_lat = round(float(lat), 4)
                            center_lon = round(float(lon), 4)
            except Exception:
                pass

        region = classify_region(center_lat, center_lon)

        records.append(
            {
                "stem": stem,
                "image_path": str(img_path.resolve()),
                "mask_path": str(matched_mask.resolve()) if matched_mask else None,
                "has_mask": matched_mask is not None,
                "scene_id": scene_meta["scene_id"],
                "path": scene_meta["path"],
                "row": scene_meta["row"],
                "date": scene_meta["date"],
                "center_lat": center_lat,
                "center_lon": center_lon,
                "region": region,
            }
        )

    return pd.DataFrame(records)


def read_image(
    path: Union[str, Path], channels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Read a multi-band GeoTIFF image as (C, H, W) numpy array.
    
    Args:
        path: Path to GeoTIFF file.
        channels: Optional 0-indexed list/tuple of channels to extract.
                  e.g., [8, 9] for thermal bands B10 and B11.
    """
    path_str = str(path)
    with rasterio.open(path_str) as src:
        if channels is None:
            arr = src.read()  # (C, H, W)
        else:
            # rasterio indexes 1-based
            indexes = [c + 1 for c in channels]
            arr = src.read(indexes)
    return arr


def read_mask(
    path: Optional[Union[str, Path]], shape: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Read a binary fire mask as a (H, W) uint8 array.
    
    If path is None or does not exist (representing a no-fire patch),
    returns an all-zero mask of the requested shape.
    """
    if path is None or not os.path.exists(path):
        return np.zeros(shape, dtype=np.uint8)

    with rasterio.open(str(path)) as src:
        arr = src.read(1)
        # Binarize: values > 0 are fire
        mask = (arr > 0).astype(np.uint8)
    return mask


def image_meta(path: Union[str, Path]) -> Dict:
    """Return dictionary of rasterio image metadata."""
    with rasterio.open(str(path)) as src:
        return {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs) if src.crs else None,
            "bounds": [float(b) for b in src.bounds],
            "nodata": src.nodata,
        }
