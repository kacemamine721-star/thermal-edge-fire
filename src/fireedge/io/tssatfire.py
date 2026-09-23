"""I/O loader for TS-SatFire VIIRS dataset (Nature Scientific Data 2025)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling

DATE_REGEX = re.compile(r"(?P<date>\d{4}[-_]?\d{2}[-_]?\d{2})")


def list_geotiffs(root: Union[str, Path]) -> List[Path]:
    """Find all GeoTIFF files under root directory recursively."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(list(root_path.rglob("*.tif")) + list(root_path.rglob("*.tiff")))


def parse_date(name: str) -> Optional[str]:
    """Extract date string from a file or folder name."""
    m = DATE_REGEX.search(name)
    if m:
        return m.group("date").replace("-", "").replace("_", "")
    return None


def describe(path: Union[str, Path]) -> Dict:
    """Extract rasterio metadata and band descriptions from a GeoTIFF."""
    with rasterio.open(str(path)) as src:
        descriptions = [src.descriptions[i] for i in range(src.count)] if src.descriptions else []
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": [str(d) for d in src.dtypes],
            "crs": str(src.crs) if src.crs else None,
            "bounds": [float(b) for b in src.bounds],
            "nodata": src.nodata,
            "descriptions": descriptions,
        }


def read_decimated(
    path: Union[str, Path], max_side: int = 512, channels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Read GeoTIFF with on-the-fly decimation to fit within max_side.
    
    Protects memory when inspecting full satellite scenes.
    """
    with rasterio.open(str(path)) as src:
        h, w = src.height, src.width
        scale = max(h / max_side, w / max_side, 1.0)
        out_h = max(int(h / scale), 1)
        out_w = max(int(w / scale), 1)

        indexes = [c + 1 for c in channels] if channels is not None else None

        arr = src.read(
            indexes=indexes,
            out_shape=(
                (len(channels) if channels else src.count),
                out_h,
                out_w,
            ),
            resampling=Resampling.bilinear,
        )
    return arr


def read_bands(
    path: Union[str, Path], channels: Sequence[int]
) -> np.ndarray:
    """Read specific 0-indexed channels from GeoTIFF."""
    with rasterio.open(str(path)) as src:
        indexes = [c + 1 for c in channels]
        return src.read(indexes)
