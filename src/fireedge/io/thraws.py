"""I/O loader for ESA-PhiLab THRawS / PyRawS Level-0 raw Sentinel-2 data."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio

# Check if pyraws is installed
try:
    import pyraws
    HAS_PYRAWS = True
except ImportError:
    HAS_PYRAWS = False


def describe_thraws(path: Union[str, Path]) -> Dict:
    """Inspect a THRawS raw granule or GeoTIFF."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"THRawS file not found: {path_obj}")

    if path_obj.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(str(path_obj)) as src:
            return {
                "format": "GeoTIFF",
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "bounds": [float(b) for b in src.bounds],
                "crs": str(src.crs) if src.crs else None,
            }
    else:
        # Raw binary telemetry or SAFE directory
        stat = path_obj.stat()
        return {
            "format": "RawBinary/SAFE",
            "size_bytes": stat.st_size,
            "path": str(path_obj),
            "has_pyraws": HAS_PYRAWS,
        }


def read_raw_granule(
    path: Union[str, Path], channel: int = 0, shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Read uncalibrated raw Level-0 DN (digital number) frame.
    
    If PyRawS is available and path is a raw SAFE granule, uses PyRawS.
    Otherwise, if path is a GeoTIFF (e.g. exported raw DNs), reads band directly.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(str(path_obj)) as src:
            idx = min(channel + 1, src.count)
            return src.read(idx)
    
    if HAS_PYRAWS:
        # Use pyraws raw extractor if available
        try:
            raw_obj = pyraws.RawProduct(str(path_obj))
            return raw_obj.get_band_data(channel)
        except Exception:
            pass

    # Fallback: read raw 16-bit binary if shape provided
    if shape is not None and path_obj.is_file():
        expected_size = shape[0] * shape[1] * 2
        with open(path_obj, "rb") as f:
            raw_bytes = f.read(expected_size)
            return np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)

    raise ValueError(f"Unable to read raw granule from {path_obj}")
