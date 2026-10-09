#!/usr/bin/env python3
"""
IASTAM Edge Fire Detection — GeoTIFF I/O, Georeferencing & Scene Tiler
======================================================================
Handles:
  1. Loading 10-band Pereira patches (extracts [C7, C6, C2] at indices [6, 5, 1])
  2. Loading 3-band exports [C7, C6, C2]
  3. Extracting geographic CRS, GSD (meters), and converting pixel centroids to WGS84 Lat/Lon
  4. Scene Tiler: splits arbitrary HxW images into 256x256 tiles with zero-padding
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import rasterio
import warnings

# Suppress NotGeoreferencedWarning for non-georeferenced GeoTIFFs
# (the pipeline handles missing CRS/transform gracefully with sensible defaults)
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass


# Ensure clean PROJ environment pointing to rasterio's modern proj.db (layout minor version 6)
try:
    rasterio_proj = str(Path(rasterio.__file__).parent / 'proj_data')
    if (Path(rasterio_proj) / 'proj.db').exists():
        os.environ['PROJ_DATA'] = rasterio_proj
        os.environ['PROJ_LIB'] = rasterio_proj
        import pyproj
        pyproj.datadir.set_data_dir(rasterio_proj)
except Exception:
    pass

import functools
import pyproj


@functools.lru_cache(maxsize=32)
def get_wgs84_transformer(crs_str: str) -> Optional[pyproj.Transformer]:
    """Caches Transformer instance to avoid expensive PROJ database lookups per event."""
    try:
        return pyproj.Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
    except Exception:
        return None


def load_patch(file_path: str | Path) -> Dict[str, Any]:
    """Loads a satellite patch/scene GeoTIFF, extracting [C7, C6, C2] as uint16."""
    file_path = str(file_path)
    with rasterio.open(file_path) as src:
        count = src.count
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

        # Approximate GSD in meters
        if transform is not None:
            gsd_m = int(round(abs(transform[0])))
            if gsd_m <= 0 or gsd_m > 1000:
                gsd_m = 30
        else:
            gsd_m = 30

        if count >= 10:
            # Pereira dataset: Band 7=SWIR2, Band 6=SWIR1, Band 2=Blue
            raw_3band = src.read([7, 6, 2])
        elif count >= 3:
            raw_3band = src.read([1, 2, 3])
        elif count == 1:
            band1 = src.read(1)
            raw_3band = np.stack([band1, band1, band1], axis=0)
        else:
            raise ValueError(f"Unsupported band count: {count}")

        if raw_3band.dtype != np.uint16:
            if np.issubdtype(raw_3band.dtype, np.floating):
                raw_3band = np.clip(raw_3band, 0, 65535).astype(np.uint16)
            elif raw_3band.dtype == np.uint8:
                raw_3band = raw_3band.astype(np.uint16)
            else:
                raw_3band = np.clip(raw_3band, 0, 65535).astype(np.uint16)

        return {
            'raw_3band': raw_3band,
            'transform': transform,
            'crs': crs,
            'gsd_m': gsd_m,
            'bounds': src.bounds,
            'count': count,
            'width': w,
            'height': h,
        }


def pixel_to_wgs84(x: float, y: float, transform: Any, crs: Any) -> Tuple[float, float]:
    """Converts pixel coordinate (x, y) to geographic WGS84 (lon, lat) in degrees."""
    if transform is None:
        return 0.0, 0.0

    # Apply affine transform to get projected coordinates
    proj_x, proj_y = transform * (x, y)

    if crs is None:
        return 0.0, 0.0

    try:
        if getattr(crs, 'is_geographic', False):
            return float(proj_x), float(proj_y)

        crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)
        transformer = get_wgs84_transformer(crs_str)
        if transformer is not None:
            lon, lat = transformer.transform(proj_x, proj_y)
            return float(lon), float(lat)
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0


class SceneTiler:
    """Tiler for scenes larger than 256x256 pixels.
    Tiles image with zero-padding on edges without overlap, and tracks tile offsets.
    """
    def __init__(self, raw_3band: np.ndarray, tile_size: int = 256,
                 transform: Optional[Any] = None, crs: Optional[Any] = None):
        self.c, self.h, self.w = raw_3band.shape
        self.raw_3band = raw_3band
        self.tile_size = tile_size
        self.base_transform = transform
        self.crs = crs

        self.n_tiles_y = int(np.ceil(self.h / tile_size))
        self.n_tiles_x = int(np.ceil(self.w / tile_size))
        self.total_tiles = self.n_tiles_y * self.n_tiles_x

    def iterate_tiles(self) -> Generator[Dict[str, Any], None, None]:
        """Yields each 256x256 tile with its grid indices, coordinates, and local transform."""
        tile_id = 0
        for ty in range(self.n_tiles_y):
            for tx in range(self.n_tiles_x):
                y0 = ty * self.tile_size
                x0 = tx * self.tile_size
                y1 = min(y0 + self.tile_size, self.h)
                x1 = min(x0 + self.tile_size, self.w)

                patch = np.zeros((self.c, self.tile_size, self.tile_size), dtype=self.raw_3band.dtype)
                patch[:, : (y1 - y0), : (x1 - x0)] = self.raw_3band[:, y0:y1, x0:x1]

                # Calculate offset transform for this tile
                if self.base_transform is not None:
                    dx = x0 * self.base_transform[0]
                    dy = y0 * self.base_transform[4]
                    tile_transform = rasterio.Affine(
                        self.base_transform[0], self.base_transform[1], self.base_transform[2] + dx,
                        self.base_transform[3], self.base_transform[4], self.base_transform[5] + dy
                    )
                else:
                    tile_transform = None

                yield {
                    'tile_id': tile_id,
                    'grid_y': ty,
                    'grid_x': tx,
                    'offset_y': y0,
                    'offset_x': x0,
                    'valid_h': y1 - y0,
                    'valid_w': x1 - x0,
                    'patch': patch,
                    'tile_transform': tile_transform,
                    'is_edge': (y1 - y0 < self.tile_size or x1 - x0 < self.tile_size)
                }
                tile_id += 1
