"""Shared test fixtures for FireEdge unit tests."""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def clean_frame_2d():
    """Generate a clean, smooth 2D gradient frame (256x256, uint16)."""
    x = np.linspace(25000, 32000, 256, dtype=np.float32)
    y = np.linspace(25000, 32000, 256, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grad = (xx + yy) / 2.0
    # Add tiny sensor noise
    np.random.seed(42)
    noise = np.random.normal(0, 15, (256, 256))
    return np.clip(grad + noise, 0, 65535).astype(np.uint16)


@pytest.fixture
def clean_multiband_frame(clean_frame_2d):
    """Generate clean 2-channel thermal frame (B10, B11)."""
    ch0 = clean_frame_2d
    ch1 = (clean_frame_2d * 0.98).astype(np.uint16)
    return np.stack([ch0, ch1], axis=0)


@pytest.fixture
def synthetic_geotiff_patch(tmp_path, clean_frame_2d):
    """Create a temporary 10-band 256x256 GeoTIFF patch on disk."""
    tif_path = tmp_path / "LC08_L1TP_227068_20200812_20200821_01_T1_patch_001.tif"
    transform = from_origin(500000, 4000000, 30, 30)

    data = np.stack([clean_frame_2d] * 10, axis=0)
    with rasterio.open(
        str(tif_path),
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=10,
        dtype=rasterio.uint16,
        transform=transform,
    ) as dst:
        dst.write(data)

    return tif_path


@pytest.fixture
def synthetic_mask_patch(tmp_path):
    """Create a paired binary mask GeoTIFF with a small 3x3 fire hotspot."""
    mask_path = tmp_path / "LC08_L1TP_227068_20200812_20200821_01_T1_patch_001_voting.tif"
    transform = from_origin(500000, 4000000, 30, 30)

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[120:125, 120:125] = 1  # 5x5 fire cluster

    with rasterio.open(
        str(mask_path),
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype=rasterio.uint8,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    return mask_path
