"""Unit tests for ESA-PhiLab THRawS / PyRawS I/O loader."""
from pathlib import Path
import numpy as np
import pytest

from fireedge.io.thraws import describe_thraws, read_raw_granule


def test_describe_and_read_thraws(synthetic_geotiff_patch):
    meta = describe_thraws(synthetic_geotiff_patch)
    assert meta["format"] == "GeoTIFF"
    assert meta["count"] == 10

    raw_band = read_raw_granule(synthetic_geotiff_patch, channel=0)
    assert raw_band.shape == (256, 256)
    assert raw_band.dtype == np.uint16
