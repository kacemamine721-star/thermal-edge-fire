"""Unit tests for TS-SatFire VIIRS I/O loader."""
from pathlib import Path
import numpy as np
import pytest

from fireedge.io.tssatfire import (
    describe,
    list_geotiffs,
    parse_date,
    read_decimated,
)


def test_parse_date():
    assert parse_date("fire_2019-08-14_scene.tif") == "20190814"
    assert parse_date("us_fire_2020_09_01.tif") == "20200901"
    assert parse_date("unrelated_file.tif") is None


def test_describe_and_decimated(synthetic_geotiff_patch):
    meta = describe(synthetic_geotiff_patch)
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["count"] == 10

    # Decimate to max 128
    dec = read_decimated(synthetic_geotiff_patch, max_side=128)
    assert dec.shape[0] == 10
    assert dec.shape[1] <= 128
    assert dec.shape[2] <= 128
