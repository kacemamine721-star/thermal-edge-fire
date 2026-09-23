"""Unit tests for Pereira ActiveFire dataset I/O loader."""
from pathlib import Path
import numpy as np
import pytest

from fireedge.io.activefire import (
    image_meta,
    index_patches,
    mask_key,
    parse_scene,
    read_image,
    read_mask,
)


def test_mask_key_strips_suffixes():
    assert mask_key("scene_patch_001_voting") == "scene_patch_001"
    assert mask_key("scene_patch_001_schroeder") == "scene_patch_001"
    assert mask_key("scene_patch_001") == "scene_patch_001"


def test_parse_scene():
    stem = "LC08_L1TP_227068_20200812_20200821_01_T1_patch_001"
    meta = parse_scene(stem)
    assert meta["path"] == "227"
    assert meta["row"] == "068"
    assert meta["date"] == "20200812"


def test_read_image_and_metadata(synthetic_geotiff_patch):
    arr = read_image(synthetic_geotiff_patch)
    assert arr.shape == (10, 256, 256)
    assert arr.dtype == np.uint16

    # Test extracting specific channels (e.g. thermal 8, 9)
    thermal = read_image(synthetic_geotiff_patch, channels=[8, 9])
    assert thermal.shape == (2, 256, 256)

    meta = image_meta(synthetic_geotiff_patch)
    assert meta["width"] == 256
    assert meta["height"] == 256
    assert meta["count"] == 10


def test_read_mask_returns_zeros_when_missing():
    mask = read_mask(None, shape=(256, 256))
    assert mask.shape == (256, 256)
    assert mask.sum() == 0
    assert mask.dtype == np.uint8


def test_index_patches_pairing(tmp_path, synthetic_geotiff_patch, synthetic_mask_patch):
    # Setup expected directory tree
    patches_dir = tmp_path / "patches"
    masks_dir = tmp_path / "masks" / "voting"
    patches_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    synthetic_geotiff_patch.rename(patches_dir / synthetic_geotiff_patch.name)
    synthetic_mask_patch.rename(masks_dir / synthetic_mask_patch.name)

    df = index_patches(tmp_path, algorithm="voting")
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["has_mask"]) is True
    assert Path(row["image_path"]).exists()
    assert Path(row["mask_path"]).exists()
