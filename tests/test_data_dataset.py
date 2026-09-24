"""Unit tests for PyTorch ActiveFireDataset and Pan-African DataLoaders."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from fireedge.config import get_reports_dir
from fireedge.data.datamodule import get_pan_african_dataloaders, scene_level_split
from fireedge.data.dataset import ActiveFireDataset


@pytest.fixture
def sample_african_df():
    """Load African inventory or build minimal fixture."""
    inv_path = get_reports_dir() / "africa_inventory.csv"
    if inv_path.exists():
        df = pd.read_csv(inv_path)
        # Use first 10 rows for fast testing
        return df.head(10).copy()

    # Synthetic fallback fixture if inventory not generated yet
    return pd.DataFrame({
        "stem": [f"LC08_L1TP_170065_20200812_p0000{i}" for i in range(6)],
        "scene_id": ["LC08_L1TP_170065_20200812"] * 3 + ["LC08_L1TP_180065_20200815"] * 3,
        "image_path": ["dummy.tif"] * 6,
        "mask_path": ["dummy_mask.tif"] * 6,
        "has_mask": [True, False, True, False, True, False],
        "region": ["SUB_SAHARAN_AFRICA"] * 3 + ["NORTH_AFRICA_MED"] * 3,
    })


def test_activefire_dataset_with_real_patches(sample_african_df):
    """Verify dataset yields PyTorch float32 tensors with expected shapes."""
    if not Path(sample_african_df.iloc[0]["image_path"]).exists():
        pytest.skip("Real raster files not found at paths; skipping live raster test.")

    ds = ActiveFireDataset(
        sample_african_df,
        channels=[8],  # Band 10 LWIR
        target_size=(256, 256),
        calibrate_temperature=True,
        augment=True,
    )
    assert len(ds) == len(sample_african_df)

    img, mask = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert img.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)
    # Normalized range check
    assert 0.0 <= img.min() <= 1.0
    assert 0.0 <= img.max() <= 1.0
    # Binary mask check
    assert set(np.unique(mask.numpy())).issubset({0.0, 1.0})


def test_pan_african_dataloader_batches(sample_african_df):
    """Verify DataLoader serves batched tensors matching Person 2 interface contract."""
    if not Path(sample_african_df.iloc[0]["image_path"]).exists():
        pytest.skip("Real raster files not found at paths; skipping live raster test.")

    train_loader, val_loader, test_loader = get_pan_african_dataloaders(
        sample_african_df,
        batch_size=2,
        channels=(8,),
        simulate_nuc=False,
    )

    batch_imgs, batch_masks = next(iter(train_loader))
    assert batch_imgs.shape == (2, 1, 256, 256)
    assert batch_masks.shape == (2, 1, 256, 256)
    assert batch_imgs.dtype == torch.float32
    assert batch_masks.dtype == torch.float32


def test_scene_level_split_no_leakage(sample_african_df):
    """Verify no scene overlap between train and test splits."""
    train_df, val_df, test_df = scene_level_split(sample_african_df, train_frac=0.6, val_frac=0.2)

    def get_scenes(d):
        return set(d["scene_id"].fillna(d["stem"].apply(lambda s: s.rsplit("_", 1)[0])).unique())

    s_train = get_scenes(train_df)
    s_test = get_scenes(test_df)

    # Intersection must be empty (strictly zero spatial leakage)
    assert len(s_train.intersection(s_test)) == 0
