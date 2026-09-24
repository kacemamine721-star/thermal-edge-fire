"""Pan-African DataLoader orchestration and leak-free spatial data splitting."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from fireedge.data.dataset import ActiveFireDataset
from fireedge.preprocessing.nuc import MicrobolometerDegrader


def scene_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split patches by parent acquisition scene to prevent spatial data leakage.

    Patches extracted from the same physical Landsat scene share identical atmospheric
    conditions and ground cover; splitting at the scene level guarantees true generalization.
    """
    # Group by scene_id (or stem prefix if scene_id is missing)
    scenes = df["scene_id"].fillna(df["stem"].apply(lambda s: s.rsplit("_", 1)[0])).unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(scenes)

    n_total = len(scenes)
    n_train = max(1, int(n_total * train_frac))
    n_val = max(1, int(n_total * val_frac))

    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train : n_train + n_val])
    test_scenes = set(scenes[n_train + n_val :])

    # If test is empty due to small pool, ensure at least 1 scene in test
    if len(test_scenes) == 0 and len(train_scenes) > 1:
        moved = train_scenes.pop()
        test_scenes.add(moved)

    train_df = df[df["scene_id"].isin(train_scenes) | df["stem"].apply(lambda s: s.rsplit("_", 1)[0]).isin(train_scenes)].copy()
    val_df = df[df["scene_id"].isin(val_scenes) | df["stem"].apply(lambda s: s.rsplit("_", 1)[0]).isin(val_scenes)].copy()
    test_df = df[df["scene_id"].isin(test_scenes) | df["stem"].apply(lambda s: s.rsplit("_", 1)[0]).isin(test_scenes)].copy()

    return train_df, val_df, test_df


class PanAfricanBatchSampler(Sampler[List[int]]):
    """Stratified batch sampler balancing active savanna fires with Tunisian hard negatives.

    In each batch of size B:
    - Guarantees at least 50% active fire patches (Sub-Saharan / African fires)
    - Injects hard-negative bare soil patches (North Africa / Mediterranean)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 8,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_seed)

        # Index sets
        self.fire_indices = np.where(df["has_mask"].values)[0]
        self.clean_indices = np.where(~df["has_mask"].values)[0]

        # If clean indices empty (all patches in subset have fires), fallback to all
        if len(self.clean_indices) == 0:
            self.clean_indices = np.arange(len(df))
        if len(self.fire_indices) == 0:
            self.fire_indices = np.arange(len(df))

        self.num_batches = len(df) // batch_size
        if self.num_batches == 0:
            self.num_batches = 1

    def __iter__(self):
        f_idx = self.fire_indices.copy()
        c_idx = self.clean_indices.copy()

        if self.shuffle:
            self.rng.shuffle(f_idx)
            self.rng.shuffle(c_idx)

        f_pos = 0
        c_pos = 0
        half_b = max(1, self.batch_size // 2)

        for _ in range(self.num_batches):
            batch = []
            # Sample fires
            for _ in range(half_b):
                batch.append(int(f_idx[f_pos % len(f_idx)]))
                f_pos += 1
            # Sample negatives / clean ground
            for _ in range(self.batch_size - half_b):
                batch.append(int(c_idx[c_pos % len(c_idx)]))
                c_pos += 1

            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def get_pan_african_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 8,
    channels: Tuple[int, ...] = (8,),
    simulate_nuc: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct leak-free Train, Validation, and Test DataLoaders for Person 2."""
    train_df, val_df, test_df = scene_level_split(df)

    degrader = MicrobolometerDegrader() if simulate_nuc else None

    train_ds = ActiveFireDataset(
        train_df,
        channels=channels,
        calibrate_temperature=True,
        degrader=degrader,
        augment=True,
    )
    val_ds = ActiveFireDataset(
        val_df,
        channels=channels,
        calibrate_temperature=True,
        degrader=None,
        augment=False,
    )
    test_ds = ActiveFireDataset(
        test_df,
        channels=channels,
        calibrate_temperature=True,
        degrader=None,
        augment=False,
    )

    sampler = PanAfricanBatchSampler(train_df, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
