"""PyTorch Dataset implementation for ActiveFire satellite thermal imagery."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fireedge.io import read_image, read_mask
from fireedge.preprocessing.calibration import dn_to_temperature
from fireedge.preprocessing.nuc import MicrobolometerDegrader
from fireedge.preprocessing.normalization import fixed_window_norm, robust_percentile_norm


class ActiveFireDataset(Dataset):
    """PyTorch Dataset yielding calibrated thermal image tensors and fire masks.

    Delivers the Stage 2 interface contract to Person 2 (Edge CNN Model):
      images: torch.Tensor [C, H, W] (float32, normalized thermal signal)
      masks:  torch.Tensor [1, H, W] (float32, binary active fire ground truth)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        channels: Sequence[int] = (8,),          # 8 = Band 10 (LWIR 10.9 um, CubeSat flight band)
        target_size: Tuple[int, int] = (256, 256),
        calibrate_temperature: bool = True,       # Convert raw DN to Kelvin
        degrader: Optional[MicrobolometerDegrader] = None, # Optional sensor degradation (§6.4)
        norm_method: str = "robust",              # "robust" or "fixed"
        augment: bool = False,                    # Data augmentation for training
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.channels = list(channels)
        self.target_size = target_size
        self.calibrate_temperature = calibrate_temperature
        self.degrader = degrader
        self.norm_method = norm_method
        self.augment = augment
        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        # 1. Read raw multi-spectral / thermal raster
        raw_img = read_image(img_path, channels=self.channels)  # Shape [C, H, W]
        C, H, W = raw_img.shape

        # 2. Read corresponding binary mask (voting / consensus fire ground truth)
        mask = read_mask(mask_path, shape=(H, W))  # Shape [H, W], uint8 {0, 1}
        mask_binary = (mask > 0).astype(np.float32)

        # 3. Optional Microbolometer Sensor Degradation Simulation (§6.4)
        if self.degrader is not None:
            degraded_channels = []
            for c in range(C):
                degraded_channels.append(self.degrader.degrade(raw_img[c]))
            raw_img = np.stack(degraded_channels, axis=0)

        # 4. Radiometric Calibration: DN ──> Brightness Temperature in Kelvin
        processed_channels = []
        for c in range(C):
            ch_data = raw_img[c]
            if self.calibrate_temperature:
                # Primary band is Landsat B10
                temp_k = dn_to_temperature(ch_data, band="B10", unit="kelvin")
            else:
                temp_k = ch_data

            # 5. Normalization: Map to [0.0, 1.0]
            if self.norm_method == "fixed":
                norm_ch = fixed_window_norm(temp_k, vmin_k=270.0, vmax_k=420.0)
            else:
                norm_ch = robust_percentile_norm(temp_k, p_min=2.0, p_max=98.0)

            processed_channels.append(norm_ch)

        img_tensor_np = np.stack(processed_channels, axis=0).astype(np.float32)  # [C, H, W]

        # 6. Ensure standard target size (e.g. crop or pad to 256x256)
        if (H, W) != self.target_size:
            tH, tW = self.target_size
            img_tensor_np = self._adjust_size(img_tensor_np, tH, tW)
            mask_binary = self._adjust_size(mask_binary[None], tH, tW)[0]

        # 7. Thermal Data Augmentations (flips, rotations, contrast scaling)
        if self.augment:
            img_tensor_np, mask_binary = self._apply_augmentations(img_tensor_np, mask_binary)

        # 8. Convert to PyTorch float32 Tensors
        tensor_img = torch.from_numpy(img_tensor_np).float()
        tensor_mask = torch.from_numpy(mask_binary[None]).float()  # Shape [1, H, W]

        return tensor_img, tensor_mask

    def _adjust_size(self, arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Crop or zero-pad 3D array [C, H, W] to exactly (target_h, target_w)."""
        C, H, W = arr.shape
        out = np.zeros((C, target_h, target_w), dtype=arr.dtype)
        h_copy = min(H, target_h)
        w_copy = min(W, target_w)
        out[:, :h_copy, :w_copy] = arr[:, :h_copy, :w_copy]
        return out

    def _apply_augmentations(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips, 90-degree rotations, and slight thermal scaling."""
        # Random Horizontal Flip
        if self.rng.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random Vertical Flip
        if self.rng.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random Orthogonal Rotation (0, 90, 180, 270 deg)
        k = self.rng.integers(0, 4)
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Thermal Contrast Jitter (0.95x - 1.05x)
        jitter = self.rng.uniform(0.95, 1.05)
        img = np.clip(img * jitter, 0.0, 1.0)

        return img, mask
