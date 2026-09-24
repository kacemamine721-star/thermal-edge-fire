"""Thermal contrast scaling and normalization routines for edge CNN backbones."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def robust_percentile_norm(
    arr: np.ndarray,
    p_min: float = 2.0,
    p_max: float = 98.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize array to [0.0, 1.0] using robust empirical percentiles.

    Protects against extreme non-physical outliers or dead pixels skewing dynamic range.
    """
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if len(valid) < 10:
        return np.zeros_like(arr, dtype=np.float32)

    vmin = float(np.percentile(valid, p_min))
    vmax = float(np.percentile(valid, p_max))

    if vmax - vmin < eps:
        return np.zeros_like(arr, dtype=np.float32)

    scaled = (arr - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def fixed_window_norm(
    temp_k: np.ndarray,
    vmin_k: float = 270.0,   # ~ -3 °C (ambient cold clouds / winter)
    vmax_k: float = 420.0,   # ~ 147 °C (intense sub-pixel wildfire emission)
) -> np.ndarray:
    """Linearly map physical temperature in Kelvin to [0.0, 1.0] across fixed physical window.

    Ensures that identical physical fire temperatures map to identical neural network
    activation values regardless of scene background, preserving thermal elevation semantics.
    """
    scaled = (temp_k - vmin_k) / max(vmax_k - vmin_k, 1e-6)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def minmax_norm(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Standard Min-Max normalization to [0.0, 1.0]."""
    _min = vmin if vmin is not None else float(np.min(arr))
    _max = vmax if vmax is not None else float(np.max(arr))
    if _max - _min < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - _min) / (_max - _min), 0.0, 1.0).astype(np.float32)
