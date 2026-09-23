"""Band-level statistics and distribution analysis for multi-spectral imagery."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class ChannelAccumulator:
    """Online / Reservoir accumulator for channel distribution percentiles and moments."""

    def __init__(self, max_reservoir: int = 100_000, nodata_val: float = 0.0, sat_val: float = 65535.0):
        self.max_reservoir = max_reservoir
        self.nodata_val = nodata_val
        self.sat_val = sat_val
        self.samples: List[np.ndarray] = []
        self.total_pixels = 0
        self.nodata_pixels = 0
        self.sat_pixels = 0
        self.sum_val = 0.0
        self.sum_sq = 0.0

    def update(self, channel_data: np.ndarray) -> None:
        flat = np.asarray(channel_data, dtype=np.float32).ravel()
        n = flat.size
        self.total_pixels += n

        nodata_mask = flat == self.nodata_val
        self.nodata_pixels += int(nodata_mask.sum())

        sat_mask = flat >= self.sat_val
        self.sat_pixels += int(sat_mask.sum())

        valid = flat[~nodata_mask & np.isfinite(flat)]
        if valid.size == 0:
            return

        self.sum_val += float(valid.sum())
        self.sum_sq += float((valid ** 2).sum())

        # Subsample for reservoir percentiles
        if valid.size > 2000:
            sub = np.random.choice(valid, size=2000, replace=False)
        else:
            sub = valid
        self.samples.append(sub)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "p01": 0.0,
                "p50": 0.0,
                "p99": 0.0,
                "p9999": 0.0,
                "nodata_pct": 0.0,
                "sat_pct": 0.0,
            }

        all_samples = np.concatenate(self.samples)
        valid_count = max(self.total_pixels - self.nodata_pixels, 1)
        mean = self.sum_val / valid_count
        variance = max((self.sum_sq / valid_count) - (mean ** 2), 0.0)
        std = float(np.sqrt(variance))

        return {
            "min": float(all_samples.min()),
            "max": float(all_samples.max()),
            "mean": float(mean),
            "std": std,
            "p01": float(np.percentile(all_samples, 0.01)),
            "p25": float(np.percentile(all_samples, 25.0)),
            "p50": float(np.median(all_samples)),
            "p75": float(np.percentile(all_samples, 75.0)),
            "p99": float(np.percentile(all_samples, 99.0)),
            "p9999": float(np.percentile(all_samples, 99.99)),
            "nodata_pct": (self.nodata_pixels / max(self.total_pixels, 1)) * 100.0,
            "sat_pct": (self.sat_pixels / max(self.total_pixels, 1)) * 100.0,
        }


def collect_band_stats(
    frames: Iterable[np.ndarray],
    band_names: Optional[Sequence[str]] = None,
    sat_val: float = 65535.0,
    nodata_val: float = 0.0,
) -> pd.DataFrame:
    """Collect summary statistics across all bands for an iterable of (C, H, W) frames."""
    accumulators: List[ChannelAccumulator] = []

    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[None]
        C = arr.shape[0]

        if not accumulators:
            accumulators = [
                ChannelAccumulator(nodata_val=nodata_val, sat_val=sat_val) for _ in range(C)
            ]

        for c in range(C):
            accumulators[c].update(arr[c])

    records = []
    for c, acc in enumerate(accumulators):
        name = band_names[c] if (band_names and c < len(band_names)) else f"Band_{c}"
        rec = {"channel": c, "band": name, **acc.summary()}
        records.append(rec)

    return pd.DataFrame(records)


def correlation_matrix(
    frames: Iterable[np.ndarray], channels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix across channels."""
    gathered: List[np.ndarray] = []

    for f in frames:
        arr = np.asarray(f)
        if arr.ndim == 2:
            arr = arr[None]
        if channels is not None:
            arr = arr[channels]
        # Sample 500 valid pixels per frame
        C, H, W = arr.shape
        flat = arr.reshape(C, -1)
        valid = (flat != 0).all(axis=0) & np.isfinite(flat).all(axis=0)
        idx = np.where(valid)[0]
        if idx.size > 200:
            sample_idx = np.random.choice(idx, size=min(200, idx.size), replace=False)
            gathered.append(flat[:, sample_idx])

    if not gathered:
        return np.eye(len(channels) if channels else 2)

    all_data = np.concatenate(gathered, axis=1)  # (C, total_samples)
    return np.corrcoef(all_data)
