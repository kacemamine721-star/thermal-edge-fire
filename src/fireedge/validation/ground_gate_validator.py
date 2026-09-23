"""Gate 1 Mode A: Ground Curation & Training Quality Gate.

Designed for multi-spectral / dual-band ground-truth dataset curation (e.g. Landsat-8 ActiveFire B10 & B11).
Exploits dual-band thermal cross-confirmation to verify that high-temperature anomalies appear
in both physical detector channels before classifying them as real active fire hotspots.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

from .base_validator import (
    Status,
    ValidationReport,
    ValidatorConfig,
    _RANK,
    analyze_base_channel,
    calibrate_from_frames,
)


class GroundGateValidator:
    """Pre-treatment quality gate for multi-channel ground truth dataset curation."""

    def __init__(
        self,
        config: Union[ValidatorConfig, Sequence[ValidatorConfig], Mapping[int, ValidatorConfig]],
    ):
        self.config = config

    def _cfg(self, c: int) -> ValidatorConfig:
        if isinstance(self.config, ValidatorConfig):
            return self.config
        if isinstance(self.config, Mapping):
            return self.config.get(c, ValidatorConfig())
        return self.config[c]

    def validate(self, frame: np.ndarray) -> ValidationReport:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[1] < 3 or arr.shape[2] < 3:
            return ValidationReport(Status.REJECT, ["BAD_SHAPE"], {})

        C = arr.shape[0]
        analyses = [analyze_base_channel(arr[c], self._cfg(c)) for c in range(C)]

        reasons: List[str] = []
        metrics: Dict[str, float] = {}
        worst = Status.PASS
        suspect = np.zeros(arr.shape, dtype=bool)

        for c in range(C):
            cfg = self._cfg(c)
            a = analyses[c]
            iso = a["isolated"].copy()

            # Multi-channel Cross-Confirmation (B10 & B11)
            # A real wildfire heats both thermal channels simultaneously;
            # a radiation transient hits only a single detector array.
            if cfg.cross_channel_confirm and C > 1 and iso.any():
                others_z = [analyses[o]["z"] for o in range(C) if o != c]
                confirmed = np.any([z > (cfg.hot_k / 2.0) for z in others_z], axis=0)
                n_confirmed = int((iso & confirmed).sum())
                metrics[f"ch{c}.hot_confirmed_real"] = n_confirmed
                iso &= ~confirmed  # Remove confirmed fires from anomaly mask

            suspect[c] = iso
            n_hot = int(iso.sum())
            metrics[f"ch{c}.hot_pixels"] = n_hot

            flags = list(a["flags"])
            rejects = list(a["rejects"])

            if n_hot > 0:
                hot_frac = n_hot / max(a["n_valid"], 1)
                if hot_frac > cfg.hot_frac_reject:
                    rejects.append("HOT_PIXELS")
                else:
                    flags.append("HOT_PIXELS")

            reasons += [f"ch{c}:{r}" for r in rejects + flags]
            metrics.update({f"ch{c}.{k}": v for k, v in a["metrics"].items()})

            level = Status.REJECT if rejects else Status.PASS_WITH_FLAGS if flags else Status.PASS
            if _RANK[level] > _RANK[worst]:
                worst = level

        cleaned = None
        if worst != Status.REJECT and suspect.any() and self._cfg(0).hot_pixel_policy == "repair":
            cleaned = arr.copy()
            for c in range(C):
                if suspect[c].any():
                    med = ndi.median_filter(arr[c], size=3, mode="nearest")
                    cleaned[c][suspect[c]] = med[suspect[c]]

        return ValidationReport(
            status=worst,
            reasons=reasons,
            metrics=metrics,
            hot_pixel_mask=suspect if suspect.any() else None,
            cleaned=cleaned,
        )
