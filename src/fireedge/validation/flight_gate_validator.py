"""Gate 1 Mode B: Flight / Orbital In-Orbit Quality Gate.

Engineered specifically for single-channel uncooled LWIR microbolometer CubeSat payloads.
Operates without requiring a second thermal band by leveraging optical physics:
Point Spread Function (PSF) Spatial Halo Verification (van Dokkum 2001; Zhukov et al. 2006).

Differentiates:
- Radiation Transients (SEUs / Cosmic Rays): Strike detector silicon directly bypassing optics;
  creates sharp Dirac-delta spike with zero optical diffusion into immediate neighbors (cliff).
- Real Wildfires (Sub-Pixel Hotspots): Thermal photons must pass through camera lens optics;
  diffraction and lens transfer function enforce optical blur (Airy disk / Gaussian halo)
  spreading radiant energy into adjacent pixels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from scipy import ndimage as ndi

from .base_validator import (
    Status,
    ValidationReport,
    ValidatorConfig,
    _RANK,
    _neighbour_max,
    analyze_base_channel,
    calibrate_from_frames,
)


class FlightGateValidator:
    """In-orbit pre-treatment quality gate for single-channel LWIR microbolometers."""

    def __init__(
        self,
        config: Optional[ValidatorConfig] = None,
        halo_min_ratio: Optional[float] = None,
        halo_threshold: Optional[float] = None,
    ):
        self.config = config if config is not None else ValidatorConfig()
        threshold = halo_threshold if halo_threshold is not None else halo_min_ratio
        if threshold is not None:
            self.config.halo_min_ratio = float(threshold)

    def validate(self, frame: np.ndarray) -> ValidationReport:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                channel = arr[0]
            else:
                # If multi-channel passed to flight validator, use primary thermal channel
                channel = arr[0]
        elif arr.ndim == 2:
            channel = arr
        else:
            return ValidationReport(Status.REJECT, ["BAD_SHAPE"], {})

        cfg = self.config
        a = analyze_base_channel(channel, cfg)

        reasons: List[str] = []
        metrics: Dict[str, float] = {}

        # 1. Base telemetry verdicts
        flags = list(a["flags"])
        rejects = list(a["rejects"])
        metrics.update({f"ch0.{k}": v for k, v in a["metrics"].items()})

        # 2. Single-Band Optical Point Spread Function (PSF) Verification
        # Candidate hot pixels with significant z-score above noise floor
        cand = a["cand"]
        n_cand = int(cand.sum())
        metrics["ch0.hot_candidates"] = n_cand

        seu_mask = np.zeros(channel.shape, dtype=bool)
        n_confirmed_optical = 0
        n_seu = 0

        if n_cand > 0:
            med = a["med"]
            residual = channel - med
            nb_max = a["nb_max"]

            # Compute optical halo elevation above local median:
            # How much did adjacent pixels bloom due to optical PSF?
            halo_elevation = np.maximum(nb_max - med, 0.0)

            # Halo Ratio: Fraction of central peak energy convolved into neighboring pixels
            with np.errstate(divide="ignore", invalid="ignore"):
                halo_ratio = np.where(residual > 0, halo_elevation / np.maximum(residual, 1e-6), 0.0)

            # Spatial cluster check: An SEU is strictly an isolated single-pixel hit.
            # If adjacent pixels are also elevated (connected component >= 2), it is an extended fire cluster.
            footprint_8 = np.ones((3, 3), dtype=bool)
            footprint_8[1, 1] = False
            has_hot_neighbor = ndi.maximum_filter(cand.astype(np.uint8), footprint=footprint_8) > 0

            # An SEU must be BOTH isolated (no hot neighbors) AND lack an optical diffusion halo
            is_seu = cand & ~has_hot_neighbor & (halo_ratio < cfg.halo_min_ratio)
            is_optical = cand & ~is_seu

            seu_mask = is_seu
            n_seu = int(is_seu.sum())
            n_confirmed_optical = int(is_optical.sum())

        metrics["ch0.seu_transients"] = n_seu
        metrics["ch0.hot_confirmed_optical"] = n_confirmed_optical

        if n_seu > 0:
            seu_frac = n_seu / max(a["n_valid"], 1)
            if seu_frac > cfg.hot_frac_reject:
                rejects.append("SEU_TRANSIENTS")
            else:
                flags.append("SEU_TRANSIENTS")

        reasons += [f"ch0:{r}" for r in rejects + flags]

        # 3. Overall Verdict
        level = Status.REJECT if rejects else Status.PASS_WITH_FLAGS if flags else Status.PASS

        # 4. Optional In-Orbit Transient Repair
        cleaned = None
        if level != Status.REJECT and n_seu > 0 and cfg.hot_pixel_policy == "repair":
            cleaned = channel.copy()
            med = ndi.median_filter(channel, size=3, mode="nearest")
            cleaned[seu_mask] = med[seu_mask]
            if arr.ndim == 3:
                cleaned = cleaned[None]

        return ValidationReport(
            status=level,
            reasons=reasons,
            metrics=metrics,
            hot_pixel_mask=seu_mask[None] if (arr.ndim == 3 and seu_mask.any()) else (seu_mask if seu_mask.any() else None),
            cleaned=cleaned,
        )
