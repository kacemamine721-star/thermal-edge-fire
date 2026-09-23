"""Base Quality Gate Validator for Satellite Telemetry.

Contains universal sensor-level integrity checks:
- Non-finite (NaN/Inf) detection
- Swath border / No-Data excess
- Dead / stuck detector check
- Dynamic range and saturation limits
- Pushbroom sensor line striping
- Mathematical helper functions and calibration routines
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import warnings

import numpy as np
from scipy import ndimage as ndi

MAD_TO_SIGMA = 1.4826
_EPS = 1e-9


class Status(str, Enum):
    PASS = "PASS"
    PASS_WITH_FLAGS = "PASS_WITH_FLAGS"
    REJECT = "REJECT"


_RANK = {Status.PASS: 0, Status.PASS_WITH_FLAGS: 1, Status.REJECT: 2}


@dataclass
class ValidatorConfig:
    # Sensor description
    bit_depth: int = 16
    saturation_level: Optional[float] = None      # default 2**bit_depth - 1
    nodata_value: Optional[float] = 0.0           # None -> no nodata concept

    # Calibrated absolute limits (None = check skipped until calibrated)
    valid_min: Optional[float] = None             # physical/DN range seen on clean data (+margin)
    valid_max: Optional[float] = None
    max_gradient: Optional[float] = None          # largest legit neighbour-to-neighbour jump (DN)
    noise_flag: Optional[float] = None            # high-frequency noise level (DN) -> flag
    noise_reject: Optional[float] = None          # -> reject

    # Thresholds (fractions of valid pixels unless stated)
    nodata_frac_reject: float = 0.30
    nonfinite_frac_reject: float = 0.01
    saturation_frac_reject: float = 0.02
    out_of_range_frac_flag: float = 0.0005
    out_of_range_frac_reject: float = 0.05
    jump_frac_flag: float = 0.005
    jump_frac_reject: float = 0.10
    noise_floor: float = 1.0                   # DN; also the "dead / stuck frame" std limit
    stripe_k: float = 10.0                     # robust z for a defective row/column
    stripe_frac_reject: float = 0.20           # fraction of rows+cols defective (Landsat-8 TIRS: ~0.06-0.13 normal)

    # Hot pixel / Transient thresholds
    hot_k: float = 8.0                         # candidate: residual > hot_k * robust sigma
    hot_isolation_ratio: float = 0.35          # neighbours must stay below ratio * residual
    halo_min_ratio: float = 0.15               # Flight PSF check: minimum optical halo fraction
    hot_frac_reject: float = 0.01
    hot_pixel_policy: str = "flag"             # "flag" | "repair"
    cross_channel_confirm: bool = True

    @property
    def sat_level(self) -> float:
        return float(self.saturation_level) if self.saturation_level is not None else float(2 ** self.bit_depth - 1)


@dataclass
class ValidationReport:
    status: Status
    reasons: List[str] = field(default_factory=list)      # e.g. "ch0:SATURATION"
    metrics: Dict[str, float] = field(default_factory=dict)
    hot_pixel_mask: Optional[np.ndarray] = None           # (C,H,W) bool - suspected transients
    cleaned: Optional[np.ndarray] = None                  # only when policy == "repair"

    @property
    def ok(self) -> bool:
        return self.status != Status.REJECT

    def to_dict(self) -> dict:
        return {"status": self.status.value, "reasons": self.reasons, **self.metrics}


def _mad_sigma(x: np.ndarray) -> float:
    """Robust standard deviation via Median Absolute Deviation."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * MAD_TO_SIGMA)


def _neighbour_max(x: np.ndarray) -> np.ndarray:
    """Maximum value in 8-neighbourhood (excluding center pixel)."""
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    return ndi.maximum_filter(x, footprint=footprint, mode="nearest")


def _neighbour_mean(x: np.ndarray) -> np.ndarray:
    """Mean value in 8-neighbourhood (excluding center pixel)."""
    kernel = np.ones((3, 3), dtype=np.float32) / 8.0
    kernel[1, 1] = 0.0
    return ndi.convolve(x.astype(np.float32), kernel, mode="nearest")


def calibrate_from_frames(
    frames: Iterable[np.ndarray],
    base: Optional[ValidatorConfig] = None,
    margin_sigma: float = 4.0,
) -> ValidatorConfig:
    """Derive physical dynamic range and gradient thresholds from clean sensor data."""
    cfg = replace(base) if base is not None else ValidatorConfig()
    mins, maxs, grads, noises = [], [], [], []

    for f in frames:
        arr = np.asarray(f, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        valid = np.isfinite(arr)
        if cfg.nodata_value is not None:
            valid &= arr != cfg.nodata_value
        if valid.sum() < 32:
            continue
        v = arr[valid]
        mins.append(float(np.percentile(v, 0.01)))
        maxs.append(float(np.percentile(v, 99.99)))

        gx = np.abs(np.diff(arr, axis=1))[valid[:, 1:] & valid[:, :-1]]
        gy = np.abs(np.diff(arr, axis=0))[valid[1:, :] & valid[:-1, :]]
        g = np.concatenate([gx, gy]) if (gx.size and gy.size) else (gx if gx.size else gy)
        if g.size:
            grads.append(float(np.percentile(g, 99.9)))

        med = ndi.median_filter(arr, size=3, mode="nearest")
        res = (arr - med)[valid]
        noises.append(_mad_sigma(res))

    if not mins:
        return cfg

    vmin_med, vmin_s = float(np.median(mins)), max(_mad_sigma(np.array(mins)), 1.0)
    vmax_med, vmax_s = float(np.median(maxs)), max(_mad_sigma(np.array(maxs)), 1.0)
    cfg.valid_min = max(0.0, vmin_med - margin_sigma * vmin_s)
    cfg.valid_max = min(cfg.sat_level, vmax_med + margin_sigma * vmax_s)

    if grads:
        cfg.max_gradient = float(np.median(grads) + margin_sigma * max(_mad_sigma(np.array(grads)), 1.0))
    if noises:
        med_noise = float(np.median(noises))
        s_noise = max(_mad_sigma(np.array(noises)), 0.1)
        cfg.noise_flag = max(med_noise + 3.0 * s_noise, 2.0)       # floor: 2 DN
        cfg.noise_reject = max(med_noise + 6.0 * s_noise, 5.0)     # floor: 5 DN (Landsat L1TP quantization)

    return cfg


def analyze_base_channel(x: np.ndarray, cfg: ValidatorConfig) -> dict:
    """Run universal physical and telemetry checks on a single 2D channel."""
    H, W = x.shape
    total_px = H * W
    flags: List[str] = []
    rejects: List[str] = []
    m: Dict[str, float] = {}

    # 1. Non-finite check (NaN/Inf)
    finite_mask = np.isfinite(x)
    n_nonfinite = total_px - int(finite_mask.sum())
    m["nonfinite_frac"] = n_nonfinite / total_px
    if m["nonfinite_frac"] > cfg.nonfinite_frac_reject:
        rejects.append("NONFINITE")

    # 2. No-data / Border excess
    nodata_mask = np.zeros((H, W), dtype=bool)
    if cfg.nodata_value is not None:
        nodata_mask = x == cfg.nodata_value
    n_nodata = int(nodata_mask.sum())
    m["nodata_frac"] = n_nodata / total_px
    if m["nodata_frac"] > cfg.nodata_frac_reject:
        rejects.append("NODATA_EXCESS")

    valid = finite_mask & ~nodata_mask
    n_valid = int(valid.sum())
    m["valid_pixels"] = n_valid
    if n_valid < 16:
        rejects.append("EMPTY_FRAME")
        return dict(flags=flags, rejects=rejects, metrics=m, isolated=np.zeros((H, W), bool),
                    valid=valid, n_valid=n_valid, z=np.zeros((H, W), np.float32), med=np.zeros((H, W), np.float32))

    xv = np.where(valid, x, np.nan)

    # 3. Dead / Stuck detector check
    raw_std = float(np.nanstd(xv))
    m["raw_std"] = raw_std
    if raw_std < cfg.noise_floor:
        rejects.append("DEAD_OR_STUCK")

    # 4. High-frequency noise level
    med = ndi.median_filter(np.where(valid, x, 0.0), size=3, mode="nearest")
    res = (x - med)[valid]
    noise = _mad_sigma(res)
    m["noise"] = noise
    if cfg.noise_reject is not None and noise > cfg.noise_reject:
        rejects.append("HIGH_NOISE")
    elif cfg.noise_flag is not None and noise > cfg.noise_flag:
        flags.append("HIGH_NOISE")

    # 5. Out-of-range physical DN limits
    if cfg.valid_min is not None or cfg.valid_max is not None:
        lo = cfg.valid_min if cfg.valid_min is not None else -np.inf
        hi = cfg.valid_max if cfg.valid_max is not None else np.inf
        oor = (x < lo) | (x > hi)
        oor_frac = float((oor & valid).sum()) / n_valid
        m["out_of_range_frac"] = oor_frac
        if oor_frac > cfg.out_of_range_frac_reject:
            rejects.append("OUT_OF_RANGE")
        elif oor_frac > cfg.out_of_range_frac_flag:
            flags.append("OUT_OF_RANGE")

    # 6. Detector well saturation
    sat = (x >= cfg.sat_level) & valid
    sat_frac = float(sat.sum()) / n_valid
    m["sat_frac"] = sat_frac
    if sat_frac > cfg.saturation_frac_reject:
        rejects.append("SATURATION_EXCESS")
    elif sat_frac > 0:
        flags.append("SATURATION")

    # 7. Non-physical spatial jumps
    if cfg.max_gradient is not None:
        gx = np.abs(np.diff(xv, axis=1))[valid[:, 1:] & valid[:, :-1]]
        gy = np.abs(np.diff(xv, axis=0))[valid[1:, :] & valid[:-1, :]]
        g = np.concatenate([gx, gy]) if (gx.size and gy.size) else (gx if gx.size else gy)
        jf = float((g > cfg.max_gradient).mean()) if g.size else 0.0
        m["jump_frac"] = jf
        if jf > cfg.jump_frac_reject:
            rejects.append("NONPHYSICAL_GRADIENT")
        elif jf > cfg.jump_frac_flag:
            flags.append("NONPHYSICAL_GRADIENT")

    # 8. Defective rows / columns (pushbroom striping)
    sig_line = max(noise / np.sqrt(max(H, W)), 1e-6)
    n_bad = 0
    for axis in (1, 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prof = np.nanmean(xv, axis=axis)
        prof = np.nan_to_num(prof, nan=0.0)
        res_prof = prof - ndi.median_filter(prof, size=3, mode="nearest")
        s = max(_mad_sigma(res_prof), sig_line)
        bad_mask = (np.abs(res_prof) / s > cfg.stripe_k) & (np.abs(res_prof) > 3 * cfg.noise_floor / np.sqrt(max(H, W)))
        n_bad += int(bad_mask.sum())
    m["stripe_frac"] = n_bad / (H + W)
    if m["stripe_frac"] > cfg.stripe_frac_reject:
        rejects.append("STRIPING")
    elif n_bad > 0:
        flags.append("STRIPING")

    # Candidate hot pixels (residual z-score)
    scale = max(noise, cfg.noise_floor)
    z = ((x - med) / scale).astype(np.float32)
    invalid = ~valid
    near_nodata = ndi.binary_dilation(invalid, structure=np.ones((3, 3), bool)) if invalid.any() else invalid
    cand = (z > cfg.hot_k) & valid & ~near_nodata

    # General isolation test
    nb_max = _neighbour_max(x)
    residual = x - med
    iso = cand & ((nb_max - med) < cfg.hot_isolation_ratio * residual)
    m["hot_candidates"] = int(iso.sum())

    return dict(
        flags=flags,
        rejects=rejects,
        metrics=m,
        isolated=iso,
        cand=cand,
        z=z,
        med=med,
        valid=valid,
        n_valid=n_valid,
        nb_max=nb_max,
        residual=residual,
    )
