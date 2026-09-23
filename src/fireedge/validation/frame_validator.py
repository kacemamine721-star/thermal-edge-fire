"""Stage 1 of the onboard CubeSat pipeline: the VALIDATION quality gate.

    capture -> [ VALIDATE ] -> normalize/NUC -> segment -> decide -> queue -> downlink

Answers J. McDonald's question "how do you guarantee your raw input is valid?"
Every check is cheap (O(pixels), NumPy/SciPy only, no learned model) so it can
run on every frame before the CNN ever sees it.

Output is a three-level verdict, NOT a boolean, because of his second point
("don't discard data that has value"):

    PASS             nothing suspicious
    PASS_WITH_FLAGS  usable; anomalies recorded (saturation, isolated hot pixels, ...)
    REJECT           frame is corrupt / unusable; do not feed it to the model

Design rules that protect recall:
  * A real fire is bright and sharp, so hot pixels are only FLAGGED by default
    (`hot_pixel_policy="flag"`); values are not modified unless "repair" is chosen.
  * An isolated hot pixel that is ALSO hot in another band is treated as real
    (a radiation transient hits one band's detector, a fire heats both B10 and B11).
  * Saturation / hot-pixel counts REJECT a frame only when they are massive.
  * All absolute thresholds are calibrated on clean data (`calibrate_from_frames`),
    never guessed, and the false-reject rate on fire frames is measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

MAD_TO_SIGMA = 1.4826
_EPS = 1e-9


class Status(str, Enum):
    PASS = "PASS"
    PASS_WITH_FLAGS = "PASS_WITH_FLAGS"
    REJECT = "REJECT"


_RANK = {Status.PASS: 0, Status.PASS_WITH_FLAGS: 1, Status.REJECT: 2}


# --------------------------------------------------------------------------- config
@dataclass
class ValidatorConfig:
    # sensor description
    bit_depth: int = 16
    saturation_level: Optional[float] = None      # default 2**bit_depth - 1
    nodata_value: Optional[float] = 0.0           # None -> no nodata concept

    # calibrated absolute limits (None = check skipped until calibrated)
    valid_min: Optional[float] = None             # physical/DN range seen on clean data (+margin)
    valid_max: Optional[float] = None
    max_gradient: Optional[float] = None          # largest legit neighbour-to-neighbour jump (DN)
    noise_flag: Optional[float] = None            # high-frequency noise level (DN) -> flag
    noise_reject: Optional[float] = None          # -> reject

    # thresholds (fractions of valid pixels unless stated)
    nodata_frac_reject: float = 0.30
    nonfinite_frac_reject: float = 0.01
    saturation_frac_reject: float = 0.02
    out_of_range_frac_flag: float = 0.0005
    out_of_range_frac_reject: float = 0.05
    jump_frac_flag: float = 0.005
    jump_frac_reject: float = 0.10
    noise_floor: float = 1.0                   # DN; also the "dead / stuck frame" std limit
    stripe_k: float = 10.0                     # robust z for a defective row/column
    stripe_frac_reject: float = 0.05           # fraction of rows+cols defective
    hot_k: float = 8.0                         # candidate: residual > hot_k * robust sigma
    hot_isolation_ratio: float = 0.35          # neighbours must stay below ratio * residual
    hot_frac_reject: float = 0.01
    hot_pixel_policy: str = "flag"             # "flag" | "repair"
    cross_channel_confirm: bool = True

    @property
    def sat_level(self) -> float:
        return float(self.saturation_level) if self.saturation_level is not None else float(2 ** self.bit_depth - 1)


# --------------------------------------------------------------------------- report
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

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- primitives
def _mad_sigma(v: np.ndarray) -> float:
    if v.size == 0:
        return 0.0
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def _neighbour_max(x: np.ndarray) -> np.ndarray:
    fp = np.ones((3, 3), bool)
    fp[1, 1] = False
    return ndi.maximum_filter(x, footprint=fp, mode="nearest")


def frame_metrics(x: np.ndarray, cfg: ValidatorConfig) -> dict:
    """Unit-free & unit-ful statistics used both for calibration and for the checks."""
    xf = np.asarray(x, np.float32)
    finite = np.isfinite(xf)
    nodata = (xf == cfg.nodata_value) if cfg.nodata_value is not None else np.zeros(xf.shape, bool)
    valid = finite & ~nodata
    out = {"valid_frac": float(valid.mean())}
    if valid.sum() < 16:
        return out
    xv = np.where(valid, xf, np.float32(np.median(xf[valid])))
    hp = xv - ndi.uniform_filter(xv, 3, mode="nearest")
    out["noise"] = _mad_sigma(hp[valid])
    gx = np.abs(np.diff(xv, axis=1))[valid[:, 1:] & valid[:, :-1]]
    gy = np.abs(np.diff(xv, axis=0))[valid[1:, :] & valid[:-1, :]]
    g = np.concatenate([gx, gy]) if (gx.size and gy.size) else (gx if gx.size else gy)
    out["grad_p9999"] = float(np.quantile(g, 0.9999)) if g.size else 0.0
    out["grad_max"] = float(g.max()) if g.size else 0.0
    vv = xf[valid]
    out["p_lo"], out["p_hi"] = float(np.quantile(vv, 1e-4)), float(np.quantile(vv, 1 - 1e-4))
    return out


def calibrate_from_frames(
    frames: Iterable[np.ndarray],
    base: Optional[ValidatorConfig] = None,
    range_pad: float = 0.25,
    noise_margin: tuple[float, float] = (1.5, 3.0),
    grad_margin: float = 1.5,
) -> ValidatorConfig:
    """Derive absolute thresholds from a set of *clean* single-channel frames.

    valid range   = [q(1e-4), q(1-1e-4)] over all pixels, padded by range_pad
    max_gradient  = 1.5 x q99.99 of the neighbour differences
    noise limits  = 1.5x / 3x the 99.9th percentile of the per-frame noise
    """
    cfg = base or ValidatorConfig()
    ms = [frame_metrics(f, cfg) for f in frames]
    ms = [m for m in ms if "noise" in m]
    if not ms:
        raise ValueError("no usable frames for calibration")
    lo = float(np.min([m["p_lo"] for m in ms]))
    hi = float(np.max([m["p_hi"] for m in ms]))
    pad = range_pad * (hi - lo) if hi > lo else 100.0
    n999 = float(np.quantile([m["noise"] for m in ms], 0.999))
    g = float(np.quantile([m["grad_p9999"] for m in ms], 0.999))
    return replace(
        cfg,
        valid_min=lo - pad,
        valid_max=hi + pad,
        max_gradient=grad_margin * g,
        noise_flag=max(noise_margin[0] * n999, cfg.noise_floor),
        noise_reject=max(noise_margin[1] * n999, 2 * cfg.noise_floor),
    )


# --------------------------------------------------------------------------- per-channel analysis
def _analyse(x: np.ndarray, cfg: ValidatorConfig) -> dict:
    flags: List[str] = []
    rejects: List[str] = []
    m: Dict[str, float] = {}
    xf = np.asarray(x, np.float32)
    H, W = xf.shape

    finite = np.isfinite(xf)
    m["nonfinite_frac"] = float(1 - finite.mean())
    if m["nonfinite_frac"] > 0:
        (rejects if m["nonfinite_frac"] > cfg.nonfinite_frac_reject else flags).append("NONFINITE")

    nodata = (xf == cfg.nodata_value) if cfg.nodata_value is not None else np.zeros(xf.shape, bool)
    invalid = nodata | ~finite
    valid = ~invalid
    m["nodata_frac"] = float(nodata.mean())
    if m["nodata_frac"] > cfg.nodata_frac_reject:
        rejects.append("NODATA_EXCESS")
    n_valid = int(valid.sum())
    if n_valid < 16:
        rejects.append("NO_VALID_PIXELS")
        return dict(
            flags=flags,
            rejects=rejects,
            metrics=m,
            isolated=np.zeros_like(valid),
            z=np.zeros(xf.shape, np.float32),
            med=xf,
            valid=valid,
            n_valid=n_valid,
        )

    xv = np.where(valid, xf, np.float32(np.median(xf[valid])))
    vals = xf[valid]

    # saturation (flag, reject only if massive)
    sat = (vals >= cfg.sat_level).mean()
    m["saturation_frac"] = float(sat)
    if sat > cfg.saturation_frac_reject:
        rejects.append("SATURATION_EXCESS")
    elif sat > 0:
        flags.append("SATURATION")

    # dead / stuck frame
    m["robust_std"] = _mad_sigma(vals)
    if m["robust_std"] < cfg.noise_floor:
        rejects.append("DEAD_OR_STUCK")

    # physical range
    if cfg.valid_min is not None and cfg.valid_max is not None:
        oor = float(((vals < cfg.valid_min) | (vals > cfg.valid_max)).mean())
        m["out_of_range_frac"] = oor
        if oor > cfg.out_of_range_frac_reject:
            rejects.append("OUT_OF_RANGE")
        elif oor > cfg.out_of_range_frac_flag:
            flags.append("OUT_OF_RANGE")

    # high-frequency noise
    hp = xv - ndi.uniform_filter(xv, 3, mode="nearest")
    noise = _mad_sigma(hp[valid])
    m["noise"] = noise
    if cfg.noise_reject is not None and noise > cfg.noise_reject:
        rejects.append("NOISE_EXCESS")
    elif cfg.noise_flag is not None and noise > cfg.noise_flag:
        flags.append("NOISE_HIGH")

    # non-physical spatial jumps
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

    # defective rows / columns (readout faults / striping)
    sig_line = max(noise / np.sqrt(max(H, W)), 1e-6)
    n_bad = 0
    for axis in (1, 0):
        prof = xv.mean(axis=axis)
        res = prof - ndi.median_filter(prof, size=3, mode="nearest")
        s = max(_mad_sigma(res), sig_line)
        bad_mask = (np.abs(res) / s > cfg.stripe_k) & (np.abs(res) > 3 * cfg.noise_floor / np.sqrt(max(H, W)))
        n_bad += int(bad_mask.sum())
    m["stripe_frac"] = n_bad / (H + W)
    if m["stripe_frac"] > cfg.stripe_frac_reject:
        rejects.append("STRIPING")
    elif n_bad > 0:
        flags.append("STRIPING")

    # isolated hot pixels (radiation transients / single event upsets)
    med = ndi.median_filter(xv, size=3, mode="nearest")
    res = xv - med
    scale = max(_mad_sigma(res[valid]), cfg.noise_floor)
    z = (res / scale).astype(np.float32)
    near_nodata = ndi.binary_dilation(invalid, structure=np.ones((3, 3), bool)) if invalid.any() else invalid
    cand = (z > cfg.hot_k) & valid & ~near_nodata
    iso = cand & ((_neighbour_max(xv) - med) < cfg.hot_isolation_ratio * res)
    m["hot_candidates"] = int(iso.sum())

    return dict(
        flags=flags,
        rejects=rejects,
        metrics=m,
        isolated=iso,
        z=z,
        med=med,
        valid=valid,
        n_valid=n_valid,
    )


# --------------------------------------------------------------------------- public API
class FrameValidator:
    """Validate a (H,W) or (C,H,W) frame against CubeSat quality-gate checks."""

    def __init__(
        self,
        config: Union[ValidatorConfig, Sequence[ValidatorConfig], Mapping[int, ValidatorConfig]],
    ):
        self.config = config

    def _cfg(self, c: int) -> ValidatorConfig:
        if isinstance(self.config, ValidatorConfig):
            return self.config
        return self.config[c]

    def validate(self, frame: np.ndarray) -> ValidationReport:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[1] < 3 or arr.shape[2] < 3:
            return ValidationReport(Status.REJECT, ["BAD_SHAPE"], {})
        C = arr.shape[0]
        an = [_analyse(arr[c], self._cfg(c)) for c in range(C)]

        reasons: List[str] = []
        metrics: Dict[str, float] = {}
        worst = Status.PASS
        suspect = np.zeros(arr.shape, bool)

        for c in range(C):
            cfg, a = self._cfg(c), an[c]
            iso = a["isolated"].copy()
            # Cross-channel confirmation: hot in both thermal channels -> confirmed real fire!
            if cfg.cross_channel_confirm and C > 1 and iso.any():
                others = [an[o]["z"] for o in range(C) if o != c]
                confirmed = np.any([z > cfg.hot_k / 2 for z in others], axis=0)
                metrics[f"ch{c}.hot_confirmed_real"] = int((iso & confirmed).sum())
                iso &= ~confirmed
            suspect[c] = iso
            n_hot = int(iso.sum())
            metrics[f"ch{c}.hot_pixels"] = n_hot
            flags, rejects = list(a["flags"]), list(a["rejects"])
            if n_hot:
                (rejects if n_hot / max(a["n_valid"], 1) > cfg.hot_frac_reject else flags).append("HOT_PIXELS")
            reasons += [f"ch{c}:{r}" for r in rejects + flags]
            metrics.update({f"ch{c}.{k}": v for k, v in a["metrics"].items()})
            level = Status.REJECT if rejects else Status.PASS_WITH_FLAGS if flags else Status.PASS
            if _RANK[level] > _RANK[worst]:
                worst = level

        cleaned = None
        if worst != Status.REJECT and suspect.any() and self._cfg(0).hot_pixel_policy == "repair":
            cleaned = arr.astype(np.float32).copy()
            for c in range(C):
                cleaned[c][suspect[c]] = an[c]["med"][suspect[c]]
        return ValidationReport(worst, reasons, metrics, suspect, cleaned)


def batch_validate(
    validator: FrameValidator, frames: Iterable[np.ndarray]
) -> Dict[str, Union[int, float, List[ValidationReport]]]:
    """Validate a batch of frames and compute pass/flag/reject totals."""
    reports: List[ValidationReport] = []
    counts = {Status.PASS: 0, Status.PASS_WITH_FLAGS: 0, Status.REJECT: 0}

    for f in frames:
        rep = validator.validate(f)
        reports.append(rep)
        counts[rep.status] += 1

    total = max(len(reports), 1)
    return {
        "total_frames": len(reports),
        "pass_count": counts[Status.PASS],
        "flag_count": counts[Status.PASS_WITH_FLAGS],
        "reject_count": counts[Status.REJECT],
        "pass_rate": counts[Status.PASS] / total,
        "flag_rate": counts[Status.PASS_WITH_FLAGS] / total,
        "reject_rate": counts[Status.REJECT] / total,
        "reports": reports,
    }
