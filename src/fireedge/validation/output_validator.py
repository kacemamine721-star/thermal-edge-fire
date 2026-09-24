"""
output_validator.py
====================
GATE 2: POST-TREATMENT OUTPUT & PHYSICAL SANITY GATE

Validates model predictions and pre-processing integrity for the onboard
wildfire-detection edge-inference pipeline, before an alert is queued for
downlink. This is the second and final validation gate; Gate 1
(frame_validator.py) already guaranteed the *input* was trustworthy before
inference ran. Gate 2 guarantees the *output* is trustworthy before it is
ever transmitted.

Execution order (see `validate_output`) is deliberately chosen for maximum
short-circuiting on a compute- and power-constrained edge processor:
cheap, catastrophic-failure checks run first; expensive, per-region checks
run last, and only on whatever survives the earlier stages.

    1. Preprocessing sanity      -- O(n) vectorized, catches corrupt tensors
    2. Catastrophic collapse     -- O(n) single reduction, catches model hallucination
    3. Spatial plausibility      -- O(n) connected-component labeling, prunes candidates
    4. Thermal consistency       -- most expensive, runs only on surviving candidates
    5. Downlink packet build/verify -- cheap, final step, only if a detection remains

Dependencies: numpy, scipy.ndimage (ground-prototype only -- see LIMITATIONS
in the module docstring at the bottom of this file for the flight-software
porting note).
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence

import numpy as np

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for this ground-prototype implementation "
        "(connected-component labeling, local filtering). See the "
        "'Limitations & Hardware Constraints' note for the flight-software "
        "porting path."
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ValidatorConfig:
    # --- Preprocessing sanity -------------------------------------------- #
    max_nonfinite_fraction: float = 0.01       # NaN/Inf: reject above 1% of pixels
    max_subnormal_fraction: float = 0.05       # denormal floats: flag above 5%
    physical_min_k: float = 180.0              # LWIR brightness temp lower bound (K)
    physical_max_k: float = 400.0              # LWIR brightness temp upper bound (K)
    max_out_of_range_fraction: float = 0.01    # fraction allowed outside physical bounds

    # --- Spatial plausibility --------------------------------------------- #
    min_component_pixels: int = 2              # below this: single-pixel blip, reject
    catastrophic_area_fraction: float = 0.50   # above this: network collapse, reject frame

    # --- Thermal consistency ----------------------------------------------- #
    ring_dilation_px: int = 3                  # background ring width around each component
    min_delta_k: float = 10.0                  # candidate must exceed local bg by this (K)
    min_delta_sigma: float = 3.0               # ...or this many local std-devs, whichever is stricter
    coastal_margin_k: float = 5.0              # extra margin required near a coastline mask

    # --- Downlink packet ---------------------------------------------------- #
    max_packet_bytes: int = 200
    confidence_scale: int = 255                # uint8 quantization of [0,1] confidence
    intensity_scale_k: float = 0.1              # uint16 quantization step, in Kelvin/LSB

    # --- Compute-budget guard ------------------------------------------------ #
    max_candidate_components: int = 64          # cap expensive stage 4 work per frame


class RejectReason(Enum):
    NONE = auto()
    NONFINITE_TENSOR = auto()
    OUT_OF_PHYSICAL_RANGE = auto()
    CATASTROPHIC_COLLAPSE = auto()              # distinct from a normal reject: implies
                                                  # the model/pipeline itself may be degraded
    ALL_COMPONENTS_SUBPIXEL = auto()
    THERMAL_INCONSISTENT = auto()
    PACKET_SCHEMA_INVALID = auto()
    CHECKSUM_MISMATCH = auto()
    OUTSIDE_FOOTPRINT = auto()


@dataclass
class ValidationResult:
    passed: bool
    reason: RejectReason = RejectReason.NONE
    detail: str = ""
    telemetry_flag: bool = False                # set True for CATASTROPHIC_COLLAPSE-class
                                                  # events: ground ops should know inference
                                                  # health is suspect, not just "no fire seen"
    surviving_mask: Optional[np.ndarray] = None
    packet: Optional[bytes] = None


# --------------------------------------------------------------------------- #
# Stage 1 -- Preprocessing sanity
# --------------------------------------------------------------------------- #

def check_preprocessing_sanity(
    tensor: np.ndarray, cfg: ValidatorConfig
) -> ValidationResult:
    """Cheapest, most fundamental check. Runs first so a corrupt tensor never
    reaches the (much more expensive) spatial/thermal stages."""
    n = tensor.size
    finite_mask = np.isfinite(tensor)
    nonfinite_frac = 1.0 - (finite_mask.sum() / n)
    if nonfinite_frac > cfg.max_nonfinite_fraction:
        return ValidationResult(
            False, RejectReason.NONFINITE_TENSOR,
            f"{nonfinite_frac:.3%} non-finite values (limit "
            f"{cfg.max_nonfinite_fraction:.1%})",
        )

    # Subnormal (denormalized) floats: not necessarily invalid, but a high
    # fraction typically indicates ADC underflow or corrupted low-order bits
    # rather than genuine near-zero radiometry -- worth flagging, not fatal
    # by itself unless combined with other symptoms.
    finite_vals = tensor[finite_mask]
    tiny = np.finfo(tensor.dtype).tiny if np.issubdtype(tensor.dtype, np.floating) else 0.0
    subnormal_frac = (
        np.mean((np.abs(finite_vals) > 0) & (np.abs(finite_vals) < tiny))
        if tiny else 0.0
    )
    # (Not returned as a hard failure -- folded into `detail` for logging;
    # promote to a hard reject if it empirically correlates with bad frames.)

    out_of_range = (finite_vals < cfg.physical_min_k) | (finite_vals > cfg.physical_max_k)
    oor_frac = out_of_range.mean() if finite_vals.size else 1.0
    if oor_frac > cfg.max_out_of_range_fraction:
        return ValidationResult(
            False, RejectReason.OUT_OF_PHYSICAL_RANGE,
            f"{oor_frac:.3%} of pixels outside [{cfg.physical_min_k}, "
            f"{cfg.physical_max_k}] K (limit {cfg.max_out_of_range_fraction:.1%})",
        )

    return ValidationResult(
        True, detail=f"subnormal_fraction={subnormal_frac:.4%}"
    )


# --------------------------------------------------------------------------- #
# Stage 2 & 3 -- Spatial plausibility
# --------------------------------------------------------------------------- #

def check_spatial_plausibility(
    fire_mask: np.ndarray, cfg: ValidatorConfig
) -> ValidationResult:
    """
    Two sub-checks, ordered cheapest-first:
      (a) catastrophic collapse -- a single reduction, checked before doing
          any labeling work at all
      (b) isolated single-pixel blips -- connected-component labeling,
          pruning the candidate set *before* the expensive thermal stage runs
    """
    total = fire_mask.size
    area_frac = fire_mask.sum() / total

    if area_frac > cfg.catastrophic_area_fraction:
        return ValidationResult(
            False, RejectReason.CATASTROPHIC_COLLAPSE,
            f"{area_frac:.1%} of frame flagged as fire (limit "
            f"{cfg.catastrophic_area_fraction:.1%}) -- likely model/pipeline "
            f"degradation, not a real detection",
            telemetry_flag=True,
        )

    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, n_components = ndimage.label(fire_mask, structure=structure)
    if n_components == 0:
        return ValidationResult(True, surviving_mask=fire_mask)

    sizes = ndimage.sum(fire_mask, labeled, index=np.arange(1, n_components + 1))
    keep_labels = np.where(sizes >= cfg.min_component_pixels)[0] + 1

    if keep_labels.size == 0:
        return ValidationResult(
            False, RejectReason.ALL_COMPONENTS_SUBPIXEL,
            f"all {n_components} candidate region(s) below "
            f"{cfg.min_component_pixels}px -- consistent with SEU noise, "
            f"not a PSF-blurred real detection",
        )

    surviving = np.isin(labeled, keep_labels)

    # Cap the number of components passed to the expensive stage 4, ranked
    # by size (a real large fire is more urgent than a marginal small one
    # anyway) -- protects the compute budget on a pathological frame with
    # many small legitimate-looking candidates.
    if keep_labels.size > cfg.max_candidate_components:
        order = np.argsort(-sizes[keep_labels - 1])[: cfg.max_candidate_components]
        keep_labels = keep_labels[order]
        surviving = np.isin(labeled, keep_labels)

    return ValidationResult(
        True,
        detail=f"{keep_labels.size}/{n_components} candidate region(s) retained",
        surviving_mask=surviving,
    )


# --------------------------------------------------------------------------- #
# Stage 4 -- Thermal consistency
# --------------------------------------------------------------------------- #

def check_thermal_consistency(
    frame: np.ndarray,
    fire_mask: np.ndarray,
    cfg: ValidatorConfig,
    coastal_mask: Optional[np.ndarray] = None,
) -> ValidationResult:
    """
    For each candidate region, compare it against a local background ring
    (the region dilated outward, minus the region itself) rather than a
    global or fixed-window baseline. This is what actually rejects cold
    cloud edges and most coastal false-triggers:

      - Cold cloud edges are, physically, COLDER than surrounding warm
        land/sea -- requiring a strictly POSITIVE delta (fire hotter than
        local background) rejects these outright, regardless of how sharp
        the edge/contrast is. A magnitude-only contrast check would not
        catch this; a signed, directional check does.
      - Coastal boundaries produce a sharp gradient from differing thermal
        inertia, but that alone does not make land "hotter than its own
        local background" -- unless it genuinely is (e.g. sun-heated rock
        at midday), which is a real ambiguity this check alone cannot fully
        resolve. `coastal_mask`, if provided, raises the required margin
        near known coastlines as a partial mitigation (see edge cases).
    """
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_components = ndimage.label(fire_mask, structure=structure)
    if n_components == 0:
        return ValidationResult(True, surviving_mask=fire_mask)

    dil_structure = np.ones(
        (2 * cfg.ring_dilation_px + 1, 2 * cfg.ring_dilation_px + 1), dtype=bool
    )
    confirmed = np.zeros_like(fire_mask, dtype=bool)

    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        dilated = ndimage.binary_dilation(component, structure=dil_structure)
        ring = dilated & ~component  # background annulus, excludes the candidate itself

        if not ring.any():
            continue  # candidate touches frame border with no usable ring; be conservative

        bg_vals = frame[ring]
        bg_median = np.median(bg_vals)
        bg_std = np.std(bg_vals)

        fg_vals = frame[component]
        fg_stat = np.median(fg_vals)  # median, not max, to resist single-pixel outliers

        delta = fg_stat - bg_median
        required_delta = max(cfg.min_delta_k, cfg.min_delta_sigma * bg_std)

        if coastal_mask is not None and coastal_mask[component].any():
            required_delta += cfg.coastal_margin_k

        if delta > required_delta:  # strictly positive AND exceeds required margin
            confirmed |= component

    if not confirmed.any():
        return ValidationResult(
            False, RejectReason.THERMAL_INCONSISTENT,
            "no candidate region exceeded its local background by the "
            "required margin -- consistent with cloud edge, coastal "
            "gradient, or other non-thermal artifact",
        )

    return ValidationResult(True, surviving_mask=confirmed)


# --------------------------------------------------------------------------- #
# Stage 5 -- Downlink packet: schema, geolocation, checksum
# --------------------------------------------------------------------------- #

# Fixed-header layout (little-endian). Variable-length crop payload follows,
# then a trailing CRC. See ALERT_HEADER_FORMAT for the exact byte layout.
#   version   : B   (1)  schema version, for forward compatibility
#   timestamp : I   (4)  unix epoch seconds
#   lat       : f   (4)  degrees
#   lon       : f   (4)  degrees
#   bbox_x    : H   (2)  crop bounding box, native frame pixel coords
#   bbox_y    : H   (2)
#   bbox_w    : H   (2)
#   bbox_h    : H   (2)
#   confidence: B   (1)  quantized 0-255 <-> [0.0, 1.0]
#   intensity : H   (2)  quantized, LSB = cfg.intensity_scale_k Kelvin
#   flags     : B   (1)  bit0: low_confidence, bit1: night_mode, bit2: coastal_proximity
#   crop_len  : H   (2)  length in bytes of the appended (optionally compressed) crop
# fixed header = 27 bytes, + crop_len bytes of crop payload, + 4 bytes CRC-32 trailer
ALERT_HEADER_FORMAT = "<BIffHHHHBHBH"
ALERT_HEADER_SIZE = struct.calcsize(ALERT_HEADER_FORMAT)  # 27 bytes


@dataclass
class AlertPacket:
    timestamp: int
    lat: float
    lon: float
    bbox: tuple  # (x, y, w, h) in native frame pixel coordinates
    confidence: float  # 0.0-1.0
    intensity_k: float  # estimated peak brightness temperature, Kelvin
    low_confidence: bool = False
    night_mode: bool = False
    coastal_proximity: bool = False
    crop_bytes: bytes = b""
    version: int = 1


def pack_alert_packet(p: AlertPacket, cfg: ValidatorConfig) -> bytes:
    flags = (
        (int(p.low_confidence) << 0)
        | (int(p.night_mode) << 1)
        | (int(p.coastal_proximity) << 2)
    )
    header = struct.pack(
        ALERT_HEADER_FORMAT,
        p.version,
        p.timestamp,
        p.lat,
        p.lon,
        *[int(v) for v in p.bbox],
        int(round(p.confidence * cfg.confidence_scale)),
        int(round(p.intensity_k / cfg.intensity_scale_k)),
        flags,
        len(p.crop_bytes),
    )
    body = header + p.crop_bytes
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """Pure-Python CRC-16-CCITT, provided as the lower-overhead alternative
    to CRC-32 for the size-constrained packet. Trade-off: CRC-16 halves the
    trailer cost (2 bytes vs 4) at slightly weaker error-detection strength
    than CRC-32. Recommendation: use CRC-16 for the actual flight packet
    given the ~200-byte budget; keep CRC-32 during ground testing where
    packet size doesn't matter and stronger validation is worth the bytes.
    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def _point_in_polygon(lat: float, lon: float, polygon: Sequence[tuple]) -> bool:
    """Ray-casting point-in-polygon, dependency-free (no shapely) -- footprint
    polygon is (lat, lon) vertices describing the current swath/footprint."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if ((lon_i > lon) != (lon_j > lon)) and (
            lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i + 1e-12) + lat_i
        ):
            inside = not inside
        j = i
    return inside


def validate_alert_packet(
    packet: bytes,
    cfg: ValidatorConfig,
    footprint_polygon: Sequence[tuple],
    footprint_bbox: Optional[tuple] = None,  # (min_lat, max_lat, min_lon, max_lon) fast path
) -> ValidationResult:
    if len(packet) > cfg.max_packet_bytes or len(packet) < ALERT_HEADER_SIZE + 4:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"packet size {len(packet)}B outside valid bounds "
            f"[{ALERT_HEADER_SIZE + 4}, {cfg.max_packet_bytes}]",
        )

    body, crc_bytes = packet[:-4], packet[-4:]
    expected_crc = struct.unpack("<I", crc_bytes)[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        return ValidationResult(
            False, RejectReason.CHECKSUM_MISMATCH,
            f"CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}",
        )

    try:
        header = struct.unpack(ALERT_HEADER_FORMAT, body[:ALERT_HEADER_SIZE])
    except struct.error:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            "packet header cannot be decoded",
        )

    (
        version,
        _ts,
        lat,
        lon,
        _bbox_x,
        _bbox_y,
        bbox_w,
        bbox_h,
        confidence,
        _intensity,
        flags,
        crop_len,
    ) = header

    if version != 1:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"unsupported packet schema version {version}",
        )
    if len(body) != ALERT_HEADER_SIZE + crop_len:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"crop length {crop_len} does not match packet body size",
        )
    if not (np.isfinite(lat) and np.isfinite(lon)) or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"invalid geographic coordinates ({lat}, {lon})",
        )
    if bbox_w == 0 or bbox_h == 0:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            "bounding box dimensions must be positive",
        )
    if confidence > cfg.confidence_scale:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"confidence quantization {confidence} exceeds {cfg.confidence_scale}",
        )
    if flags & ~0x07:
        return ValidationResult(
            False, RejectReason.PACKET_SCHEMA_INVALID,
            f"reserved packet flags set: {flags:#04x}",
        )

    # Cheap bounding-box pre-check before the more expensive polygon test --
    # short-circuits the common case (well inside or well outside the swath).
    if footprint_bbox is not None:
        min_lat, max_lat, min_lon, max_lon = footprint_bbox
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return ValidationResult(
                False, RejectReason.OUTSIDE_FOOTPRINT,
                f"({lat:.4f}, {lon:.4f}) outside footprint bbox "
                f"fast-path pre-check",
            )

    if not _point_in_polygon(lat, lon, footprint_polygon):
        return ValidationResult(
            False, RejectReason.OUTSIDE_FOOTPRINT,
            f"({lat:.4f}, {lon:.4f}) fails full polygon containment -- "
            f"likely a geolocation computation error, not a real detection "
            f"outside the sensor swath",
        )

    return ValidationResult(True, packet=packet)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def validate_output(
    raw_tensor: np.ndarray,
    frame: np.ndarray,
    fire_mask: np.ndarray,
    cfg: ValidatorConfig,
    coastal_mask: Optional[np.ndarray] = None,
) -> ValidationResult:
    """Runs stages 1-4 in short-circuit order. Stage 5 (packet build/verify)
    is intentionally a separate call (see `pack_alert_packet` /
    `validate_alert_packet`) since it needs geolocation/ephemeris data that
    lives outside this function's scope in the real flight software."""

    r1 = check_preprocessing_sanity(raw_tensor, cfg)
    if not r1.passed:
        return r1

    r2 = check_spatial_plausibility(fire_mask, cfg)
    if not r2.passed:
        return r2

    if r2.surviving_mask is None or not r2.surviving_mask.any():
        return ValidationResult(True, detail="no candidates after spatial filtering")

    r3 = check_thermal_consistency(frame, r2.surviving_mask, cfg, coastal_mask=coastal_mask)
    return r3


# --------------------------------------------------------------------------- #
# Self-test / usage demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = ValidatorConfig()
    rng = np.random.default_rng(0)

    # Synthetic 64x64 LWIR-style frame: background ~290K, one real hot spot,
    # one single-pixel SEU spike, one cold cloud-edge artifact.
    frame = rng.normal(290.0, 2.0, size=(64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=bool)

    frame[30:34, 30:34] += 45.0   # real fire: 4x4 blurred hot region
    mask[30:34, 30:34] = True

    frame[10, 50] += 60.0          # SEU: single-pixel spike, no PSF blur
    mask[10, 50] = True

    frame[45:50, 5:10] -= 30.0     # cold cloud edge: flagged by mask, but COLDER
    mask[45:50, 5:10] = True

    result = validate_output(frame, frame, mask, cfg)
    print("validate_output ->", result.passed, result.reason, "|", result.detail)
    if result.surviving_mask is not None:
        ys, xs = np.where(result.surviving_mask)
        print(f"Surviving pixels: {len(ys)} "
              f"(expect only the real 4x4 fire region, i.e. 16 px)")

    if result.passed and result.surviving_mask is not None and result.surviving_mask.any():
        ys, xs = np.where(result.surviving_mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        packet_obj = AlertPacket(
            timestamp=int(time.time()),
            lat=36.8065, lon=10.1815,  # example: Tunis
            bbox=bbox,
            confidence=0.91,
            intensity_k=float(frame[result.surviving_mask].max()),
        )
        packet = pack_alert_packet(packet_obj, cfg)
        print(f"Packet size: {len(packet)} bytes (budget {cfg.max_packet_bytes})")

        footprint = [(30.0, 5.0), (30.0, 15.0), (40.0, 15.0), (40.0, 5.0)]
        pr = validate_alert_packet(packet, cfg, footprint_polygon=footprint)
        print("validate_alert_packet ->", pr.passed, pr.reason, "|", pr.detail)





