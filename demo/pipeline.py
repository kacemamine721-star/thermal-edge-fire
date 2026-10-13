#!/usr/bin/env python3
"""
Onboard-style fire-detection pipeline (demo implementation)
============================================================
Python port of the notebook reference pipeline (cells 13, 13B and 14), with the constants read from
models/manifest.json (models/manifest_v7.json is still accepted as a fallback file name).

Stages:
  1. Gate 1: input validation and transient detection (optional scrub, optional dead-column repair)
  2. Fixed radiometric preprocessing: linear DN -> TOA reflectance
  3. Neural segmentation: Tiny U-Net, ONNX Runtime INT8 / FP32
  4. Probability calibration (temperature scaling) and operating threshold (T = 2.1228, tau = 0.96)
  5. Event extraction: 8-connected components, contrast metric, geolocation
  6. Gate 2: output sanity checks, then the compact binary alert packet (version 3, CRC32)

Known differences from the notebook (documented in MODEL_ARCHITECTURE.md, section 3.1):
  * event confidence is the raw mean temperature-scaled probability (the notebook maps it through an isotonic table);
  * geolocation uses the file's affine transform + pyproj (the notebook uses a linear model per patch).
"""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy import ndimage
from scipy.ndimage import median_filter
from scipy.special import expit

# ───────────────── Manifest constants ─────────────────
MANIFEST_NAMES = ("manifest.json", "manifest_v7.json")


def load_manifest(manifest_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads the model manifest so that operational constants are read from it, not hard-coded."""
    candidates = []
    if manifest_path:
        candidates.append(Path(manifest_path))
    here = Path(__file__).resolve().parent
    for base in (here / "models", here.parent / "models",
                 Path.cwd() / "thermal-edge-fire" / "demo" / "models", Path.cwd() / "demo" / "models",
                 Path.cwd() / "models"):
        for name in MANIFEST_NAMES:
            candidates.append(base / name)
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


MANIFEST_DATA = load_manifest()

# ───────────────── Operational constants (synchronised with the manifest) ─────────────────
PKT_MAGIC = 0xF1
PKT_VERSION = int(MANIFEST_DATA.get("packet", {}).get("version", 3))
MODEL_ID = int(MANIFEST_DATA.get("model_ids", {}).get("pereira3c", 1))

HEADER_FMT = '<BBBIQBB'        # magic, version, model_id, frame_id, acq_ts, n_events, flags
EVENT_FMT = '<iiHHHHHHHHHB'    # lat_e7, lon_e7, x, y, w, h, area_ha_x10, conf_u16, metric_u16, gsd_m, geo_uncert, flags

HEADER_BYTES = int(MANIFEST_DATA.get("packet", {}).get("header_bytes", struct.calcsize(HEADER_FMT)))
EVENT_BYTES = int(MANIFEST_DATA.get("packet", {}).get("event_bytes", struct.calcsize(EVENT_FMT)))
MAX_EVENTS = int(MANIFEST_DATA.get("packet", {}).get("max_events", 6))
MAX_PACKET_BYTES = int(MANIFEST_DATA.get("packet", {}).get("max_alert_packet_bytes", HEADER_BYTES + MAX_EVENTS * EVENT_BYTES + 4))
EMPTY_PACKET_BYTES = int(MANIFEST_DATA.get("packet", {}).get("max_empty_or_heartbeat_bytes", HEADER_BYTES + 4))
SINGLE_EVENT_BYTES = HEADER_BYTES + EVENT_BYTES + 4

RAW_BYTES_PER_TILE = int(MANIFEST_DATA.get("pipeline", {}).get("raw_bytes_per_frame_selected_bands", 3 * 256 * 256 * 2))
HEARTBEAT_EVERY_N_FRAMES = int(MANIFEST_DATA.get("heartbeat", {}).get("every_n_frames", 30))

TEMPERATURE_DEFAULT = float(MANIFEST_DATA.get("probability_calibration", {}).get("temperature", 2.122759125440246))
THRESHOLD_DEFAULT = float(MANIFEST_DATA.get("threshold", {}).get("benchmark_threshold", 0.96))
MIN_EVENT_PIXELS = 1

# Packet flags (header, 8 bits - all assigned)
FLAG_INPUT_REJECTED = 1
FLAG_OUTPUT_REJECTED = 2
FLAG_TRANSIENTS_SCRUBBED = 4      # bit 4: Gate 1 modified the input (transient scrub and/or dead-column repair)
FLAG_INPUT_REPAIRED = FLAG_TRANSIENTS_SCRUBBED
FLAG_SATURATED_INPUT = 8
FLAG_GATE2_WARN = 16
FLAG_HEARTBEAT = 32
FLAG_TIME_UNAVAILABLE = 64
FLAG_REFERENCE_ONLY = 128

# Event flags (per event record, 8 bits: bits 0-3 are flags, bits 4-7 are the dropped-event counter, see encode_packet)
EVENT_FLAG_NO_GEO = 1
EVENT_FLAG_GEO_UNCERT_UNKN = 2
EVENT_FLAG_SWIR_METRIC = 4
EVENT_FLAG_THERMAL_METRIC = 8
EVENT_FLAG_MASK = 0x0F
EVENT_DROPPED_SHIFT = 4

FRAME_SIZE = 256
N_CHANNELS = 3
BAND_NAMES = ("c7", "c6", "c2")
GSD_M_DEFAULT = 30
GEO_UNCERT_UNKNOWN_M = 65535
RING_PX = 5                       # width of the local background ring used by the contrast metric

REFL_SCALE = 2.0e-5
REFL_OFFSET = -0.1
REFL_MAX = 1.6
SAT_DN = 65535

_S8 = np.ones((3, 3), dtype=bool)


# ───────────────── Stage 1: Gate 1 ─────────────────
def repair_dead_columns_band(band: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """Replaces each dead column by linear interpolation between its nearest live neighbours
    (copy of the nearest live column at the frame edge). `dead` is a bool vector over columns."""
    out = band.copy()
    live = np.flatnonzero(~dead)
    if live.size == 0:
        return out
    for c in np.flatnonzero(dead):
        i = int(np.searchsorted(live, c))
        left = int(live[i - 1]) if i > 0 else None
        right = int(live[i]) if i < live.size else None
        if left is not None and right is not None:
            w = (c - left) / float(right - left)
            col = (1.0 - w) * band[:, left].astype(np.float32) + w * band[:, right].astype(np.float32)
            out[:, c] = np.clip(np.rint(col), 0, SAT_DN).astype(band.dtype)
        else:
            out[:, c] = band[:, left if left is not None else right]
    return out


class Gate1:
    """Input validation and transient detection (same checks as the notebook reference, cell 13B).

    Rejects: wrong shape / channel count, non-finite values, <5% valid area, a constant band, a band that is >=50%
    saturated, a band with >20% dead columns. Flags: >2% saturated pixels. Counts isolated saturated pixels
    (transients) in every band; `scrub=True` replaces them by the 3x3 median. `repair_dead_columns=True` interpolates
    columns that are exactly zero in one band while the other bands hold data (OFF by default, not part of the
    reported evaluation unless the dead-column repair experiment has been run)."""

    def __init__(self, scrub: bool = False, max_saturated_frac: float = 0.02,
                 min_valid_frac: float = 0.05, reject_saturated_frac: float = 0.50,
                 max_dead_col_frac: float = 0.20, repair_dead_columns: bool = False,
                 names: Tuple[str, ...] = BAND_NAMES):
        self.names = tuple(names)
        self.scrub = bool(scrub)
        self.repair_dead_columns = bool(repair_dead_columns)
        self.max_saturated_frac = max_saturated_frac
        self.min_valid_frac = min_valid_frac
        self.reject_saturated_frac = reject_saturated_frac
        self.max_dead_col_frac = max_dead_col_frac

    def _reject(self, dn: np.ndarray, reason: str) -> Dict[str, Any]:
        return {
            'status': 'REJECT', 'reasons': [reason], 'flags': FLAG_INPUT_REJECTED, 'dn': dn,
            'n_transients': 0, 'n_dead_columns': 0, 'n_dead_columns_repaired': 0,
            'sat_frac': 0.0, 'dead_col_frac': 0.0, 'valid_frac': 0.0,
        }

    def validate(self, dn: np.ndarray) -> Dict[str, Any]:
        dn = np.asarray(dn)
        C = len(self.names)
        if dn.ndim != 3 or dn.shape != (C, FRAME_SIZE, FRAME_SIZE):
            return self._reject(dn, f'bad shape {dn.shape}, expected {(C, FRAME_SIZE, FRAME_SIZE)}')
        if np.issubdtype(dn.dtype, np.floating):
            if not np.isfinite(dn).all():
                return self._reject(dn, 'non-finite input values')
            dn = np.clip(dn, 0, SAT_DN)
        dn = dn.astype(np.uint16, copy=False)

        valid = (dn > 0).any(0)                      # zero-padded / no-data border is zero in every band
        valid_frac = float(valid.mean())
        if valid_frac < self.min_valid_frac:
            return self._reject(dn, f'valid area {valid_frac:.1%} < {self.min_valid_frac:.0%}')

        reasons, flags = [], 0
        sat_max = dead_max = 0.0
        dead_masks: Dict[int, np.ndarray] = {}
        for k, nm in enumerate(self.names):
            band = dn[k]
            if band[valid].min() == band[valid].max():
                return self._reject(dn, f'band {nm} is constant (dead channel)')
            sat = float((band >= SAT_DN).mean())
            sat_max = max(sat_max, sat)
            if sat >= self.reject_saturated_frac:
                return self._reject(dn, f'band {nm} saturated on {sat:.0%} of pixels')
            if sat > self.max_saturated_frac:
                flags |= FLAG_SATURATED_INPUT
                reasons.append(f'band {nm}: {sat:.1%} saturated pixels')
            if C > 1:
                others_have_data = (np.delete(dn, k, axis=0) > 0).any(axis=(0, 1))     # per column
                dead = (band == 0).all(axis=0) & others_have_data
                dead_max = max(dead_max, float(dead.mean()))
                if dead.mean() > self.max_dead_col_frac:
                    return self._reject(dn, f'band {nm}: {dead.mean():.0%} dead columns')
                if dead.any():
                    dead_masks[k] = dead
                    reasons.append(f'band {nm}: {int(dead.sum())} dead columns'
                                   + (' (repaired)' if self.repair_dead_columns else ' (not repaired)'))

        out = dn
        n_dead_total = int(sum(int(m.sum()) for m in dead_masks.values()))
        n_repaired = 0
        if self.repair_dead_columns and dead_masks:
            out = dn.copy()
            for k, dead in dead_masks.items():
                out[k] = repair_dead_columns_band(dn[k], dead)
                n_repaired += int(dead.sum())
            flags |= FLAG_INPUT_REPAIRED

        n_trans = 0
        kernel = np.ones((3, 3), np.int32)
        for k in range(C):
            band = out[k]
            sat_px = band >= SAT_DN
            if not sat_px.any():
                continue
            # isolated = saturated pixel with at most ONE saturated neighbour in its 3x3 window
            n_sat_nb = ndimage.convolve(sat_px.astype(np.int32), kernel, mode='constant') - 1
            spike = sat_px & (n_sat_nb <= 1)
            n = int(spike.sum())
            if n:
                n_trans += n
                if self.scrub:
                    if out is dn:
                        out = dn.copy()
                    med = median_filter(band, size=3, mode='nearest')
                    out[k][spike] = med[spike]
        if n_trans:
            reasons.append(f'{n_trans} isolated saturated pixel(s)' + (' scrubbed' if self.scrub else ' (not scrubbed)'))
            if self.scrub:
                flags |= FLAG_TRANSIENTS_SCRUBBED
        return {
            'status': 'OK', 'reasons': reasons, 'flags': flags, 'dn': out,
            'n_transients': n_trans, 'n_dead_columns': n_dead_total, 'n_dead_columns_repaired': n_repaired,
            'sat_frac': sat_max, 'dead_col_frac': dead_max, 'valid_frac': valid_frac,
        }


# ───────────────── Stage 2: Preprocessing ─────────────────
def preprocess_dn(dn: np.ndarray) -> np.ndarray:
    """Fixed radiometric preprocessing: linear DN -> TOA reflectance.
    Input: dn uint16 [3, 256, 256] in order [C7, C6, C2]; output: float32 clipped to [0.0, 1.6]."""
    return np.clip(dn.astype(np.float32) * REFL_SCALE + REFL_OFFSET, 0.0, REFL_MAX)


# ───────────────── Stage 3: Inference engine ─────────────────
class InferenceEngine:
    def __init__(self, model_path: str, num_threads: int = 1):
        so = ort.SessionOptions()
        # On small cloud containers force 1 thread to avoid contention
        if num_threads <= 0 and (os.environ.get("RENDER") or os.environ.get("RENDER_INSTANCE_ID")):
            num_threads = 1
        # num_threads <= 0 means "all cores" (ONNX Runtime value 0)
        intra = int(num_threads) if num_threads > 0 else 0
        so.intra_op_num_threads = intra
        so.inter_op_num_threads = 1
        self.threads_requested = intra
        self.threads_label = (f"{intra} thread" + ("s" if intra != 1 else "")) if intra > 0 \
            else f"all cores ({os.cpu_count() or 1} logical)"
        so.enable_cpu_mem_arena = False          # avoids a large pre-allocated arena (saves RAM on small hosts)
        self.session = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        dummy = np.zeros((1, N_CHANNELS, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)      # warm-up run
        self.session.run(None, {self.input_name: dummy})

    def run(self, x: np.ndarray) -> np.ndarray:
        """x: float32 [1, 3, 256, 256] -> raw logits [256, 256]"""
        logits = self.session.run(None, {self.input_name: x})[0]
        return logits.squeeze()


# ───────────────── Stage 4: Calibration ─────────────────
def calibrate_probabilities(logits: np.ndarray, temperature: float = TEMPERATURE_DEFAULT) -> np.ndarray:
    """Temperature-scaled sigmoid: p = sigmoid(z / T)."""
    return expit(logits / float(temperature))


# ───────────────── Stage 5: Event extraction ─────────────────
def extract_events(prob: np.ndarray, dn: np.ndarray, threshold: float = THRESHOLD_DEFAULT,
                   min_pixels: int = MIN_EVENT_PIXELS, geo_transform: Optional[Any] = None,
                   crs: Optional[Any] = None,
                   gsd_m: int = GSD_M_DEFAULT) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """8-connected components -> events, sorted by (confidence, area) descending. ALL events are returned;
    the cap of MAX_EVENTS and the dropped-event counter are applied by encode_packet().

    confidence : mean temperature-scaled probability inside the component (raw, no isotonic table)
    metric     : 95th percentile of C7 (SWIR2) reflectance in the component minus the median reflectance of a
                 5-pixel ring around it (same definition as the notebook), in reflectance x 10000
    area       : pixels x gsd^2 in 0.1-ha units, floored at 1 (0.1 ha) so a real event is never encoded as 0 ha
    """
    from io_tiff import get_wgs84_transformer

    binary_mask = (prob >= threshold).astype(np.uint8)
    labels, _ = ndimage.label(binary_mask, structure=_S8)
    slices = ndimage.find_objects(labels)

    events: List[Dict[str, Any]] = []
    c7_refl = np.clip(dn[0].astype(np.float32) * REFL_SCALE + REFL_OFFSET, 0.0, REFL_MAX)

    transformer = None
    if crs is not None and not getattr(crs, 'is_geographic', False):
        try:
            crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)
            transformer = get_wgs84_transformer(crs_str)
        except Exception:
            transformer = None

    geo_ok = (geo_transform is not None and crs is not None
              and not getattr(geo_transform, 'is_identity', False)
              and (getattr(crs, 'is_geographic', False) or transformer is not None))

    for eid, slc in enumerate(slices, 1):
        if slc is None:
            continue
        sub_mask = (labels[slc] == eid)
        area_px = int(sub_mask.sum())
        if area_px < min_pixels:
            continue

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        ys, xs = np.where(sub_mask)
        centroid_y = y0 + float(ys.mean())
        centroid_x = x0 + float(xs.mean())

        # Geolocation: the affine transform maps pixel CORNERS, hence the +0.5 to reach the pixel centre.
        # Valid only with a real CRS and a non-identity transform (rasterio returns identity for un-georeferenced files).
        if geo_ok:
            proj_x, proj_y = geo_transform * (centroid_x + 0.5, centroid_y + 0.5)
            if transformer is not None:
                lon, lat = transformer.transform(proj_x, proj_y)
            else:
                lon, lat = proj_x, proj_y
            lat_e7 = int(round(float(np.clip(lat, -90.0, 90.0)) * 1e7))
            lon_e7 = int(round(float(np.clip(lon, -180.0, 180.0)) * 1e7))
            flags = EVENT_FLAG_SWIR_METRIC
        else:
            lat_e7, lon_e7 = 0, 0
            flags = EVENT_FLAG_SWIR_METRIC | EVENT_FLAG_NO_GEO
        flags |= EVENT_FLAG_GEO_UNCERT_UNKN          # the uncertainty field always carries 65535 = unknown here

        # Area in 0.1-ha units (1 px at 30 m = 0.09 ha = 0.9 units), floored at 1
        area_ha = area_px * (gsd_m ** 2) / 10000.0
        area_ha_x10 = int(np.clip(int(round(area_px * (gsd_m ** 2) / 1000.0)), 1, 65535))

        mean_conf = float(prob[slc][sub_mask].mean())
        conf_u16 = int(round(float(np.clip(mean_conf, 0.0, 1.0)) * 65535))

        # Contrast metric on a crop with a RING_PX margin (exact: the ring never leaves the crop)
        cy0, cy1 = max(0, y0 - RING_PX), min(FRAME_SIZE, y1 + RING_PX)
        cx0, cx1 = max(0, x0 - RING_PX), min(FRAME_SIZE, x1 + RING_PX)
        comp = labels[cy0:cy1, cx0:cx1] == eid
        ring = ndimage.binary_dilation(comp, structure=_S8, iterations=RING_PX) & ~comp
        field = c7_refl[cy0:cy1, cx0:cx1]
        contrast = (float(np.percentile(field[comp], 95) - np.median(field[ring])) if ring.any() else 0.0)
        contrast = max(contrast, 0.0)
        metric_u16 = int(np.clip(round(contrast * 10000.0), 0, 65535))

        events.append({
            'event_id': eid,
            'lat_e7': lat_e7, 'lon_e7': lon_e7,
            'x': int(x0), 'y': int(y0), 'w': int(x1 - x0), 'h': int(y1 - y0),
            'area_px': area_px,
            'area_ha': round(area_ha, 2),
            'area_ha_x10': area_ha_x10,
            'confidence': round(mean_conf, 4),
            'confidence_u16': conf_u16,
            'metric_u16': metric_u16,
            'contrast_val': round(contrast, 3),
            'gsd_m_u16': int(gsd_m),
            'geo_uncertainty_m_u16': GEO_UNCERT_UNKNOWN_M,
            'flags': int(flags),
        })

    events.sort(key=lambda e: (e['confidence_u16'], e['area_ha_x10']), reverse=True)
    for rank, e in enumerate(events, 1):            # event_id = rank (1 = most confident)
        e['event_id'] = rank
    return events, binary_mask


# ───────────────── Stage 6: Gate 2 ─────────────────
class Gate2:
    """Output sanity checks. This gate verifies the integrity of the model output (valid probabilities, event boxes
    inside the frame, positive area) and warns when more than 5% of the tile is flagged. It does NOT test thermal
    contrast, cloud, sun glint or any other physical plausibility."""

    def __init__(self, threshold: float = THRESHOLD_DEFAULT, max_fire_frac: float = 0.05):
        self.threshold = threshold
        self.max_fire_frac = max_fire_frac

    def validate(self, prob: np.ndarray, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not np.isfinite(prob).all() or prob.min() < -1e-3 or prob.max() > 1 + 1e-3:
            return {'status': 'REJECT', 'reasons': ['Probability map values invalid'], 'flags': FLAG_OUTPUT_REJECTED}
        for e in events:
            if e['x'] + e['w'] > FRAME_SIZE or e['y'] + e['h'] > FRAME_SIZE or e['area_ha_x10'] <= 0:
                return {'status': 'REJECT', 'reasons': ['Event geometry or area invalid'], 'flags': FLAG_OUTPUT_REJECTED}
        fire_fraction = float((prob >= self.threshold).mean())
        if fire_fraction > self.max_fire_frac:
            return {'status': 'WARN', 'reasons': [f'Large detected extent: {fire_fraction:.1%} of the tile'],
                    'flags': FLAG_GATE2_WARN}
        return {'status': 'OK', 'reasons': [], 'flags': 0}


# ───────────────── Packet serialisation and parsing ─────────────────
def encode_packet(frame_id: int, model_id: int, timestamp_ms: int,
                  events: List[Dict[str, Any]], flags: int = 0) -> bytes:
    """Serialises the compact alert packet (version 3, little-endian, CRC32).

    At most MAX_EVENTS events are written (the list must be sorted, most important first). If events are dropped, the
    number dropped (saturating at 15) is stored in the upper 4 bits of the LAST event record's flags byte, which are
    unused in version 3: the layout and the packet size do not change."""
    sent = list(events[:MAX_EVENTS])
    dropped = max(0, len(events) - len(sent))

    pkt = bytearray(struct.pack(
        HEADER_FMT, PKT_MAGIC, PKT_VERSION, int(model_id) & 0xFF,
        int(frame_id) & 0xFFFFFFFF, int(timestamp_ms) & 0xFFFFFFFFFFFFFFFF,
        len(sent) & 0xFF, int(flags) & 0xFF))

    for i, e in enumerate(sent):
        ef = int(e['flags']) & EVENT_FLAG_MASK
        if dropped and i == len(sent) - 1:
            ef |= (min(dropped, 15) << EVENT_DROPPED_SHIFT)
        pkt.extend(struct.pack(
            EVENT_FMT,
            e['lat_e7'], e['lon_e7'], e['x'], e['y'], e['w'], e['h'],
            e['area_ha_x10'], e['confidence_u16'], e['metric_u16'],
            e['gsd_m_u16'], e['geo_uncertainty_m_u16'], ef))

    pkt.extend(struct.pack('<I', zlib.crc32(bytes(pkt)) & 0xFFFFFFFF))
    return bytes(pkt)


def decode_packet(data: bytes) -> Dict[str, Any]:
    """Parses a packet. A CRC or length problem does not raise: it is reported in `crc_valid` / `length_valid`
    (the notebook decoder raises instead). Bad magic / version / minimum length do raise."""
    if len(data) < HEADER_BYTES + 4:
        raise ValueError(f'Packet length {len(data)} too short (minimum {HEADER_BYTES + 4} B)')

    body = data[:-4]
    expected_crc = struct.unpack('<I', data[-4:])[0]
    crc_valid = (expected_crc == (zlib.crc32(body) & 0xFFFFFFFF))

    magic, ver, mid, fid, acq_ts, n_events, flags = struct.unpack_from(HEADER_FMT, body, 0)
    if magic != PKT_MAGIC or ver != PKT_VERSION:
        raise ValueError(f'Invalid packet magic/version: magic=0x{magic:02X}, ver={ver}')
    length_valid = (len(body) == HEADER_BYTES + n_events * EVENT_BYTES)

    events = []
    off = HEADER_BYTES
    names = ['lat_e7', 'lon_e7', 'x', 'y', 'w', 'h', 'area_ha_x10', 'confidence_u16',
             'metric_u16', 'gsd_m_u16', 'geo_uncertainty_m_u16', 'flags']
    events_dropped = 0
    for _ in range(n_events):
        if off + EVENT_BYTES > len(body):
            break
        vals = struct.unpack_from(EVENT_FMT, body, off)
        off += EVENT_BYTES
        e = dict(zip(names, vals))
        events_dropped = max(events_dropped, e['flags'] >> EVENT_DROPPED_SHIFT)
        e['flags'] = e['flags'] & EVENT_FLAG_MASK
        e['lat'] = round(e['lat_e7'] / 1e7, 5)
        e['lon'] = round(e['lon_e7'] / 1e7, 5)
        e['confidence'] = round(e['confidence_u16'] / 65535.0, 4)
        e['area_ha'] = round(e['area_ha_x10'] / 10.0, 2)
        e['contrast_val'] = round(e['metric_u16'] / 10000.0, 4)      # C7 reflectance contrast (event p95 - ring median)
        events.append(e)

    return {
        'magic': f'0x{magic:02X}',
        'version': ver,
        'model_id': mid,
        'frame_id': fid,
        'timestamp_ms': acq_ts,
        'n_events': n_events,
        'events_dropped': events_dropped,
        'flags': flags,
        'flags_hex': f'0x{flags:02X}',
        'events': events,
        'crc32': f'0x{expected_crc:08X}',
        'crc_valid': crc_valid,
        'length_valid': length_valid,
        'packet_bytes': len(data),
    }


# ───────────────── Master pipeline class ─────────────────
class EndToEndPipeline:
    def __init__(self, model_path: str, num_threads: int = 1,
                 threshold: float = THRESHOLD_DEFAULT,
                 temperature: float = TEMPERATURE_DEFAULT,
                 gate1_scrub: bool = False,
                 gate1_repair_dead_columns: bool = False,
                 heartbeat_enabled: bool = True,
                 heartbeat_every_n_frames: int = HEARTBEAT_EVERY_N_FRAMES):
        # heartbeat_enabled=True reproduces the onboard behaviour (21 B link-health packet on the first frame and every
        # N-th frame). The web demo creates its pipelines with heartbeat_enabled=False so that a fire-free tile shows
        # what matters: nothing is transmitted.
        self.heartbeat_enabled = bool(heartbeat_enabled)
        self.heartbeat_every_n_frames = int(heartbeat_every_n_frames)
        self.engine = InferenceEngine(model_path, num_threads=num_threads)
        self.gate1 = Gate1(scrub=gate1_scrub, repair_dead_columns=gate1_repair_dead_columns)
        self.gate2 = Gate2(threshold=threshold)
        self.threshold = threshold
        self.temperature = temperature
        self.frames_processed = 0

    def process_frame(self, raw_bands: np.ndarray, frame_id: int = 0,
                      geo_transform: Optional[Any] = None,
                      crs: Optional[Any] = None,
                      gsd_m: int = GSD_M_DEFAULT,
                      timestamp_ms: int = 0) -> Dict[str, Any]:
        """Runs the six stages on a raw uint16 [3, 256, 256] frame ordered [C7, C6, C2]."""
        self.frames_processed += 1
        timings: Dict[str, float] = {}
        base_flags = FLAG_TIME_UNAVAILABLE if int(timestamp_ms) <= 0 else 0

        t0 = time.perf_counter_ns()
        g1 = self.gate1.validate(raw_bands)
        timings['gate1_ms'] = (time.perf_counter_ns() - t0) / 1e6

        if g1['status'] == 'REJECT':
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, [], flags=base_flags | g1['flags'])
            timings['total_ms'] = timings['gate1_ms']
            return {
                'status': 'GATE1_REJECT', 'gate1_info': g1, 'gate2_info': None,
                'events': [], 'n_events_total': 0, 'n_events_dropped': 0,
                'packet': pkt, 'packet_json': decode_packet(pkt),
                'prob_map': None, 'binary_mask': None, 'timings': timings,
                'raw_bytes': RAW_BYTES_PER_TILE, 'packet_bytes': len(pkt),
                'reduction_ratio': RAW_BYTES_PER_TILE / len(pkt),
            }

        t0 = time.perf_counter_ns()
        x = preprocess_dn(g1['dn'])[np.newaxis, ...]
        timings['preprocess_ms'] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        logits = self.engine.run(x)
        timings['infer_ms'] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        prob_map = calibrate_probabilities(logits, self.temperature)
        timings['calibration_ms'] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        events, binary_mask = extract_events(
            prob_map, g1['dn'], threshold=self.threshold,
            geo_transform=geo_transform, crs=crs, gsd_m=gsd_m)
        timings['event_extract_ms'] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        g2 = self.gate2.validate(prob_map, events)
        timings['gate2_ms'] = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        flags = base_flags | g1['flags'] | g2['flags']
        n_total = len(events)
        n_dropped = max(0, n_total - MAX_EVENTS)

        if g2['status'] == 'REJECT':
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, [], flags=flags | FLAG_OUTPUT_REJECTED)
            status = 'GATE2_REJECT'
            n_total = n_dropped = 0
        elif n_total > 0:
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, events, flags=flags)
            status = 'ALERT'
        else:
            heartbeat_due = self.heartbeat_enabled and (
                self.frames_processed == 1 or
                (self.heartbeat_every_n_frames > 0 and self.frames_processed % self.heartbeat_every_n_frames == 0))
            if heartbeat_due:
                pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, [], flags=flags | FLAG_HEARTBEAT)
                status = 'HEARTBEAT_NO_FIRE'
            else:
                pkt = b''
                status = 'NO_FIRE_SILENCE'
        timings['encode_ms'] = (time.perf_counter_ns() - t0) / 1e6
        timings['total_ms'] = sum(timings.values())

        pkt_len = len(pkt)
        return {
            'status': status,
            'gate1_info': g1,
            'gate2_info': g2,
            'events': events,
            'n_events_total': n_total,
            'n_events_dropped': n_dropped,
            'prob_map': prob_map,
            'binary_mask': binary_mask,
            'packet': pkt,
            'packet_json': decode_packet(pkt) if pkt_len > 0 else None,
            'timings': timings,
            'raw_bytes': RAW_BYTES_PER_TILE,
            'packet_bytes': pkt_len,
            'reduction_ratio': RAW_BYTES_PER_TILE / pkt_len if pkt_len > 0 else float('inf'),
        }