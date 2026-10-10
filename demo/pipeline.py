#!/usr/bin/env python3
"""
IASTAM Final Production Model — Onboard Edge Fire-Detection Pipeline
=====================================================================
Ported directly from V7 Production Notebook (CELL 13, 13B & 14) and manifest_v7.json.

Stages:
  1. Gate 1: Input Quality Screening & Transient Noise (SEU) Gate
  2. Fixed Radiometric Preprocessing: Linear DN -> TOA Reflectance
  3. Neural Segmentation: Tiny U-Net INT8 / FP32 ONNX Runtime Engine
  4. Probability Calibration & Operating Thresholding (T=2.1228, tau=0.96)
  5. Event Extraction: 8-connectivity Connected Component Labeling & Geolocation
  6. Gate 2: Physical Plausibility Gate & Compact Binary Alert Packet (V3)
"""

from __future__ import annotations

import json
import math
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

# ───────────────── Manifest Constants Loading (TRD §1.6) ─────────────────
def load_manifest(manifest_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads manifest_v7.json to ensure all operational constants are read from the manifest (TRD §1.6)."""
    candidates = []
    if manifest_path:
        candidates.append(Path(manifest_path))
    candidates.extend([
        Path(__file__).resolve().parent / "models" / "manifest_v7.json",
        Path(__file__).resolve().parent.parent / "models" / "manifest_v7.json",
        Path.cwd() / "thermal-edge-fire" / "demo" / "models" / "manifest_v7.json",
        Path.cwd() / "demo" / "models" / "manifest_v7.json",
        Path.cwd() / "models" / "manifest_v7.json",
    ])
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

MANIFEST_DATA = load_manifest()

# ───────────────── Operational Constants (Synchronized with manifest_v7.json) ─────────────────
PKT_MAGIC = 0xF1
PKT_VERSION = int(MANIFEST_DATA.get("packet", {}).get("version", 3))
MODEL_ID = int(MANIFEST_DATA.get("model_ids", {}).get("pereira3c", 1))

HEADER_FMT = '<BBBIQBB'        # magic, version, model_id, frame_id, acq_ts, n_events, flags
EVENT_FMT  = '<iiHHHHHHHHHB'   # lat_e7, lon_e7, x, y, w, h, area_ha_x10, conf_u16, metric_u16, gsd_m, geo_uncert, flags

HEADER_BYTES = int(MANIFEST_DATA.get("packet", {}).get("header_bytes", struct.calcsize(HEADER_FMT)))
EVENT_BYTES  = int(MANIFEST_DATA.get("packet", {}).get("event_bytes", struct.calcsize(EVENT_FMT)))
MAX_EVENTS   = int(MANIFEST_DATA.get("packet", {}).get("max_events", 6))
MAX_PACKET_BYTES = int(MANIFEST_DATA.get("packet", {}).get("max_alert_packet_bytes", HEADER_BYTES + MAX_EVENTS * EVENT_BYTES + 4))
EMPTY_PACKET_BYTES = int(MANIFEST_DATA.get("packet", {}).get("max_empty_or_heartbeat_bytes", HEADER_BYTES + 4))
SINGLE_EVENT_BYTES = HEADER_BYTES + EVENT_BYTES + 4

RAW_BYTES_PER_TILE = int(MANIFEST_DATA.get("pipeline", {}).get("raw_bytes_per_frame_selected_bands", 3 * 256 * 256 * 2))
HEARTBEAT_EVERY_N_FRAMES = int(MANIFEST_DATA.get("heartbeat", {}).get("every_n_frames", 30))

TEMPERATURE_DEFAULT = float(MANIFEST_DATA.get("probability_calibration", {}).get("temperature", 2.122759125440246))
THRESHOLD_DEFAULT = float(MANIFEST_DATA.get("threshold", {}).get("benchmark_threshold", 0.96))
MIN_EVENT_PIXELS = 1

# Status & packet flags
FLAG_INPUT_REJECTED      = 1
FLAG_OUTPUT_REJECTED     = 2
FLAG_TRANSIENTS_SCRUBBED = 4
FLAG_SATURATED_INPUT     = 8
FLAG_GATE2_WARN          = 16
FLAG_HEARTBEAT           = 32
FLAG_TIME_UNAVAILABLE    = 64
FLAG_REFERENCE_ONLY      = 128

# Event flags
EVENT_FLAG_NO_GEO            = 1
EVENT_FLAG_GEO_UNCERT_UNKN   = 2
EVENT_FLAG_SWIR_METRIC       = 4
EVENT_FLAG_THERMAL_METRIC    = 8

FRAME_SIZE = 256
GSD_M_DEFAULT = 30
PIXEL_AREA_HA = (GSD_M_DEFAULT ** 2) / 10_000.0  # 0.09 ha per pixel

REFL_SCALE  = 2.0e-5
REFL_OFFSET = -0.1
REFL_MAX    = 1.6
SAT_DN      = 65535


# ───────────────── Stage 1: Gate 1 ─────────────────
class Gate1:
    """Input quality screening & transient radiation noise filter."""
    def __init__(self, scrub: bool = False, max_saturated_frac: float = 0.02,
                 min_valid_frac: float = 0.05, reject_saturated_frac: float = 0.50,
                 max_dead_col_frac: float = 0.20):
        self.scrub = bool(scrub)
        self.max_saturated_frac = max_saturated_frac
        self.min_valid_frac = min_valid_frac
        self.reject_saturated_frac = reject_saturated_frac
        self.max_dead_col_frac = max_dead_col_frac

    def _reject(self, dn: np.ndarray, reason: str) -> Dict[str, Any]:
        return {
            'status': 'REJECT',
            'reasons': [reason],
            'flags': FLAG_INPUT_REJECTED,
            'dn': dn,
            'n_transients': 0,
            'sat_frac': 0.0,
            'dead_col_frac': 0.0
        }

    def validate(self, dn: np.ndarray) -> Dict[str, Any]:
        dn = np.asarray(dn)
        C = dn.shape[0]
        if dn.ndim != 3 or dn.shape != (C, FRAME_SIZE, FRAME_SIZE):
            return self._reject(dn, f'Invalid shape {dn.shape}, expected ({C}, {FRAME_SIZE}, {FRAME_SIZE})')

        if np.issubdtype(dn.dtype, np.floating):
            if not np.isfinite(dn).all():
                return self._reject(dn, 'Non-finite input values detected')
            dn = np.clip(dn, 0, SAT_DN)
        dn = dn.astype(np.uint16, copy=False)

        valid = (dn > 0).any(0)
        valid_frac = float(valid.mean())
        if valid_frac < self.min_valid_frac:
            return self._reject(dn, f'Valid pixel area {valid_frac:.1%} < {self.min_valid_frac:.0%}')

        col_std = dn.std(axis=1)  # (C, W)
        # Columns that are zero in every band are no-data / zero-padding (scene-tiler edges), not dead detector columns.
        data_cols = (dn > 0).any(axis=(0, 1))                       # (W,)
        dead_cols = ((col_std < 1.0) & data_cols[None, :]).sum(axis=1)
        dead_frac = float(dead_cols.max() / FRAME_SIZE)
        if dead_frac > self.max_dead_col_frac:
            return self._reject(dn, f'Dead columns {dead_frac:.1%} exceed limit {self.max_dead_col_frac:.0%}')

        sat_frac = float((dn == SAT_DN).mean())
        if sat_frac > self.reject_saturated_frac:
            return self._reject(dn, f'Gross saturation {sat_frac:.1%} exceeds limit {self.reject_saturated_frac:.0%}')

        flags = 0
        reasons = []
        if sat_frac > self.max_saturated_frac:
            flags |= FLAG_SATURATED_INPUT
            reasons.append(f'Elevated saturation {sat_frac:.2%}')

        # Transient detection (single-pixel saturated spikes from cosmic rays/SEUs)
        c7 = dn[0]
        sat_mask = (c7 == SAT_DN)
        n_transients = 0
        cleaned_dn = dn

        if sat_mask.any():
            kernel = np.ones((3, 3), dtype=int)
            kernel[1, 1] = 0
            sat_neighbors = ndimage.convolve(sat_mask.astype(int), kernel, mode='constant', cval=0)
            isolated = sat_mask & (sat_neighbors <= 1)
            n_transients = int(isolated.sum())

            if n_transients > 0 and self.scrub:
                cleaned_c7 = c7.copy()
                med = median_filter(c7, size=3)
                cleaned_c7[isolated] = med[isolated]
                cleaned_dn = dn.copy()
                cleaned_dn[0] = cleaned_c7
                flags |= FLAG_TRANSIENTS_SCRUBBED
                reasons.append(f'Scrubbed {n_transients} isolated transient pixel(s)')

        return {
            'status': 'OK',
            'reasons': reasons,
            'flags': flags,
            'dn': cleaned_dn,
            'n_transients': n_transients,
            'sat_frac': sat_frac,
            'dead_col_frac': dead_frac,
            'valid_frac': valid_frac
        }


# ───────────────── Stage 2: Preprocessing ─────────────────
def preprocess_dn(dn: np.ndarray) -> np.ndarray:
    """Fixed radiometric preprocessing: Linear DN -> TOA Reflectance.
    Input: dn uint16 [3, 256, 256] in order [C7, C6, C2]
    Output: float32 [3, 256, 256] clipped to [0.0, 1.6]
    """
    return np.clip(dn.astype(np.float32) * REFL_SCALE + REFL_OFFSET, 0.0, REFL_MAX)


# ───────────────── Stage 3: Inference Engine ─────────────────
class InferenceEngine:
    def __init__(self, model_path: str, num_threads: int = 1):
        so = ort.SessionOptions()
        # In cloud containers (Render 512MB RAM / CFS quota), force 1 thread to avoid thread contention
        if num_threads <= 0 and (os.environ.get("RENDER") or os.environ.get("RENDER_INSTANCE_ID")):
            num_threads = 1
        # num_threads <= 0 means "all cores": ONNX Runtime's value 0 = use every available core.
        # (Before, <= 0 was silently turned into 1 thread, so the two UI modes ran identically.)
        intra = int(num_threads) if num_threads > 0 else 0
        so.intra_op_num_threads = intra
        so.inter_op_num_threads = 1
        self.threads_requested = intra
        self.threads_label = f"{intra} thread" + ("s" if intra != 1 else "") if intra > 0 else f"all cores ({os.cpu_count() or 1} logical)"
        # Prevent ONNX Runtime from pre-allocating a huge memory arena (saves ~80-120MB RAM)
        so.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # Warmup run
        dummy = np.zeros((1, 3, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def run(self, x: np.ndarray) -> np.ndarray:
        """x: float32 [1, 3, 256, 256] -> returns raw logits [256, 256]"""
        logits = self.session.run(None, {self.input_name: x})[0]
        return logits.squeeze()


# ───────────────── Stage 4: Calibration ─────────────────
def calibrate_probabilities(logits: np.ndarray, temperature: float = TEMPERATURE_DEFAULT) -> np.ndarray:
    """Temperature-scaled sigmoid probability calibration."""
    scaled = logits / float(temperature)
    return expit(scaled)          # numerically stable, no overflow warnings on very negative logits


# ───────────────── Stage 5: Event Extraction ─────────────────
def extract_events(prob: np.ndarray, dn: np.ndarray, threshold: float = THRESHOLD_DEFAULT,
                   min_pixels: int = MIN_EVENT_PIXELS, geo_transform: Optional[Any] = None,
                   crs: Optional[Any] = None,
                   gsd_m: int = GSD_M_DEFAULT) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """8-connectivity connected components labeling, bounding boxes & geolocation."""
    from io_tiff import get_wgs84_transformer

    binary_mask = (prob >= threshold).astype(np.uint8)
    _S8 = np.ones((3, 3), dtype=int)
    labels, n_components = ndimage.label(binary_mask, structure=_S8)
    slices = ndimage.find_objects(labels)

    events = []
    c7 = dn[0]  # SWIR2 band
    c7_median = float(np.median(c7))

    # Pre-create transformer once for the entire frame
    transformer = None
    if crs is not None and not getattr(crs, 'is_geographic', False):
        try:
            crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)
            transformer = get_wgs84_transformer(crs_str)
        except Exception:
            transformer = None

    for eid, slc in enumerate(slices, 1):
        if slc is None:
            continue
        sub_mask = (labels[slc] == eid)
        area_px = int(sub_mask.sum())
        if area_px < min_pixels:
            continue

        ys, xs = np.where(sub_mask)
        ymin = slc[0].start
        xmin = slc[1].start
        h = slc[0].stop - ymin
        w = slc[1].stop - xmin

        centroid_y = ymin + float(ys.mean())
        centroid_x = xmin + float(xs.mean())

        # Geolocation via affine transform and cached CRS reprojection.
        # Valid only with a real CRS and a non-identity transform: rasterio returns an identity transform for files
        # without georeferencing, which used to yield pixel coordinates reported as latitude/longitude.
        geo_ok = (geo_transform is not None and crs is not None
                  and not getattr(geo_transform, 'is_identity', False)
                  and (getattr(crs, 'is_geographic', False) or transformer is not None))
        if geo_ok:
            # +0.5: centroid is in pixel-index units; the affine transform maps pixel CORNERS
            proj_x, proj_y = geo_transform * (centroid_x + 0.5, centroid_y + 0.5)
            if transformer is not None:
                lon, lat = transformer.transform(proj_x, proj_y)
            else:
                lon, lat = proj_x, proj_y
            lat_e7 = int(round(np.clip(lat, -90.0, 90.0) * 1e7))
            lon_e7 = int(round(np.clip(lon, -180.0, 180.0) * 1e7))
            flags = EVENT_FLAG_SWIR_METRIC
        else:
            lat_e7, lon_e7 = 0, 0
            flags = EVENT_FLAG_SWIR_METRIC | EVENT_FLAG_NO_GEO | EVENT_FLAG_GEO_UNCERT_UNKN

        # Area in 0.1 hectare units (1 px = 0.09 ha = 0.9 units of 0.1 ha)
        area_ha = area_px * (gsd_m ** 2) / 10000.0
        area_ha_x10 = max(1, int(round(area_ha * 10.0)))

        # Calibrated confidence (mean probability over event cluster)
        mean_conf = float(prob[slc][sub_mask].mean())
        conf_u16 = int(round(np.clip(mean_conf, 0.0, 1.0) * 65535))

        # Local spectral contrast metric: SWIR2 peak reflectance above background * 10000
        c7_peak = float(c7[slc][sub_mask].max())
        contrast_dn = max(0.0, c7_peak - c7_median)
        contrast_rho = contrast_dn * REFL_SCALE
        metric_u16 = int(np.clip(contrast_rho * 10000.0, 0, 65535))

        events.append({
            'event_id': eid,
            'lat_e7': lat_e7,
            'lon_e7': lon_e7,
            'x': xmin,
            'y': ymin,
            'w': w,
            'h': h,
            'area_px': area_px,
            'area_ha': round(area_ha, 2),
            'area_ha_x10': area_ha_x10,
            'confidence': round(mean_conf, 4),
            'confidence_u16': conf_u16,
            'metric_u16': metric_u16,
            'contrast_val': round(contrast_rho, 3),
            'gsd_m_u16': gsd_m,
            'geo_uncertainty_m_u16': 65535,
            'flags': flags
        })

    # Sort descending by confidence
    events.sort(key=lambda e: e['confidence_u16'], reverse=True)
    return events, binary_mask


# ───────────────── Stage 6: Gate 2 ─────────────────
class Gate2:
    """Physical plausibility and area validation gate."""
    def __init__(self, threshold: float = THRESHOLD_DEFAULT, max_fire_frac: float = 0.05):
        self.threshold = threshold
        self.max_fire_frac = max_fire_frac

    def validate(self, prob: np.ndarray, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not np.isfinite(prob).all() or prob.min() < -1e-3 or prob.max() > 1 + 1e-3:
            return {'status': 'REJECT', 'reasons': ['Probability map values invalid'], 'flags': FLAG_OUTPUT_REJECTED}

        for e in events:
            if e['x'] + e['w'] > FRAME_SIZE or e['y'] + e['h'] > FRAME_SIZE or e['area_ha_x10'] <= 0:
                return {'status': 'REJECT', 'reasons': ['Event geometry or area non-physical'], 'flags': FLAG_OUTPUT_REJECTED}

        fire_fraction = float((prob >= self.threshold).mean())
        if fire_fraction > self.max_fire_frac:
            return {'status': 'WARN', 'reasons': [f'Implausibly large extent: {fire_fraction:.1%} of tile'], 'flags': FLAG_GATE2_WARN}

        return {'status': 'OK', 'reasons': [], 'flags': 0}


# ───────────────── Packet Serialization & Parsing ─────────────────
def encode_packet(frame_id: int, model_id: int, timestamp_ms: int,
                  events: List[Dict[str, Any]], flags: int = 0) -> bytes:
    """Serialize Compact Binary Alert Packet (Version 3, Little-Endian)."""
    transmitted_events = events[:MAX_EVENTS]
    n_events = len(transmitted_events)

    pkt = bytearray(struct.pack(
        HEADER_FMT,
        PKT_MAGIC,
        PKT_VERSION,
        int(model_id) & 0xFF,
        int(frame_id) & 0xFFFFFFFF,
        int(timestamp_ms) & 0xFFFFFFFFFFFFFFFF,
        n_events & 0xFF,
        int(flags) & 0xFF
    ))

    for e in transmitted_events:
        pkt.extend(struct.pack(
            EVENT_FMT,
            e['lat_e7'], e['lon_e7'],
            e['x'], e['y'], e['w'], e['h'],
            e['area_ha_x10'], e['confidence_u16'], e['metric_u16'],
            e['gsd_m_u16'], e['geo_uncertainty_m_u16'], e['flags']
        ))

    # Append 32-bit CRC checksum
    crc = zlib.crc32(bytes(pkt)) & 0xFFFFFFFF
    pkt.extend(struct.pack('<I', crc))
    return bytes(pkt)


def decode_packet(data: bytes) -> Dict[str, Any]:
    """Parse Compact Binary Alert Packet and verify CRC32."""
    if len(data) < HEADER_BYTES + 4:
        raise ValueError(f'Packet length {len(data)} too short (minimum {HEADER_BYTES + 4} B)')

    body = data[:-4]
    expected_crc = struct.unpack('<I', data[-4:])[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    crc_valid = (expected_crc == actual_crc)

    magic, ver, mid, fid, acq_ts, n_events, flags = struct.unpack_from(HEADER_FMT, body, 0)
    if magic != PKT_MAGIC or ver != PKT_VERSION:
        raise ValueError(f'Invalid packet magic/version: magic=0x{magic:02X}, ver={ver}')

    events = []
    off = HEADER_BYTES
    names = ['lat_e7', 'lon_e7', 'x', 'y', 'w', 'h', 'area_ha_x10', 'confidence_u16',
             'metric_u16', 'gsd_m_u16', 'geo_uncertainty_m_u16', 'flags']

    for _ in range(n_events):
        if off + EVENT_BYTES > len(body):
            break
        vals = struct.unpack_from(EVENT_FMT, body, off)
        off += EVENT_BYTES
        e_dict = dict(zip(names, vals))
        e_dict['lat'] = round(e_dict['lat_e7'] / 1e7, 5)
        e_dict['lon'] = round(e_dict['lon_e7'] / 1e7, 5)
        e_dict['confidence'] = round(e_dict['confidence_u16'] / 65535.0, 4)
        e_dict['area_ha'] = round(e_dict['area_ha_x10'] / 10.0, 2)
        events.append(e_dict)

    return {
        'magic': f'0x{magic:02X}',
        'version': ver,
        'model_id': mid,
        'frame_id': fid,
        'timestamp_ms': acq_ts,
        'n_events': n_events,
        'flags': flags,
        'flags_hex': f'0x{flags:02X}',
        'events': events,
        'crc32': f'0x{expected_crc:08X}',
        'crc_valid': crc_valid,
        'packet_bytes': len(data)
    }


# ───────────────── Master Pipeline Class ─────────────────
class EndToEndPipeline:
    def __init__(self, model_path: str, num_threads: int = 1,
                 threshold: float = THRESHOLD_DEFAULT,
                 temperature: float = TEMPERATURE_DEFAULT,
                 gate1_scrub: bool = False,
                 heartbeat_enabled: bool = True,
                 heartbeat_every_n_frames: int = HEARTBEAT_EVERY_N_FRAMES):
        # heartbeat_enabled=True reproduces the onboard behaviour (21 B link-health packet on the first frame and every
        # N-th frame). The demo creates single-frame pipelines with heartbeat_enabled=False so a fire-free tile
        # shows what matters: nothing is transmitted.
        self.heartbeat_enabled = bool(heartbeat_enabled)
        self.heartbeat_every_n_frames = int(heartbeat_every_n_frames)
        self.engine = InferenceEngine(model_path, num_threads=num_threads)
        self.gate1 = Gate1(scrub=gate1_scrub)
        self.gate2 = Gate2(threshold=threshold)
        self.threshold = threshold
        self.temperature = temperature
        self.frames_processed = 0

    def process_frame(self, raw_bands: np.ndarray, frame_id: int = 0,
                      geo_transform: Optional[Any] = None,
                      crs: Optional[Any] = None,
                      gsd_m: int = GSD_M_DEFAULT,
                      timestamp_ms: int = 0) -> Dict[str, Any]:
        """Executes full 6-stage pipeline on raw uint16 [3, 256, 256] bands."""
        self.frames_processed += 1
        timings = {}

        # Stage 1: Gate 1
        t0 = time.perf_counter_ns()
        g1 = self.gate1.validate(raw_bands)
        timings['gate1_ms'] = (time.perf_counter_ns() - t0) / 1e6

        if g1['status'] == 'REJECT':
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, [], flags=FLAG_INPUT_REJECTED)
            timings['total_ms'] = timings['gate1_ms']
            return {
                'status': 'GATE1_REJECT',
                'gate1_info': g1,
                'events': [],
                'packet': pkt,
                'packet_json': decode_packet(pkt),
                'prob_map': None,
                'binary_mask': None,
                'timings': timings,
                'raw_bytes': RAW_BYTES_PER_TILE,
                'packet_bytes': len(pkt),
                'reduction_ratio': RAW_BYTES_PER_TILE / len(pkt)
            }

        # Stage 2: Preprocessing
        t0 = time.perf_counter_ns()
        x = preprocess_dn(g1['dn'])[np.newaxis, ...]
        timings['preprocess_ms'] = (time.perf_counter_ns() - t0) / 1e6

        # Stage 3: Inference
        t0 = time.perf_counter_ns()
        logits = self.engine.run(x)
        timings['infer_ms'] = (time.perf_counter_ns() - t0) / 1e6

        # Stage 4: Calibration
        t0 = time.perf_counter_ns()
        prob_map = calibrate_probabilities(logits, self.temperature)
        timings['calibration_ms'] = (time.perf_counter_ns() - t0) / 1e6

        # Stage 5: Event Extraction
        t0 = time.perf_counter_ns()
        events, binary_mask = extract_events(
            prob_map, g1['dn'], threshold=self.threshold,
            geo_transform=geo_transform, crs=crs, gsd_m=gsd_m
        )
        timings['event_extract_ms'] = (time.perf_counter_ns() - t0) / 1e6

        # Stage 6: Gate 2
        t0 = time.perf_counter_ns()
        g2 = self.gate2.validate(prob_map, events)
        timings['gate2_ms'] = (time.perf_counter_ns() - t0) / 1e6

        # Packet encoding
        t0 = time.perf_counter_ns()
        flags = g1['flags'] | g2['flags']
        if timestamp_ms == 0:
            flags |= FLAG_TIME_UNAVAILABLE

        if g2['status'] == 'REJECT':
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, [], flags=flags | FLAG_OUTPUT_REJECTED)
            status = 'GATE2_REJECT'
        elif len(events) > 0:
            pkt = encode_packet(frame_id, MODEL_ID, timestamp_ms, events, flags=flags)
            status = 'ALERT'
        else:
            # Heartbeat on the 1st frame and every N-th frame (onboard link-health behaviour, optional in the demo)
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
        reduction = RAW_BYTES_PER_TILE / pkt_len if pkt_len > 0 else float('inf')

        return {
            'status': status,
            'gate1_info': g1,
            'gate2_info': g2,
            'events': events,
            'prob_map': prob_map,
            'binary_mask': binary_mask,
            'packet': pkt,
            'packet_json': decode_packet(pkt) if pkt_len > 0 else None,
            'timings': timings,
            'raw_bytes': RAW_BYTES_PER_TILE,
            'packet_bytes': pkt_len,
            'reduction_ratio': reduction
        }