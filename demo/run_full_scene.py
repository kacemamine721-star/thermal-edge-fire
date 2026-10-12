#!/usr/bin/env python3
"""Stream a full Landsat GeoTIFF through the IASTAM six-stage pipeline.

This tool is intentionally separate from the Gradio dashboard.  It reads one
256 x 256 window at a time, so it can verify a complete Landsat scene without
loading the scene or full-resolution visualisations into RAM.  Every emitted
per-tile packet is kept verbatim and indexed by a JSON manifest.

Input contract:
  * one raw Landsat Level-1 DN GeoTIFF with at least 10 bands (B1...B10), or
  * one three-band DN GeoTIFF ordered [C7, C6, C2].

Example (from the demo directory):
  python run_full_scene.py sample_data/custom_tif/LC08_192035_20260722_L1_DN_B1_to_B10.tif
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from io_tiff import to_uint16_dn
from pipeline import (
    FLAG_TRANSIENTS_SCRUBBED,
    MANIFEST_DATA,
    MODEL_ID,
    THRESHOLD_DEFAULT,
    EndToEndPipeline,
    decode_packet,
)
from render import make_binary_mask_img, make_false_color, make_heatmap, make_overlay


TILE_SIZE = 256
STAGE_KEYS = (
    "gate1_ms", "preprocess_ms", "infer_ms", "calibration_ms",
    "event_extract_ms", "gate2_ms", "encode_ms",
)
PREVIEW_MAX_SIDE = 1024


def selected_band_indexes(count: int) -> Tuple[int, int, int]:
    """Return rasterio's one-based indexes in the trained-model input order."""
    if count >= 10:
        return (7, 6, 2)  # Landsat B7 (SWIR2), B6 (SWIR1), B2 (Blue)
    if count == 3:
        return (1, 2, 3)  # Explicit three-band export contract: [C7, C6, C2]
    raise ValueError(
        f"Unsupported band count: {count}. Expected a 10+ band Landsat stack, "
        "or a three-band [C7, C6, C2] GeoTIFF."
    )


def iter_tiles(src: rasterio.DatasetReader, indexes: Tuple[int, int, int]) -> Iterator[Dict[str, Any]]:
    """Read and pad one tile at a time; no full-scene data array is allocated."""
    tile_id = 0
    for y0 in range(0, src.height, TILE_SIZE):
        for x0 in range(0, src.width, TILE_SIZE):
            valid_h = min(TILE_SIZE, src.height - y0)
            valid_w = min(TILE_SIZE, src.width - x0)
            window = Window(x0, y0, valid_w, valid_h)
            raw = to_uint16_dn(src.read(indexes, window=window))
            patch = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint16)
            patch[:, :valid_h, :valid_w] = raw
            yield {
                "tile_id": tile_id,
                "offset_x": x0,
                "offset_y": y0,
                "valid_w": valid_w,
                "valid_h": valid_h,
                "patch": patch,
                "transform": src.window_transform(window),
            }
            tile_id += 1


def json_safe(value: Any) -> Any:
    """Convert NumPy values retained in decoded packets to standard JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def resolve_model(model: str) -> Path:
    file_name = "unet_pereira3c_int8.onnx" if model == "int8" else "unet_pereira3c_fp32.onnx"
    model_path = DEMO_DIR / "models" / file_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def preview_shape(width: int, height: int) -> Tuple[int, int]:
    """Choose a bounded overview grid without changing the inference grid."""
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def preview_window(offset: int, valid_size: int, source_size: int, preview_size: int) -> Tuple[int, int]:
    """Map a source tile extent to a non-empty pixel range in the overview grid."""
    start = int(np.floor(offset * preview_size / source_size))
    end = int(np.ceil((offset + valid_size) * preview_size / source_size))
    return start, max(start + 1, min(preview_size, end))


def resize_tile(arr: np.ndarray, width: int, height: int, resampling: Image.Resampling) -> np.ndarray:
    """Resize a probability or binary tile for an overview without full-scene allocation."""
    return np.asarray(Image.fromarray(arr).resize((width, height), resample=resampling))


def write_preview_products(output_dir: Path, raw_preview: np.ndarray, probability_preview: np.ndarray,
                           mask_preview: np.ndarray, scene_events: list[Dict[str, Any]],
                           source_width: int, source_height: int, threshold: float) -> Dict[str, str]:
    """Write small overview products that preserve the full-scene inference result."""
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    false_color = make_false_color(raw_preview)
    scale_x = false_color.shape[1] / source_width
    scale_y = false_color.shape[0] / source_height
    display_events = []
    for event in scene_events:
        display_event = dict(event)
        display_event["x"] = int(round(event["x"] * scale_x))
        display_event["y"] = int(round(event["y"] * scale_y))
        display_event["w"] = max(1, int(round(event["w"] * scale_x)))
        display_event["h"] = max(1, int(round(event["h"] * scale_y)))
        display_events.append(display_event)

    products = {
        "false_color": ("false_color_overview.png", false_color),
        "probability": ("probability_overview.png", make_heatmap(probability_preview)),
        "binary_mask": ("binary_mask_overview.png", make_binary_mask_img(mask_preview, threshold=threshold)),
        "overlay": ("detection_overlay_overview.png", make_overlay(false_color, mask_preview, display_events)),
    }
    written = {}
    for name, (file_name, image) in products.items():
        path = preview_dir / file_name
        Image.fromarray(image).save(path)
        written[name] = str(path.relative_to(output_dir))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a complete GeoTIFF scene through IASTAM using bounded memory."
    )
    parser.add_argument("input", type=Path, help="Raw Level-1 DN GeoTIFF to process")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Artifact directory (default: <input-stem>_iastam_scene beside the input)",
    )
    parser.add_argument("--model", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime inference threads")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    parser.add_argument("--scrub", action="store_true", help="Enable Gate 1 transient-pixel scrubbing")
    parser.add_argument(
        "--max-tiles", type=int, default=0,
        help="Optional smoke-test limit; 0 means process the complete scene",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    if not source_path.is_file():
        parser.error(f"Input GeoTIFF not found: {source_path}")
    if args.threads == 0:
        parser.error("--threads must be non-zero; use a positive count or -1 for all cores")

    output_dir = (args.output.expanduser().resolve() if args.output else
                  source_path.with_name(f"{source_path.stem}_iastam_scene"))
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model(args.model)
    pipeline = EndToEndPipeline(
        model_path=str(model_path),
        num_threads=args.threads,
        threshold=args.threshold,
        gate1_scrub=args.scrub,
        heartbeat_enabled=False,
    )

    stage_totals: Dict[str, float] = defaultdict(float)
    stage_samples: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    gate1_reason_counts: Counter[str] = Counter()
    gate2_status_counts: Counter[str] = Counter()
    gate2_reason_counts: Counter[str] = Counter()
    gate1_transients_detected = 0
    gate1_transients_scrubbed = 0
    packet_manifest = []
    transmitted_events = []
    detected_events = 0
    packets_bytes = 0
    tiles_processed = 0
    started = time.perf_counter()

    with rasterio.open(source_path) as src:
        indexes = selected_band_indexes(src.count)
        total_tiles = ((src.width + TILE_SIZE - 1) // TILE_SIZE) * ((src.height + TILE_SIZE - 1) // TILE_SIZE)
        preview_w, preview_h = preview_shape(src.width, src.height)
        raw_preview = to_uint16_dn(src.read(
            indexes,
            out_shape=(3, preview_h, preview_w),
            resampling=Resampling.nearest,
        ))
        probability_preview = np.zeros((preview_h, preview_w), dtype=np.float32)
        mask_preview = np.zeros((preview_h, preview_w), dtype=np.uint8)
        scene_events = []
        print(
            f"Input: {src.width}x{src.height}, {src.count} bands, {src.dtypes[0]}, "
            f"{total_tiles} tiles; selected bands={indexes}",
            flush=True,
        )

        for tile in iter_tiles(src, indexes):
            result = pipeline.process_frame(
                raw_bands=tile["patch"],
                frame_id=tile["tile_id"],
                geo_transform=tile["transform"],
                crs=src.crs,
                gsd_m=max(1, int(round(abs(src.transform.a)))),
            )
            tiles_processed += 1
            status_counts[result["status"]] += 1
            detected_events += len(result["events"])
            gate1_info = result.get("gate1_info", {})
            gate1_reason_counts.update(gate1_info.get("reasons", []))
            gate1_transients_detected += int(gate1_info.get("n_transients", 0))
            if gate1_info.get("flags", 0) & FLAG_TRANSIENTS_SCRUBBED:
                gate1_transients_scrubbed += int(gate1_info.get("n_transients", 0))
            gate2_info = result.get("gate2_info")
            if gate2_info is not None:
                gate2_status_counts[gate2_info["status"]] += 1
                gate2_reason_counts.update(gate2_info.get("reasons", []))

            if result["prob_map"] is not None and result["binary_mask"] is not None:
                x0, x1 = preview_window(tile["offset_x"], tile["valid_w"], src.width, preview_w)
                y0, y1 = preview_window(tile["offset_y"], tile["valid_h"], src.height, preview_h)
                tile_prob = result["prob_map"][:tile["valid_h"], :tile["valid_w"]]
                tile_mask = result["binary_mask"][:tile["valid_h"], :tile["valid_w"]]
                overview_prob = resize_tile(tile_prob, x1 - x0, y1 - y0, Image.Resampling.NEAREST)
                overview_mask = resize_tile(tile_mask, x1 - x0, y1 - y0, Image.Resampling.NEAREST)
                probability_preview[y0:y1, x0:x1] = np.maximum(probability_preview[y0:y1, x0:x1], overview_prob)
                mask_preview[y0:y1, x0:x1] = np.maximum(mask_preview[y0:y1, x0:x1], overview_mask)

            for event in result["events"]:
                scene_event = dict(event)
                scene_event["x"] += tile["offset_x"]
                scene_event["y"] += tile["offset_y"]
                scene_events.append(scene_event)

            for key in STAGE_KEYS:
                if key in result["timings"]:
                    stage_totals[key] += float(result["timings"][key])
                    stage_samples[key] += 1

            packet = result["packet"]
            if packet:
                packet_name = f"tile_{tile['tile_id']:06d}.bin"
                packet_path = packet_dir / packet_name
                packet_path.write_bytes(packet)
                decoded = json_safe(decode_packet(packet))
                packets_bytes += len(packet)
                packet_manifest.append({
                    "tile_id": tile["tile_id"],
                    "grid_offset_px": [tile["offset_x"], tile["offset_y"]],
                    "valid_shape_px": [tile["valid_w"], tile["valid_h"]],
                    "status": result["status"],
                    "packet_file": f"packets/{packet_name}",
                    "packet_bytes": len(packet),
                    "crc_valid": decoded.get("crc_valid"),
                    "events_in_packet": decoded.get("n_events", 0),
                })
                for event in decoded.get("events", []):
                    event["tile_id"] = tile["tile_id"]
                    event["tile_offset_x"] = tile["offset_x"]
                    event["tile_offset_y"] = tile["offset_y"]
                    transmitted_events.append(event)

            if args.progress_every > 0 and (
                tiles_processed == 1 or tiles_processed % args.progress_every == 0 or tiles_processed == total_tiles
            ):
                elapsed = time.perf_counter() - started
                rate = tiles_processed / elapsed if elapsed else 0.0
                print(
                    f"Processed {tiles_processed}/{total_tiles} tiles "
                    f"({rate:.2f} tiles/s; packets={len(packet_manifest)})",
                    flush=True,
                )
            if args.max_tiles and tiles_processed >= args.max_tiles:
                print(f"Stopped after requested smoke-test limit of {args.max_tiles} tiles.", flush=True)
                break

        raw_selected_bytes = src.width * src.height * 3 * np.dtype(src.dtypes[0]).itemsize
        metadata = {
            "source_file": str(source_path),
            "source_size_bytes": source_path.stat().st_size,
            "source_shape": [src.count, src.height, src.width],
            "source_dtype": src.dtypes[0],
            "source_crs": str(src.crs) if src.crs else None,
            "source_transform": list(src.transform)[:6],
            "selected_input_bands": list(indexes),
            "selected_raw_bytes": raw_selected_bytes,
            "tile_size_px": TILE_SIZE,
            "tiles_available": total_tiles,
        }

    preview_files = write_preview_products(
        output_dir,
        raw_preview,
        probability_preview,
        mask_preview,
        scene_events,
        metadata["source_shape"][2],
        metadata["source_shape"][1],
        args.threshold,
    )

    elapsed_s = time.perf_counter() - started
    mean_stage_ms = {
        key: (stage_totals[key] / stage_samples[key] if stage_samples[key] else None)
        for key in STAGE_KEYS
    }
    report = {
        "run_complete": tiles_processed == metadata["tiles_available"],
        "metadata": metadata,
        "configuration": {
            "model": args.model,
            "model_id": MODEL_ID,
            "threads": args.threads,
            "threshold": args.threshold,
            "gate1_scrub": args.scrub,
        },
        "summary": {
            "tiles_processed": tiles_processed,
            "wall_time_seconds": elapsed_s,
            "throughput_tiles_per_second": tiles_processed / elapsed_s if elapsed_s else 0.0,
            "statuses": dict(status_counts),
            "events_detected_before_packet_limit": detected_events,
            "packets_emitted": len(packet_manifest),
            "packet_bytes_total": packets_bytes,
            "raw_to_packet_reduction": raw_selected_bytes / packets_bytes if packets_bytes else None,
        },
        "pipeline_execution_audit": {
            "implementation": {
                "gate1": "shape/type/range, valid-area, dead-column, saturation, and optional isolated saturated-pixel scrub",
                "conditioning": "fixed Landsat DN-to-TOA-reflectance transform",
                "gate2": "probability-range, event geometry/area, and per-tile 5% fire-occupancy plausibility check",
                "packet": "custom V3 packet: 17-byte header, up to six 27-byte events, CRC32",
            },
            "gate1": {
                "tiles_inspected": tiles_processed,
                "tiles_passed_to_preprocessing": stage_samples["preprocess_ms"],
                "tiles_rejected": status_counts["GATE1_REJECT"],
                "rejection_or_flag_reasons": dict(gate1_reason_counts),
                "isolated_transients_detected": gate1_transients_detected,
                "isolated_transients_scrubbed": gate1_transients_scrubbed,
            },
            "gate2": {
                "tiles_reached": sum(gate2_status_counts.values()),
                "statuses": dict(gate2_status_counts),
                "rejection_or_warning_reasons": dict(gate2_reason_counts),
            },
            "packet_output": {
                "packets_emitted": len(packet_manifest),
                "packet_bytes_total": packets_bytes,
                "crc_valid_packets": sum(1 for packet in packet_manifest if packet["crc_valid"]),
            },
        },
        "benchmark_reference": next(
            (item for item in MANIFEST_DATA["test_results_by_config"] if item["config"] == "pereira3c"),
            None,
        ),
        "stage_totals_ms": dict(stage_totals),
        "stage_mean_ms_per_reached_tile": mean_stage_ms,
        "preview_files": preview_files,
        "packets": packet_manifest,
        "transmitted_events": transmitted_events,
    }
    report_path = output_dir / "scene_manifest.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")

    print("\nComplete scene result", flush=True)
    print(f"  tiles: {tiles_processed}/{metadata['tiles_available']}", flush=True)
    print(f"  wall time: {elapsed_s:.2f} s", flush=True)
    print(f"  statuses: {dict(status_counts)}", flush=True)
    print(f"  packets: {len(packet_manifest)} ({packets_bytes:,} B total)", flush=True)
    print(f"  manifest: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())