#!/usr/bin/env python3
"""
Patch app.py and render.py for the corrected pipeline / metrics modules
=======================================================================
Usage (from the demo folder, after copying the new pipeline.py, metrics.py, decode_packet.py):

    python apply_app_fixes.py --check     # shows what would change, writes nothing
    python apply_app_fixes.py             # patches app.py and render.py (backups: app.py.bak, render.py.bak)

Each patch is independent. A patch whose target text is not found is reported as SKIPPED (the file may already be
patched, or may differ from the version this script was written against). Nothing is written unless at least one patch
applies, and the exit code is 1 when anything was skipped, so you notice. Line endings (CRLF / LF) are preserved.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RESULTS = []   # (file, name, status)


def rep(text, name, old, new, fname, count=1):
    n = text.count(old)
    if n == 0:
        RESULTS.append((fname, name, "SKIPPED (text not found)"))
        return text
    if count and n != count:
        RESULTS.append((fname, name, f"SKIPPED (found {n} times, expected {count})"))
        return text
    RESULTS.append((fname, name, "ok"))
    return text.replace(old, new)


def span(text, name, start, end, new, fname):
    """Replaces everything from `start` (included) up to `end` (excluded)."""
    if text.count(start) != 1 or text.count(end) != 1:
        RESULTS.append((fname, name, "SKIPPED (markers not found exactly once)"))
        return text
    i, j = text.index(start), text.index(end)
    if j <= i:
        RESULTS.append((fname, name, "SKIPPED (markers out of order)"))
        return text
    RESULTS.append((fname, name, "ok"))
    return text[:i] + new + text[j:]


# ═════════════════════════════════════════════════════════════════ app.py
def patch_app(t):
    f = "app.py"

    # ---- wording ----
    t = rep(t, "docstring: Gate 1", "Stage 1: Gate 1 (Input Quality Screening & Radiation SEU Filter)",
            "Stage 1: Gate 1 (Input Quality Screening & Transient Filter)", f)
    t = rep(t, "docstring: packet version", "Compact Binary Alert Packet (V3)", "Compact Binary Alert Packet (version 3)", f)

    # ---- imports ----
    t = rep(t, "import FLAG_TIME_UNAVAILABLE, MAX_EVENTS",
            "    EMPTY_PACKET_BYTES,\n    MANIFEST_DATA,\n    MAX_PACKET_BYTES,",
            "    EMPTY_PACKET_BYTES,\n    FLAG_TIME_UNAVAILABLE,\n    MANIFEST_DATA,\n    MAX_EVENTS,\n    MAX_PACKET_BYTES,", f)

    # ---- model fallback path (no internal version name in code) ----
    t = rep(t, "model fallback via MODEL_FALLBACK_DIR",
            'if not INT8_MODEL_PATH.exists():\n'
            '    fallback_v7 = REPO_ROOT.parent / "U_NET_MODEL_V7" / "fire_detection_V7_outputs_20261008_202441"\n'
            '    INT8_MODEL_PATH = fallback_v7 / "unet_pereira3c_int8.onnx"\n'
            '    FP32_MODEL_PATH = fallback_v7 / "unet_pereira3c_fp32.onnx"',
            'if not INT8_MODEL_PATH.exists() and os.environ.get("MODEL_FALLBACK_DIR"):\n'
            '    _fallback_dir = Path(os.environ["MODEL_FALLBACK_DIR"])\n'
            '    INT8_MODEL_PATH = _fallback_dir / "unet_pereira3c_int8.onnx"\n'
            '    FP32_MODEL_PATH = _fallback_dir / "unet_pereira3c_fp32.onnx"', f)

    # ---- preset label ('sub-pixel' is wrong for a 1-pixel fire) ----
    t = rep(t, "preset label sub-pixel", '"[Sub-Pixel] Emerging Fire (1 Fire Pixel · Sensitivity Test)"',
            '"[Small] Emerging Fire (1 Fire Pixel · Sensitivity Test)"', f)

    # ---- reference mask: same rule as the training labels (>= 2 of 3 algorithms) ----
    t = rep(t, "mask suffixes", 'for suffix in ("_gt.tif", ".tif"):', 'for suffix in ("_gt.tif", "_voting.tif", ".tif"):', f)
    t = span(t, "reference mask voting rule",
             "    algo_masks = []\n    for algo in (",
             "\n\ndef format_hex_dump",
             '    algo_masks = []\n'
             '    seen = set()\n'
             '    candidates = sorted(list(MASK_DIR.glob(f"{scene}_*_{pid}.tif")) + list(MASK_DIR.glob(f"{scene}_{pid}_*.tif")))\n'
             '    for mpath in candidates:\n'
             '        low = mpath.name.lower()\n'
             '        if "kumar" in low and "roy" in low:\n'
             '            algo = "kumar_roy"\n'
             '        elif "murphy" in low:\n'
             '            algo = "murphy"\n'
             '        elif "schroeder" in low:\n'
             '            algo = "schroeder"\n'
             '        else:\n'
             '            continue\n'
             '        if algo in seen:\n'
             '            continue\n'
             '        seen.add(algo)\n'
             '        with rasterio.open(mpath) as ds:\n'
             '            algo_masks.append((ds.read(1) > 0).astype(np.uint8))\n'
             '\n'
             '    if not algo_masks:\n'
             '        return None\n'
             '\n'
             '    # Same rule as the training labels: at least 2 of the 3 algorithms must agree (fewer masks -> no voting fire).\n'
             '    return (np.sum(algo_masks, axis=0) >= 2).astype(np.uint8)\n', f)

    # ---- dissector ----
    t = rep(t, "dissector flag 4 name", '4: "TRANSIENTS_SCRUBBED"', '4: "INPUT_MODIFIED (scrub / dead-column repair)"', f)
    t = rep(t, "dissector location text",
            "        contrast = ev.get('contrast_val', 0.0)\n",
            "        contrast = ev.get('contrast_val', 0.0)\n"
            "        loc_txt = (\"Location unavailable (NO_GEO)\" if int(ev.get('flags', 0)) & 1\n"
            "                   else f\"Lat: {lat:+.4f}°, Lon: {lon:+.4f}°\")\n", f)
    t = rep(t, "dissector location cell", "**Lat: {lat:+.4f}°, Lon: {lon:+.4f}°**", "**{loc_txt}**", f)
    t = rep(t, "dissector dropped-events row",
            '    crc_status = badge("VALIDATED (Integrity Confirmed)", "optimal")',
            '    n_drop = int(packet_json.get("events_dropped", 0))\n'
            '    if n_drop:\n'
            '        md += (f"| `Event #{n_ev} flags[7:4]` | **Dropped-event counter** | `{n_drop}` | **{n_drop}{\'+\' if n_drop >= 15 else \'\'} event(s) not sent** "\n'
            '               f"| Detected events that did not fit in the {MAX_EVENTS}-event packet (counter saturates at 15) |\\n")\n\n'
            '    crc_status = badge("VALIDATED (Integrity Confirmed)", "optimal")', f)

    # ---- scene mode: the scene packet carries the most confident events, flagged as time-unavailable ----
    t = rep(t, "scene packet ordering and flags",
            "            packet_bytes = encode_packet(0, MODEL_ID, 0, all_events)",
            '            all_events.sort(key=lambda e: (e["confidence_u16"], e["area_ha_x10"]), reverse=True)\n'
            "            packet_bytes = encode_packet(0, MODEL_ID, 0, all_events, flags=FLAG_TIME_UNAVAILABLE)", f)

    # ---- metrics ----
    t = rep(t, "event recall takes the mask", "ev_metrics = compute_event_recall(events, gt_mask)",
            "ev_metrics = compute_event_recall(binary_mask, gt_mask)", f)
    new_metrics = '''        if not metrics['has_reference_fire']:
            frame_badge = badge("CORRECTLY SILENT", "optimal") if metrics['fp'] == 0 else badge("FALSE ALARM", "alert")
            metrics_summary_md = f"""### {SVG_TARGET} Verification Metrics (vs Pereira voting mask: SWIR-algorithm labels, not field ground truth)
*The reference mask holds no fire in this tile, so precision, recall, F1 and IoU are undefined.*

| Check | Result |
| :--- | :---: |
| **Predicted fire pixels (all false alarms)** | **{metrics['fp']:,} px** |
| **Predicted events (all false events)** | **{ev_metrics['n_pred_events']}** |
| **Frame decision** | {frame_badge} |

> Test-split reference: frame false-alarm rate **6.3 %** (53 of 835 fire-free test patches), i.e. about one fire-free tile in sixteen raises an alert.
"""
        else:
            def _vs_ref(value, ref):
                return badge("AT/ABOVE REF", "optimal") if value >= ref else badge("BELOW REF", "margin")

            prec_badge = _vs_ref(metrics['precision'], 0.9496)
            rec_badge = _vs_ref(metrics['recall'], 0.9489)
            f1_badge = _vs_ref(metrics['f1'], 0.9492)
            iou_badge = _vs_ref(metrics['iou'], 0.9034)
            all_found = ev_metrics['n_matched_events'] == ev_metrics['n_gt_events']
            ev_badge = badge("ALL FOUND", "optimal") if all_found else badge("PARTIAL", "margin")

            metrics_summary_md = f"""### {SVG_TARGET} Verification Metrics (vs Pereira voting mask: SWIR-algorithm labels, not field ground truth)
| Metric | This image | Test-split reference (4,781 patches) | Status |
| :--- | :---: | :---: | :---: |
| **Precision** | **{metrics['precision']:.4f}** | 0.9496 | {prec_badge} |
| **Recall** | **{metrics['recall']:.4f}** | 0.9489 | {rec_badge} |
| **F1-Score** | **{metrics['f1']:.4f}** | 0.9492 | {f1_badge} |
| **IoU (Jaccard)** | **{metrics['iou']:.4f}** | 0.9034 | {iou_badge} |
| **Event Recall** | **{ev_metrics['event_recall']*100:.1f}%** ({ev_metrics['n_matched_events']}/{ev_metrics['n_gt_events']} labelled clusters) | 95.5% (all test events) | {ev_badge} |

- **True Positives**: {metrics['tp']:,} px &nbsp;|&nbsp; **False Positives**: {metrics['fp']:,} px &nbsp;|&nbsp; **False Negatives**: {metrics['fn']:,} px &nbsp;|&nbsp; **False events**: {ev_metrics['false_events']}
- Single tiles scatter widely around the test-split averages (most fires are a few pixels), so one image is an illustration, not a result.
"""
            if metrics['recall'] == 1.0 and metrics['precision'] < 0.90:
                metrics_summary_md += f"""
> **Note on single-image metrics**:  
> Recall is **100.0%** (all {metrics['tp']} labelled fire pixels detected, none missed).  
> The {metrics['fp']} extra pixels are predicted fire outside the voting mask. They may be flame-front pixels the 3-algorithm vote missed, or false positives; one image cannot tell which, and the voting masks are SWIR-algorithm output, not field ground truth.
"""
'''
    t = span(t, "metrics block",
             "        # Badges use exactly the reference values shown in the table",
             "    else:\n        metrics_summary_md = f\"\"\"### {SVG_INFO} Operational Context",
             new_metrics, f)
    t = rep(t, "operational context wording", "*No ground truth mask is associated with this imagery",
            "*No reference mask is associated with this imagery", f)

    # ---- reduction panel ----
    t = rep(t, "events text variable",
            '    edge_adv_txt = "nothing to send" if packet_size == 0 else f"{alert_vs_zlib_x:,.0f}× smaller"\n',
            '    edge_adv_txt = "nothing to send" if packet_size == 0 else f"{alert_vs_zlib_x:,.0f}× smaller"\n'
            '    events_txt = (f"{MAX_EVENTS} most confident sent, {len(events) - MAX_EVENTS} dropped (count carried in the packet)"\n'
            '                  if len(events) > MAX_EVENTS else "All clusters encoded")\n', f)
    t = rep(t, "events cell", "{'Transmitted top 6 in alert' if len(events)>6 else 'All clusters encoded'}", "{events_txt}", f)

    # ---- telemetry ----
    t = rep(t, "stage label gate 1", '"Gate 1 (Quality & Radiation)"', '"Gate 1 (Input Checks)"', f)
    t = rep(t, "gate 1 notes row",
            '        telemetry_md += f"| **Inference threads** | **{pipeline.engine.threads_label}** | {model_variant} |\\n"\n',
            '        telemetry_md += f"| **Inference threads** | **{pipeline.engine.threads_label}** | {model_variant} |\\n"\n'
            '        _g1_notes = "; ".join((res.get("gate1_info") or {}).get("reasons", []))\n'
            '        if _g1_notes:\n'
            '            telemetry_md += f"| **Gate 1 notes** | {_g1_notes} | input checks (flag only unless repair is enabled) |\\n"\n', f)
    t = rep(t, "FP32 size in note", "514.8 KiB vs 1.89 MiB", "514.8 KiB vs 1.85 MiB", f)
    t = rep(t, "FP32 label", "FP32 (1.89 MB)", "FP32 (1.85 MiB)", f, count=0)

    # ---- benchmark table: only numbers that exist in the latency results ----
    t = rep(t, "benchmark min row", "| **Minimum Latency** | **`{min_t:.2f} ms`** | 58.42 ms (Inference floor) |",
            "| **Mean Latency** | **`{float(np.mean(latencies)):.2f} ms`** | 65.68 ms |", f)
    t = rep(t, "benchmark max row", "| **Maximum Latency** | **`{max_t:.2f} ms`** | 85.10 ms |",
            "| **99th Percentile (p99, 30 runs)** | **`{float(np.percentile(latencies, 99)):.2f} ms`** | 91.30 ms |", f)
    t = rep(t, "benchmark fps row", "| **Sustained Throughput** | **`{fps:.1f} FPS`** | 15.2 FPS |",
            "| **Throughput at p50** | **`{fps:.1f} FPS`** | 15.5 FPS |", f)

    # ---- header badges ----
    t = rep(t, "header fps", '64.38 ms <span class="badge-sub">15.2 FPS</span>', '64.38 ms <span class="badge-sub">15.5 FPS</span>', f)
    t = rep(t, "header reduction", '8,192× <span class="badge-sub">384 KiB → 48 B</span>',
            '5,132× <span class="badge-sub">384 KiB → 76.6 B mean alert</span>', f)

    # ---- controls ----
    t = rep(t, "int8 info", 'info="INT8 achieves 3.68x compression with 99.999% decision agreement"',
            'info="INT8 file is 3.7x smaller; alert decisions agree with PyTorch FP32 on 99% of 1,500 test frames"', f)
    t = rep(t, "scrub info", 'info="Applies 3×3 median scrub on isolated cosmic-ray/SEU spikes"',
            'info="Replaces isolated saturated pixels (possible radiation hits / hot pixels) by the 3×3 median; can also erase a real 1-pixel fire"', f)
    t = rep(t, "threshold info", "determined from manifest_v7.json", "read from the model manifest", f)
    t = rep(t, "dropdown label", "Select Curated Satellite Patch (with Ground Truth)", "Select Curated Satellite Patch (with reference mask)", f)
    t = rep(t, "gt view label", "Ground Truth Evaluation (Prediction vs GT vs Error Map)",
            "Reference-Mask Evaluation (Prediction vs Voting Mask vs Error Map)", f)
    return t


# ═════════════════════════════════════════════════════════════════ render.py
def patch_render(t):
    f = "render.py"
    t = rep(t, "stale comment", "    # Fire enhancement: actively boost fire front visibility in false color\n", "", f)
    t = rep(t, "docstring ground truth", "  - Ground Truth Error Map (Green=TP, Red=FP, Blue=FN)", "  - Reference-mask error map (Green=TP, Red=FP, Blue=FN)", f)
    t = rep(t, "panel 2 title", 'f"Ground Truth ({int(gt_mask.sum())} px)"', 'f"Reference Mask ({int(gt_mask.sum())} px)"', f)
    t = rep(t, "function docstring", "Prediction vs Ground Truth vs Classification Error Map", "Prediction vs Reference Mask vs Error Map", f)
    return t


def process(name, fn, check):
    path = DEMO_DIR / name
    if not path.exists():
        print(f"[{name}] not found next to this script, skipped")
        RESULTS.append((name, "file", "SKIPPED (file missing)"))
        return
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    new = fn(text)
    if new == text:
        print(f"[{name}] nothing changed")
        return
    if check:
        print(f"[{name}] would change (--check, nothing written)")
        return
    shutil.copyfile(path, str(path) + ".bak")
    out = new.replace("\n", "\r\n") if crlf else new
    path.write_bytes(out.encode("utf-8"))
    print(f"[{name}] patched ({'CRLF' if crlf else 'LF'} kept), backup: {name}.bak")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    process("app.py", patch_app, args.check)
    process("render.py", patch_render, args.check)
    skipped = [r for r in RESULTS if r[2] != "ok"]
    print()
    for fname, name, status in RESULTS:
        print(f"  {fname:<10} {name:<38} {status}")
    print(f"\n{len(RESULTS) - len(skipped)} applied, {len(skipped)} skipped")
    if skipped:
        print("Skipped patches: apply them by hand (see demo_corrections.md) or tell me which one failed.")
    sys.exit(1 if skipped else 0)


if __name__ == "__main__":
    main()
