#!/usr/bin/env python3
"""
IASTAM Edge Fire Detection — High-Fidelity Rendering & Visualization
======================================================================
Produces aerospace-grade visualization outputs:
  - SWIR False-Color Composite (C7 -> Red, C6 -> Green, C2 -> Blue)
  - Fire Probability Heatmap with calibrated colorbar
  - Binary Fire Mask (tau = 0.96)
  - Detection Overlay with event bounding boxes, confidence badges, and areas
  - Reference-mask error map (Green=TP, Red=FP, Blue=FN)
"""

from __future__ import annotations

import gc
import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

# Aesthetic theme colors
COLOR_BG = "#0d1117"
COLOR_TEXT = "#e6edf3"
COLOR_PRIMARY = "#ff6b35"
COLOR_CYAN = "#38bdf8"
COLOR_EMERALD = "#10b981"
COLOR_RED = "#ef4444"
COLOR_BLUE = "#3b82f6"


def _percentile_stretch(band: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Robust 2-98% percentile contrast stretch for satellite bands."""
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo = float(np.percentile(valid, pmin))
    hi = float(np.percentile(valid, pmax))
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.clip((band.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return stretched


def make_false_color(raw_3band: np.ndarray) -> np.ndarray:
    """SWIR false-color composite:
    R = C7 (SWIR2, 2.2 μm) -> Intense flame emission appears bright red/orange
    G = C6 (SWIR1, 1.6 μm) -> Vegetated/smoldering areas
    B = C2 (Blue, 0.48 μm) -> Clouds, water, smoke disambiguation
    Input: uint16 [3, H, W] in order [C7, C6, C2]
    Output: uint8 [H, W, 3] RGB
    """
    r = _percentile_stretch(raw_3band[0])  # C7
    g = _percentile_stretch(raw_3band[1])  # C6
    b = _percentile_stretch(raw_3band[2])  # C2

    rgb = np.stack([r, g, b], axis=-1)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return rgb_u8


def make_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Renders fire probability map with inferno colormap and calibrated scale."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    im = ax.imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.axis("off")
    ax.set_title("Calibrated Fire Probability (T=2.12)", color=COLOR_TEXT, fontsize=10, fontweight="bold", pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=COLOR_TEXT, labelcolor=COLOR_TEXT)
    cbar.outline.set_edgecolor("#30363d")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=8, color=COLOR_TEXT)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    plt.close("all")
    buf.seek(0)
    with Image.open(buf) as im:
        img_arr = np.array(im.convert("RGB"))
    buf.close()
    return img_arr


def make_binary_mask_img(binary_mask: np.ndarray, threshold: float = 0.96) -> np.ndarray:
    """Renders crisp binary fire mask at operating threshold."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.imshow(binary_mask, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fire_px = int(binary_mask.sum())
    title_text = f"Binary Fire Mask (τ={threshold:.2f}) · {fire_px:,} Fire Pixels"
    ax.set_title(title_text, color=COLOR_TEXT, fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    plt.close("all")
    buf.seek(0)
    with Image.open(buf) as im:
        img_arr = np.array(im.convert("RGB"))
    buf.close()
    return img_arr


def make_overlay(false_color: np.ndarray, binary_mask: np.ndarray,
                 events: List[Dict[str, Any]]) -> np.ndarray:
    """Renders composite with semi-transparent fire front and bounded event detections."""
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.imshow(false_color)

    # Superimpose semi-transparent fire mask
    fire_overlay = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
    fire_overlay[binary_mask > 0] = [1.0, 0.25, 0.0, 0.65]  # Bright neon orange-red
    ax.imshow(fire_overlay)

    # Draw bounding boxes for detected events
    for idx, ev in enumerate(events[:6]):  # Transmitted events (max 6)
        x, y, w, h = ev['x'], ev['y'], ev['w'], ev['h']
        rect = Rectangle(
            (x - 0.5, y - 0.5), w, h,
            linewidth=1.8, edgecolor=COLOR_EMERALD, facecolor="none",
            linestyle="--", alpha=0.95
        )
        ax.add_patch(rect)

        badge_text = f"#{ev['event_id']} ({ev['confidence']:.2f}) {ev['area_ha']}ha"
        ax.text(
            x, max(0, y - 3), badge_text,
            color=COLOR_EMERALD, fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COLOR_BG, alpha=0.85, edgecolor=COLOR_EMERALD, linewidth=0.8)
        )

    ax.axis("off")
    n_ev = len(events)
    n_tx = min(n_ev, 6)
    title = f"Events: {n_ev} Detected / {n_tx} Transmitted" if n_ev > 0 else "Detection Overlay: No Active Fire"
    ax.set_title(title, color=COLOR_TEXT, fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    plt.close("all")
    buf.seek(0)
    with Image.open(buf) as im:
        img_arr = np.array(im.convert("RGB"))
    buf.close()
    return img_arr


def make_gt_comparison(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """3-panel visual comparison: Prediction vs Reference Mask vs Error Map."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)

    # Panel 1: Prediction
    axes[0].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Predicted Fire ({int(pred_mask.sum())} px)", color=COLOR_TEXT, fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Reference mask
    axes[1].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Reference Mask ({int(gt_mask.sum())} px)", color=COLOR_TEXT, fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Error Classification Map
    # TP (Green) = [16, 185, 129]
    # FP (Red)   = [239, 68, 68]
    # FN (Blue)  = [59, 130, 246]
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)

    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    error_map[p & g]  = [16, 185, 129]  # True Positive (Emerald)
    error_map[p & ~g] = [239, 68, 68]   # False Positive (Red)
    error_map[~p & g] = [59, 130, 246]  # False Negative (Blue)

    axes[2].imshow(error_map)
    tp_cnt = int((p & g).sum())
    fp_cnt = int((p & ~g).sum())
    fn_cnt = int((~p & g).sum())
    axes[2].set_title(f"Error Map: TP={tp_cnt} · FP={fp_cnt} · FN={fn_cnt}", color=COLOR_TEXT, fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    plt.close("all")
    buf.seek(0)
    with Image.open(buf) as im:
        img_arr = np.array(im.convert("RGB"))
    buf.close()
    gc.collect()
    return img_arr
