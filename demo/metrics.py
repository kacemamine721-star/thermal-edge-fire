#!/usr/bin/env python3
"""
Evaluation metrics for the demo
===============================
Pixel-level and event-level metrics between a predicted binary mask and a reference mask. The reference is the
Pereira voting mask (agreement of SWIR-based algorithms), not field ground truth.

The event-level matching is the same as in the notebook (match_events): 8-connected components, a reference event
counts as found when any predicted pixel lies within `tolerance_px` pixels of it (8-connected dilation), and a
predicted event is false when no reference pixel lies within `tolerance_px` pixels of it.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import ndimage

_S8 = ndimage.generate_binary_structure(2, 2)


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    """Pixel precision / recall / F1 / IoU. Both inputs are 2-D binary arrays of the same shape.
    When the reference holds no fire, precision and recall are undefined; the values returned for that case (1.0 if
    nothing is predicted, else 0.0) are placeholders: check `has_reference_fire` before displaying them."""
    p = pred_mask.astype(bool).ravel()
    g = gt_mask.astype(bool).ravel()

    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())

    total_gt_fire = tp + fn
    total_pred_fire = tp + fp

    precision = tp / total_pred_fire if total_pred_fire > 0 else (1.0 if total_gt_fire == 0 else 0.0)
    recall = tp / total_gt_fire if total_gt_fire > 0 else (1.0 if total_pred_fire == 0 else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if total_gt_fire == 0 and total_pred_fire == 0 else 0.0)
    accuracy = (tp + tn) / len(p) if len(p) > 0 else 1.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_fire_pixels": total_gt_fire,
        "pred_fire_pixels": total_pred_fire,
        "has_reference_fire": total_gt_fire > 0,
    }


def compute_event_recall(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 3) -> Dict[str, Any]:
    """Event-level recall and false events, matching as in the notebook. `pred_mask` is the binary mask
    (not the list of events). `event_recall` is None when the reference holds no fire event."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    g_lab, n_g = ndimage.label(gt, structure=_S8)
    p_lab, n_p = ndimage.label(pred, structure=_S8)

    n_found = 0
    if n_g:
        near = ndimage.binary_dilation(pred, structure=_S8, iterations=tolerance_px) if n_p else pred
        found = np.atleast_1d(ndimage.maximum(near.astype(np.uint8), g_lab, np.arange(1, n_g + 1))) > 0
        n_found = int(found.sum())

    n_false = 0
    if n_p:
        gnear = ndimage.binary_dilation(gt, structure=_S8, iterations=tolerance_px) if n_g else gt
        touched = np.atleast_1d(ndimage.maximum(gnear.astype(np.uint8), p_lab, np.arange(1, n_p + 1))) > 0
        n_false = int((~touched).sum())

    return {
        "n_gt_events": int(n_g),
        "n_matched_events": n_found,
        "event_recall": round(n_found / n_g, 4) if n_g else None,
        "n_pred_events": int(n_p),
        "false_events": n_false,
    }