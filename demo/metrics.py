#!/usr/bin/env python3
"""
IASTAM Edge Fire Detection — Evaluation Metrics Module
======================================================
Computes pixel-level and event-level segmentation and detection performance
when a ground truth voting mask is available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    """Computes spatial segmentation metrics between predicted binary mask and ground truth.
    Both inputs must be 2D binary numpy arrays (0 and 1).
    """
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_fire_pixels": total_gt_fire,
        "pred_fire_pixels": total_pred_fire
    }


def compute_event_recall(pred_events: List[Dict[str, Any]], gt_mask: np.ndarray,
                         tolerance_px: int = 3) -> Dict[str, Any]:
    """Matches detected events against ground truth connected clusters within spatial tolerance."""
    from scipy import ndimage
    gt_labels, n_gt = ndimage.label(gt_mask.astype(bool))
    if n_gt == 0:
        return {
            "n_gt_events": 0,
            "n_matched_events": 0,
            "event_recall": 1.0 if len(pred_events) == 0 else 1.0,
            "false_events": len(pred_events)
        }

    matched_gt = set()
    for ev in pred_events:
        # Check if bounding box expands and touches any gt cluster
        ymin = max(0, ev['y'] - tolerance_px)
        ymax = min(gt_mask.shape[0], ev['y'] + ev['h'] + tolerance_px)
        xmin = max(0, ev['x'] - tolerance_px)
        xmax = min(gt_mask.shape[1], ev['x'] + ev['w'] + tolerance_px)

        sub_labels = gt_labels[ymin:ymax, xmin:xmax]
        touching = set(sub_labels[sub_labels > 0])
        matched_gt.update(touching)

    event_recall = len(matched_gt) / n_gt if n_gt > 0 else 1.0
    return {
        "n_gt_events": n_gt,
        "n_matched_events": len(matched_gt),
        "event_recall": round(event_recall, 4),
        "false_events": max(0, len(pred_events) - len(matched_gt))
    }
