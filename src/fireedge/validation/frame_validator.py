"""Gate 1: Pre-treatment Input Quality Gate wrapper.

Maintains backward-compatibility by aliasing FrameValidator to GroundGateValidator
and re-exporting base validation structures.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Union

import numpy as np

from .base_validator import (
    Status,
    ValidationReport,
    ValidatorConfig,
    calibrate_from_frames,
)
from .ground_gate_validator import GroundGateValidator

# Backward-compatible alias
FrameValidator = GroundGateValidator


def frame_metrics(report: ValidationReport) -> Dict[str, float]:
    """Extract metrics dictionary from validation report."""
    return dict(report.metrics)


def batch_validate(
    validator: GroundGateValidator, frames: Iterable[np.ndarray]
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
