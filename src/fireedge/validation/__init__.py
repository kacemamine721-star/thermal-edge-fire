"""Validation Quality Gates for Onboard CubeSat Pipeline."""
from .frame_validator import (
    FrameValidator,
    Status,
    ValidationReport,
    ValidatorConfig,
    batch_validate,
    calibrate_from_frames,
    frame_metrics,
)
from .output_validator import (
    OutputValidationReport,
    OutputValidator,
    OutputValidatorConfig,
)

__all__ = [
    # Gate 1: Pre-treatment Input Quality Gate
    "Status",
    "ValidatorConfig",
    "ValidationReport",
    "FrameValidator",
    "calibrate_from_frames",
    "frame_metrics",
    "batch_validate",
    # Gate 2: Post-treatment Output & Physical Sanity Gate
    "OutputValidatorConfig",
    "OutputValidationReport",
    "OutputValidator",
]
