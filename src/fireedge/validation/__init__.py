"""Validation Quality Gates for Onboard CubeSat Pipeline (Dual-Gate Architecture)."""
from .base_validator import (
    Status,
    ValidationReport,
    ValidatorConfig,
    calibrate_from_frames,
)
from .ground_gate_validator import GroundGateValidator
from .flight_gate_validator import FlightGateValidator
from .frame_validator import FrameValidator, batch_validate, frame_metrics
from .output_validator import (
    AlertPacket,
    RejectReason,
    ValidationResult as OutputValidationReport,
    ValidatorConfig as OutputValidatorConfig,
    check_preprocessing_sanity,
    check_spatial_plausibility,
    check_thermal_consistency,
    crc16_ccitt,
    pack_alert_packet,
    validate_alert_packet,
    validate_output,
)

__all__ = [
    # Core Data Structures
    "Status",
    "ValidatorConfig",
    "ValidationReport",
    "calibrate_from_frames",
    # Gate 1 Mode A: Multi-channel Ground Curation (ActiveFire B10/B11)
    "GroundGateValidator",
    "FrameValidator",
    "batch_validate",
    "frame_metrics",
    # Gate 1 Mode B: Single-channel Flight Microbolometer (PSF Optical Halo)
    "FlightGateValidator",
    # Gate 2: Post-treatment Output & Physical Sanity Gate
    "OutputValidatorConfig",
    "OutputValidationReport",
    "RejectReason",
    "AlertPacket",
    "check_preprocessing_sanity",
    "check_spatial_plausibility",
    "check_thermal_consistency",
    "validate_output",
    "pack_alert_packet",
    "validate_alert_packet",
    "crc16_ccitt",
]
