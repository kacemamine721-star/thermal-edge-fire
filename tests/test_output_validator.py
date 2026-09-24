"""Tests for Gate 2 post-treatment and physical sanity validation."""

import numpy as np

from fireedge.validation import (
    AlertPacket,
    OutputValidatorConfig,
    RejectReason,
    check_preprocessing_sanity,
    pack_alert_packet,
    validate_alert_packet,
    validate_output,
)


def _clean_tensor() -> np.ndarray:
    return np.full((1, 64, 64), 290.0, dtype=np.float32)


def _fire_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = np.full((64, 64), 290.0, dtype=np.float32)
    frame[30:34, 30:34] = 330.0
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 30:34] = True
    return _clean_tensor(), frame, mask


def test_preprocessing_sanity_passes_clean_tensor():
    result = check_preprocessing_sanity(_clean_tensor(), OutputValidatorConfig())

    assert result.passed is True


def test_preprocessing_sanity_rejects_nonfinite_values():
    tensor = _clean_tensor()
    tensor[:, :8, :] = np.nan

    result = check_preprocessing_sanity(tensor, OutputValidatorConfig())

    assert result.passed is False
    assert result.reason is RejectReason.NONFINITE_TENSOR


def test_preprocessing_sanity_rejects_out_of_physical_range():
    tensor = _clean_tensor()
    tensor[:, :8, :] = 450.0

    result = check_preprocessing_sanity(tensor, OutputValidatorConfig())

    assert result.passed is False
    assert result.reason is RejectReason.OUT_OF_PHYSICAL_RANGE


def test_spatial_filter_rejects_single_pixel_blip():
    _, frame, mask = _fire_case()
    mask[:, :] = False
    mask[10, 10] = True

    result = validate_output(_clean_tensor(), frame, mask, OutputValidatorConfig())

    assert result.passed is False
    assert result.reason is RejectReason.ALL_COMPONENTS_SUBPIXEL


def test_spatial_filter_rejects_catastrophic_collapse():
    _, frame, _ = _fire_case()
    mask = np.ones((64, 64), dtype=bool)

    result = validate_output(_clean_tensor(), frame, mask, OutputValidatorConfig())

    assert result.passed is False
    assert result.reason is RejectReason.CATASTROPHIC_COLLAPSE
    assert result.telemetry_flag is True


def test_thermal_consistency_accepts_hot_candidate():
    tensor, frame, mask = _fire_case()

    result = validate_output(tensor, frame, mask, OutputValidatorConfig())

    assert result.passed is True
    assert result.surviving_mask is not None
    assert result.surviving_mask.sum() == mask.sum()


def test_thermal_consistency_rejects_cold_candidate():
    tensor, frame, mask = _fire_case()
    frame[30:34, 30:34] = 270.0

    result = validate_output(tensor, frame, mask, OutputValidatorConfig())

    assert result.passed is False
    assert result.reason is RejectReason.THERMAL_INCONSISTENT


def test_alert_packet_round_trip_validates_crc_and_footprint():
    cfg = OutputValidatorConfig()
    packet = pack_alert_packet(
        AlertPacket(
            timestamp=1_758_000_000,
            lat=36.8,
            lon=10.2,
            bbox=(30, 30, 4, 4),
            confidence=0.91,
            intensity_k=330.0,
        ),
        cfg,
    )
    footprint = [(30.0, 5.0), (30.0, 15.0), (40.0, 15.0), (40.0, 5.0)]

    result = validate_alert_packet(packet, cfg, footprint)

    assert result.passed is True
    assert result.packet == packet
    assert len(packet) <= cfg.max_packet_bytes


def test_alert_packet_rejects_checksum_corruption():
    cfg = OutputValidatorConfig()
    packet = bytearray(
        pack_alert_packet(
            AlertPacket(1_758_000_000, 36.8, 10.2, (0, 0, 4, 4), 0.9, 330.0),
            cfg,
        )
    )
    packet[0] ^= 0x01
    footprint = [(30.0, 5.0), (30.0, 15.0), (40.0, 15.0), (40.0, 5.0)]

    result = validate_alert_packet(bytes(packet), cfg, footprint)

    assert result.passed is False
    assert result.reason is RejectReason.CHECKSUM_MISMATCH


def test_alert_packet_rejects_coordinate_outside_footprint():
    cfg = OutputValidatorConfig()
    packet = pack_alert_packet(
        AlertPacket(1_758_000_000, 20.0, 10.2, (0, 0, 4, 4), 0.9, 330.0),
        cfg,
    )
    footprint = [(30.0, 5.0), (30.0, 15.0), (40.0, 15.0), (40.0, 5.0)]

    result = validate_alert_packet(packet, cfg, footprint)

    assert result.passed is False
    assert result.reason is RejectReason.OUTSIDE_FOOTPRINT


def test_alert_packet_rejects_zero_sized_bounding_box():
    cfg = OutputValidatorConfig()
    packet = pack_alert_packet(
        AlertPacket(1_758_000_000, 36.8, 10.2, (0, 0, 0, 4), 0.9, 330.0),
        cfg,
    )
    footprint = [(30.0, 5.0), (30.0, 15.0), (40.0, 15.0), (40.0, 5.0)]

    result = validate_alert_packet(packet, cfg, footprint)

    assert result.passed is False
    assert result.reason is RejectReason.PACKET_SCHEMA_INVALID