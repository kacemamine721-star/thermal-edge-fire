"""Unit tests for the Stage 1 Validation Quality Gate."""
import numpy as np
import pytest

from fireedge.validation import (
    FrameValidator,
    Status,
    ValidatorConfig,
    calibrate_from_frames,
)


def test_clean_frame_passes(clean_multiband_frame):
    """Clean frame with normal gradient must achieve Status.PASS."""
    cfg = ValidatorConfig(bit_depth=16, noise_floor=1.0)
    validator = FrameValidator(cfg)
    rep = validator.validate(clean_multiband_frame)
    assert rep.status == Status.PASS
    assert rep.ok is True
    assert len(rep.reasons) == 0


def test_dead_frame_rejects(clean_frame_2d):
    """Constant value stuck sensor must trigger REJECT (DEAD_OR_STUCK)."""
    dead = np.full((256, 256), 25000, dtype=np.uint16)
    cfg = ValidatorConfig(bit_depth=16, noise_floor=1.0)
    validator = FrameValidator(cfg)
    rep = validator.validate(dead)
    assert rep.status == Status.REJECT
    assert any("DEAD_OR_STUCK" in r for r in rep.reasons)


def test_massive_saturation_rejects(clean_frame_2d):
    """Frame with >2% saturated pixels must be rejected."""
    saturated = clean_frame_2d.copy()
    # Saturate 3% of the image
    saturated[:50, :50] = 65535
    cfg = ValidatorConfig(bit_depth=16, saturation_frac_reject=0.02)
    validator = FrameValidator(cfg)
    rep = validator.validate(saturated)
    assert rep.status == Status.REJECT
    assert any("SATURATION_EXCESS" in r for r in rep.reasons)


def test_nodata_excess_rejects(clean_frame_2d):
    """Frame with >30% zeros (dropped scanlines / missing data) must be rejected."""
    corrupt = clean_frame_2d.copy()
    corrupt[:100, :] = 0  # ~39% missing
    cfg = ValidatorConfig(bit_depth=16, nodata_frac_reject=0.30)
    validator = FrameValidator(cfg)
    rep = validator.validate(corrupt)
    assert rep.status == Status.REJECT
    assert any("NODATA_EXCESS" in r for r in rep.reasons)


def test_isolated_hot_pixel_flagged(clean_frame_2d):
    """A radiation transient hitting a single detector pixel in ONE channel must be flagged."""
    frame = clean_frame_2d.copy()
    # Inject isolated high spike
    frame[128, 128] = np.uint16(frame[128, 128] + 8000)

    cfg = ValidatorConfig(bit_depth=16, hot_k=5.0)
    validator = FrameValidator(cfg)
    rep = validator.validate(frame)

    assert rep.status in [Status.PASS_WITH_FLAGS, Status.PASS]
    assert rep.metrics.get("ch0.hot_pixels", 0) >= 1


def test_cross_channel_confirmed_fire_preserves_recall(clean_multiband_frame):
    """Hot pixel showing up in BOTH B10 and B11 must be confirmed as real fire candidate."""
    frame = clean_multiband_frame.copy()
    # Spike at same pixel in both thermal channels
    frame[0, 100, 100] = np.uint16(frame[0, 100, 100] + 9000)
    frame[1, 100, 100] = np.uint16(frame[1, 100, 100] + 9000)

    cfg = ValidatorConfig(bit_depth=16, hot_k=5.0, cross_channel_confirm=True)
    validator = FrameValidator(cfg)
    rep = validator.validate(frame)

    # Must confirm as real fire candidate, protecting recall!
    assert rep.metrics.get("ch0.hot_confirmed_real", 0) >= 1
    assert rep.status != Status.REJECT


def test_striping_detection(clean_frame_2d):
    """Inject defective readout column to test detector striping detection."""
    striped = clean_frame_2d.copy()
    striped[:, 50] = np.uint16(striped[:, 50] + 4000)  # Defective readout column

    cfg = ValidatorConfig(bit_depth=16, stripe_k=5.0)
    validator = FrameValidator(cfg)
    rep = validator.validate(striped)

    assert any("STRIPING" in r for r in rep.reasons)


def test_calibration_from_frames(clean_frame_2d):
    """Test automatic empirical threshold calibration from clean frames."""
    frames = [clean_frame_2d, (clean_frame_2d * 0.95).astype(np.uint16)]
    calib_cfg = calibrate_from_frames(frames)

    assert calib_cfg.valid_min is not None
    assert calib_cfg.valid_max is not None
    assert calib_cfg.max_gradient is not None
    assert calib_cfg.noise_flag is not None
    assert calib_cfg.noise_reject is not None
    assert calib_cfg.valid_max > calib_cfg.valid_min
