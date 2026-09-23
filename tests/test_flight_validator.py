"""Unit tests for Gate 1 Mode B: Flight In-Orbit Quality Gate (FlightGateValidator).

Validates Point Spread Function (PSF) Spatial Halo verification on single-channel LWIR microbolometers:
- Confirms that non-optical Dirac-delta spikes (SEUs / Cosmic Rays) are detected and flagged.
- Confirms that optical Gaussian-blurred thermal hotspots (wildfires) are confirmed and preserved.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from fireedge.validation import FlightGateValidator, Status, ValidatorConfig


def test_flight_clean_frame_passes(clean_frame_2d):
    """Clean single-channel frame must achieve Status.PASS with 0 SEUs."""
    cfg = ValidatorConfig(bit_depth=16, noise_floor=1.0)
    validator = FlightGateValidator(cfg)
    rep = validator.validate(clean_frame_2d)
    assert rep.status == Status.PASS
    assert rep.ok is True
    assert rep.metrics.get("ch0.seu_transients", 0) == 0


def test_flight_single_event_upset_detected(clean_frame_2d):
    """A direct radiation transient (SEU) hit without optical diffusion halo must be flagged."""
    frame = clean_frame_2d.copy()
    # Inject isolated Dirac delta spike (single pixel cliff, no optical halo)
    frame[100, 100] = np.uint16(frame[100, 100] + 8000)

    cfg = ValidatorConfig(bit_depth=16, hot_k=8.0, halo_min_ratio=0.15)
    validator = FlightGateValidator(cfg)
    rep = validator.validate(frame)

    assert rep.status == Status.PASS_WITH_FLAGS
    assert rep.metrics.get("ch0.seu_transients", 0) >= 1
    assert any("SEU_TRANSIENTS" in r for r in rep.reasons)


def test_flight_optical_wildfire_preserved(clean_frame_2d):
    """A sub-pixel wildfire blurred by the camera lens optical PSF must be confirmed and preserved."""
    frame = clean_frame_2d.astype(np.float32).copy()

    # Simulate sub-pixel wildfire passing through optics:
    # 2D Gaussian Point Spread Function with sigma = 0.8 pixels
    y, x = np.ogrid[-3:4, -3:4]
    psf_kernel = np.exp(-(x**2 + y**2) / (2 * 0.8**2))
    psf_kernel /= psf_kernel.max()  # Peak normalized to 1.0
    fire_spot = (8000.0 * psf_kernel).astype(np.float32)

    # Add optical thermal spot to image
    frame[97:104, 97:104] += fire_spot
    frame = np.clip(frame, 0, 65535).astype(np.uint16)

    cfg = ValidatorConfig(bit_depth=16, hot_k=8.0, halo_min_ratio=0.15)
    validator = FlightGateValidator(cfg)
    rep = validator.validate(frame)

    # Optical blur circle MUST be recognized as real optical thermal emission, NOT an SEU
    assert rep.metrics.get("ch0.hot_confirmed_optical", 0) >= 1
    assert rep.metrics.get("ch0.seu_transients", 0) == 0
    assert not any("SEU_TRANSIENTS" in r for r in rep.reasons)


def test_flight_repair_policy_cleans_transient(clean_frame_2d):
    """When policy is 'repair', isolated radiation transient is replaced by local median."""
    frame = clean_frame_2d.copy()
    frame[120, 120] = np.uint16(frame[120, 120] + 10000)

    cfg = ValidatorConfig(bit_depth=16, hot_k=8.0, halo_min_ratio=0.15, hot_pixel_policy="repair")
    validator = FlightGateValidator(cfg)
    rep = validator.validate(frame)

    assert rep.cleaned is not None
    assert rep.cleaned[120, 120] < frame[120, 120]
