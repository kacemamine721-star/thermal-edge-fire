"""Unit tests for Stage 2 radiometric calibration, NUC, and normalization."""
import numpy as np
import pytest

from fireedge.preprocessing.calibration import (
    dn_to_radiance,
    dn_to_temperature,
    radiance_to_temperature,
    temperature_to_dn,
)
from fireedge.preprocessing.normalization import (
    fixed_window_norm,
    minmax_norm,
    robust_percentile_norm,
)
from fireedge.preprocessing.nuc import (
    MicrobolometerConfig,
    MicrobolometerDegrader,
    TwoPointNUC,
)


def test_radiometric_calibration_roundtrip():
    """Verify that DN -> Temp -> DN preserves fidelity."""
    original_dn = np.array([20000, 25000, 30000, 40000], dtype=np.uint16)
    temp_k = dn_to_temperature(original_dn, band="B10", unit="kelvin")

    # Ambient earth temperatures are ~280 - 320 K
    assert 270.0 < temp_k[1] < 320.0
    # Higher DN must produce strictly higher physical temperature
    assert np.all(np.diff(temp_k) > 0)

    # Invert back to DN
    reconstructed_dn = temperature_to_dn(temp_k, band="B10", unit="kelvin")
    # Due to floating point quantization, difference should be <= 2 DN
    np.testing.assert_allclose(original_dn, reconstructed_dn, atol=2)


def test_temperature_celsius_conversion():
    """Verify Kelvin to Celsius consistency."""
    dn = np.array([25000], dtype=np.uint16)
    temp_k = dn_to_temperature(dn, unit="kelvin")
    temp_c = dn_to_temperature(dn, unit="celsius")
    np.testing.assert_allclose(temp_k - 273.15, temp_c, atol=1e-4)


def test_microbolometer_degrader_and_nuc():
    """Verify FPN noise injection and two-point NUC uniformity restoration."""
    # Synthetic flat field at ambient 300 K
    clean_frame = np.full((128, 128), 26000.0, dtype=np.float32)

    cfg = MicrobolometerConfig(
        array_shape=(128, 128),
        fpn_gain_sigma=0.04,
        fpn_offset_sigma_dn=50.0,
        netd_noise_dn=10.0,
        bad_pixel_fraction=0.001,
        random_seed=42,
    )
    degrader = MicrobolometerDegrader(cfg)
    degraded = degrader.degrade(clean_frame)

    # Degraded image must exhibit significant column variance compared to clean
    assert np.std(degraded) > np.std(clean_frame)

    # Two-point NUC calibration
    cold_flat = np.full((128, 128), 20000.0, dtype=np.float32)
    warm_flat = np.full((128, 128), 35000.0, dtype=np.float32)
    deg_cold = degrader.degrade(cold_flat)
    deg_warm = degrader.degrade(warm_flat)

    nuc = TwoPointNUC(deg_cold, deg_warm)
    corrected = nuc.correct(degraded)

    # NUC must substantially reduce FPN variance
    assert np.std(corrected) < np.std(degraded)


def test_thermal_normalization_bounds():
    """Verify normalized values strictly reside within [0.0, 1.0]."""
    temps = np.linspace(250.0, 450.0, 500)
    norm_fixed = fixed_window_norm(temps, vmin_k=270.0, vmax_k=420.0)
    assert np.min(norm_fixed) >= 0.0
    assert np.max(norm_fixed) <= 1.0

    raw_noisy = np.random.normal(25000, 2000, (64, 64))
    norm_robust = robust_percentile_norm(raw_noisy)
    assert np.min(norm_robust) >= 0.0
    assert np.max(norm_robust) <= 1.0
