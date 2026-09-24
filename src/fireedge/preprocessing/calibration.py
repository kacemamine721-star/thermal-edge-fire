"""Radiometric calibration module for satellite thermal infrared telemetry.

Converts raw Digital Numbers (DN) to:
1. Top-of-Atmosphere (TOA) Spectral Radiance (W / (m^2 * sr * um))
2. At-Satellite Brightness Temperature in Kelvin (K) or Celsius (°C)
via the physical Planck function inversion.

Coefficients calibrated from USGS Landsat-8 TIRS documentation:
- Band 10 (10.60 - 11.19 um): ML = 0.0003342, AL = 0.1, K1 = 774.8853, K2 = 1321.0789
- Band 11 (11.50 - 12.51 um): ML = 0.0003342, AL = 0.1, K1 = 480.8883, K2 = 1201.1442
"""
from __future__ import annotations

from typing import Dict, Literal, Union
import numpy as np

# Landsat-8 TIRS Radiance Rescaling and Thermal Constants
TIRS_CONSTANTS: Dict[str, Dict[str, float]] = {
    "B10": {
        "ML": 3.3420e-4,     # Radiance multiplicative scaling factor
        "AL": 0.1,           # Radiance additive scaling factor
        "K1": 774.8853,      # Thermal conversion constant 1 (W / (m^2 * sr * um))
        "K2": 1321.0789,     # Thermal conversion constant 2 (Kelvin)
    },
    "B11": {
        "ML": 3.3420e-4,
        "AL": 0.1,
        "K1": 480.8883,
        "K2": 1201.1442,
    },
}


def dn_to_radiance(
    dn: Union[np.ndarray, float],
    band: str = "B10",
) -> Union[np.ndarray, float]:
    """Convert raw integer Digital Numbers (DN) to TOA Spectral Radiance L_lambda.

    L_lambda = ML * Q_cal + AL
    """
    if band not in TIRS_CONSTANTS:
        raise ValueError(f"Unknown thermal band '{band}'. Supported: {list(TIRS_CONSTANTS.keys())}")
    
    cfg = TIRS_CONSTANTS[band]
    radiance = cfg["ML"] * np.asarray(dn, dtype=np.float32) + cfg["AL"]
    return np.maximum(radiance, 0.0)


def radiance_to_temperature(
    radiance: Union[np.ndarray, float],
    band: str = "B10",
    unit: Literal["kelvin", "celsius"] = "kelvin",
) -> Union[np.ndarray, float]:
    """Convert TOA Spectral Radiance to At-Satellite Brightness Temperature.

    T = K2 / ln(K1 / L_lambda + 1)
    """
    if band not in TIRS_CONSTANTS:
        raise ValueError(f"Unknown thermal band '{band}'. Supported: {list(TIRS_CONSTANTS.keys())}")
    
    cfg = TIRS_CONSTANTS[band]
    rad = np.asarray(radiance, dtype=np.float32)
    # Avoid division by zero or negative radiance
    safe_rad = np.maximum(rad, 1e-6)
    
    temp_k = cfg["K2"] / np.log((cfg["K1"] / safe_rad) + 1.0)
    
    if unit == "celsius":
        return temp_k - 273.15
    return temp_k


def dn_to_temperature(
    dn: Union[np.ndarray, float],
    band: str = "B10",
    unit: Literal["kelvin", "celsius"] = "kelvin",
) -> Union[np.ndarray, float]:
    """Direct end-to-end conversion from raw Digital Numbers to physical temperature."""
    radiance = dn_to_radiance(dn, band=band)
    return radiance_to_temperature(radiance, band=band, unit=unit)


def temperature_to_dn(
    temp: Union[np.ndarray, float],
    band: str = "B10",
    unit: Literal["kelvin", "celsius"] = "kelvin",
) -> np.ndarray:
    """Inverse Planck function: converts physical temperature back to raw Digital Numbers.

    Used by physics-based sensor simulators (§6.4) to synthesize raw telemetry.
    """
    if band not in TIRS_CONSTANTS:
        raise ValueError(f"Unknown thermal band '{band}'. Supported: {list(TIRS_CONSTANTS.keys())}")
    
    temp_k = np.asarray(temp, dtype=np.float32)
    if unit == "celsius":
        temp_k = temp_k + 273.15
        
    cfg = TIRS_CONSTANTS[band]
    # L_lambda = K1 / (exp(K2 / T) - 1)
    safe_temp = np.maximum(temp_k, 50.0)
    radiance = cfg["K1"] / (np.exp(cfg["K2"] / safe_temp) - 1.0)
    
    # Q_cal = (L_lambda - AL) / ML
    dn = (radiance - cfg["AL"]) / cfg["ML"]
    return np.clip(np.round(dn), 0, 65535).astype(np.uint16)
