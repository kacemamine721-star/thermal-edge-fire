"""Microbolometer Non-Uniformity Correction (NUC) and Degradation Simulation.

Uncooled microbolometer focal plane arrays (FPA) exhibit pixel-to-pixel gain and
offset non-uniformity that drifts with temperature and time. This module provides:

1. **TwoPointNUC**: Classical two-point correction calibrated from blackbody
   reference frames (hot + cold uniform targets).
2. **MicrobolometerDegrader**: Simulates realistic FPA degradation for CNN training
   robustness: fixed-pattern noise (FPN), 1/f temporal drift, and random dead pixels.
3. **MicrobolometerConfig**: Configurable degradation parameters.

Reference: KITSUNE CM3+ payload documentation §4.2 (NUC procedure).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class MicrobolometerConfig:
    """Configuration for microbolometer degradation simulation.

    Parameters calibrated against ULIS Pico384 / FLIR Lepton 3.5 datasheets:
    - NETD (Noise Equivalent Temperature Difference): ~50 mK typical
    - FPN: 0.1-0.3% of dynamic range after factory NUC
    - 1/f drift: ~0.01 K/min at stable ambient
    - Dead pixel rate: < 0.5% contractual threshold
    """

    # Fixed-Pattern Noise (gain non-uniformity as fractional std dev)
    fpn_gain_sigma: float = 0.02       # 2% gain variation across FPA
    fpn_offset_sigma: float = 5.0      # 5 DN additive offset spread

    # Temporal 1/f drift (simulated as slow random walk)
    drift_sigma: float = 0.5           # DN per frame temporal drift magnitude

    # Dead / hot pixel simulation
    dead_pixel_frac: float = 0.003     # 0.3% of pixels are dead (stuck at 0 or saturated)

    # Random read noise (Gaussian, per-pixel per-frame)
    read_noise_sigma: float = 2.0      # DN read noise (NETD ~ 50 mK -> ~2 DN at 16-bit)

    # Bit depth for saturation clipping
    bit_depth: int = 16

    # Random seed for reproducibility
    seed: Optional[int] = None

    # Compatibility aliases
    array_shape: Optional[Tuple[int, int]] = None
    fpn_offset_sigma_dn: Optional[float] = None
    netd_noise_dn: Optional[float] = None
    bad_pixel_fraction: Optional[float] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.fpn_offset_sigma_dn is not None:
            self.fpn_offset_sigma = float(self.fpn_offset_sigma_dn)
        if self.netd_noise_dn is not None:
            self.read_noise_sigma = float(self.netd_noise_dn)
        if self.bad_pixel_fraction is not None:
            self.dead_pixel_frac = float(self.bad_pixel_fraction)
        if self.random_seed is not None:
            self.seed = self.random_seed

    @property
    def sat_level(self) -> float:
        return float(2 ** self.bit_depth - 1)


class TwoPointNUC:
    """Classical Two-Point Non-Uniformity Correction with Bad Pixel Replacement.

    Calibration procedure:
    1. Acquire a frame at low irradiance/temp (cold blackbody reference)
    2. Acquire a frame at high irradiance/temp (hot blackbody reference)
    3. Compute per-pixel gain and offset correction coefficients:
       gain[i,j]   = mean(delta) / (hot[i,j] - cold[i,j])
       offset[i,j] = mean(cold) - gain[i,j] * cold[i,j]
    4. Detect bad/unresponsive pixels and replace them with local neighborhood median.
    """

    def __init__(
        self,
        cold_frame: Optional[np.ndarray] = None,
        hot_frame: Optional[np.ndarray] = None,
        cold_temp: Optional[float] = None,
        hot_temp: Optional[float] = None,
        min_dn_delta: float = 10.0,
    ):
        self.cold_temp = cold_temp
        self.hot_temp = hot_temp
        self.min_dn_delta = min_dn_delta
        self.gain: Optional[np.ndarray] = None
        self.offset: Optional[np.ndarray] = None
        self.bad_pixel_mask: Optional[np.ndarray] = None

        if cold_frame is not None and hot_frame is not None:
            self.calibrate(cold_frame, hot_frame)

    def calibrate(self, cold_frame: np.ndarray, hot_frame: np.ndarray) -> None:
        """Compute per-pixel gain and offset from blackbody reference frames."""
        cold = cold_frame.astype(np.float64)
        hot = hot_frame.astype(np.float64)

        delta = hot - cold
        mean_delta = float(np.mean(delta))
        mean_cold = float(np.mean(cold))
        std_delta = float(np.std(delta))

        # Detect dead / unresponsive / extreme pixels
        bad_mask = (delta <= self.min_dn_delta)
        if std_delta > 0:
            bad_mask = bad_mask | (np.abs(delta - mean_delta) > 3.0 * std_delta)
        self.bad_pixel_mask = bad_mask

        safe_delta = np.where(bad_mask, 1.0, delta)

        if self.cold_temp is not None and self.hot_temp is not None:
            # Calibrate to Kelvin
            delta_temp = self.hot_temp - self.cold_temp
            self.gain = np.where(bad_mask, 0.0, delta_temp / safe_delta)
            self.offset = np.where(bad_mask, self.cold_temp, self.cold_temp - self.gain * cold)
        else:
            # Calibrate in DN space (standard flat-field uniformity restoration)
            self.gain = np.where(bad_mask, 1.0, mean_delta / safe_delta)
            self.offset = np.where(bad_mask, 0.0, mean_cold - self.gain * cold)

    def correct(self, raw_frame: np.ndarray) -> np.ndarray:
        """Apply two-point NUC correction with bad-pixel replacement."""
        if self.gain is None or self.offset is None:
            raise RuntimeError(
                "NUC not calibrated. Call calibrate(cold_frame, hot_frame) first, "
                "or pass reference frames to the constructor."
            )

        raw = raw_frame.astype(np.float64)

        # Handle shape broadcasting
        if raw.ndim == 3 and self.gain.ndim == 2:
            gain = self.gain[np.newaxis, :, :]
            offset = self.offset[np.newaxis, :, :]
            bad_mask = self.bad_pixel_mask[np.newaxis, :, :] if self.bad_pixel_mask is not None else None
        else:
            gain = self.gain
            offset = self.offset
            bad_mask = self.bad_pixel_mask

        corrected = gain * raw + offset

        # Bad pixel replacement via 3x3 local median filter
        if bad_mask is not None and np.any(bad_mask):
            from scipy.ndimage import median_filter
            med = median_filter(corrected, size=3)
            corrected = np.where(bad_mask, med, corrected)

        return corrected.astype(np.float32)

    @property
    def is_calibrated(self) -> bool:
        return self.gain is not None and self.offset is not None


class MicrobolometerDegrader:
    """Simulates realistic uncooled microbolometer sensor degradation.

    Used during training to improve CNN robustness against on-orbit sensor
    artifacts: fixed-pattern noise, temporal drift, dead pixels, and read noise.

    Usage:
        degrader = MicrobolometerDegrader(cfg)
        degraded_frame = degrader.degrade(clean_frame)
    """

    def __init__(
        self,
        config: Optional[MicrobolometerConfig] = None,
        shape: Tuple[int, int] = (256, 256),
    ):
        self.cfg = config or MicrobolometerConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._shape = shape
        self._initialized = False

        # Lazily initialized per-pixel patterns
        self._gain_map: Optional[np.ndarray] = None
        self._offset_map: Optional[np.ndarray] = None
        self._dead_mask: Optional[np.ndarray] = None
        self._drift_state: float = 0.0

    def _init_patterns(self, shape: Tuple[int, int]) -> None:
        """Generate fixed-pattern noise maps and dead pixel mask."""
        H, W = shape
        self._gain_map = 1.0 + self.rng.normal(0, self.cfg.fpn_gain_sigma, (H, W)).astype(np.float32)
        self._offset_map = self.rng.normal(0, self.cfg.fpn_offset_sigma, (H, W)).astype(np.float32)

        # Dead pixel mask: True = dead
        n_dead = int(H * W * self.cfg.dead_pixel_frac)
        dead_flat = np.zeros(H * W, dtype=bool)
        if n_dead > 0:
            dead_indices = self.rng.choice(H * W, size=n_dead, replace=False)
            dead_flat[dead_indices] = True
        self._dead_mask = dead_flat.reshape(H, W)

        self._initialized = True

    def degrade(self, frame: np.ndarray) -> np.ndarray:
        """Apply simulated microbolometer degradation to a clean 2D frame.

        Args:
            frame: Clean sensor frame, shape (H, W), any numeric dtype.

        Returns:
            Degraded frame, shape (H, W), float32, clipped to [0, sat_level].
        """
        arr = frame.astype(np.float32)
        H, W = arr.shape

        # Lazy initialization on first call (or shape change)
        if not self._initialized or self._gain_map.shape != (H, W):
            self._init_patterns((H, W))

        # 1. Apply fixed-pattern noise (multiplicative gain + additive offset)
        degraded = self._gain_map * arr + self._offset_map

        # 2. Temporal 1/f drift (random walk accumulator)
        self._drift_state += self.rng.normal(0, self.cfg.drift_sigma)
        degraded += self._drift_state

        # 3. Per-pixel read noise
        if self.cfg.read_noise_sigma > 0:
            read_noise = self.rng.normal(0, self.cfg.read_noise_sigma, (H, W)).astype(np.float32)
            degraded += read_noise

        # 4. Dead pixels: stuck at 0 or saturated
        if self._dead_mask is not None and self._dead_mask.any():
            dead_vals = self.rng.choice(
                [0.0, self.cfg.sat_level], size=int(self._dead_mask.sum())
            )
            degraded[self._dead_mask] = dead_vals

        # 5. Clip to valid detector range
        degraded = np.clip(degraded, 0.0, self.cfg.sat_level)

        return degraded

    def reset(self) -> None:
        """Reset temporal drift state (simulates NUC recalibration event)."""
        self._drift_state = 0.0
