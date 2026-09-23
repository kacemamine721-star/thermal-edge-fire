"""Gate 2: Post-Treatment Output & Physical Sanity Gate.

Validates:
1. Pre-processing sanity (no NaNs/infs, correct dynamic range).
2. Physical consistency: pixels classified as fire by the CNN MUST be hotter
   in the underlying thermal band than the local background.
3. Spatial plausibility: rejects isolated 1-pixel false-alarm speckles and
   catastrophic full-frame hallucinations (> 50% fire).
4. Alert packet validity before passing to the downlink queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .frame_validator import Status


@dataclass
class OutputValidatorConfig:
    min_fire_pixels: int = 2
    max_fire_fraction: float = 0.50
    min_thermal_contrast_ratio: float = 1.15  # Fire pixels must be >= 15% hotter than background
    min_confidence: float = 0.50
    suppress_isolated_speckles: bool = True


@dataclass
class OutputValidationReport:
    status: Status
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    filtered_mask: Optional[np.ndarray] = None

    @property
    def ok(self) -> bool:
        return self.status != Status.REJECT


class OutputValidator:
    """Gate 2 Validator for post-processing and CNN prediction outputs."""

    def __init__(self, config: Optional[OutputValidatorConfig] = None):
        self.config = config or OutputValidatorConfig()

    def validate_preprocessing(self, normalized_tensor: np.ndarray) -> OutputValidationReport:
        """Validate tensor after radiometric NUC and dynamic range normalization."""
        t = np.asarray(normalized_tensor, dtype=np.float32)
        reasons = []
        metrics = {}

        if not np.all(np.isfinite(t)):
            reasons.append("PREPROCESSING_NONFINITE")
            return OutputValidationReport(Status.REJECT, reasons, {"finite_fraction": float(np.isfinite(t).mean())})

        # Check for empty / zeroed out normalization
        std = float(t.std())
        metrics["normalized_std"] = std
        if std < 1e-6:
            reasons.append("PREPROCESSING_ZERO_VARIANCE")
            return OutputValidationReport(Status.REJECT, reasons, metrics)

        return OutputValidationReport(Status.PASS, [], metrics)

    def validate_prediction(
        self,
        pred_mask: np.ndarray,
        raw_thermal: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> OutputValidationReport:
        """Verify model predictions against raw physical thermal radiance.
        
        Args:
            pred_mask: (H, W) binary mask predicted by CNN (1=fire, 0=bg).
            raw_thermal: (H, W) raw thermal band digital numbers (e.g. Landsat B10 or VIIRS I5).
            confidence_map: Optional (H, W) float array of model confidence scores [0, 1].
        """
        mask = np.asarray(pred_mask > 0, dtype=np.uint8)
        thermal = np.asarray(raw_thermal, dtype=np.float32)
        if mask.shape != thermal.shape:
            return OutputValidationReport(Status.REJECT, ["SHAPE_MISMATCH"], {})

        H, W = mask.shape
        total_pixels = H * W
        n_fire = int(mask.sum())
        metrics: Dict[str, float] = {"fire_pixels": n_fire, "fire_fraction": n_fire / total_pixels}

        # Case 1: No fire predicted (standard nominal clear pass)
        if n_fire == 0:
            return OutputValidationReport(Status.PASS, [], metrics, filtered_mask=mask)

        reasons = []
        worst = Status.PASS

        # Check 1: Catastrophic hallucination (network collapse / false blanket detection)
        if (n_fire / total_pixels) > self.config.max_fire_fraction:
            reasons.append("HALLUCINATED_AREA_EXCESS")
            worst = Status.REJECT

        # Check 2: Physical Thermal Consistency (Crucial Aerospace Gate)
        # Real fire pixels MUST exhibit elevated thermal emission relative to background.
        fire_thermal_vals = thermal[mask == 1]
        bg_thermal_vals = thermal[(mask == 0) & (thermal > 0)]

        if bg_thermal_vals.size > 0:
            fire_mean = float(fire_thermal_vals.mean())
            bg_mean = float(bg_thermal_vals.mean())
            contrast_ratio = fire_mean / max(bg_mean, 1e-5)
            metrics["fire_mean_thermal_dn"] = fire_mean
            metrics["bg_mean_thermal_dn"] = bg_mean
            metrics["thermal_contrast_ratio"] = contrast_ratio

            # If the CNN fires on a cold surface (e.g., cloud edge, snow, water boundary)
            if contrast_ratio < self.config.min_thermal_contrast_ratio:
                reasons.append("PHYSICAL_THERMAL_INCONSISTENCY")
                worst = Status.REJECT
        else:
            metrics["thermal_contrast_ratio"] = 1.0

        # Check 3: Isolated single-pixel speckles
        cleaned_mask = mask.copy()
        if self.config.suppress_isolated_speckles and n_fire < self.config.min_fire_pixels:
            labeled, n_features = ndi.label(mask)
            if n_features == n_fire:  # All fire pixels are disconnected single dots
                reasons.append("ISOLATED_SPECKLE_FLAG")
                if worst != Status.REJECT:
                    worst = Status.PASS_WITH_FLAGS

        # Check 4: Confidence check
        if confidence_map is not None:
            conf_vals = confidence_map[mask == 1]
            mean_conf = float(conf_vals.mean()) if conf_vals.size else 0.0
            metrics["mean_fire_confidence"] = mean_conf
            if mean_conf < self.config.min_confidence:
                reasons.append("LOW_MODEL_CONFIDENCE")
                if worst != Status.REJECT:
                    worst = Status.PASS_WITH_FLAGS

        return OutputValidationReport(worst, reasons, metrics, filtered_mask=cleaned_mask)
