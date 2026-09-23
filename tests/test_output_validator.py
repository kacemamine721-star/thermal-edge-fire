"""Unit tests for Gate 2: Post-Treatment Output & Physical Sanity Gate."""
import numpy as np
import pytest

from fireedge.validation import (
    OutputValidator,
    OutputValidatorConfig,
    Status,
)


def test_preprocessing_validation_passes_clean():
    clean_tensor = np.random.uniform(0.1, 0.9, size=(2, 256, 256)).astype(np.float32)
    val = OutputValidator()
    rep = val.validate_preprocessing(clean_tensor)
    assert rep.status == Status.PASS
    assert rep.ok is True


def test_preprocessing_validation_rejects_nans():
    corrupt_tensor = np.random.uniform(0.1, 0.9, size=(2, 256, 256)).astype(np.float32)
    corrupt_tensor[0, 10, 10] = np.nan
    val = OutputValidator()
    rep = val.validate_preprocessing(corrupt_tensor)
    assert rep.status == Status.REJECT
    assert any("PREPROCESSING_NONFINITE" in r for r in rep.reasons)


def test_no_fire_prediction_passes():
    raw_thermal = np.full((256, 256), 28000, dtype=np.uint16)
    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    val = OutputValidator()
    rep = val.validate_prediction(pred_mask, raw_thermal)
    assert rep.status == Status.PASS
    assert rep.metrics["fire_pixels"] == 0


def test_physically_consistent_fire_passes():
    # Background thermal at 28000 DN, Fire cluster at 45000 DN (strong thermal anomaly)
    raw_thermal = np.full((256, 256), 28000, dtype=np.float32)
    raw_thermal[100:105, 100:105] = 45000.0  # +60% thermal contrast

    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask[100:105, 100:105] = 1

    val = OutputValidator()
    rep = val.validate_prediction(pred_mask, raw_thermal)
    assert rep.status == Status.PASS
    assert rep.ok is True
    assert rep.metrics["thermal_contrast_ratio"] > 1.15


def test_cold_fire_inconsistency_rejects():
    # CNN false-alarm hallucinated fire on a cold cloud edge (thermal = 18000 vs bg 28000)
    raw_thermal = np.full((256, 256), 28000, dtype=np.float32)
    raw_thermal[100:105, 100:105] = 18000.0  # Cold cloud edge!

    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask[100:105, 100:105] = 1

    val = OutputValidator()
    rep = val.validate_prediction(pred_mask, raw_thermal)
    assert rep.status == Status.REJECT
    assert any("PHYSICAL_THERMAL_INCONSISTENCY" in r for r in rep.reasons)


def test_hallucinated_massive_area_rejects():
    # Model collapse predicting >50% of the entire scene is on fire
    raw_thermal = np.full((256, 256), 35000, dtype=np.float32)
    pred_mask = np.ones((256, 256), dtype=np.uint8)  # 100% fire

    val = OutputValidator(OutputValidatorConfig(max_fire_fraction=0.50))
    rep = val.validate_prediction(pred_mask, raw_thermal)
    assert rep.status == Status.REJECT
    assert any("HALLUCINATED_AREA_EXCESS" in r for r in rep.reasons)


def test_isolated_speckle_flagged():
    # Single 1-pixel hot detection without cluster confirmation
    raw_thermal = np.full((256, 256), 28000, dtype=np.float32)
    raw_thermal[50, 50] = 40000.0

    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask[50, 50] = 1  # 1 pixel only

    val = OutputValidator(OutputValidatorConfig(min_fire_pixels=2))
    rep = val.validate_prediction(pred_mask, raw_thermal)
    assert rep.status == Status.PASS_WITH_FLAGS
    assert any("ISOLATED_SPECKLE_FLAG" in r for r in rep.reasons)
