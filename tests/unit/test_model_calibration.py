"""Unit tests for model calibration helpers."""

from nbaprop.models.calibration import apply_calibration, fit_linear_calibration


def test_fit_linear_calibration_and_apply():
    rows = [
        {"result": "WIN", "raw_model_probability": 0.7, "prop_type": "points"},
        {"result": "LOSS", "raw_model_probability": 0.3, "prop_type": "points"},
        {"result": "WIN", "raw_model_probability": 0.8, "prop_type": "rebounds"},
        {"result": "LOSS", "raw_model_probability": 0.2, "prop_type": "rebounds"},
    ]
    calibration = fit_linear_calibration(
        rows,
        prob_key="raw_model_probability",
        min_samples=2,
    )

    assert calibration["global"] is not None
    assert "points" in calibration["by_prop_type"]

    calibrated, meta = apply_calibration(0.6, "points", calibration)
    assert 0 <= calibrated <= 1
    assert meta["source"] == "points"

    fallback, meta_fallback = apply_calibration(0.6, "assists", calibration)
    assert 0 <= fallback <= 1
    assert meta_fallback["source"] == "global"

