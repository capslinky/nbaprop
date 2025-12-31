"""Unit tests for scoring logic."""

import pytest

from nbaprop.models.scoring import score_prop


def test_score_prop_injury_out_pass():
    row = {"prop_id": "p1", "features": {"line": 10, "injury_status": "OUT"}}
    result = score_prop(row)
    assert result["pick"] == "PASS"
    assert result["reason"] == "INJURY_OUT"


def test_score_prop_market_blend():
    row = {
        "prop_id": "p2",
        "features": {
            "line": 10,
            "recent_avg": 12,
            "season_avg": 12,
            "odds": -110,
            "side": "over",
            "market_prob": 0.4,
        },
    }
    result = score_prop(row, market_blend=0.5)
    expected = (result["model_probability"] + 0.4) / 2
    assert result["probability"] == pytest.approx(expected, rel=1e-3)


def test_score_prop_odds_range_passes():
    row = {
        "prop_id": "p3",
        "features": {
            "line": 10,
            "recent_avg": 12,
            "season_avg": 12,
            "odds": -150,
            "side": "over",
        },
    }
    result = score_prop(row, odds_min=-110, odds_max=110)
    assert result["pick"] == "PASS"
    assert result["reason"] == "ODDS_OUT_OF_RANGE"


def test_score_prop_confidence_cap_outside_band():
    row = {
        "prop_id": "p4",
        "features": {
            "line": 10,
            "recent_avg": 16,
            "season_avg": 15,
            "odds": 140,
            "side": "over",
        },
    }
    result = score_prop(
        row,
        min_confidence=0.2,
        odds_min=-150,
        odds_max=150,
        odds_confidence_min=-120,
        odds_confidence_max=120,
        confidence_cap_outside_band=0.6,
    )
    assert result["confidence"] <= 0.6
