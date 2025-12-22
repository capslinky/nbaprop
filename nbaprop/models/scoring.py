"""Odds-aware scoring helpers."""

from typing import Dict
import math


def _american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def score_prop(row: Dict, min_edge: float = 0.03, min_confidence: float = 0.4) -> Dict:
    """Score a single prop row with probability, edge, and confidence."""
    features = row.get("features", {})
    line = features.get("line")
    if line in (None, 0, 0.0):
        return {
            "prop_id": row.get("prop_id"),
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
        }

    odds = features.get("odds", -110)
    try:
        odds = int(odds)
    except (TypeError, ValueError):
        odds = -110

    recent_avg = features.get("recent_avg")
    season_avg = features.get("season_avg")

    if recent_avg in (None, 0, 0.0) and season_avg in (None, 0, 0.0):
        projection = float(line)
    elif season_avg in (None, 0, 0.0):
        projection = float(recent_avg)
    elif recent_avg in (None, 0, 0.0):
        projection = float(season_avg)
    else:
        projection = (float(recent_avg) * 0.6) + (float(season_avg) * 0.4)

    std = max(1.0, abs(projection) * 0.25)
    z = (float(line) - projection) / std
    prob_over = 1 - _normal_cdf(z)

    side = (features.get("side") or "").lower()
    implied = _american_to_implied(odds)
    prob_under = 1 - prob_over

    if side == "under":
        prob_win = prob_under
        pick = "UNDER"
        edge = prob_win - implied
    elif side == "over":
        prob_win = prob_over
        pick = "OVER"
        edge = prob_win - implied
    else:
        edge_over = prob_over - implied
        edge_under = prob_under - implied
        if edge_over >= edge_under:
            prob_win = prob_over
            pick = "OVER"
            edge = edge_over
        else:
            prob_win = prob_under
            pick = "UNDER"
            edge = edge_under

    confidence = max(0.2, min(0.95, abs(prob_win - 0.5) * 2))

    if edge < min_edge or confidence < min_confidence:
        pick = "PASS"

    return {
        "prop_id": row.get("prop_id"),
        "projection": round(projection, 2),
        "probability": round(prob_win, 4),
        "edge": round(edge, 4),
        "confidence": round(confidence, 4),
        "pick": pick,
        "odds": odds,
    }
