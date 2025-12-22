"""Odds-aware scoring helpers."""

from typing import Dict


def score_prop(row: Dict) -> Dict:
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

    edge = 0.0
    confidence = 0.2
    pick = "PASS"
    if abs(edge) >= 0.03:
        pick = "OVER" if edge > 0 else "UNDER"
    return {
        "prop_id": row.get("prop_id"),
        "edge": edge,
        "confidence": confidence,
        "pick": pick,
        "odds": odds,
    }
