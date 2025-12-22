"""Odds-aware scoring helpers."""

from typing import Dict


def score_prop(row: Dict) -> Dict:
    """Score a single prop row with probability, edge, and confidence."""
    return {
        "prop_id": row.get("prop_id"),
        "edge": 0.0,
        "confidence": 0.0,
        "pick": "PASS",
    }
