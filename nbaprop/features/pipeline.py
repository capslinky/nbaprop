"""Build feature sets for scoring."""

from typing import List, Dict


def build_features(props: List[Dict]) -> List[Dict]:
    """Build model-ready features for each prop row."""
    rows: List[Dict] = []
    for prop in props:
        rows.append({
            "prop_id": prop.get("prop_id"),
            "features": {
                "recent_avg": 0.0,
                "season_avg": 0.0,
                "minutes_trend": 1.0,
                "line": prop.get("line"),
                "odds": prop.get("odds"),
                "prop_type": prop.get("prop_type"),
                "side": prop.get("side"),
            },
        })
    return rows
