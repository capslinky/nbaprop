"""Build feature sets for scoring."""

from typing import List, Dict, Optional


def _stat_key(prop_type: Optional[str]) -> Optional[str]:
    if not prop_type:
        return None
    prop_type = prop_type.lower()
    if prop_type == "threes":
        return "fg3m"
    if prop_type == "pra":
        return "pra"
    return prop_type


def _extract_history(logs: List[Dict], key: str) -> List[float]:
    values = []
    for row in logs:
        val = row.get(key)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    return values


def _calc_recent_avg(values: List[float], window: int = 5) -> float:
    if not values:
        return 0.0
    subset = values[:window]
    return sum(subset) / len(subset)


def _calc_season_avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_features(props: List[Dict], raw_player_logs: Optional[List[Dict]] = None) -> List[Dict]:
    """Build model-ready features for each prop row."""
    player_logs_map = {}
    if raw_player_logs:
        for row in raw_player_logs:
            player_logs_map[row.get("player")] = row.get("logs", [])

    rows: List[Dict] = []
    for prop in props:
        player = prop.get("player")
        prop_type = prop.get("prop_type")
        stat_key = _stat_key(prop_type)
        logs = player_logs_map.get(player, [])
        values = _extract_history(logs, stat_key) if stat_key else []
        recent_avg = _calc_recent_avg(values)
        season_avg = _calc_season_avg(values)

        rows.append({
            "prop_id": prop.get("prop_id"),
            "features": {
                "recent_avg": recent_avg,
                "season_avg": season_avg,
                "minutes_trend": 1.0,
                "line": prop.get("line"),
                "odds": prop.get("odds"),
                "prop_type": prop_type,
                "side": prop.get("side"),
            },
        })
    return rows
