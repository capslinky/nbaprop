"""Build feature sets for scoring."""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

from nbaprop.normalization.ids import canonicalize_player_name, canonicalize_team_abbrev
from nbaprop.normalization.name_utils import load_roster_map, normalize_name_for_matching


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


def _calc_recent_std(values: List[float], window: int = 10) -> Optional[float]:
    if not values:
        return None
    subset = values[:window]
    if len(subset) < 2:
        return None
    mean = sum(subset) / len(subset)
    var = sum((val - mean) ** 2 for val in subset) / (len(subset) - 1)
    return var ** 0.5


def _calc_hit_rate(values: List[float], line: Optional[float], side: str, window: int) -> Optional[float]:
    if not values:
        return None
    if line in (None, 0, 0.0):
        return None
    sample = values[:window]
    if not sample:
        return None
    line_val = float(line)
    if side == "over":
        hits = sum(1 for value in sample if value > line_val)
    elif side == "under":
        hits = sum(1 for value in sample if value < line_val)
    else:
        return None
    return hits / len(sample)


def _calc_streak(values: List[float], line: Optional[float], side: str) -> Optional[int]:
    if not values:
        return None
    if line in (None, 0, 0.0):
        return None
    line_val = float(line)
    results = []
    if side == "over":
        results = [value > line_val for value in values]
    elif side == "under":
        results = [value < line_val for value in values]
    else:
        return None
    if not results:
        return None
    first = results[0]
    streak = 0
    for result in results:
        if result != first:
            break
        streak += 1
    return streak if first else -streak


def _trend_label(avg_short: Optional[float], avg_long: Optional[float], threshold: float = 0.5) -> Optional[str]:
    if avg_short is None or avg_long is None:
        return None
    delta = avg_short - avg_long
    if delta >= threshold:
        return "UP"
    if delta <= -threshold:
        return "DOWN"
    return "FLAT"


def _extract_usage(logs: List[Dict]) -> List[float]:
    values = []
    for row in logs:
        try:
            fga = float(row.get("fga", 0) or 0)
            fta = float(row.get("fta", 0) or 0)
            tov = float(row.get("turnovers", 0) or 0)
            minutes = float(row.get("minutes", 0) or 0)
        except (TypeError, ValueError):
            continue
        if minutes <= 0:
            continue
        usage = (fga + (0.44 * fta) + tov) / minutes
        values.append(usage)
    return values


def _calc_true_shooting(logs: List[Dict], window: int = 10) -> Optional[Dict]:
    """Calculate true shooting percentage from recent games.

    TS% = PTS / (2 * (FGA + 0.44 * FTA))
    Returns dict with ts_pct, league_avg (assumed 57%), and ts_factor.
    """
    league_avg_ts = 0.57
    pts_total = 0.0
    fga_total = 0.0
    fta_total = 0.0
    count = 0

    for row in logs[:window]:
        try:
            pts = float(row.get("pts", 0) or row.get("points", 0) or 0)
            fga = float(row.get("fga", 0) or 0)
            fta = float(row.get("fta", 0) or 0)
        except (TypeError, ValueError):
            continue
        if fga <= 0:
            continue
        pts_total += pts
        fga_total += fga
        fta_total += fta
        count += 1

    if count < 3 or fga_total <= 0:
        return None

    tsa = fga_total + (0.44 * fta_total)
    if tsa <= 0:
        return None

    ts_pct = pts_total / (2 * tsa)
    deviation = ts_pct - league_avg_ts
    # Regression factor: if shooting hot, expect regression down; if cold, expect bounce back
    # Use 30% weight toward league average
    ts_factor = 1.0 - (deviation * 0.3)
    # Clamp to reasonable range
    ts_factor = max(0.92, min(1.08, ts_factor))

    regression = "NONE"
    if deviation > 0.05:
        regression = "DOWN"
    elif deviation < -0.05:
        regression = "UP"

    return {
        "ts_pct": round(ts_pct, 4),
        "league_avg": league_avg_ts,
        "deviation": round(deviation, 4),
        "ts_factor": round(ts_factor, 4),
        "regression": regression,
    }


def _calc_vs_team_history(
    logs: List[Dict], opponent: Optional[str], stat_key: Optional[str], window: int = 5
) -> Optional[Dict]:
    """Calculate player's history against a specific opponent.

    Returns dict with vs_avg, overall_avg, and vs_factor.
    """
    if not opponent or not stat_key or not logs:
        return None

    opponent_norm = canonicalize_team_abbrev(opponent)
    if not opponent_norm:
        return None

    vs_values = []
    all_values = []

    for row in logs[:20]:  # Look at last 20 games for context
        try:
            val = float(row.get(stat_key, 0) or 0)
        except (TypeError, ValueError):
            continue

        all_values.append(val)

        # Check if this game was against the opponent
        matchup = row.get("matchup", "") or ""
        if opponent_norm in matchup or opponent in matchup:
            vs_values.append(val)

    if len(all_values) < 5:
        return None

    overall_avg = sum(all_values) / len(all_values) if all_values else 0

    if len(vs_values) < 2:
        # Not enough history vs this team
        return None

    vs_avg = sum(vs_values[:window]) / len(vs_values[:window]) if vs_values else overall_avg

    if overall_avg <= 0:
        return None

    # Factor based on vs-team performance relative to overall
    vs_factor = vs_avg / overall_avg
    # Clamp to reasonable range (±10%)
    vs_factor = max(0.90, min(1.10, vs_factor))

    return {
        "vs_avg": round(vs_avg, 2),
        "overall_avg": round(overall_avg, 2),
        "vs_factor": round(vs_factor, 4),
        "vs_games": len(vs_values),
    }


def _parse_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_minutes(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ":" in text:
        parts = text.split(":", 1)
        try:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins + (secs / 60.0)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_log_date(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    parsed = _parse_datetime(value)
    return parsed


def _team_from_matchup(matchup: Optional[str]) -> Optional[str]:
    if not matchup:
        return None
    text = str(matchup)
    team_part = None
    if " vs. " in text:
        team_part = text.split(" vs. ", 1)[0].strip()
    elif " @ " in text:
        team_part = text.split(" @ ", 1)[0].strip()
    elif "vs." in text:
        team_part = text.split("vs.", 1)[0].strip()
    elif "@" in text:
        team_part = text.split("@", 1)[0].strip()
    else:
        parts = text.split()
        if parts:
            team_part = parts[0]
    if not team_part:
        return None
    return canonicalize_team_abbrev(team_part)


def _team_from_logs(logs: List[Dict]) -> Optional[str]:
    if not logs:
        return None
    for row in logs:
        for key in ("team_abbrev", "TEAM_ABBREVIATION", "team", "TEAM"):
            team = row.get(key)
            if team:
                return canonicalize_team_abbrev(str(team))
        matchup = row.get("matchup")
        team = _team_from_matchup(matchup)
        if team:
            return team
    return None


def _usage_proxy_series(logs: List[Dict], max_games: int = 25) -> List[Tuple[datetime, float]]:
    if not logs:
        return []
    if logs and "date" in logs[0]:
        logs_sorted = sorted(
            logs,
            key=lambda row: _parse_log_date(row.get("date")) or datetime.min,
            reverse=True,
        )
    else:
        logs_sorted = list(logs)
    series: List[Tuple[datetime, float]] = []
    for row in logs_sorted:
        log_date = _parse_log_date(row.get("date"))
        minutes = _parse_minutes(row.get("minutes"))
        if log_date is None or minutes is None or minutes <= 0:
            continue
        try:
            fga = float(row.get("fga", 0) or 0)
            fta = float(row.get("fta", 0) or 0)
        except (TypeError, ValueError):
            continue
        usage = (fga + (0.44 * fta)) / minutes
        series.append((log_date, usage))
        if len(series) >= max_games:
            break
    return series


def _usage_proxy_avg(logs: List[Dict], max_games: int = 10) -> Optional[float]:
    series = _usage_proxy_series(logs, max_games=max_games)
    if not series:
        return None
    values = [val for _, val in series]
    if not values:
        return None
    return sum(values) / len(values)


def _expected_usage_factor(
    player_name: str,
    player_logs: List[Dict],
    team_abbrev: Optional[str],
    team_players: Dict[str, Dict[str, List[Dict]]],
    injury_by_name: Dict[str, Dict],
    min_samples: int = 5,
    max_teammates: int = 4,
    weight: float = 0.5,
) -> Tuple[float, Optional[str]]:
    if not player_logs or not team_abbrev:
        return 1.0, None
    player_series = _usage_proxy_series(player_logs, max_games=25)
    if not player_series:
        return 1.0, None
    player_last_date = player_series[0][0].date()

    candidates: List[Tuple[float, str, List[Dict]]] = []
    for teammate_name, logs in team_players.get(team_abbrev, {}).items():
        if teammate_name == player_name:
            continue
        avg_usage = _usage_proxy_avg(logs, max_games=10)
        if avg_usage is None:
            continue
        candidates.append((avg_usage, teammate_name, logs))

    if not candidates:
        return 1.0, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:max_teammates]

    factor = 1.0
    notes: List[str] = []

    for _, teammate_name, logs in candidates:
        teammate_series = _usage_proxy_series(logs, max_games=25)
        if not teammate_series:
            continue
        teammate_dates = {entry[0].date() for entry in teammate_series}
        if not teammate_dates:
            continue

        with_vals = [val for dt, val in player_series if dt.date() in teammate_dates]
        without_vals = [val for dt, val in player_series if dt.date() not in teammate_dates]
        if len(with_vals) < min_samples or len(without_vals) < min_samples:
            continue
        avg_with = sum(with_vals) / len(with_vals)
        avg_without = sum(without_vals) / len(without_vals)
        if avg_with <= 0 or avg_without <= 0:
            continue

        roster_key = normalize_name_for_matching(teammate_name)
        status = (injury_by_name.get(roster_key, {}).get("status") or "").upper()
        played_last_game = player_last_date in teammate_dates

        ratio = None
        label = None
        if status in _OUT_STATUSES and played_last_game:
            ratio = avg_without / avg_with
            label = "OUT"
        elif status in _RISK_STATUSES and not played_last_game:
            ratio = avg_with / avg_without
            label = "RETURN"
        if ratio is None:
            continue
        if abs(ratio - 1.0) < 0.03:
            continue

        adj = 1 + (ratio - 1.0) * weight
        adj = max(0.85, min(1.15, adj))
        factor *= adj
        notes.append(f"{teammate_name} {label} x{adj:.2f}")

    factor = max(0.85, min(1.15, factor))
    return factor, "; ".join(notes) if notes else None


def _calc_rest_days(logs: List[Dict], game_time) -> Tuple[Optional[int], Optional[bool]]:
    if not logs:
        return None, None
    last_game = _parse_datetime(logs[0].get("date"))
    if last_game is None:
        return None, None
    game_dt = _parse_datetime(game_time)
    if game_dt is None:
        return None, None
    rest_days = (game_dt.date() - last_game.date()).days
    b2b = rest_days <= 1 if rest_days is not None else None
    return rest_days, b2b


def _american_to_implied(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds_val = int(odds)
    except (TypeError, ValueError):
        return None
    if odds_val > 0:
        return 100 / (odds_val + 100)
    return abs(odds_val) / (abs(odds_val) + 100)


def _market_key(prop: Dict) -> Tuple[str, str, float]:
    player_key = prop.get("player_id") or canonicalize_player_name(prop.get("player_name") or prop.get("player") or "")
    prop_type = (prop.get("prop_type") or "").lower()
    line = prop.get("line")
    try:
        line_val = float(line)
    except (TypeError, ValueError):
        line_val = None
    return player_key, prop_type, line_val


def _build_player_name_map(players: Optional[List[Dict]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not players:
        return mapping
    for row in players:
        player_id = row.get("player_id")
        player_name = row.get("player_name")
        if player_id and player_name:
            mapping[player_id] = player_name
    return mapping


def _build_player_team_map(players: Optional[List[Dict]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not players:
        return mapping
    for row in players:
        player_id = row.get("player_id")
        team_abbrev = row.get("team_abbrev")
        if player_id and team_abbrev:
            mapping[player_id] = team_abbrev
    return mapping


def _build_injury_lookup(injuries: Optional[List[Dict]]) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    if not injuries:
        return lookup
    for row in injuries:
        player_id = row.get("player_id")
        if player_id and player_id not in lookup:
            lookup[player_id] = row
    return lookup


def _build_injury_name_lookup(injuries: Optional[List[Dict]]) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    if not injuries:
        return lookup
    for row in injuries:
        player_name = row.get("player_name")
        if not player_name:
            continue
        key = normalize_name_for_matching(player_name)
        if key and key not in lookup:
            lookup[key] = row
    return lookup


def _build_team_injury_counts(injuries: Optional[List[Dict]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    if not injuries:
        return counts
    roster_map = load_roster_map()
    for row in injuries:
        team = row.get("team_abbrev")
        if not team:
            player_name = row.get("player_name") or ""
            team = roster_map.get(normalize_name_for_matching(player_name))
        if not team:
            continue
        status = (row.get("status") or "").upper()
        bucket = "other"
        if status in {"OUT", "O", "DNP", "INJURED", "SUSPENDED", "DOUBTFUL", "D"}:
            bucket = "out"
        elif status in {"QUESTIONABLE", "Q", "GTD", "GAME TIME DECISION", "GAME-TIME DECISION"}:
            bucket = "questionable"
        elif status in {"PROBABLE", "P"}:
            bucket = "probable"
        team_counts = counts.setdefault(team, {"out": 0, "questionable": 0, "probable": 0})
        team_counts[bucket] = team_counts.get(bucket, 0) + 1
    return counts


_OUT_STATUSES = {
    "OUT",
    "O",
    "DNP",
    "INJURED",
    "SUSPENDED",
    "DOUBTFUL",
    "D",
}
_RISK_STATUSES = {
    "QUESTIONABLE",
    "Q",
    "GTD",
    "GAME TIME DECISION",
    "GAME-TIME DECISION",
}
_SOFT_STATUSES = {
    "PROBABLE",
    "P",
}


def _build_team_injury_details(injuries: Optional[List[Dict]]) -> Dict[str, Dict[str, List[str]]]:
    details: Dict[str, Dict[str, List[str]]] = {}
    if not injuries:
        return details
    roster_map = load_roster_map()
    for row in injuries:
        team = row.get("team_abbrev")
        player_name = row.get("player_name") or row.get("player") or ""
        if not team:
            roster_key = normalize_name_for_matching(player_name)
            team = roster_map.get(roster_key)
        if not team or not player_name:
            continue
        status = (row.get("status") or "").upper()
        injury = row.get("injury") or row.get("injury_detail") or ""
        if injury:
            label = f"{player_name} ({injury})"
        else:
            label = player_name
        bucket = None
        if status in _OUT_STATUSES:
            bucket = "out"
        elif status in _RISK_STATUSES:
            bucket = "questionable"
        elif status in _SOFT_STATUSES:
            bucket = "probable"
        if not bucket:
            continue
        team_bucket = details.setdefault(team, {"out": [], "questionable": [], "probable": []})
        team_bucket[bucket].append(label)
    return details


def build_features(
    props: List[Dict],
    raw_player_logs: Optional[List[Dict]] = None,
    injuries: Optional[List[Dict]] = None,
    players: Optional[List[Dict]] = None,
    team_stats: Optional[List[Dict]] = None,
    excluded_prop_types: Optional[List[str]] = None,
) -> List[Dict]:
    """Build model-ready features for each prop row."""
    excluded = {str(item).lower() for item in (excluded_prop_types or [])}
    player_logs_map = {}
    if raw_player_logs:
        for row in raw_player_logs:
            player_name = row.get("player")
            if not player_name:
                continue
            canonical_name = canonicalize_player_name(player_name)
            player_logs_map[canonical_name] = row.get("logs", [])

    player_name_map = _build_player_name_map(players)
    player_team_map = _build_player_team_map(players)
    injury_by_id = _build_injury_lookup(injuries)
    injury_by_name = _build_injury_name_lookup(injuries)
    roster_map = load_roster_map()
    team_injury_counts = _build_team_injury_counts(injuries)
    team_injury_details = _build_team_injury_details(injuries)

    player_team_by_name: Dict[str, str] = {}
    if players:
        for row in players:
            name = row.get("player_name")
            team = row.get("team_abbrev")
            key = normalize_name_for_matching(name or "")
            if key and team:
                player_team_by_name[key] = team

    team_players: Dict[str, Dict[str, List[Dict]]] = {}
    for player_name, logs in player_logs_map.items():
        roster_key = normalize_name_for_matching(player_name)
        team = player_team_by_name.get(roster_key) or roster_map.get(roster_key)
        if not team:
            team = _team_from_logs(logs)
        if not team:
            continue
        team_players.setdefault(team, {})[player_name] = logs

    expected_usage_cache: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}

    defense_map: Dict[str, float] = {}
    pace_map: Dict[str, float] = {}
    def_values: List[float] = []
    pace_values: List[float] = []
    if team_stats:
        for row in team_stats:
            for defense_row in row.get("defense_ratings", []) or []:
                team = defense_row.get("team_abbrev")
                rating = defense_row.get("def_rating")
                if team is not None and rating is not None:
                    try:
                        rating_val = float(rating)
                    except (TypeError, ValueError):
                        continue
                    defense_map[team] = rating_val
                    def_values.append(rating_val)
            for pace_row in row.get("pace", []) or []:
                team = pace_row.get("team_abbrev")
                pace = pace_row.get("pace")
                if team is not None and pace is not None:
                    try:
                        pace_val = float(pace)
                    except (TypeError, ValueError):
                        continue
                    pace_map[team] = pace_val
                    pace_values.append(pace_val)

    league_def_avg = sum(def_values) / len(def_values) if def_values else None
    league_pace_avg = sum(pace_values) / len(pace_values) if pace_values else None

    market_map: Dict[Tuple[str, str, float], Dict[str, Optional[float]]] = {}
    for prop in props:
        prop_type = (prop.get("prop_type") or "").lower()
        if prop_type in excluded:
            continue
        key = _market_key(prop)
        if key[2] is None:
            continue
        side = (prop.get("side") or "").lower()
        odds = prop.get("odds")
        entry = market_map.setdefault(key, {})
        if side == "over":
            entry["over_odds"] = odds
        elif side == "under":
            entry["under_odds"] = odds

    rows: List[Dict] = []
    for prop in props:
        player_id = prop.get("player_id")
        player = prop.get("player") or prop.get("player_name") or player_name_map.get(player_id)
        canonical_name = canonicalize_player_name(player) if player else None
        roster_key = normalize_name_for_matching(player or "")
        logs = player_logs_map.get(canonical_name, [])
        player_team = prop.get("team_abbrev") or player_team_map.get(player_id) or roster_map.get(roster_key)
        if not player_team:
            player_team = _team_from_logs(logs)
        home_team = prop.get("home_team")
        away_team = prop.get("away_team")
        opponent_team = None
        is_home = None
        if player_team and home_team and away_team:
            if player_team == home_team:
                opponent_team = away_team
                is_home = True
            elif player_team == away_team:
                opponent_team = home_team
                is_home = False
        prop_type = prop.get("prop_type")
        stat_key = _stat_key(prop_type)
        values = _extract_history(logs, stat_key) if stat_key else []
        n_games = len(values)
        recent_avg = _calc_recent_avg(values)
        season_avg = _calc_season_avg(values)
        recent_std = _calc_recent_std(values)
        avg_10 = _calc_recent_avg(values, window=10)
        avg_15 = _calc_recent_avg(values, window=15)
        trend = _trend_label(recent_avg, avg_15)
        hit_rate_over_10 = _calc_hit_rate(values, prop.get("line"), "over", 10)
        hit_rate_under_10 = _calc_hit_rate(values, prop.get("line"), "under", 10)
        hit_rate_over_15 = _calc_hit_rate(values, prop.get("line"), "over", 15)
        hit_rate_under_15 = _calc_hit_rate(values, prop.get("line"), "under", 15)
        streak_over = _calc_streak(values, prop.get("line"), "over")
        streak_under = _calc_streak(values, prop.get("line"), "under")
        minutes_values = _extract_history(logs, "minutes")
        minutes_avg_5 = _calc_recent_avg(minutes_values)
        minutes_avg_15 = _calc_recent_avg(minutes_values, window=15)
        minutes_trend = (minutes_avg_5 / minutes_avg_15) if minutes_avg_15 else None

        usage_values = _extract_usage(logs)
        usage_avg_5 = _calc_recent_avg(usage_values)
        usage_avg_15 = _calc_recent_avg(usage_values, window=15)

        expected_usage_factor = 1.0
        expected_usage_notes = None
        if canonical_name and player_team:
            cache_key = (canonical_name, str(prop.get("game_time") or ""))
            cached = expected_usage_cache.get(cache_key)
            if cached:
                expected_usage_factor, expected_usage_notes = cached
            else:
                expected_usage_factor, expected_usage_notes = _expected_usage_factor(
                    canonical_name,
                    logs,
                    player_team,
                    team_players,
                    injury_by_name,
                )
                expected_usage_cache[cache_key] = (expected_usage_factor, expected_usage_notes)

        rest_days, b2b = _calc_rest_days(logs, prop.get("game_time"))

        # Calculate true shooting efficiency regression
        ts_info = _calc_true_shooting(logs) if stat_key in ("points", "pra") else None

        # Calculate vs-team history
        vs_team_info = _calc_vs_team_history(logs, opponent_team, stat_key)

        opp_def = defense_map.get(opponent_team) if opponent_team else None
        team_pace = pace_map.get(player_team) if player_team else None
        team_injuries = team_injury_counts.get(player_team, {}) if player_team else {}
        team_injury_detail = team_injury_details.get(player_team, {}) if player_team else {}
        team_out_detail = "; ".join(team_injury_detail.get("out", []))
        team_questionable_detail = "; ".join(team_injury_detail.get("questionable", []))
        team_probable_detail = "; ".join(team_injury_detail.get("probable", []))
        market_key = _market_key(prop)
        market_entry = market_map.get(market_key, {})
        over_odds = market_entry.get("over_odds")
        under_odds = market_entry.get("under_odds")
        over_implied = _american_to_implied(over_odds)
        under_implied = _american_to_implied(under_odds)
        market_prob = None
        if over_implied is not None and under_implied is not None:
            denom = over_implied + under_implied
            market_prob_over = (over_implied / denom) if denom else None
            market_prob_under = (under_implied / denom) if denom else None
        else:
            market_prob_over = over_implied
            market_prob_under = under_implied
        side = (prop.get("side") or "").lower()
        if side == "over":
            market_prob = market_prob_over
        elif side == "under":
            market_prob = market_prob_under

        injury_entry = None
        if player_id:
            injury_entry = injury_by_id.get(player_id)
        if injury_entry is None and roster_key:
            injury_entry = injury_by_name.get(roster_key)

        rows.append({
            "prop_id": prop.get("prop_id"),
            "features": {
                "recent_avg": recent_avg,
                "season_avg": season_avg,
                "n_games": n_games,
                "recent_std": recent_std,
                "avg_10": avg_10,
                "avg_15": avg_15,
                "trend": trend,
                "hit_rate_over_10": hit_rate_over_10,
                "hit_rate_under_10": hit_rate_under_10,
                "hit_rate_over_15": hit_rate_over_15,
                "hit_rate_under_15": hit_rate_under_15,
                "streak_over": streak_over,
                "streak_under": streak_under,
                "minutes_avg_5": minutes_avg_5,
                "minutes_avg_15": minutes_avg_15,
                "minutes_trend": minutes_trend,
                "usage_avg_5": usage_avg_5,
                "usage_avg_15": usage_avg_15,
                "expected_usage_factor": expected_usage_factor,
                "expected_usage_notes": expected_usage_notes,
                "rest_days": rest_days,
                "b2b": b2b,
                "line": prop.get("line"),
                "odds": prop.get("odds"),
                "prop_type": prop_type,
                "side": prop.get("side"),
                "event_id": prop.get("event_id"),
                "home_team": prop.get("home_team"),
                "away_team": prop.get("away_team"),
                "game_time": prop.get("game_time"),
                "player_team": player_team,
                "opponent_team": opponent_team,
                "is_home": is_home,
                "opp_def_rating": opp_def,
                "team_pace": team_pace,
                "league_def_avg": league_def_avg,
                "league_pace_avg": league_pace_avg,
                "market_over_odds": over_odds,
                "market_under_odds": under_odds,
                "market_prob": market_prob,
                "team_out_count": team_injuries.get("out", 0),
                "team_questionable_count": team_injuries.get("questionable", 0),
                "team_probable_count": team_injuries.get("probable", 0),
                "team_out_detail": team_out_detail or None,
                "team_questionable_detail": team_questionable_detail or None,
                "team_probable_detail": team_probable_detail or None,
                "player_id": player_id,
                "player_name": player,
                "injury_status": injury_entry.get("status") if injury_entry else None,
                "injury_source": injury_entry.get("source") if injury_entry else None,
                "injury_detail": injury_entry.get("injury") if injury_entry else None,
                "injury_team": injury_entry.get("team_abbrev") if injury_entry else None,
                # Game total and spread from odds (if available)
                "game_total": prop.get("game_total") or prop.get("over_under"),
                "spread": prop.get("spread"),
                # True shooting efficiency regression
                "ts_pct": ts_info.get("ts_pct") if ts_info else None,
                "ts_factor": ts_info.get("ts_factor") if ts_info else None,
                "ts_regression": ts_info.get("regression") if ts_info else None,
                # Vs-team history
                "vs_team_avg": vs_team_info.get("vs_avg") if vs_team_info else None,
                "vs_team_factor": vs_team_info.get("vs_factor") if vs_team_info else None,
                "vs_team_games": vs_team_info.get("vs_games") if vs_team_info else None,
            },
        })
    return rows
