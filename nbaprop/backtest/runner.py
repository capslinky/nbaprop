"""Backtest runner for recent games."""

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import sqlite3

from nbaprop.features.pipeline import build_features
from nbaprop.models.baseline import BaselineModel
from nbaprop.normalization.ids import canonicalize_player_name
from nbaprop.ingestion.nba_stats import fetch_player_logs
from nbaprop.storage import FileCache


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


def _calc_avg(values: List[float], window: int) -> Optional[float]:
    if not values:
        return None
    sample = values[:window]
    if not sample:
        return None
    return sum(sample) / len(sample)


def _count_hits(values: List[float], line: Optional[float], side: str, window: int) -> Tuple[int, int]:
    if not values or line in (None, 0, 0.0):
        return 0, 0
    if side not in {"OVER", "UNDER"}:
        return 0, 0
    sample = values[:window]
    if not sample:
        return 0, 0
    if side == "OVER":
        hits = sum(1 for v in sample if v > float(line))
    elif side == "UNDER":
        hits = sum(1 for v in sample if v < float(line))
    else:
        hits = 0
    return hits, len(sample)


def _calc_streak(values: List[float], line: Optional[float], side: str) -> int:
    if not values or line in (None, 0, 0.0):
        return 0
    if side not in {"OVER", "UNDER"}:
        return 0
    hits = []
    for v in values:
        if side == "OVER":
            hits.append(v > float(line))
        else:
            hits.append(v < float(line))
    if not hits:
        return 0
    first = hits[0]
    streak = 0
    for hit in hits:
        if hit != first:
            break
        streak += 1
    return streak if first else -streak


def _trend_label(avg_short: Optional[float], avg_long: Optional[float], threshold: float = 0.5) -> str:
    if avg_short is None or avg_long is None:
        return "UNKNOWN"
    delta = avg_short - avg_long
    if delta >= threshold:
        return "UP"
    if delta <= -threshold:
        return "DOWN"
    return "FLAT"


def _parse_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _season_for_date(game_date: date) -> str:
    if game_date.month >= 10:
        start_year = game_date.year
    else:
        start_year = game_date.year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def load_historical_props_from_db(
    db_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict]:
    if not db_path.exists():
        return []
    query = (
        "SELECT date, player, prop_type, line, odds_over, odds_under, actual, hit_over "
        "FROM historical_props"
    )
    params: List[str] = []
    if start_date and end_date:
        query += " WHERE date BETWEEN ? AND ?"
        params = [start_date, end_date]
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error:
        return []

    output = []
    for row in rows:
        date_str, player, prop_type, line, odds_over, odds_under, actual, hit_over = row
        output.append({
            "date": date_str,
            "player": player,
            "prop_type": prop_type,
            "line": line,
            "odds_over": odds_over,
            "odds_under": odds_under,
            "actual": actual,
            "hit_over": hit_over,
        })
    return output


def run_historical_model_backtest(
    historical_props: List[Dict],
    cache_dir: str,
    min_edge: float = 0.03,
    min_confidence: float = 0.4,
    prop_type_min_edge: Optional[Dict[str, float]] = None,
    prop_type_min_confidence: Optional[Dict[str, float]] = None,
    excluded_prop_types: Optional[List[str]] = None,
    injury_risk_edge_multiplier: float = 1.0,
    injury_risk_confidence_multiplier: float = 1.0,
    market_blend: float = 0.7,
    calibration: Optional[Dict] = None,
    odds_min: Optional[int] = None,
    odds_max: Optional[int] = None,
    odds_confidence_min: Optional[int] = None,
    odds_confidence_max: Optional[int] = None,
    confidence_cap_outside_band: Optional[float] = 0.65,
    base_delay: Optional[float] = None,
    cache_only: bool = False,
) -> Dict:
    """Backtest the current model against historical props with actual outcomes."""
    excluded = {str(item).lower() for item in (excluded_prop_types or [])}
    players_by_season: Dict[str, set] = {}
    prop_rows = []
    for row in historical_props:
        game_date = _parse_date(row.get("date"))
        if not game_date:
            continue
        player = row.get("player")
        prop_type = row.get("prop_type")
        line = row.get("line")
        if not player or not prop_type or line is None:
            continue
        if str(prop_type).lower() in excluded:
            continue
        season = _season_for_date(game_date)
        players_by_season.setdefault(season, set()).add(player)
        prop_rows.append((row, game_date, season))

    logs_by_player_season: Dict[Tuple[str, str], List[Dict]] = {}
    for season, players in players_by_season.items():
        fetched = fetch_player_logs(
            sorted(players),
            cache=FileCache(Path(cache_dir) / "snapshots"),
            ttl_seconds=3600,
            season=season,
            base_delay=base_delay,
            cache_dir=cache_dir,
            cache_only=cache_only,
        )
        for row in fetched:
            player = row.get("player")
            if not player:
                continue
            key = (canonicalize_player_name(player), season)
            logs_by_player_season[key] = row.get("logs", [])

    model = BaselineModel(
        min_edge=min_edge,
        min_confidence=min_confidence,
        prop_type_min_edge=prop_type_min_edge,
        prop_type_min_confidence=prop_type_min_confidence,
        excluded_prop_types=excluded_prop_types,
        injury_risk_edge_multiplier=injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=injury_risk_confidence_multiplier,
        market_blend=market_blend,
        calibration=calibration,
        odds_min=odds_min,
        odds_max=odds_max,
        odds_confidence_min=odds_confidence_min,
        odds_confidence_max=odds_confidence_max,
        confidence_cap_outside_band=confidence_cap_outside_band,
    )
    results: List[Dict] = []
    wins = losses = pushes = 0
    pick_counts = {"OVER": 0, "UNDER": 0, "PASS": 0}

    for row, game_date, season in prop_rows:
        player = row.get("player")
        prop_type = (row.get("prop_type") or "").lower()
        try:
            line_val = float(row.get("line"))
        except (TypeError, ValueError):
            continue

        odds_over = row.get("odds_over")
        odds_under = row.get("odds_under")
        props = []
        if odds_over is not None:
            props.append({
                "player": player,
                "prop_type": prop_type,
                "line": line_val,
                "side": "over",
                "odds": odds_over,
                "game_time": game_date.isoformat(),
            })
        if odds_under is not None:
            props.append({
                "player": player,
                "prop_type": prop_type,
                "line": line_val,
                "side": "under",
                "odds": odds_under,
                "game_time": game_date.isoformat(),
            })
        if not props:
            continue

        logs = logs_by_player_season.get((canonicalize_player_name(player), season), [])
        filtered_logs = []
        for log in logs:
            log_date = _parse_datetime(log.get("date"))
            if log_date is None or log_date.date() >= game_date:
                continue
            filtered_logs.append(log)

        feature_rows = build_features(
            props,
            raw_player_logs=[{"player": player, "logs": filtered_logs}],
            injuries=None,
            players=None,
            team_stats=None,
            excluded_prop_types=excluded_prop_types,
        )
        if not feature_rows:
            continue
        picks = model.predict(feature_rows)
        if not picks:
            continue

        candidates = [pick for pick in picks if pick.get("pick") in ("OVER", "UNDER")]
        if candidates:
            best_pick = max(
                candidates,
                key=lambda p: (p.get("edge") or 0.0, p.get("confidence") or 0.0),
            )
        else:
            best_pick = picks[0]

        pick_side = best_pick.get("pick", "PASS")
        pick_counts[pick_side] = pick_counts.get(pick_side, 0) + 1

        actual = row.get("actual")
        try:
            actual_val = float(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_val = None

        result = None
        if pick_side in ("OVER", "UNDER") and actual_val is not None:
            if actual_val == line_val:
                result = "PUSH"
                pushes += 1
            elif pick_side == "OVER":
                result = "WIN" if actual_val > line_val else "LOSS"
            else:
                result = "WIN" if actual_val < line_val else "LOSS"

            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1

        results.append({
            "player_name": player,
            "prop_type": prop_type,
            "line": line_val,
            "date": game_date.isoformat(),
            "pick": pick_side,
            "edge": best_pick.get("edge"),
            "confidence": best_pick.get("confidence"),
            "projection": best_pick.get("projection"),
            "probability": best_pick.get("probability"),
            "model_probability": best_pick.get("model_probability"),
            "odds": best_pick.get("odds"),
            "actual": actual_val,
            "result": result,
            "source": "nba_historical_cache",
        })

    total = wins + losses + pushes
    summary = {
        "total_props": len(results),
        "graded_props": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
        "pick_counts": pick_counts,
    }
    return {"results": results, "summary": summary}


def run_recent_backtest(
    props: List[Dict],
    raw_player_logs: List[Dict],
    injuries: Optional[List[Dict]] = None,
    players: Optional[List[Dict]] = None,
    team_stats: Optional[List[Dict]] = None,
    min_edge: float = 0.03,
    min_confidence: float = 0.4,
    prop_type_min_edge: Optional[Dict[str, float]] = None,
    prop_type_min_confidence: Optional[Dict[str, float]] = None,
    excluded_prop_types: Optional[List[str]] = None,
    injury_risk_edge_multiplier: float = 1.0,
    injury_risk_confidence_multiplier: float = 1.0,
    market_blend: float = 0.7,
    calibration: Optional[Dict] = None,
    odds_min: Optional[int] = None,
    odds_max: Optional[int] = None,
    odds_confidence_min: Optional[int] = None,
    odds_confidence_max: Optional[int] = None,
    confidence_cap_outside_band: Optional[float] = 0.65,
    windows: Sequence[int] = (10, 15),
) -> Dict:
    """Evaluate model picks against last N games per player."""
    features = build_features(
        props,
        raw_player_logs=raw_player_logs,
        injuries=injuries,
        players=players,
        team_stats=team_stats,
        excluded_prop_types=excluded_prop_types,
    )
    model = BaselineModel(
        min_edge=min_edge,
        min_confidence=min_confidence,
        prop_type_min_edge=prop_type_min_edge,
        prop_type_min_confidence=prop_type_min_confidence,
        excluded_prop_types=excluded_prop_types,
        injury_risk_edge_multiplier=injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=injury_risk_confidence_multiplier,
        market_blend=market_blend,
        calibration=calibration,
        odds_min=odds_min,
        odds_max=odds_max,
        odds_confidence_min=odds_confidence_min,
        odds_confidence_max=odds_confidence_max,
        confidence_cap_outside_band=confidence_cap_outside_band,
    )
    picks = model.predict(features)
    picks_by_prop = {row.get("prop_id"): row for row in picks}

    logs_map: Dict[str, List[Dict]] = {}
    for row in raw_player_logs:
        player_name = row.get("player")
        if not player_name:
            continue
        logs_map[canonicalize_player_name(player_name)] = row.get("logs", [])

    totals = {int(window): {"hits": 0, "samples": 0} for window in windows}
    pick_counts = {"OVER": 0, "UNDER": 0, "PASS": 0}

    results: List[Dict] = []
    for row in features:
        prop_id = row.get("prop_id")
        feature_payload = row.get("features", {})
        pick_row = picks_by_prop.get(prop_id, {})
        pick_side = pick_row.get("pick", "PASS")
        pick_counts[pick_side] = pick_counts.get(pick_side, 0) + 1

        player_name = feature_payload.get("player_name")
        canonical_name = canonicalize_player_name(player_name) if player_name else None
        logs = logs_map.get(canonical_name, [])
        stat_key = _stat_key(feature_payload.get("prop_type"))
        values = _extract_history(logs, stat_key) if stat_key else []

        avg_5 = _calc_avg(values, 5)
        avg_10 = _calc_avg(values, 10)
        avg_15 = _calc_avg(values, 15)
        trend = _trend_label(avg_5, avg_15)

        line = feature_payload.get("line")
        streak = _calc_streak(values, line, pick_side)

        row_out = {
            "prop_id": prop_id,
            "player_name": player_name,
            "prop_type": feature_payload.get("prop_type"),
            "line": line,
            "odds": feature_payload.get("odds"),
            "pick": pick_side,
            "edge": pick_row.get("edge"),
            "confidence": pick_row.get("confidence"),
            "projection": pick_row.get("projection"),
            "probability": pick_row.get("probability"),
            "injury_status": pick_row.get("injury_status"),
            "injury_source": pick_row.get("injury_source"),
            "trend": trend,
            "avg_5": avg_5,
            "avg_10": avg_10,
            "avg_15": avg_15,
            "streak": streak,
        }

        for window in windows:
            window = int(window)
            hits, sample = _count_hits(values, line, pick_side, window)
            if sample:
                totals[window]["hits"] += hits
                totals[window]["samples"] += sample
            hit_rate = round(hits / sample, 4) if sample else None
            row_out[f"hit_rate_{window}"] = hit_rate
            row_out[f"samples_{window}"] = sample

        results.append(row_out)

    summary = {
        "total_props": len(features),
        "pick_counts": pick_counts,
        "window_metrics": {
            str(window): {
                "hits": totals[int(window)]["hits"],
                "samples": totals[int(window)]["samples"],
                "hit_rate": round(
                    totals[int(window)]["hits"] / totals[int(window)]["samples"], 4
                ) if totals[int(window)]["samples"] else None,
            }
            for window in windows
        },
    }

    return {
        "results": results,
        "summary": summary,
    }


def run_historical_backtest(
    historical_props: List[Dict],
    raw_player_logs: List[Dict],
    injuries: Optional[List[Dict]] = None,
    players: Optional[List[Dict]] = None,
    team_stats: Optional[List[Dict]] = None,
    min_edge: float = 0.03,
    min_confidence: float = 0.4,
    market_blend: float = 0.7,
    calibration: Optional[Dict] = None,
    odds_min: Optional[int] = None,
    odds_max: Optional[int] = None,
    odds_confidence_min: Optional[int] = None,
    odds_confidence_max: Optional[int] = None,
    confidence_cap_outside_band: Optional[float] = 0.65,
    excluded_prop_types: Optional[List[str]] = None,
) -> Dict:
    """Leakage-free backtest using historical props with actual outcomes."""
    logs_map: Dict[str, List[Dict]] = {}
    for row in raw_player_logs:
        player_name = row.get("player")
        if not player_name:
            continue
        logs_map[canonicalize_player_name(player_name)] = row.get("logs", [])

    model = BaselineModel(
        min_edge=min_edge,
        min_confidence=min_confidence,
        market_blend=market_blend,
        calibration=calibration,
        odds_min=odds_min,
        odds_max=odds_max,
        odds_confidence_min=odds_confidence_min,
        odds_confidence_max=odds_confidence_max,
        confidence_cap_outside_band=confidence_cap_outside_band,
    )
    results: List[Dict] = []
    wins = 0
    losses = 0
    pushes = 0

    excluded = {str(item).lower() for item in (excluded_prop_types or [])}
    for prop in historical_props:
        prop_type = (prop.get("prop_type") or "").lower()
        if prop_type in excluded:
            continue
        player_name = prop.get("player_name") or prop.get("player")
        if not player_name:
            continue
        canonical_name = canonicalize_player_name(player_name)
        logs = logs_map.get(canonical_name, [])

        game_time = _parse_datetime(prop.get("game_time") or prop.get("date"))
        if game_time is None:
            continue

        filtered_logs = []
        for row in logs:
            row_date = _parse_datetime(row.get("date"))
            if row_date is None or row_date >= game_time:
                continue
            filtered_logs.append(row)

        feature_rows = build_features(
            [prop],
            raw_player_logs=[{"player": player_name, "logs": filtered_logs}],
            injuries=injuries,
            players=players,
            team_stats=team_stats,
            excluded_prop_types=excluded_prop_types,
        )
        if not feature_rows:
            continue
        pick = model.predict(feature_rows)[0]

        actual = prop.get("actual")
        try:
            actual_val = float(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_val = None

        result = None
        if pick.get("pick") in ("OVER", "UNDER") and actual_val is not None and pick.get("line") is not None:
            line = float(pick.get("line"))
            if actual_val == line:
                result = "PUSH"
                pushes += 1
            elif pick["pick"] == "OVER":
                result = "WIN" if actual_val > line else "LOSS"
            else:
                result = "WIN" if actual_val < line else "LOSS"

            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1

        output = dict(prop)
        output.update({
            "pick": pick.get("pick"),
            "edge": pick.get("edge"),
            "confidence": pick.get("confidence"),
            "projection": pick.get("projection"),
            "probability": pick.get("probability"),
            "result": result,
        })
        results.append(output)

    total = wins + losses + pushes
    summary = {
        "total_props": len(historical_props),
        "graded_props": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
    }
    return {"results": results, "summary": summary}
