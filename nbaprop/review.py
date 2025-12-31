"""Review and calibration utilities for daily picks."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sqlite3
from zoneinfo import ZoneInfo
import csv
import os

from nbaprop.ingestion.nba_stats import fetch_player_logs
from nbaprop.normalization.name_utils import normalize_name_for_matching
from nbaprop.storage import FileCache


def _parse_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _game_date(value, timezone: str = "America/New_York") -> Optional[datetime.date]:
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    if parsed.tzinfo is None:
        return parsed.date()
    return parsed.astimezone(ZoneInfo(timezone)).date()


def _parse_date(value: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _stat_key(prop_type: str) -> str:
    prop = (prop_type or "").lower()
    if prop in ("threes", "3pt", "fg3m"):
        return "fg3m"
    if prop in ("pra", "pts_reb_ast"):
        return "pra"
    if prop in ("points", "pts"):
        return "points"
    if prop in ("rebounds", "reb"):
        return "rebounds"
    if prop in ("assists", "ast"):
        return "assists"
    if prop in ("blocks", "blk"):
        return "blocks"
    if prop in ("steals", "stl"):
        return "steals"
    if prop in ("turnovers", "tov"):
        return "turnovers"
    return prop


def _normalize_prop_type(prop_type: Optional[str]) -> str:
    prop = (prop_type or "").lower()
    if prop in ("pts", "points"):
        return "points"
    if prop in ("reb", "rebounds"):
        return "rebounds"
    if prop in ("ast", "assists"):
        return "assists"
    if prop in ("threes", "3pt", "fg3m"):
        return "threes"
    if prop in ("pra", "pts_reb_ast"):
        return "pra"
    if prop in ("stl", "steals"):
        return "steals"
    if prop in ("blk", "blocks"):
        return "blocks"
    if prop in ("tov", "turnovers"):
        return "turnovers"
    return prop or "unknown"


def find_latest_picks_for_date(picks_dir: Path, target_date: str) -> Path:
    target = _parse_date(target_date)
    if target is None:
        raise ValueError(f"Invalid date: {target_date}")

    candidates = sorted(
        picks_dir.glob("picks_filtered_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                game_date = _game_date(row.get("game_time"))
                if not game_date:
                    continue
                if game_date == target:
                    return path
    raise FileNotFoundError(f"No picks found for {target_date} in {picks_dir}")


def load_picks(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def load_picks_from_paths(paths: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def load_picks_from_dir(picks_dir: Path, pattern: str = "picks_filtered_*.csv") -> List[Dict]:
    paths = sorted(picks_dir.glob(pattern))
    return load_picks_from_paths(paths)


def _load_cached_logs_from_db(
    db_path: Path,
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> Dict[str, Dict]:
    logs_by_player_date: Dict[str, Dict] = {}
    query = (
        "SELECT player_name, game_date, points, rebounds, assists, threes, pra, "
        "steals, blocks, turnovers FROM game_logs"
    )
    params: List[str] = []
    if start_date and end_date:
        query += " WHERE game_date BETWEEN ? AND ?"
        params = [start_date.isoformat(), end_date.isoformat()]
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error:
        return logs_by_player_date

    for player_name, game_date, points, rebounds, assists, threes, pra, steals, blocks, turnovers in rows:
        if not player_name or not game_date:
            continue
        try:
            date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        key = normalize_name_for_matching(player_name)
        date_map = logs_by_player_date.setdefault(key, {})
        date_map[date_obj] = {
            "points": points,
            "rebounds": rebounds,
            "assists": assists,
            "fg3m": threes,
            "pra": pra,
            "steals": steals,
            "blocks": blocks,
            "turnovers": turnovers,
        }
    return logs_by_player_date


def evaluate_picks(
    picks: List[Dict],
    target_date: str,
    cache_dir: str = ".cache",
    base_delay: Optional[float] = None,
    timezone: str = "America/New_York",
    odds_snapshot: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    target = _parse_date(target_date)
    if target is None:
        raise ValueError(f"Invalid date: {target_date}")

    players = sorted({row.get("player_name") for row in picks if row.get("player_name")})
    cache = FileCache(Path(cache_dir) / "snapshots")
    player_logs = fetch_player_logs(
        players,
        cache,
        ttl_seconds=3600,
        base_delay=base_delay,
        cache_dir=cache_dir,
    )
    logs_map = {
        row.get("player"): row.get("logs", [])
        for row in player_logs
        if row.get("player")
    }

    results: List[Dict] = []
    wins = losses = pushes = 0
    pass_count = 0
    missing_actuals = 0
    by_prop = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0, "total": 0})
    for row in picks:
        game_date = _game_date(row.get("game_time"), timezone=timezone)
        if not game_date or game_date != target:
            continue

        player = row.get("player_name")
        prop_type = row.get("prop_type") or ""
        side = (row.get("pick") or row.get("side") or "").upper()
        line = row.get("line")
        if not player or not prop_type or not side:
            continue
        if side not in ("OVER", "UNDER"):
            pass_count += 1
            out_row = dict(row)
            out_row["actual"] = None
            out_row["result"] = None
            results.append(out_row)
            continue

        stat_key = _stat_key(prop_type)
        actual = None
        for log in logs_map.get(player, []):
            log_date = _parse_datetime(log.get("date"))
            if not log_date or log_date.date() != target:
                continue
            actual = log.get(stat_key)
            break

        result = None
        if actual is not None and line not in (None, "") and side in ("OVER", "UNDER"):
            try:
                actual_val = float(actual)
                line_val = float(line)
            except (TypeError, ValueError):
                actual_val = None
                line_val = None

            if actual_val is not None and line_val is not None:
                if actual_val == line_val:
                    result = "PUSH"
                    pushes += 1
                elif side == "OVER":
                    result = "WIN" if actual_val > line_val else "LOSS"
                else:
                    result = "WIN" if actual_val < line_val else "LOSS"

                if result == "WIN":
                    wins += 1
                elif result == "LOSS":
                    losses += 1
                by_prop[prop_type]["total"] += 1
                if result == "WIN":
                    by_prop[prop_type]["wins"] += 1
                elif result == "LOSS":
                    by_prop[prop_type]["losses"] += 1
                else:
                    by_prop[prop_type]["pushes"] += 1
        elif actual is None:
            missing_actuals += 1

        out_row = dict(row)
        out_row["actual"] = actual
        out_row["result"] = result
        results.append(out_row)

    graded = wins + losses + pushes
    by_prop_summary = {}
    for prop_type, counts in by_prop.items():
        win_rate = None
        if counts["wins"] + counts["losses"] > 0:
            win_rate = round(counts["wins"] / (counts["wins"] + counts["losses"]), 4)
        by_prop_summary[prop_type] = {
            **counts,
            "win_rate": win_rate,
        }
    summary = {
        "total_picks": len(results),
        "graded_picks": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
        "pass_count": pass_count,
        "missing_actuals": missing_actuals,
        "by_prop_type": by_prop_summary,
        "by_pick": _build_group_summary(results, "pick"),
        "by_trend": _build_group_summary(results, "trend"),
        "by_confidence_tier": _build_group_summary(results, "confidence_tier"),
    }
    if odds_snapshot:
        results, clv_summary = attach_clv_snapshot(results, odds_snapshot)
        summary["clv"] = clv_summary
    return results, summary


def _build_bin_summary(rows: List[Dict], value_key: str, bins: List[Tuple[float, float]]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for row in rows:
        result = row.get("result")
        if result not in ("WIN", "LOSS"):
            continue
        raw_value = row.get(value_key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        label = None
        for low, high in bins:
            if low <= value < high:
                label = f"{low:.2f}-{high:.2f}"
                break
        if label is None:
            last_high = bins[-1][1]
            label = f"{last_high:.2f}+"

        bucket = summary.setdefault(label, {"wins": 0, "losses": 0, "total": 0})
        bucket["total"] += 1
        if result == "WIN":
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1

    for label, bucket in summary.items():
        wins = bucket["wins"]
        losses = bucket["losses"]
        bucket["win_rate"] = round(wins / (wins + losses), 4) if (wins + losses) else None
    return summary


def _build_group_summary(rows: List[Dict], value_key: str) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for row in rows:
        result = row.get("result")
        if result not in ("WIN", "LOSS"):
            continue
        label = row.get(value_key) or "UNKNOWN"
        bucket = summary.setdefault(label, {"wins": 0, "losses": 0, "total": 0})
        bucket["total"] += 1
        if result == "WIN":
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
    for label, bucket in summary.items():
        wins = bucket["wins"]
        losses = bucket["losses"]
        bucket["win_rate"] = round(wins / (wins + losses), 4) if (wins + losses) else None
    return summary


def _build_market_index(odds_snapshot: Dict) -> Tuple[Dict[Tuple, List[Dict]], Dict[Tuple, List[Dict]]]:
    event_index: Dict[Tuple, List[Dict]] = {}
    team_index: Dict[Tuple, List[Dict]] = {}
    for prop in odds_snapshot.get("props", []) or []:
        player = prop.get("player")
        prop_type = _normalize_prop_type(prop.get("prop_type"))
        side = (prop.get("side") or "").upper()
        if not player or not prop_type or not side:
            continue
        player_key = normalize_name_for_matching(player)
        event_id = prop.get("event_id") or prop.get("game_id")
        home_team = prop.get("home_team") or ""
        away_team = prop.get("away_team") or ""
        if event_id:
            event_index.setdefault((player_key, prop_type, side, event_id), []).append(prop)
        team_index.setdefault((player_key, prop_type, side, home_team, away_team), []).append(prop)
    return event_index, team_index


def attach_clv_snapshot(rows: List[Dict], odds_snapshot: Optional[Dict]) -> Tuple[List[Dict], Dict]:
    if not odds_snapshot:
        return rows, {"samples": 0}
    event_index, team_index = _build_market_index(odds_snapshot)
    clv_samples = 0
    positive = 0
    signed_total = 0.0
    line_total = 0.0

    output: List[Dict] = []
    for row in rows:
        out_row = dict(row)
        player = row.get("player_name")
        prop_type = _normalize_prop_type(row.get("prop_type"))
        side = (row.get("pick") or row.get("side") or "").upper()
        if not player or not prop_type or side not in ("OVER", "UNDER"):
            output.append(out_row)
            continue

        player_key = normalize_name_for_matching(player)
        event_id = row.get("event_id")
        home_team = row.get("home_team") or ""
        away_team = row.get("away_team") or ""
        candidates = []
        if event_id:
            candidates = event_index.get((player_key, prop_type, side, event_id), [])
        if not candidates:
            candidates = team_index.get((player_key, prop_type, side, home_team, away_team), [])
        if not candidates:
            output.append(out_row)
            continue

        try:
            line_val = float(row.get("line"))
        except (TypeError, ValueError):
            line_val = None

        def _line_distance(prop_row: Dict) -> float:
            try:
                return abs(float(prop_row.get("line")) - line_val) if line_val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        chosen = min(candidates, key=_line_distance)
        try:
            closing_line = float(chosen.get("line"))
        except (TypeError, ValueError):
            closing_line = None

        out_row["closing_line"] = closing_line
        out_row["closing_odds"] = chosen.get("odds")

        if line_val is not None and closing_line is not None:
            clv_delta = closing_line - line_val
            if side == "OVER":
                clv_signed = line_val - closing_line
            else:
                clv_signed = closing_line - line_val
            out_row["clv_line_delta"] = round(clv_delta, 3)
            out_row["clv_signed"] = round(clv_signed, 3)
            clv_samples += 1
            signed_total += clv_signed
            line_total += clv_delta
            if clv_signed > 0:
                positive += 1
        output.append(out_row)

    summary = {
        "samples": clv_samples,
        "avg_line_delta": round(line_total / clv_samples, 4) if clv_samples else None,
        "avg_signed": round(signed_total / clv_samples, 4) if clv_samples else None,
        "positive_rate": round(positive / clv_samples, 4) if clv_samples else None,
    }
    return output, summary


def evaluate_picks_range(
    picks: List[Dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: str = ".cache",
    base_delay: Optional[float] = None,
    timezone: str = "America/New_York",
    cache_only: bool = False,
    odds_snapshot: Optional[Dict] = None,
) -> Tuple[List[Dict], Dict]:
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None

    filtered: List[Dict] = []
    for row in picks:
        game_date = _game_date(row.get("game_time"), timezone=timezone)
        if not game_date:
            continue
        if start and game_date < start:
            continue
        if end and game_date > end:
            continue
        row = dict(row)
        row["game_date"] = game_date.isoformat()
        filtered.append(row)

    players = sorted({row.get("player_name") for row in filtered if row.get("player_name")})

    logs_by_player_date: Dict[str, Dict] = {}
    repo_root = Path(__file__).resolve().parents[2]
    gamelog_db = repo_root / "nba_gamelog_cache.db"

    if gamelog_db.exists():
        logs_by_player_date = _load_cached_logs_from_db(
            gamelog_db,
            start_date=start,
            end_date=end,
        )

    missing_players = set()
    for player in players:
        player_key = normalize_name_for_matching(player)
        if player_key not in logs_by_player_date:
            missing_players.add(player)

    for row in filtered:
        game_date = _game_date(row.get("game_time"), timezone=timezone)
        if not game_date:
            continue
        player = row.get("player_name")
        if not player:
            continue
        player_key = normalize_name_for_matching(player)
        date_map = logs_by_player_date.get(player_key)
        if date_map is None or game_date not in date_map:
            missing_players.add(player)

    if missing_players and not cache_only:
        cache = FileCache(Path(cache_dir) / "snapshots")
        player_logs = fetch_player_logs(
            sorted(missing_players),
            cache,
            ttl_seconds=3600,
            base_delay=base_delay,
            cache_dir=cache_dir,
        )
        for row in player_logs:
            player = row.get("player")
            if not player:
                continue
            key = normalize_name_for_matching(player)
            date_map = logs_by_player_date.setdefault(key, {})
            for log in row.get("logs", []) or []:
                log_date = _parse_datetime(log.get("date"))
                if not log_date:
                    continue
                date_map[log_date.date()] = log

    results: List[Dict] = []
    wins = losses = pushes = 0
    pass_count = 0
    missing_actuals = 0
    by_prop = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0, "total": 0})

    for row in filtered:
        game_date = _game_date(row.get("game_time"), timezone=timezone)
        if not game_date:
            continue

        player = row.get("player_name")
        prop_type = row.get("prop_type") or ""
        side = (row.get("pick") or row.get("side") or "").upper()
        line = row.get("line")
        if not player or not prop_type or not side:
            continue
        if side not in ("OVER", "UNDER"):
            pass_count += 1
            out_row = dict(row)
            out_row["actual"] = None
            out_row["result"] = None
            results.append(out_row)
            continue

        stat_key = _stat_key(prop_type)
        actual = row.get("actual")
        result = (row.get("result") or "").upper() if row.get("result") else None
        if result not in ("WIN", "LOSS", "PUSH"):
            result = None

        actual_val = None
        if actual not in (None, ""):
            try:
                actual_val = float(actual)
            except (TypeError, ValueError):
                actual_val = None

        if result is None and actual_val is None:
            actual = None
            actual_val = None
            result = None
        elif result is None and actual_val is not None:
            if line not in (None, "") and side in ("OVER", "UNDER"):
                try:
                    line_val = float(line)
                except (TypeError, ValueError):
                    line_val = None
                if line_val is not None:
                    if actual_val == line_val:
                        result = "PUSH"
                    elif side == "OVER":
                        result = "WIN" if actual_val > line_val else "LOSS"
                    else:
                        result = "WIN" if actual_val < line_val else "LOSS"

        if result is not None:
            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1
            else:
                pushes += 1
            by_prop[prop_type]["total"] += 1
            if result == "WIN":
                by_prop[prop_type]["wins"] += 1
            elif result == "LOSS":
                by_prop[prop_type]["losses"] += 1
            else:
                by_prop[prop_type]["pushes"] += 1
            out_row = dict(row)
            out_row["actual"] = actual_val if actual_val is not None else actual
            out_row["result"] = result
            results.append(out_row)
            continue

        actual = None
        player_key = normalize_name_for_matching(player)
        log_row = logs_by_player_date.get(player_key, {}).get(game_date)
        if log_row:
            actual = log_row.get(stat_key)

        if actual is not None and line not in (None, "") and side in ("OVER", "UNDER"):
            try:
                actual_val = float(actual)
                line_val = float(line)
            except (TypeError, ValueError):
                actual_val = None
                line_val = None

            if actual_val is not None and line_val is not None:
                if actual_val == line_val:
                    result = "PUSH"
                    pushes += 1
                elif side == "OVER":
                    result = "WIN" if actual_val > line_val else "LOSS"
                else:
                    result = "WIN" if actual_val < line_val else "LOSS"

                if result == "WIN":
                    wins += 1
                elif result == "LOSS":
                    losses += 1
                by_prop[prop_type]["total"] += 1
                if result == "WIN":
                    by_prop[prop_type]["wins"] += 1
                elif result == "LOSS":
                    by_prop[prop_type]["losses"] += 1
                else:
                    by_prop[prop_type]["pushes"] += 1
        elif actual is None:
            missing_actuals += 1

        out_row = dict(row)
        out_row["actual"] = actual
        out_row["result"] = result
        results.append(out_row)

    graded = wins + losses + pushes
    by_prop_summary = {}
    for prop_type, counts in by_prop.items():
        win_rate = None
        if counts["wins"] + counts["losses"] > 0:
            win_rate = round(counts["wins"] / (counts["wins"] + counts["losses"]), 4)
        by_prop_summary[prop_type] = {
            **counts,
            "win_rate": win_rate,
        }

    confidence_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    edge_bins = [(-0.2, -0.05), (-0.05, -0.02), (-0.02, 0.0), (0.0, 0.02),
                 (0.02, 0.05), (0.05, 0.1)]

    summary = {
        "total_picks": len(results),
        "graded_picks": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
        "pass_count": pass_count,
        "missing_actuals": missing_actuals,
        "by_prop_type": by_prop_summary,
        "by_confidence_bin": _build_bin_summary(results, "confidence", confidence_bins),
        "by_edge_bin": _build_bin_summary(results, "edge", edge_bins),
        "by_pick": _build_group_summary(results, "pick"),
        "by_trend": _build_group_summary(results, "trend"),
        "by_confidence_tier": _build_group_summary(results, "confidence_tier"),
    }
    if odds_snapshot:
        results, clv_summary = attach_clv_snapshot(results, odds_snapshot)
        summary["clv"] = clv_summary
    return results, summary


def calibrate_market_blend(results: List[Dict], step: float = 0.1) -> Dict:
    candidates = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    scores: Dict[float, float] = {}

    for blend in candidates:
        brier_sum = 0.0
        count = 0
        for row in results:
            result = row.get("result")
            pick = row.get("pick")
            if result not in ("WIN", "LOSS"):
                continue
            model_prob = row.get("model_probability")
            market_prob = row.get("market_probability")
            if model_prob in (None, "") or market_prob in (None, ""):
                continue
            try:
                model_prob = float(model_prob)
                market_prob = float(market_prob)
            except (TypeError, ValueError):
                continue
            prob = (model_prob * blend) + (market_prob * (1 - blend))
            outcome = 1.0 if result == "WIN" else 0.0
            brier_sum += (prob - outcome) ** 2
            count += 1
        if count:
            scores[blend] = brier_sum / count

    if not scores:
        return {"best_blend": None, "scores": {}}

    best_blend = min(scores, key=scores.get)
    return {"best_blend": best_blend, "scores": scores}


def update_env_value(env_path: Path, key: str, value: str) -> None:
    lines = []
    found = False
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            updated_lines.append(f"{key}={value}")
            found = True
        else:
            updated_lines.append(line)
    if not found:
        updated_lines.append(f"{key}={value}")
    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
