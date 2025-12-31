"""Legacy picks normalization helpers."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import logging
import re

logger = logging.getLogger(__name__)


LEGACY_COLUMNS = [
    "player_name",
    "prop_type",
    "line",
    "pick",
    "game_time",
    "probability",
    "model_probability",
    "edge",
    "confidence",
    "projection",
    "odds",
    "actual",
    "result",
    "source_file",
    "source_row",
]


def discover_legacy_pick_files(repo_root: Path) -> List[Path]:
    patterns = [
        "nba_daily_picks_*.csv",
        "nba_daily_picks_*.xlsx",
        "nba_picks_*.csv",
        "nba_picks_*.xlsx",
        "nba_picks_v2_*.csv",
        "nba_picks_*_*.csv",
        "daily_reports/*.csv",
        "daily_reports/*.xlsx",
        "pick_results_tracking.csv",
        "nba_prop_backtest_results.xlsx",
    ]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(repo_root.glob(pattern))
    # De-duplicate and filter out any normalized output
    seen = set()
    ordered = []
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        if ".cache" in path.parts:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _first_value(row: Dict, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _to_decimal(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if abs(value) > 1.0:
        return value / 100.0
    return value


def _normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    match = re.match(r"(\d{2})-(\d{2})-(\d{4})", text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month}-{day}"
    match = re.match(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month}-{day}"
    return None


def _date_from_filename(path: Path) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}-\d{2}-\d{4})", path.name)
    if match:
        month, day, year = match.group(1).split("-")
        return f"{year}-{month}-{day}"
    return None


def _normalize_prop_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip().lower()
    mapping = {
        "pts": "points",
        "pt": "points",
        "points": "points",
        "reb": "rebounds",
        "rebounds": "rebounds",
        "ast": "assists",
        "assists": "assists",
        "pra": "pra",
        "pts_reb_ast": "pra",
        "points+rebounds+assists": "pra",
        "pts+reb+ast": "pra",
        "3pt": "threes",
        "3pm": "threes",
        "fg3m": "threes",
        "threes": "threes",
        "blocks": "blocks",
        "blk": "blocks",
        "steals": "steals",
        "stl": "steals",
        "turnovers": "turnovers",
        "tov": "turnovers",
    }
    return mapping.get(text, text)


def _normalize_pick(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip().upper()
    if text in {"OVER", "UNDER"}:
        return text
    if text in {"O", "OV"}:
        return "OVER"
    if text in {"U", "UN"}:
        return "UNDER"
    return None


def _derive_pick(line: Optional[float], projection: Optional[float]) -> Optional[str]:
    if line is None or projection is None:
        return None
    if projection > line:
        return "OVER"
    if projection < line:
        return "UNDER"
    return None


def normalize_legacy_pick_files(paths: List[Path]) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    seen: Dict[Tuple, int] = {}
    skipped = 0

    for path in paths:
        file_date = _date_from_filename(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                import pandas as pd
            except Exception as exc:
                logger.warning("pandas unavailable for %s: %s", path, exc)
                continue

            try:
                df = pd.read_excel(path)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                continue
            raw_rows = df.to_dict(orient="records")
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                raw_rows = list(reader)

        for idx, row in enumerate(raw_rows, start=2):
            player = _first_value(row, ["player", "Player", "PLAYER", "player_name", "Player Name"])
            prop_raw = _first_value(row, ["prop_type", "prop", "Prop", "PROP", "stat", "Stat"])
            prop_type = _normalize_prop_type(prop_raw)
            line_val = _parse_float(_first_value(row, ["line", "Line", "LINE"]))

            pick_raw = _first_value(row, ["pick", "Pick", "recommended_side", "recommended_pick", "Side", "side"])
            pick = _normalize_pick(pick_raw)

            projection = _parse_float(_first_value(row, ["projection", "Proj", "proj", "model_projection", "base_projection"]))
            if pick is None:
                pick = _derive_pick(line_val, projection)

            date_value = _first_value(row, ["report_date", "date", "game_date"])
            game_date = _normalize_date(date_value) or file_date

            if not player or not prop_type or line_val is None or not pick or not game_date:
                skipped += 1
                continue

            edge_val = _parse_float(_first_value(row, ["edge", "avg_edge", "raw_edge", "edge_pct", "Edge"]))
            confidence_val = _parse_float(_first_value(row, ["confidence", "confidence_pct", "Confidence"]))
            prob_val = _parse_float(_first_value(row, ["probability", "our_prob", "P(Win)", "Pwin", "prob_win"]))
            model_prob_val = _parse_float(_first_value(row, ["model_probability", "model_prob"]))

            edge_val = _to_decimal(edge_val)
            confidence_val = _to_decimal(confidence_val)
            prob_val = _to_decimal(prob_val)
            model_prob_val = _to_decimal(model_prob_val)

            odds_val = _parse_float(_first_value(row, ["odds", "Odds", "over_odds", "under_odds"]))
            odds = int(odds_val) if odds_val is not None else None

            if prob_val is None and odds is not None and edge_val is not None:
                if odds > 0:
                    implied = 100 / (odds + 100)
                else:
                    implied = abs(odds) / (abs(odds) + 100)
                prob_val = max(0.0, min(1.0, implied + edge_val))
            if model_prob_val is None:
                model_prob_val = prob_val

            actual_val = _parse_float(_first_value(row, ["actual", "Actual"]))
            result_val = None
            raw_result = _first_value(row, ["result", "Result"])
            if raw_result:
                raw = str(raw_result).strip().upper()
                if raw in {"WIN", "LOSS", "PUSH"}:
                    result_val = raw
            if result_val is None and actual_val is not None and line_val is not None:
                if actual_val == line_val:
                    result_val = "PUSH"
                elif pick == "OVER":
                    result_val = "WIN" if actual_val > line_val else "LOSS"
                elif pick == "UNDER":
                    result_val = "WIN" if actual_val < line_val else "LOSS"

            normalized = {
                "player_name": str(player).strip(),
                "prop_type": prop_type,
                "line": line_val,
                "pick": pick,
                "game_time": game_date,
                "probability": prob_val,
                "model_probability": model_prob_val,
                "edge": edge_val,
                "confidence": confidence_val,
                "projection": projection,
                "odds": odds,
                "actual": actual_val,
                "result": result_val,
                "source_file": str(path),
                "source_row": idx,
            }

            key = (
                game_date,
                normalized["player_name"].lower(),
                prop_type,
                round(line_val, 3),
                pick,
            )
            existing_idx = seen.get(key)
            if existing_idx is None:
                seen[key] = len(rows)
                rows.append(normalized)
            else:
                existing = rows[existing_idx]
                existing_conf = existing.get("confidence") or -1
                new_conf = confidence_val or -1
                if new_conf > existing_conf:
                    rows[existing_idx] = normalized

    summary = {
        "input_files": [str(path) for path in paths],
        "rows": len(rows),
        "skipped": skipped,
        "deduped": len(seen),
    }
    return rows, summary


def load_pick_tracker_db(db_path: Path) -> List[Dict]:
    if not db_path.exists():
        return []
    rows: List[Dict] = []
    try:
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            results = conn.execute(
                """
                SELECT p.date, p.player, p.prop_type, p.line, p.pick, p.edge,
                       p.confidence, p.projection, p.odds, r.actual
                  FROM picks p
             LEFT JOIN results r
                    ON p.date = r.date
                   AND p.player = r.player
                   AND p.prop_type = r.prop_type
                """
            ).fetchall()
    except Exception:
        return rows

    for date_str, player, prop_type, line, pick, edge, confidence, projection, odds, actual in results:
        prop_type = _normalize_prop_type(prop_type)
        pick_norm = _normalize_pick(pick)
        if not date_str or not player or not prop_type or not pick_norm:
            continue
        try:
            line_val = float(line)
        except (TypeError, ValueError):
            continue

        edge_val = _to_decimal(_parse_float(edge))
        confidence_val = _to_decimal(_parse_float(confidence))
        projection_val = _parse_float(projection)
        odds_val = _parse_float(odds)
        odds_int = int(odds_val) if odds_val is not None else None

        implied = None
        if odds_int is not None:
            if odds_int > 0:
                implied = 100 / (odds_int + 100)
            else:
                implied = abs(odds_int) / (abs(odds_int) + 100)
        probability = None
        if implied is not None and edge_val is not None:
            probability = max(0.0, min(1.0, implied + edge_val))

        result = None
        actual_val = None
        try:
            actual_val = float(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_val = None

        if actual_val is not None:
            if actual_val == line_val:
                result = "PUSH"
            elif pick_norm == "OVER":
                result = "WIN" if actual_val > line_val else "LOSS"
            elif pick_norm == "UNDER":
                result = "WIN" if actual_val < line_val else "LOSS"

        rows.append({
            "player_name": str(player).strip(),
            "prop_type": prop_type,
            "line": line_val,
            "pick": pick_norm,
            "game_time": str(date_str),
            "probability": probability,
            "model_probability": probability,
            "edge": edge_val,
            "confidence": confidence_val,
            "projection": projection_val,
            "odds": odds_int,
            "actual": actual_val,
            "result": result,
            "source_file": str(db_path),
            "source_row": None,
        })

    return rows


def dedupe_rows(rows: List[Dict]) -> List[Dict]:
    deduped: Dict[Tuple, Dict] = {}
    for row in rows:
        game_time = row.get("game_time")
        player = row.get("player_name") or ""
        prop_type = row.get("prop_type") or ""
        line = row.get("line")
        pick = row.get("pick") or ""
        try:
            line_key = round(float(line), 3)
        except (TypeError, ValueError):
            line_key = line
        key = (game_time, player.lower(), prop_type, line_key, pick)

        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue

        existing_result = existing.get("result")
        new_result = row.get("result")
        if existing_result in ("WIN", "LOSS", "PUSH"):
            if new_result in ("WIN", "LOSS", "PUSH") and new_result != existing_result:
                deduped[key] = row
            continue
        if new_result in ("WIN", "LOSS", "PUSH"):
            deduped[key] = row
            continue

        existing_conf = existing.get("confidence") or -1
        new_conf = row.get("confidence") or -1
        if new_conf > existing_conf:
            deduped[key] = row
            continue

    return list(deduped.values())
