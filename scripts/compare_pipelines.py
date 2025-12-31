#!/usr/bin/env python3
"""
Compare legacy UnifiedPropModel vs v2 BaselineModel on historical props.

Cache-only by default to avoid network calls. This is intended for pipeline
selection and ROI backtesting only.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import pickle
import re

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import CONFIG as LEGACY_CONFIG
from models.unified import UnifiedPropModel
from nbaprop.config import Config as V2Config
from nbaprop.features.pipeline import build_features
from nbaprop.ingestion.nba_stats import fetch_player_logs
from nbaprop.models.baseline import BaselineModel
from nbaprop.models.calibration import load_calibration, resolve_calibration_path
from nbaprop.normalization.ids import canonicalize_player_name
from nbaprop.storage import FileCache


@dataclass
class PickRow:
    player: str
    prop_type: str
    line: float
    date: date
    pick: str
    edge: float
    confidence: float
    odds: Optional[int]
    result: Optional[str]
    units: Optional[float]
    game: str
    source: str
    n_games: Optional[int]


class NoInjuryTracker:
    def get_player_status(self, player_name: str) -> dict:
        return {
            "status": "HEALTHY",
            "is_out": False,
            "is_gtd": False,
            "is_questionable": False,
            "injury": None,
            "source": "cache_only",
        }

    def get_teammate_boost(self, player_name: str, team_abbrev: str, prop_type: str = "points") -> dict:
        return {
            "boost_factor": 1.0,
            "stars_out": [],
            "reason": "cache_only",
        }


class CacheOnlyFetcher:
    """Provide minimal fetcher behavior without network calls."""

    def __init__(self) -> None:
        self._game_date: Optional[datetime] = None

    def set_game_date(self, game_date: Optional[date]) -> None:
        if game_date:
            self._game_date = datetime.combine(game_date, datetime.min.time())
        else:
            self._game_date = None

    def get_team_defense_vs_position(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_team_pace(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_player_vs_team_stats(self, player_name: str, opponent: str, prop_type: str = "points") -> dict:
        return {}

    def get_player_usage(self, player_name: str, season: str = None) -> dict:
        return {}

    def check_back_to_back(self, player_logs: pd.DataFrame, game_date: datetime = None) -> dict:
        if player_logs.empty or "date" not in player_logs.columns:
            return {"is_b2b": False, "rest_days": None}

        if game_date is None:
            game_date = self._game_date or datetime.now()

        logs = player_logs.sort_values("date", ascending=False)
        last_game = logs.iloc[0]["date"]
        if isinstance(last_game, str):
            last_game = pd.to_datetime(last_game, errors="coerce")

        if last_game is None or pd.isna(last_game):
            return {"is_b2b": False, "rest_days": None}

        days_rest = (game_date - last_game).days
        recent_b2b = False
        if len(logs) >= 2:
            second_last = logs.iloc[1]["date"]
            if isinstance(second_last, str):
                second_last = pd.to_datetime(second_last, errors="coerce")
            if second_last is not None and not pd.isna(second_last):
                days_between = (last_game - second_last).days
                recent_b2b = days_between <= 1

        return {
            "is_b2b": days_rest <= 1,
            "rest_days": days_rest,
            "recent_b2b_played": recent_b2b,
            "last_game_date": last_game,
        }

    def get_player_minutes_trend(self, player_logs: pd.DataFrame) -> Dict[str, float]:
        if player_logs.empty or "minutes" not in player_logs.columns:
            return {}

        logs = player_logs.sort_values("date", ascending=False)
        mins = logs["minutes"].apply(_parse_minutes)
        mins = mins.dropna()
        if mins.empty:
            return {}

        recent_5 = mins.head(5).mean()
        season_avg = mins.mean()
        return {
            "recent_avg": round(recent_5, 1),
            "season_avg": round(season_avg, 1),
            "min": round(mins.min(), 1),
            "max": round(mins.max(), 1),
            "trend": "UP" if recent_5 > season_avg * 1.05 else "DOWN" if recent_5 < season_avg * 0.95 else "STABLE",
            "minutes_factor": round(recent_5 / season_avg, 3) if season_avg else 1.0,
        }

    def calculate_usage_trend(self, game_logs: pd.DataFrame) -> dict:
        if game_logs.empty:
            return {"usage_factor": 1.0, "trend": "STABLE"}
        required_cols = ["fga", "fta", "minutes"]
        if not all(col in game_logs.columns for col in required_cols):
            return {"usage_factor": 1.0, "trend": "STABLE"}

        logs = game_logs.sort_values("date", ascending=False) if "date" in game_logs.columns else game_logs
        mins = logs["minutes"].apply(_parse_minutes)
        valid_mask = mins >= 10
        if valid_mask.sum() < 5:
            return {"usage_factor": 1.0, "trend": "STABLE"}

        logs_valid = logs[valid_mask].copy()
        mins_valid = mins[valid_mask]
        usage_scores = (logs_valid["fga"] + 0.44 * logs_valid["fta"]) / mins_valid
        recent_5 = usage_scores.head(5).mean()
        older_10 = usage_scores.iloc[5:15].mean() if len(usage_scores) > 5 else usage_scores.mean()

        if not older_10 or math.isnan(older_10):
            return {"usage_factor": 1.0, "trend": "STABLE"}

        usage_trend = recent_5 / older_10
        usage_factor = max(0.92, min(1.08, usage_trend))
        return {
            "usage_factor": round(usage_factor, 3),
            "recent_usage": round(recent_5, 3),
            "season_usage": round(older_10, 3),
            "trend": "UP" if usage_trend > 1.03 else "DOWN" if usage_trend < 0.97 else "STABLE",
            "raw_trend": round(usage_trend, 3),
        }

    def calculate_ts_efficiency(self, game_logs: pd.DataFrame) -> dict:
        if game_logs.empty:
            return {"ts_factor": 1.0, "regression": "NONE"}
        required_cols = ["points", "fga", "fta"]
        if not all(col in game_logs.columns for col in required_cols):
            return {"ts_factor": 1.0, "regression": "NONE"}

        logs = game_logs.sort_values("date", ascending=False) if "date" in game_logs.columns else game_logs

        def calc_ts(df: pd.DataFrame) -> float:
            pts = df["points"].sum()
            fga = df["fga"].sum()
            fta = df["fta"].sum()
            denom = 2 * (fga + 0.44 * fta)
            return pts / denom if denom > 0 else 0.5

        recent_ts = calc_ts(logs.head(5))
        season_ts = calc_ts(logs)
        if season_ts == 0:
            return {"ts_factor": 1.0, "regression": "NONE"}

        deviation = (recent_ts - season_ts) / season_ts
        threshold = 0.05
        weight = 0.3
        max_adj = 0.05

        if abs(deviation) > threshold:
            if deviation > 0:
                ts_factor = 1.0 + min(max_adj, deviation * weight)
                momentum = "HOT"
            else:
                ts_factor = 1.0 - min(max_adj, abs(deviation) * weight)
                momentum = "COLD"
        else:
            ts_factor = 1.0
            momentum = "NONE"

        return {
            "ts_factor": round(ts_factor, 3),
            "recent_ts_pct": round(recent_ts * 100, 1),
            "season_ts_pct": round(season_ts * 100, 1),
            "deviation_pct": round(deviation * 100, 1),
            "regression": momentum,
        }


WEIGHT_SETS = [
    {"name": "roi_heavy", "roi": 0.6, "win_rate": 0.2, "drawdown": 0.1, "volume": 0.1, "edge": 0.0},
    {"name": "balanced", "roi": 0.4, "win_rate": 0.3, "drawdown": 0.2, "volume": 0.1, "edge": 0.0},
    {"name": "risk_adjusted", "roi": 0.3, "win_rate": 0.2, "drawdown": 0.4, "volume": 0.1, "edge": 0.0},
    {"name": "volume", "roi": 0.4, "win_rate": 0.2, "drawdown": 0.1, "volume": 0.3, "edge": 0.0},
    {"name": "edge_balance", "roi": 0.35, "win_rate": 0.2, "drawdown": 0.15, "volume": 0.1, "edge": 0.2},
    {"name": "edge_heavy", "roi": 0.3, "win_rate": 0.1, "drawdown": 0.1, "volume": 0.1, "edge": 0.4},
    {"name": "roi_edge", "roi": 0.5, "win_rate": 0.1, "drawdown": 0.1, "volume": 0.0, "edge": 0.3},
]


def _parse_minutes(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ":" in text:
        mins, secs = text.split(":", 1)
        try:
            return float(mins) + (float(secs) / 60.0)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_log_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.date()


def _season_for_date(game_date: date) -> str:
    if game_date.month >= 10:
        start_year = game_date.year
    else:
        start_year = game_date.year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def _normalize_matchup(matchup: str) -> str:
    raw = matchup.strip()
    cleaned = raw.replace("vs.", "vs").replace("VS.", "vs").replace("Vs.", "vs")
    parts = cleaned.split()
    if len(parts) >= 3:
        team = parts[0]
        sep = parts[1].replace(".", "").lower()
        opp = parts[2]
        if sep == "vs":
            home = team
            away = opp
            return f"{away} @ {home}"
        if sep == "@":
            home = opp
            away = team
            return f"{away} @ {home}"
    return raw


def _american_to_implied(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds_val = float(odds)
    except (TypeError, ValueError):
        return None
    if odds_val < 0:
        return abs(odds_val) / (abs(odds_val) + 100)
    return 100 / (odds_val + 100)


def _odds_to_profit(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds_val = float(odds)
    except (TypeError, ValueError):
        return None
    if odds_val < 0:
        return 100.0 / abs(odds_val)
    return odds_val / 100.0


def _resolve_date_range(
    db_path: Path,
    bookmaker: str,
    days: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if start_date and end_date:
        return start_date, end_date
    query = "SELECT MAX(date) FROM historical_props WHERE bookmaker = ?"
    params = (bookmaker,)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(query, params).fetchone()
    max_date = _parse_date(row[0]) if row and row[0] else None
    if not max_date:
        return None, None
    start = max_date - timedelta(days=max(days, 1) - 1)
    return start.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")


def _load_historical_props(
    db_path: Path,
    bookmaker: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict]:
    if not db_path.exists():
        return []
    query = (
        "SELECT date, player, prop_type, line, odds_over, odds_under, actual, "
        "hit_over, hit_under, push "
        "FROM historical_props WHERE bookmaker = ?"
    )
    params: List = [bookmaker]
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    output = []
    for row in rows:
        date_str, player, prop_type, line, odds_over, odds_under, actual, hit_over, hit_under, push = row
        output.append({
            "date": date_str,
            "player": player,
            "prop_type": (prop_type or "").lower(),
            "line": line,
            "odds_over": odds_over,
            "odds_under": odds_under,
            "actual": actual,
            "hit_over": hit_over,
            "hit_under": hit_under,
            "push": push,
        })
    return output


def _build_logs_by_player_season(
    props: List[Dict],
    cache_dir: str,
    cache_only: bool,
) -> Dict[Tuple[str, str], List[Dict]]:
    players_by_season: Dict[str, set] = {}
    for row in props:
        game_date = _parse_date(row.get("date"))
        if not game_date:
            continue
        player = row.get("player")
        if not player:
            continue
        season = _season_for_date(game_date)
        players_by_season.setdefault(season, set()).add(player)

    logs_by_player_season: Dict[Tuple[str, str], List[Dict]] = {}
    for season, players in players_by_season.items():
        fetched = fetch_player_logs(
            sorted(players),
            cache=FileCache(Path(cache_dir) / "snapshots"),
            ttl_seconds=3600,
            season=season,
            cache_dir=cache_dir,
            cache_only=cache_only,
        )
        for row in fetched:
            player = row.get("player")
            if not player:
                continue
            key = (canonicalize_player_name(player), season)
            logs = row.get("logs", [])
            if not logs:
                legacy_logs = _load_legacy_cached_logs(Path(cache_dir), player, season)
                if legacy_logs:
                    logs = legacy_logs
            logs_by_player_season[key] = logs
    return logs_by_player_season


def _legacy_cache_path(cache_dir: Path, player_name: str, season: str) -> Optional[Path]:
    if not cache_dir:
        return None
    key_str = f"{player_name.lower()}_{season}"
    digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    safe_player = re.sub(r"[^a-z0-9]+", "_", player_name.lower()).strip("_")[:40]
    if not safe_player:
        safe_player = "player"
    return cache_dir / f"game_logs_{safe_player}_{season}_{digest}.pkl"


def _load_legacy_cached_logs(cache_dir: Path, player_name: str, season: str) -> List[Dict]:
    path = _legacy_cache_path(cache_dir, player_name, season)
    if not path or not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if data is None:
        return []
    try:
        return data.to_dict(orient="records")
    except Exception:
        return []


def _build_game_key_map(
    logs_by_player_season: Dict[Tuple[str, str], List[Dict]],
) -> Dict[Tuple[str, date], str]:
    game_keys: Dict[Tuple[str, date], str] = {}
    for (player_key, _season), logs in logs_by_player_season.items():
        for log in logs:
            log_date = _parse_log_date(log.get("date"))
            if not log_date:
                continue
            matchup = log.get("matchup") or log.get("MATCHUP") or log.get("Matchup")
            if not matchup:
                continue
            game_keys.setdefault((player_key, log_date), _normalize_matchup(str(matchup)))
    return game_keys


def _grade_result(row: Dict, pick_side: str) -> Optional[str]:
    actual = row.get("actual")
    line = row.get("line")
    try:
        line_val = float(line)
    except (TypeError, ValueError):
        return None

    if actual is not None:
        try:
            actual_val = float(actual)
        except (TypeError, ValueError):
            actual_val = None
        if actual_val is not None:
            if actual_val == line_val:
                return "PUSH"
            if pick_side == "OVER":
                return "WIN" if actual_val > line_val else "LOSS"
            if pick_side == "UNDER":
                return "WIN" if actual_val < line_val else "LOSS"
            return None

    if row.get("push"):
        return "PUSH"
    if pick_side == "OVER" and row.get("hit_over") is not None:
        return "WIN" if row.get("hit_over") else "LOSS"
    if pick_side == "UNDER" and row.get("hit_under") is not None:
        return "WIN" if row.get("hit_under") else "LOSS"
    return None


def _apply_thresholds(
    picks: Iterable[PickRow],
    thresholds: Dict[str, Dict[str, float]],
    min_samples: Dict[str, int],
    ranking_mode: str,
    top_per_game: int,
) -> List[PickRow]:
    filtered: List[PickRow] = []
    for pick in picks:
        thresholds_for = thresholds.get(pick.prop_type, None)
        min_edge = thresholds_for.get("min_edge") if thresholds_for else None
        min_conf = thresholds_for.get("min_confidence") if thresholds_for else None
        if min_edge is None or min_conf is None:
            min_edge = 0.0
            min_conf = 0.0

        if pick.edge is None or pick.confidence is None:
            continue
        if pick.edge < min_edge:
            continue
        if pick.confidence < min_conf:
            continue
        required = min_samples.get(pick.prop_type, 0)
        if required and pick.n_games is not None and pick.n_games < required:
            continue
        filtered.append(pick)

    grouped: Dict[str, List[PickRow]] = {}
    for pick in filtered:
        grouped.setdefault(pick.game, []).append(pick)

    ranked: List[PickRow] = []
    for rows in grouped.values():
        rows.sort(key=lambda r: _rank_score(r, ranking_mode), reverse=True)
        if top_per_game and top_per_game > 0:
            ranked.extend(rows[:top_per_game])
        else:
            ranked.extend(rows)

    ranked.sort(key=lambda r: _rank_score(r, ranking_mode), reverse=True)
    return ranked


def _rank_score(pick: PickRow, ranking_mode: str) -> float:
    edge_val = pick.edge or 0.0
    confidence = pick.confidence or 0.0
    if ranking_mode == "edge":
        return edge_val
    return abs(edge_val) * confidence


def _compute_metrics(picks: List[PickRow]) -> Dict:
    graded = [p for p in picks if p.result in ("WIN", "LOSS", "PUSH") and p.units is not None]
    wins = sum(1 for p in graded if p.result == "WIN")
    losses = sum(1 for p in graded if p.result == "LOSS")
    pushes = sum(1 for p in graded if p.result == "PUSH")
    total_bets = wins + losses
    total_units = sum(p.units for p in graded if p.units is not None)
    roi = (total_units / total_bets) if total_bets else None
    win_rate = (wins / total_bets) if total_bets else None
    avg_edge = None
    if graded:
        avg_edge = sum(p.edge for p in graded if p.edge is not None) / len(graded)

    max_drawdown = 0.0
    peak = 0.0
    cumulative = 0.0
    for pick in sorted(graded, key=lambda p: p.date):
        cumulative += pick.units or 0.0
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "total_picks": len(picks),
        "graded_picks": len(graded),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi": roi,
        "total_units": total_units,
        "max_drawdown": max_drawdown,
        "avg_edge": avg_edge,
    }


def _compute_prop_metrics(picks: List[PickRow]) -> Dict[str, Dict]:
    by_prop: Dict[str, List[PickRow]] = {}
    for pick in picks:
        by_prop.setdefault(pick.prop_type, []).append(pick)
    return {prop: _compute_metrics(rows) for prop, rows in by_prop.items()}


def _score_picks(picks: List[PickRow], weights: Dict[str, float]) -> Optional[float]:
    metrics = _compute_metrics(picks)
    if not metrics.get("graded_picks"):
        return None
    total_bets = metrics["wins"] + metrics["losses"]
    if total_bets == 0:
        return None

    roi = metrics.get("roi") or 0.0
    win_rate = metrics.get("win_rate") or 0.0
    avg_edge = metrics.get("avg_edge") or 0.0
    max_drawdown = metrics.get("max_drawdown") or 0.0
    drawdown_norm = max_drawdown / max(total_bets, 1)
    volume_norm = min(total_bets / 100.0, 1.0)

    return (
        weights["roi"] * roi +
        weights["win_rate"] * (win_rate - 0.5) +
        weights["edge"] * avg_edge +
        weights["volume"] * volume_norm -
        weights["drawdown"] * drawdown_norm
    )


def _fold_dates(picks: List[PickRow], folds: int = 3) -> List[List[date]]:
    unique_dates = sorted({p.date for p in picks})
    if not unique_dates:
        return []
    if len(unique_dates) <= folds:
        return [[d] for d in unique_dates]
    chunk = math.ceil(len(unique_dates) / folds)
    return [unique_dates[i:i + chunk] for i in range(0, len(unique_dates), chunk)]


def _run_weight_search(picks: List[PickRow]) -> List[Dict]:
    folds = _fold_dates(picks, folds=3)
    results = []
    for weights in WEIGHT_SETS:
        overall_score = _score_picks(picks, weights)
        fold_scores = []
        for fold in folds:
            fold_picks = [p for p in picks if p.date in fold]
            score = _score_picks(fold_picks, weights)
            if score is None:
                score = -1.0
            fold_scores.append(score)
        avg_score = sum(fold_scores) / len(fold_scores) if fold_scores else None
        stdev = float(np.std(fold_scores)) if fold_scores else None
        stability = None if avg_score is None or stdev is None else avg_score - stdev
        results.append({
            "weights": weights,
            "overall_score": overall_score,
            "fold_scores": fold_scores,
            "avg_score": avg_score,
            "stdev": stdev,
            "stability": stability,
        })
    return results


def _score_legacy(
    props: List[Dict],
    logs_by_player_season: Dict[Tuple[str, str], List[Dict]],
    game_keys: Dict[Tuple[str, date], str],
    excluded_props: set,
    thresholds: Dict[str, Dict[str, float]],
    min_samples: Dict[str, int],
    ranking_mode: str,
    top_per_game: int,
) -> List[PickRow]:
    fetcher = CacheOnlyFetcher()
    model = UnifiedPropModel(data_fetcher=fetcher, injury_tracker=NoInjuryTracker())

    logs_df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_picks: List[PickRow] = []

    for row in props:
        prop_type = (row.get("prop_type") or "").lower()
        if prop_type in excluded_props:
            continue
        game_date = _parse_date(row.get("date"))
        if not game_date:
            continue
        player = row.get("player")
        if not player:
            continue
        try:
            line_val = float(row.get("line"))
        except (TypeError, ValueError):
            continue

        season = _season_for_date(game_date)
        key = (canonicalize_player_name(player), season)
        logs = logs_by_player_season.get(key, [])
        if not logs:
            continue

        if key not in logs_df_cache:
            df = pd.DataFrame(logs)
            if "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            logs_df_cache[key] = df

        logs_df = logs_df_cache[key]
        filtered = logs_df
        if "date" in logs_df.columns:
            filtered = logs_df[logs_df["date"].dt.date < game_date]
        if filtered.empty:
            continue
        if "date" in filtered.columns:
            filtered = filtered.sort_values("date", ascending=False)

        fetcher.set_game_date(game_date)
        analysis = model.analyze(
            player_name=player,
            prop_type=prop_type,
            line=line_val,
            odds=row.get("odds_over") or -110,
            game_logs=filtered,
            game_date=game_date,
        )
        if analysis.games_analyzed == 0:
            continue

        raw_edge = analysis.edge
        if raw_edge == 0:
            continue
        pick_side = "OVER" if raw_edge > 0 else "UNDER"

        odds_over = row.get("odds_over")
        odds_under = row.get("odds_under")
        over_implied = _american_to_implied(odds_over)
        under_implied = _american_to_implied(odds_under)
        if over_implied is None or under_implied is None:
            continue
        total_implied = over_implied + under_implied
        if total_implied == 0:
            continue

        if pick_side == "OVER":
            breakeven = over_implied / total_implied
            our_prob = min(0.85, max(0.15, breakeven + raw_edge * 0.3))
            pick_odds = odds_over
        else:
            breakeven = under_implied / total_implied
            our_prob = min(0.85, max(0.15, breakeven + abs(raw_edge) * 0.3))
            pick_odds = odds_under

        vig_edge = our_prob - breakeven
        result = _grade_result(row, pick_side)
        units = None
        profit = _odds_to_profit(pick_odds)
        if profit is not None and result:
            if result == "WIN":
                units = profit
            elif result == "LOSS":
                units = -1.0
            elif result == "PUSH":
                units = 0.0

        game_key = game_keys.get((canonicalize_player_name(player), game_date))
        if not game_key:
            game_key = f"{game_date.isoformat()}::{player}"

        raw_picks.append(PickRow(
            player=player,
            prop_type=prop_type,
            line=line_val,
            date=game_date,
            pick=pick_side,
            edge=vig_edge,
            confidence=analysis.confidence,
            odds=pick_odds,
            result=result,
            units=units,
            game=game_key,
            source="legacy",
            n_games=analysis.games_analyzed,
        ))

    return _apply_thresholds(raw_picks, thresholds, min_samples, ranking_mode, top_per_game)


def _score_v2(
    props: List[Dict],
    logs_by_player_season: Dict[Tuple[str, str], List[Dict]],
    game_keys: Dict[Tuple[str, date], str],
    config: V2Config,
    calibration: Optional[Dict],
    thresholds: Dict[str, Dict[str, float]],
    min_samples: Dict[str, int],
    ranking_mode: str,
    top_per_game: int,
) -> List[PickRow]:
    zero_edges = {k: 0.0 for k in config.prop_type_min_edge}
    zero_conf = {k: 0.0 for k in config.prop_type_min_confidence}
    model = BaselineModel(
        min_edge=0.0,
        min_confidence=0.0,
        prop_type_min_edge=zero_edges,
        prop_type_min_confidence=zero_conf,
        excluded_prop_types=config.excluded_prop_types,
        injury_risk_edge_multiplier=config.injury_risk_edge_multiplier,
        injury_risk_confidence_multiplier=config.injury_risk_confidence_multiplier,
        market_blend=config.market_blend,
        calibration=calibration,
        odds_min=config.odds_min,
        odds_max=config.odds_max,
        odds_confidence_min=config.odds_confidence_min,
        odds_confidence_max=config.odds_confidence_max,
        confidence_cap_outside_band=config.confidence_cap_outside_band,
    )

    raw_picks: List[PickRow] = []
    for row in props:
        prop_type = (row.get("prop_type") or "").lower()
        if prop_type in config.excluded_prop_types:
            continue
        game_date = _parse_date(row.get("date"))
        if not game_date:
            continue
        player = row.get("player")
        if not player:
            continue
        try:
            line_val = float(row.get("line"))
        except (TypeError, ValueError):
            continue

        odds_over = row.get("odds_over")
        odds_under = row.get("odds_under")
        props_list = []
        if odds_over is not None:
            props_list.append({
                "player": player,
                "prop_type": prop_type,
                "line": line_val,
                "side": "over",
                "odds": odds_over,
                "game_time": game_date.isoformat(),
            })
        if odds_under is not None:
            props_list.append({
                "player": player,
                "prop_type": prop_type,
                "line": line_val,
                "side": "under",
                "odds": odds_under,
                "game_time": game_date.isoformat(),
            })
        if not props_list:
            continue

        season = _season_for_date(game_date)
        logs = logs_by_player_season.get((canonicalize_player_name(player), season), [])
        filtered_logs = []
        for log in logs:
            log_date = _parse_log_date(log.get("date"))
            if log_date is None or log_date >= game_date:
                continue
            filtered_logs.append(log)

        feature_rows = build_features(
            props_list,
            raw_player_logs=[{"player": player, "logs": filtered_logs}],
            injuries=None,
            players=None,
            team_stats=None,
        )
        if not feature_rows:
            continue

        picks = model.predict(feature_rows)
        if not picks:
            continue

        candidates = [pick for pick in picks if pick.get("pick") in ("OVER", "UNDER")]
        if not candidates:
            continue
        best_pick = max(
            candidates,
            key=lambda p: (p.get("edge") or 0.0, p.get("confidence") or 0.0),
        )

        pick_side = best_pick.get("pick")
        pick_odds = best_pick.get("odds")
        result = _grade_result(row, pick_side)
        units = None
        profit = _odds_to_profit(pick_odds)
        if profit is not None and result:
            if result == "WIN":
                units = profit
            elif result == "LOSS":
                units = -1.0
            elif result == "PUSH":
                units = 0.0

        features_by_side = {}
        for feature_row in feature_rows:
            side = (feature_row.get("features", {}).get("side") or "").lower()
            features_by_side[side] = feature_row.get("features", {})
        feature_payload = features_by_side.get(pick_side.lower(), {})
        n_games = feature_payload.get("n_games")

        game_key = game_keys.get((canonicalize_player_name(player), game_date))
        if not game_key:
            game_key = f"{game_date.isoformat()}::{player}"

        raw_picks.append(PickRow(
            player=player,
            prop_type=prop_type,
            line=line_val,
            date=game_date,
            pick=pick_side,
            edge=best_pick.get("edge") or 0.0,
            confidence=best_pick.get("confidence") or 0.0,
            odds=pick_odds,
            result=result,
            units=units,
            game=game_key,
            source="v2",
            n_games=n_games if isinstance(n_games, int) else None,
        ))

    return _apply_thresholds(raw_picks, thresholds, min_samples, ranking_mode, top_per_game)


def _load_actual_pick_history(db_path: Path) -> List[PickRow]:
    if not db_path.exists():
        return []
    query = (
        "SELECT date, player, prop_type, line, pick, odds, actual, hit "
        "FROM picks_with_results"
    )
    rows = []
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(query).fetchall():
            date_str, player, prop_type, line, pick, odds, actual, hit = row
            game_date = _parse_date(date_str)
            if not game_date:
                continue
            try:
                line_val = float(line)
            except (TypeError, ValueError):
                continue
            result = None
            if hit is not None:
                if hit == 1:
                    result = "WIN"
                elif hit == 0:
                    result = "LOSS"
            if actual is not None and result is None:
                try:
                    actual_val = float(actual)
                except (TypeError, ValueError):
                    actual_val = None
                if actual_val is not None:
                    if actual_val == line_val:
                        result = "PUSH"
            units = None
            profit = _odds_to_profit(odds)
            if profit is not None and result:
                units = profit if result == "WIN" else -1.0 if result == "LOSS" else 0.0
            rows.append(PickRow(
                player=player,
                prop_type=(prop_type or "").lower(),
                line=line_val,
                date=game_date,
                pick=pick,
                edge=0.0,
                confidence=0.0,
                odds=odds,
                result=result,
                units=units,
                game=f"{game_date.isoformat()}::{player}",
                source="actual",
                n_games=None,
            ))
    return rows


def _print_metrics(label: str, metrics: Dict) -> None:
    roi = metrics.get("roi")
    win_rate = metrics.get("win_rate")
    avg_edge = metrics.get("avg_edge")
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Total picks:     {metrics.get('total_picks')}")
    print(f"Graded picks:    {metrics.get('graded_picks')}")
    print(f"Wins/Losses:     {metrics.get('wins')}/{metrics.get('losses')} (push {metrics.get('pushes')})")
    print(f"Win rate:        {win_rate:.2%}" if win_rate is not None else "Win rate:        n/a")
    print(f"ROI (unit):      {roi:.4f}" if roi is not None else "ROI (unit):      n/a")
    print(f"Total units:     {metrics.get('total_units'):.2f}")
    print(f"Max drawdown:    {metrics.get('max_drawdown'):.2f}")
    print(f"Avg edge:        {avg_edge:.4f}" if avg_edge is not None else "Avg edge:        n/a")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy and v2 pipelines on historical props.")
    parser.add_argument("--db", default="nba_historical_cache.db", help="Path to historical props DB")
    parser.add_argument("--pick-db", default="clv_tracking.db", help="Path to pick tracker DB")
    parser.add_argument("--book", default="fanduel", help="Bookmaker filter")
    parser.add_argument("--days", type=int, default=60, help="Lookback window in days")
    parser.add_argument("--start-date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--config", default=None, help="Optional v2 config path")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory for v2 logs")
    parser.add_argument("--cache-only", action="store_true", help="Use cache-only mode (no network)")
    parser.add_argument("--ranking-mode", default=None, help="Override ranking mode")
    parser.add_argument("--top-per-game", type=int, default=None, help="Override top picks per game")
    parser.add_argument("--output", default=None, help="Optional path to JSON summary output")
    args = parser.parse_args()

    db_path = Path(args.db)
    pick_db = Path(args.pick_db)
    v2_config = V2Config.load(config_path=args.config)

    start_date, end_date = _resolve_date_range(
        db_path,
        args.book,
        args.days,
        args.start_date,
        args.end_date,
    )
    props = _load_historical_props(db_path, args.book, start_date, end_date)
    if not props:
        print("No historical props found for selection.")
        return 1

    cache_only = args.cache_only
    if cache_only:
        print("Cache-only mode enabled (no network calls).")

    logs_by_player_season = _build_logs_by_player_season(props, args.cache_dir, cache_only)
    game_keys = _build_game_key_map(logs_by_player_season)

    excluded = set(v2_config.excluded_prop_types or [])
    excluded.update(LEGACY_CONFIG.EXCLUDED_PROP_TYPES or [])

    ranking_mode = args.ranking_mode or v2_config.pick_ranking_mode
    top_per_game = args.top_per_game if args.top_per_game is not None else v2_config.top_picks_per_game

    thresholds_v2 = {}
    for prop in v2_config.prop_type_min_edge.keys():
        thresholds_v2[prop] = v2_config.prop_thresholds_for(prop)
    thresholds_legacy = {}
    for prop in LEGACY_CONFIG.PROP_TYPE_MIN_EDGE.keys():
        thresholds_legacy[prop] = LEGACY_CONFIG.prop_thresholds_for(prop)

    min_samples = {
        prop: LEGACY_CONFIG.get_min_sample_size(prop) for prop in ["points", "rebounds", "assists", "pra", "threes"]
    }

    calibration_path = resolve_calibration_path(v2_config.calibration_path, v2_config.cache_dir, Path.cwd())
    calibration = load_calibration(calibration_path)

    legacy_picks = _score_legacy(
        props,
        logs_by_player_season,
        game_keys,
        excluded,
        thresholds_legacy,
        min_samples,
        ranking_mode,
        top_per_game,
    )

    v2_picks = _score_v2(
        props,
        logs_by_player_season,
        game_keys,
        v2_config,
        calibration,
        thresholds_v2,
        min_samples,
        ranking_mode,
        top_per_game,
    )

    print(f"\nData range: {start_date} to {end_date} ({args.book})")
    print(f"Historical props: {len(props)}")
    print(f"Game keys mapped: {len(game_keys)}")

    _print_metrics("Legacy (UnifiedPropModel)", _compute_metrics(legacy_picks))
    _print_metrics("v2 (BaselineModel)", _compute_metrics(v2_picks))

    print("\nPer-prop metrics (legacy):")
    for prop, metrics in _compute_prop_metrics(legacy_picks).items():
        roi = metrics.get("roi")
        win_rate = metrics.get("win_rate")
        print(f"  {prop}: bets {metrics.get('graded_picks')}, ROI {roi:.4f}" if roi is not None else f"  {prop}: bets {metrics.get('graded_picks')}, ROI n/a")
        if win_rate is not None:
            print(f"    win_rate {win_rate:.2%}, units {metrics.get('total_units'):.2f}")

    print("\nPer-prop metrics (v2):")
    for prop, metrics in _compute_prop_metrics(v2_picks).items():
        roi = metrics.get("roi")
        win_rate = metrics.get("win_rate")
        print(f"  {prop}: bets {metrics.get('graded_picks')}, ROI {roi:.4f}" if roi is not None else f"  {prop}: bets {metrics.get('graded_picks')}, ROI n/a")
        if win_rate is not None:
            print(f"    win_rate {win_rate:.2%}, units {metrics.get('total_units'):.2f}")

    legacy_scores = _run_weight_search(legacy_picks)
    v2_scores = _run_weight_search(v2_picks)

    print("\nWeight search (overall score + stability = mean - stdev):")
    def _fmt_score(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    best_weight = None
    best_pipeline = None
    best_score = None

    for idx, weights in enumerate(WEIGHT_SETS):
        legacy = legacy_scores[idx]
        v2 = v2_scores[idx]
        legacy_stability = legacy.get("stability")
        v2_stability = v2.get("stability")
        legacy_overall = legacy.get("overall_score")
        v2_overall = v2.get("overall_score")
        print(
            f"  {weights['name']}: "
            f"legacy {_fmt_score(legacy_overall)} (stab {_fmt_score(legacy_stability)}) "
            f"v2 {_fmt_score(v2_overall)} (stab {_fmt_score(v2_stability)})"
        )

        if legacy_overall is None or v2_overall is None:
            continue
        if legacy_overall >= v2_overall:
            candidate_score = legacy_overall
            candidate_pipeline = "legacy"
        else:
            candidate_score = v2_overall
            candidate_pipeline = "v2"

        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_weight = weights
            best_pipeline = candidate_pipeline

    if best_weight:
        print(f"\nSelected weight set: {best_weight['name']}")
        print(f"Recommended pipeline: {best_pipeline}")

    actual_picks = _load_actual_pick_history(pick_db)
    if actual_picks:
        _print_metrics("Actual pick history (clv_tracking.db)", _compute_metrics(actual_picks))

    if args.output:
        payload = {
            "date_range": {"start": start_date, "end": end_date},
            "bookmaker": args.book,
            "legacy": {
                "metrics": _compute_metrics(legacy_picks),
                "prop_metrics": _compute_prop_metrics(legacy_picks),
            },
            "v2": {
                "metrics": _compute_metrics(v2_picks),
                "prop_metrics": _compute_prop_metrics(v2_picks),
            },
            "weight_search": {
                "legacy": legacy_scores,
                "v2": v2_scores,
                "selected": {"weights": best_weight, "pipeline": best_pipeline},
            },
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSummary saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
