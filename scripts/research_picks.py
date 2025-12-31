#!/usr/bin/env python3
"""Deep research and re-rank top picks with news intelligence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbaprop.config import Config  # noqa: E402
from nbaprop.review import find_latest_picks_for_date, load_picks  # noqa: E402
from nbaprop.storage import FileCache  # noqa: E402
from nbaprop.ops import get_rate_limiter  # noqa: E402
from nbaprop.ingestion.injuries import fetch_injury_report  # noqa: E402
from nbaprop.ingestion.odds import fetch_odds_snapshot  # noqa: E402
from nbaprop.normalization.name_utils import (  # noqa: E402
    normalize_name_for_matching,
    load_roster_map,
)

from core.constants import (  # noqa: E402
    normalize_team_abbrev,
    STAR_PLAYERS,
    STAR_OUT_BOOST,
)
from core.news_intelligence import (  # noqa: E402
    NewsContext,
    NewsIntelligence,
    create_perplexity_api_search,
)


_OUT_STATUSES = {
    "OUT",
    "O",
    "DNP",
    "INJURED",
    "SUSPENDED",
    "DOUBTFUL",
    "D",
}
_QUESTIONABLE_STATUSES = {
    "QUESTIONABLE",
    "Q",
    "GTD",
    "GAME TIME DECISION",
    "GAME-TIME DECISION",
}
_PROBABLE_STATUSES = {
    "PROBABLE",
    "P",
}


def _parse_float(value, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_bool(value) -> bool:
    if value in (None, ""):
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: Optional[float]) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(ts, ZoneInfo("America/New_York"))
    return dt.strftime("%Y-%m-%d %I:%M %p ET")


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    dt = value.astimezone(ZoneInfo("America/New_York"))
    return dt.strftime("%Y-%m-%d %I:%M %p ET")


def _parse_injury_report_url_timestamp(url: str) -> Optional[datetime]:
    if not url:
        return None
    match = re.search(
        r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf",
        url,
    )
    if match:
        date_str, hour_str, minute_str, meridiem = match.groups()
        try:
            return datetime.strptime(
                f"{date_str} {hour_str}:{minute_str}{meridiem}",
                "%Y-%m-%d %I:%M%p",
            ).replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            return None
    match = re.search(
        r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(AM|PM)\.pdf",
        url,
    )
    if match:
        date_str, hour_str, meridiem = match.groups()
        try:
            return datetime.strptime(
                f"{date_str} {hour_str}{meridiem}",
                "%Y-%m-%d %I%p",
            ).replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            return None
    return None


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


def _prop_boost_key(prop_type: Optional[str]) -> Optional[str]:
    prop = _normalize_prop_type(prop_type)
    if prop in ("points", "rebounds", "assists", "pra", "threes"):
        return prop
    return None


def _game_date(value: Optional[str], timezone: str = "America/New_York") -> Optional[str]:
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    return parsed.astimezone(ZoneInfo(timezone)).date().isoformat()


def _resolve_run_date(config: Config) -> str:
    if config.run_date:
        return config.run_date
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.strftime("%Y-%m-%d")


def _resolve_latest_picks(config: Config) -> Path:
    cache_dir = Path(config.cache_dir)
    runs_dir = cache_dir / "runs"
    run_date = _resolve_run_date(config)
    return find_latest_picks_for_date(runs_dir, run_date)


def _normalize_team(team: Optional[str]) -> str:
    if not team:
        return ""
    return normalize_team_abbrev(team)


def _news_label(context: Optional[NewsContext]) -> str:
    if not context:
        return "NO_NEWS"
    return context.status or "NO_NEWS"


def _serialize_list(values: Optional[Iterable[str]]) -> str:
    if not values:
        return ""
    return "; ".join(str(v).strip() for v in values if str(v).strip())


def _extract_injury_report_meta(report: Optional[Dict]) -> Dict[str, Optional[str]]:
    if not report:
        return {
            "injury_report_fetched_at": None,
            "injury_report_as_of": None,
            "injury_report_url": None,
            "injury_report_sources": None,
        }
    fetched_at = _format_timestamp(report.get("fetched_at"))
    debug = report.get("injury_debug")
    report_url = None
    if isinstance(debug, dict):
        report_url = debug.get("official_report_url")
    report_as_of = _format_datetime(_parse_injury_report_url_timestamp(report_url)) if report_url else None
    sources = _serialize_list(report.get("sources"))
    return {
        "injury_report_fetched_at": fetched_at,
        "injury_report_as_of": report_as_of,
        "injury_report_url": report_url,
        "injury_report_sources": sources,
    }


def _build_injury_indexes(entries: Optional[List[Dict]]) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    by_name: Dict[str, Dict] = {}
    by_team: Dict[str, List[Dict]] = {}
    if not entries:
        return by_name, by_team
    for entry in entries:
        player = entry.get("player") or entry.get("PLAYER_NAME") or ""
        if not player:
            continue
        team_raw = entry.get("team") or entry.get("TEAM_ABBREVIATION") or ""
        team = _normalize_team(team_raw)
        status_raw = entry.get("status") or entry.get("INJURY_STATUS") or ""
        status = str(status_raw).strip().upper() if status_raw else ""
        injury = entry.get("injury") or entry.get("INJURY_DESCRIPTION") or ""
        source = entry.get("source") or "Unknown"
        player_key = normalize_name_for_matching(player)
        record = {
            "player": player,
            "team": team,
            "status": status,
            "injury": injury,
            "source": source,
        }
        if player_key and player_key not in by_name:
            by_name[player_key] = record
        if team:
            by_team.setdefault(team, []).append(record)
    return by_name, by_team


def _team_injury_names(
    team: str,
    by_team: Dict[str, List[Dict]],
    statuses: set,
    exclude_key: Optional[str],
) -> List[str]:
    if not team:
        return []
    entries = by_team.get(team, [])
    if not entries:
        return []
    labels = []
    for entry in entries:
        status = (entry.get("status") or "").upper()
        if status not in statuses:
            continue
        player = entry.get("player") or ""
        if exclude_key and normalize_name_for_matching(player) == exclude_key:
            continue
        injury = entry.get("injury") or ""
        if injury:
            labels.append(f"{player} ({injury})")
        else:
            labels.append(player)
    return labels


def _stars_out_for_team(
    team: str,
    by_name: Dict[str, Dict],
    exclude_key: Optional[str],
) -> List[str]:
    if not team:
        return []
    stars = STAR_PLAYERS.get(team, [])
    out: List[str] = []
    for star in stars:
        if exclude_key and normalize_name_for_matching(star) == exclude_key:
            continue
        entry = by_name.get(normalize_name_for_matching(star))
        if not entry:
            continue
        status = (entry.get("status") or "").upper()
        if status in _OUT_STATUSES:
            out.append(star)
    return out


def _build_odds_indexes(odds_snapshot: Optional[Dict]) -> Tuple[Dict[Tuple, List[Dict]], Dict[Tuple, List[Dict]]]:
    event_index: Dict[Tuple, List[Dict]] = {}
    team_index: Dict[Tuple, List[Dict]] = {}
    if not odds_snapshot:
        return event_index, team_index
    for prop in odds_snapshot.get("props", []) or []:
        player = prop.get("player")
        prop_type = _normalize_prop_type(prop.get("prop_type"))
        side = (prop.get("side") or "").upper()
        if not player or not prop_type or side not in ("OVER", "UNDER"):
            continue
        player_key = normalize_name_for_matching(player)
        event_id = prop.get("event_id") or prop.get("game_id")
        home_team = prop.get("home_team") or ""
        away_team = prop.get("away_team") or ""
        if event_id:
            event_index.setdefault((player_key, prop_type, side, event_id), []).append(prop)
        team_index.setdefault((player_key, prop_type, side, home_team, away_team), []).append(prop)
    return event_index, team_index


def _select_best_odds(candidates: List[Dict], line_value: Optional[float]) -> Optional[Dict]:
    if not candidates:
        return None

    def _line_distance(prop_row: Dict) -> float:
        if line_value is None:
            return 0.0
        try:
            return abs(float(prop_row.get("line")) - line_value)
        except (TypeError, ValueError):
            return 0.0

    return min(candidates, key=_line_distance)


def _odds_cache_key(
    max_events: int,
    target_date: Optional[str],
    markets: Optional[List[str]],
    bookmakers: Optional[List[str]],
) -> str:
    markets_key = ",".join(markets) if markets else "default"
    books_key = ",".join(bookmakers) if bookmakers else "all_books"
    return f"odds_snapshot:{max_events}:{target_date or 'all'}:{markets_key}:{books_key}"


def _resolve_target_date(picks: List[Dict], override: Optional[str], config: Config) -> Optional[str]:
    if override:
        return override
    if config.run_date:
        return config.run_date
    for row in picks:
        date_val = _game_date(row.get("game_time"))
        if date_val:
            return date_val
    return None


def _load_player_team_map_from_nba_api() -> Dict[str, str]:
    try:
        from nba_api.stats.endpoints import playerindex
    except Exception:
        return {}

    try:
        from core.constants import get_current_nba_season
    except Exception:
        return {}

    limiter = get_rate_limiter()
    limiter.wait("nba_api", interval=0.6)

    try:
        players = playerindex.PlayerIndex(season=get_current_nba_season())
        frames = players.get_data_frames()
    except Exception:
        return {}

    if not frames:
        return {}

    df = frames[0]
    if df is None or df.empty:
        return {}

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        first = row.get("PLAYER_FIRST_NAME")
        last = row.get("PLAYER_LAST_NAME")
        if first and last:
            name = f"{first} {last}".strip()
        else:
            name = row.get("PLAYER_NAME") or row.get("PLAYER_SLUG")
        team = row.get("TEAM_ABBREVIATION")
        if not name or not team:
            continue
        mapping[normalize_name_for_matching(str(name))] = _normalize_team(team)
    return mapping


def _market_divergence(row: Dict) -> Optional[float]:
    model_prob = row.get("model_probability") or row.get("raw_model_probability")
    market_prob = row.get("market_probability")
    if model_prob in (None, "") or market_prob in (None, ""):
        return None
    try:
        model_val = float(model_prob)
        market_val = float(market_prob)
    except (TypeError, ValueError):
        return None
    return abs(model_val - market_val)


def _apply_usage_adjustment(score: float, pick: str, usage_text: str, notes: List[str]) -> float:
    if not usage_text:
        return score
    usage_lower = usage_text.lower()
    if "usage up" in usage_lower or "teammates out" in usage_lower:
        if pick == "OVER":
            score += 2.0
            notes.append("usage up (over boost)")
        elif pick == "UNDER":
            score -= 2.0
            notes.append("usage up (under penalty)")
    if "usage down" in usage_lower:
        if pick == "OVER":
            score -= 2.0
            notes.append("usage down (over penalty)")
        elif pick == "UNDER":
            score += 2.0
            notes.append("usage down (under boost)")
    return score


def _apply_injury_report_adjustment(
    score: float,
    pick: str,
    prop_type: Optional[str],
    injury_context: Optional[Dict],
    notes: List[str],
) -> Tuple[float, bool]:
    if not injury_context:
        return score, False

    status = (injury_context.get("player_status") or "").upper()
    if status in _OUT_STATUSES:
        notes.append(f"injury report: {status} (skip)")
        return -999.0, True
    if status in _QUESTIONABLE_STATUSES:
        score -= 6.0
        notes.append(f"injury report: {status.lower()}")
    elif status in _PROBABLE_STATUSES:
        score -= 1.5
        notes.append("injury report: probable")

    stars_out = injury_context.get("stars_out") or []
    if stars_out:
        boost_key = _prop_boost_key(prop_type)
        base_boost = STAR_OUT_BOOST.get(boost_key, 1.03) if boost_key else 1.02
        impact = max(1.5, (base_boost - 1.0) * 50.0)
        if len(stars_out) > 1:
            impact += (len(stars_out) - 1) * 0.75
        if pick == "OVER":
            score += impact
            notes.append(f"star out: {', '.join(stars_out)} (usage boost)")
        elif pick == "UNDER":
            score -= impact
            notes.append(f"star out: {', '.join(stars_out)} (usage risk)")

    teammates_questionable = injury_context.get("teammates_questionable") or []
    if teammates_questionable:
        score -= 0.5
        notes.append("teammates questionable")

    return score, False


def _apply_player_news_adjustment(
    score: float,
    pick: str,
    context: Optional[NewsContext],
    notes: List[str],
) -> Tuple[float, bool]:
    if not context:
        return score, False
    if context.should_skip():
        notes.append(f"player status {context.status}: skip")
        return -999.0, True

    if context.adjustment_factor != 1.0:
        score *= context.adjustment_factor
        notes.append(f"news adj {context.adjustment_factor:.2f}")

    status = context.status
    if status in ("GTD_LEANING_OUT", "GTD_UNCERTAIN"):
        score -= 6.0
        notes.append(f"{status.lower().replace('_', ' ')}")
    elif status == "GTD_LEANING_PLAY":
        score -= 2.0
        notes.append("gtd leaning play")
    elif status == "RETURNING":
        score -= 2.0
        notes.append("returning from injury")

    if context.minutes_impact == "REDUCED":
        score -= 4.0
        notes.append("minutes restriction")
    elif context.minutes_impact == "INCREASED":
        score += 2.0
        notes.append("minutes increase")

    if context.confidence < 0.6:
        score -= 1.0
        notes.append("low news confidence")

    flags_lower = " ".join(context.flags).lower()
    if "usage boost" in flags_lower:
        if pick == "OVER":
            score += 2.0
            notes.append("news usage boost (over)")
        elif pick == "UNDER":
            score -= 2.0
            notes.append("news usage boost (under)")

    return score, False


def _apply_team_news_adjustment(
    score: float,
    pick: str,
    context: Optional[NewsContext],
    notes: List[str],
) -> float:
    if not context:
        return score
    flags_lower = " ".join(context.flags).lower()
    if "players out" in flags_lower:
        if pick == "OVER":
            score += 1.0
            notes.append("team injuries (over boost)")
        elif pick == "UNDER":
            score -= 1.0
            notes.append("team injuries (under penalty)")
    if "star player" in flags_lower:
        if pick == "OVER":
            score += 1.5
            notes.append("star teammate news (over boost)")
        elif pick == "UNDER":
            score -= 1.5
            notes.append("star teammate news (under penalty)")
    if context.confidence < 0.6:
        score -= 0.5
        notes.append("low team-news confidence")
    return score


def _compute_research_score(
    row: Dict,
    player_news: Optional[NewsContext],
    team_news: Optional[NewsContext],
    injury_context: Optional[Dict],
) -> Tuple[float, List[str], bool]:
    edge = _parse_float(row.get("edge"))
    confidence = _parse_float(row.get("confidence"))
    pick = (row.get("pick") or "").upper()
    prop_type = row.get("prop_type")
    notes: List[str] = []

    score = edge * 100.0
    notes.append(f"edge {edge:.3f}")
    score += (confidence - 0.5) * 20.0
    notes.append(f"confidence {confidence:.2f}")

    hit_rates = []
    hit10 = _parse_float(row.get("recent_hit_rate_10"), default=math.nan)
    hit15 = _parse_float(row.get("recent_hit_rate_15"), default=math.nan)
    if not math.isnan(hit10):
        hit_rates.append(hit10)
    if not math.isnan(hit15):
        hit_rates.append(hit15)
    if hit_rates:
        avg_hit = sum(hit_rates) / len(hit_rates)
        score += (avg_hit - 0.5) * 10.0
        notes.append(f"hit-rate {avg_hit:.2f}")

    trend = (row.get("trend") or "").upper()
    if trend == "UP":
        score += 2.0
        notes.append("trend up")
    elif trend == "DOWN":
        score -= 2.0
        notes.append("trend down")

    volatility_ratio = _parse_float(row.get("volatility_ratio"), default=math.nan)
    if not math.isnan(volatility_ratio):
        if volatility_ratio >= 0.35:
            penalty = min(8.0, (volatility_ratio - 0.3) * 20.0)
            score -= penalty
            notes.append(f"volatility {volatility_ratio:.2f} (-{penalty:.1f})")

    market_diff = _market_divergence(row)
    if market_diff is not None:
        if market_diff >= 0.15:
            score -= 3.0
            notes.append("market divergence")
        elif market_diff >= 0.1:
            score -= 1.5
            notes.append("market mild divergence")

    rest_days = _parse_int(row.get("rest_days"))
    if rest_days is not None:
        if rest_days <= 1:
            score -= 1.0
            notes.append("short rest")
        elif rest_days >= 3:
            score += 1.0
            notes.append("extra rest")

    if _parse_bool(row.get("b2b")):
        score -= 1.0
        notes.append("back-to-back")

    usage_text = row.get("usage_expectation") or ""
    score = _apply_usage_adjustment(score, pick, usage_text, notes)

    odds_val = _parse_int(row.get("latest_odds") or row.get("odds"))
    if odds_val is not None and abs(odds_val) >= 140:
        score -= 1.5
        notes.append("price stretch")
    elif odds_val is not None and abs(odds_val) >= 120:
        score -= 0.8
        notes.append("price outside core")

    score, skip = _apply_injury_report_adjustment(score, pick, prop_type, injury_context, notes)
    if skip:
        return score, notes, True

    score, skip = _apply_player_news_adjustment(score, pick, player_news, notes)
    if skip:
        return score, notes, True

    score = _apply_team_news_adjustment(score, pick, team_news, notes)

    return score, notes, False


def _build_news_intel(config: Config, disable_news: bool) -> Optional[NewsIntelligence]:
    if disable_news:
        return None
    if not config.perplexity_api_key:
        return None
    search_fn = create_perplexity_api_search(config.perplexity_api_key)
    return NewsIntelligence(search_fn=search_fn)


def _load_filtered_picks(picks_path: Path) -> List[Dict]:
    rows = load_picks(picks_path)
    filtered = []
    for row in rows:
        pick = (row.get("pick") or row.get("side") or "").upper()
        if pick not in ("OVER", "UNDER"):
            continue
        filtered.append(row)
    return filtered


def _resolve_output_path(output_dir: Path, picks_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = picks_path.stem
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"research_{stem}_{timestamp}.csv"


def _load_config(config_path: Optional[str]) -> Config:
    if config_path:
        return Config.load(config_path=config_path)
    candidate_paths = [ROOT / ".env", ROOT.parent / ".env"]
    for path in candidate_paths:
        if path.exists():
            return Config.load(config_path=str(path))
    return Config.load()


def run_research(
    picks_path: Optional[str],
    top_n: int,
    output_dir: str,
    disable_news: bool,
    game_date: Optional[str],
    config_path: Optional[str],
    refresh_injuries: bool,
    refresh_odds: bool,
    odds_max_events: int,
) -> Path:
    config = _load_config(config_path)
    resolved_picks_path = Path(picks_path) if picks_path else _resolve_latest_picks(config)
    picks = _load_filtered_picks(resolved_picks_path)
    if not picks:
        raise SystemExit(f"No picks found in {resolved_picks_path}")

    picks.sort(key=lambda row: _parse_float(row.get("edge")), reverse=True)
    top_picks = picks[: max(1, top_n)]

    cache = FileCache(config.cache_dir)
    if refresh_injuries:
        cache.invalidate("injury_report")
    injury_report = fetch_injury_report(
        cache,
        ttl_seconds=300,
        perplexity_api_key=config.perplexity_api_key or None,
    )
    injury_meta = _extract_injury_report_meta(injury_report)
    injury_entries = injury_report.get("entries") if injury_report else []
    injury_by_name, injury_by_team = _build_injury_indexes(injury_entries)

    target_date = _resolve_target_date(top_picks, game_date, config)
    odds_snapshot = None
    odds_meta = {
        "odds_snapshot_fetched_at": None,
        "odds_snapshot_source": None,
        "odds_snapshot_remaining": None,
    }
    odds_event_index: Dict[Tuple, List[Dict]] = {}
    odds_team_index: Dict[Tuple, List[Dict]] = {}
    if config.odds_api_key:
        max_events = max(0, odds_max_events)
        cache_key = _odds_cache_key(
            max_events,
            target_date,
            config.odds_prop_markets,
            config.odds_bookmakers,
        )
        if refresh_odds:
            cache.invalidate(cache_key)
        odds_snapshot = fetch_odds_snapshot(
            cache,
            ttl_seconds=90,
            api_key=config.odds_api_key,
            max_events=max_events,
            include_props=True,
            target_date=target_date,
            markets=config.odds_prop_markets,
            bookmakers=config.odds_bookmakers,
        )
        odds_meta = {
            "odds_snapshot_fetched_at": _format_timestamp(odds_snapshot.get("fetched_at")),
            "odds_snapshot_source": odds_snapshot.get("source"),
            "odds_snapshot_remaining": odds_snapshot.get("remaining_requests"),
            "odds_snapshot_bookmakers": _serialize_list(config.odds_bookmakers),
        }
        odds_event_index, odds_team_index = _build_odds_indexes(odds_snapshot)

    news_intel = _build_news_intel(config, disable_news)
    team_news_cache: Dict[Tuple[str, str, str], Dict[str, NewsContext]] = {}
    player_news_cache: Dict[Tuple[str, str], NewsContext] = {}
    roster_map = load_roster_map()
    nba_team_map: Optional[Dict[str, str]] = None

    enriched: List[Dict] = []
    for idx, row in enumerate(top_picks, start=1):
        player = row.get("player_name") or ""
        team = _normalize_team(row.get("player_team"))
        home = _normalize_team(row.get("home_team"))
        away = _normalize_team(row.get("away_team"))
        row_game_date = game_date or _game_date(row.get("game_time"))

        player_key = normalize_name_for_matching(player)
        injury_entry = injury_by_name.get(player_key)
        if not team and injury_entry:
            team = _normalize_team(injury_entry.get("team"))
        if not team:
            team = _normalize_team(roster_map.get(player_key, ""))
        if not team:
            if nba_team_map is None:
                nba_team_map = _load_player_team_map_from_nba_api()
            if nba_team_map:
                team = _normalize_team(nba_team_map.get(player_key, ""))

        injury_status = (injury_entry.get("status") if injury_entry else "") or ""
        injury_detail = (injury_entry.get("injury") if injury_entry else "") or ""
        injury_source = (injury_entry.get("source") if injury_entry else "") or ""

        teammates_out = _team_injury_names(team, injury_by_team, _OUT_STATUSES, player_key)
        teammates_questionable = _team_injury_names(team, injury_by_team, _QUESTIONABLE_STATUSES, player_key)
        stars_out = _stars_out_for_team(team, injury_by_name, player_key)

        pick_side = (row.get("pick") or row.get("side") or "").upper()
        injury_context_note = ""
        if injury_status and injury_status.upper() in _OUT_STATUSES:
            injury_context_note = f"Player listed {injury_status}: {injury_detail}".strip()
        elif injury_status and injury_status.upper() in _QUESTIONABLE_STATUSES:
            injury_context_note = f"Player listed {injury_status}: {injury_detail}".strip()
        elif injury_status and injury_status.upper() in _PROBABLE_STATUSES:
            injury_context_note = f"Player listed {injury_status}: {injury_detail}".strip()
        elif stars_out:
            impact = "expected usage increase" if pick_side == "OVER" else "usage risk for under"
            injury_context_note = f"Teammates out: {', '.join(stars_out)}; {impact}."
        elif teammates_out:
            injury_context_note = f"Teammates out: {', '.join(teammates_out)}; rotation/usage volatility."

        if teammates_questionable:
            suffix = f" Questionable: {', '.join(teammates_questionable)}."
            injury_context_note = f"{injury_context_note}{suffix}".strip()

        injury_context = {
            "player_status": injury_status,
            "player_source": injury_source,
            "player_injury": injury_detail,
            "team": team,
            "stars_out": stars_out,
            "teammates_out": teammates_out,
            "teammates_questionable": teammates_questionable,
        }

        latest_odds = None
        latest_line = None
        if odds_snapshot:
            prop_type = _normalize_prop_type(row.get("prop_type"))
            side = pick_side
            if player and prop_type and side in ("OVER", "UNDER"):
                event_id = row.get("event_id")
                candidates: List[Dict] = []
                if event_id:
                    candidates = odds_event_index.get((player_key, prop_type, side, event_id), [])
                if not candidates:
                    candidates = odds_team_index.get((player_key, prop_type, side, home, away), [])
                line_value = _parse_float(row.get("line"), default=math.nan)
                line_value = None if math.isnan(line_value) else line_value
                chosen = _select_best_odds(candidates, line_value)
                if chosen:
                    latest_odds = chosen.get("odds")
                    latest_line = chosen.get("line")

        player_news = None
        team_news = None

        if news_intel:
            player_key = (player, row_game_date or "")
            if player_key not in player_news_cache:
                player_news_cache[player_key] = news_intel.fetch_player_news(
                    player_name=player,
                    team=team or None,
                    game_date=row_game_date,
                )
            player_news = player_news_cache[player_key]

            matchup_key = (home, away, row_game_date or "")
            if matchup_key not in team_news_cache:
                team_news_cache[matchup_key] = news_intel.fetch_game_news(
                    home_team=home,
                    away_team=away,
                    game_date=row_game_date,
                )
            team_news = team_news_cache[matchup_key].get(team)

        if latest_odds is not None:
            row["latest_odds"] = latest_odds
        if latest_line is not None:
            row["latest_line"] = latest_line

        score, score_notes, skip = _compute_research_score(row, player_news, team_news, injury_context)

        output_row = dict(row)
        output_row.update({
            "base_rank": idx,
            "research_score": round(score, 3),
            "research_skip": skip,
            "research_notes": " | ".join(score_notes),
            "injury_report_status": injury_status or None,
            "injury_report_detail": injury_detail or None,
            "injury_report_source": injury_source or None,
            "injury_report_team": team or None,
            "injury_teammates_out": _serialize_list(teammates_out),
            "injury_teammates_questionable": _serialize_list(teammates_questionable),
            "injury_stars_out": _serialize_list(stars_out),
            "injury_context_note": injury_context_note,
            "latest_odds": latest_odds,
            "latest_line": latest_line,
            "latest_odds_fetched_at": odds_meta.get("odds_snapshot_fetched_at"),
            "latest_odds_source": odds_meta.get("odds_snapshot_source"),
            "player_news_status": _news_label(player_news),
            "player_news_confidence": round(player_news.confidence, 3) if player_news else None,
            "player_news_flags": _serialize_list(player_news.flags) if player_news else "",
            "player_news_notes": _serialize_list(player_news.notes) if player_news else "",
            "player_news_sources": _serialize_list(player_news.sources) if player_news else "",
            "team_news_status": _news_label(team_news),
            "team_news_confidence": round(team_news.confidence, 3) if team_news else None,
            "team_news_flags": _serialize_list(team_news.flags) if team_news else "",
            "team_news_notes": _serialize_list(team_news.notes) if team_news else "",
            "team_news_sources": _serialize_list(team_news.sources) if team_news else "",
        })
        output_row.update(injury_meta)
        output_row.update(odds_meta)
        if team and not output_row.get("player_team"):
            output_row["player_team"] = team
        enriched.append(output_row)

    enriched.sort(key=lambda row: row.get("research_score", -999), reverse=True)
    for idx, row in enumerate(enriched, start=1):
        row["research_rank"] = idx

    output_path = _resolve_output_path(Path(output_dir), resolved_picks_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: List[str] = []
        for row in enriched:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    summary_path = output_path.with_suffix(".json")
    summary_payload = {
        "picks_source": str(resolved_picks_path),
        "generated_at": datetime.utcnow().isoformat(),
        "top_n": top_n,
        "news_enabled": bool(news_intel),
        "injury_report": injury_meta,
        "odds_snapshot": odds_meta,
        "results": enriched,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep research and re-rank top NBA prop picks.")
    parser.add_argument("--config", dest="config_path", help="Path to config file (optional).")
    parser.add_argument("--picks-path", help="Path to picks CSV (defaults to latest filtered picks).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of picks to research (default: 10).")
    parser.add_argument("--output-dir", default=".cache/research", help="Output directory for research report.")
    parser.add_argument("--no-news", action="store_true", help="Skip Perplexity news lookup.")
    parser.add_argument("--game-date", help="Override game date (YYYY-MM-DD) for news queries.")
    parser.add_argument("--refresh-injuries", action="store_true", help="Force refresh the injury report.")
    parser.add_argument("--refresh-odds", action="store_true", help="Force refresh the odds snapshot.")
    parser.add_argument(
        "--odds-max-events",
        type=int,
        default=0,
        help="Max events to fetch for odds snapshot (0 = all events).",
    )
    args = parser.parse_args()

    output_path = run_research(
        picks_path=args.picks_path,
        top_n=args.top_n,
        output_dir=args.output_dir,
        disable_news=args.no_news,
        game_date=args.game_date,
        config_path=args.config_path,
        refresh_injuries=args.refresh_injuries,
        refresh_odds=args.refresh_odds,
        odds_max_events=args.odds_max_events,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
