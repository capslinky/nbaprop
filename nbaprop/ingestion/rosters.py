"""Roster and depth chart refresh utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional
import json
import logging
import os
import unicodedata

from core.constants import get_current_nba_season
from core.rosters import PlayerRoster, TeamRoster, save_rosters_to_json
from nbaprop.normalization.name_utils import normalize_name_for_matching

logger = logging.getLogger(__name__)


_DEFAULT_ROSTER_PATH = Path(__file__).resolve().parents[2] / "data" / "rosters_2025_26.json"


def _ascii_name(value: str) -> str:
    if not value:
        return ""
    return (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
    )


def _parse_last_updated(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    last_updated = None
    for team in payload.values():
        stamp = _parse_last_updated(team.get("last_updated") or "")
        if stamp and (last_updated is None or stamp > last_updated):
            last_updated = stamp
    if not last_updated:
        return True
    delta = datetime.utcnow() - last_updated
    return delta.days >= max_age_days


def _load_existing_rosters(path: Path) -> Dict[str, TeamRoster]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rosters: Dict[str, TeamRoster] = {}
    for team_abbrev, team_data in payload.items():
        players = []
        for player in team_data.get("players", []) or []:
            try:
                players.append(PlayerRoster(**player))
            except TypeError:
                continue
        rosters[team_abbrev] = TeamRoster(
            team_abbrev=team_data.get("team_abbrev", team_abbrev),
            team_name=team_data.get("team_name", team_abbrev),
            players=players,
            last_updated=team_data.get("last_updated", ""),
        )
    return rosters


def build_rosters_from_nba_api(
    season: Optional[str] = None,
    base_delay: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, TeamRoster]:
    """Build full team rosters + depth chart roles from NBA API data."""
    try:
        from nba_api.stats.static import teams
        from nba_api.stats.endpoints import commonteamroster
    except Exception as exc:
        raise RuntimeError(f"nba_api unavailable: {exc}") from exc

    from data.fetchers.nba_fetcher import NBADataFetcher

    season = season or get_current_nba_season()
    fetcher = NBADataFetcher(cache_dir=cache_dir, base_delay=base_delay)
    advanced = fetcher.get_player_advanced_stats(season=season, force_refresh=True)

    minutes_map: Dict[str, float] = {}
    if not advanced.empty:
        for _, row in advanced.iterrows():
            name = row.get("player_name")
            if not name:
                continue
            key = normalize_name_for_matching(str(name))
            try:
                minutes = float(row.get("minutes") or 0.0)
            except (TypeError, ValueError):
                minutes = 0.0
            minutes_map[key] = minutes

    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    roster_map: Dict[str, TeamRoster] = {}

    for team in teams.get_teams():
        team_id = team.get("id")
        team_abbrev = team.get("abbreviation")
        team_name = team.get("full_name")
        if not team_id or not team_abbrev:
            continue

        try:
            response = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = response.get_data_frames()[0]
        except Exception as exc:
            logger.warning("Roster fetch failed for %s: %s", team_abbrev, exc)
            continue

        players = []
        for _, row in roster_df.iterrows():
            name = row.get("PLAYER") or row.get("Player")
            if not name:
                continue
            name = _ascii_name(name)
            position = row.get("POSITION") or row.get("Position") or "UNK"
            key = normalize_name_for_matching(name)
            minutes = minutes_map.get(key, 0.0)
            players.append({
                "name": name,
                "position": position,
                "avg_minutes": minutes,
            })

        players.sort(key=lambda entry: entry["avg_minutes"], reverse=True)
        roster_players = []
        for idx, player in enumerate(players):
            if idx < 5:
                role = "STARTER"
            elif idx < 10:
                role = "ROTATION"
            else:
                role = "BENCH"
            is_star = idx < 2 or player["avg_minutes"] >= 32
            roster_players.append(PlayerRoster(
                name=player["name"],
                position=player["position"],
                role=role,
                avg_minutes=round(float(player["avg_minutes"]), 1),
                is_star=bool(is_star),
            ))

        roster_map[team_abbrev] = TeamRoster(
            team_abbrev=team_abbrev,
            team_name=team_name or team_abbrev,
            players=roster_players,
            last_updated=updated_at,
        )

    return roster_map


def refresh_rosters(
    output_path: Optional[Path] = None,
    season: Optional[str] = None,
    base_delay: Optional[float] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
    max_age_days: int = 1,
) -> Path:
    """Refresh roster file if stale or forced; returns path to roster JSON."""
    path = Path(
        output_path
        or os.environ.get("NBAPROP_ROSTERS_PATH", "")
        or _DEFAULT_ROSTER_PATH
    )
    fallback = _load_existing_rosters(path)
    if not force and not _is_stale(path, max_age_days=max_age_days):
        logger.info("Roster file is fresh: %s", path)
        return path

    rosters = build_rosters_from_nba_api(
        season=season,
        base_delay=base_delay,
        cache_dir=cache_dir,
    )
    if fallback:
        for team_abbrev, roster in fallback.items():
            rosters.setdefault(team_abbrev, roster)
    save_rosters_to_json(rosters, filepath=str(path))
    logger.info("Saved roster file to %s (%d teams).", path, len(rosters))
    return path
