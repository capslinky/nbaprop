"""Normalize raw ingestion data into canonical tables."""

from typing import Dict, List, Tuple
import time

from nbaprop.normalization.ids import (
    make_player_id,
    make_team_id,
    make_game_id,
    make_prop_key,
    canonicalize_player_name,
    canonicalize_team_abbrev,
)
from nbaprop.normalization.name_utils import load_roster_map, normalize_name_for_matching
from nbaprop.normalization.schema import validate_table

try:
    from core.constants import normalize_team_abbrev as _normalize_team_abbrev
except Exception:  # pragma: no cover
    def _normalize_team_abbrev(team: str) -> str:
        return canonicalize_team_abbrev(team or "")


def normalize_raw_data(
    raw_odds: Dict,
    raw_player_logs: List[Dict],
    raw_injuries: Dict,
) -> Dict[str, List[Dict]]:
    players: Dict[str, Dict] = {}
    teams: Dict[str, Dict] = {}
    games: Dict[str, Dict] = {}
    props: Dict[str, Dict] = {}
    injuries: Dict[str, Dict] = {}
    roster_map = load_roster_map()

    for row in raw_player_logs:
        player_name = canonicalize_player_name(row.get("player", "Unknown Player"))
        player_id = make_player_id(player_name)
        roster_key = normalize_name_for_matching(player_name)
        team_abbrev = roster_map.get(roster_key, "")
        players[player_id] = {
            "player_id": player_id,
            "player_name": player_name,
            "team_abbrev": team_abbrev,
        }

    if raw_injuries and isinstance(raw_injuries, dict):
        for entry in raw_injuries.get("entries", []):
            if not isinstance(entry, dict):
                continue
            player_name = canonicalize_player_name(entry.get("player", ""))
            if not player_name:
                continue
            player_id = make_player_id(player_name)
            if player_id in injuries:
                continue
            status = (entry.get("status") or entry.get("INJURY_STATUS") or "").upper()
            team_raw = entry.get("team") or entry.get("TEAM_ABBREVIATION") or ""
            if team_raw:
                team_abbrev = canonicalize_team_abbrev(team_raw)
            else:
                roster_key = normalize_name_for_matching(player_name)
                team_abbrev = roster_map.get(roster_key, "")
            injuries[player_id] = {
                "player_id": player_id,
                "player_name": player_name,
                "team_abbrev": team_abbrev,
                "status": status,
                "injury": entry.get("injury") or entry.get("INJURY_DESCRIPTION") or "",
                "source": entry.get("source") or raw_injuries.get("source"),
            }
            players.setdefault(player_id, {
                "player_id": player_id,
                "player_name": player_name,
                "team_abbrev": team_abbrev,
            })

    # Build event map for reliable home/away/team resolution
    event_map: Dict[str, Dict[str, str]] = {}
    if raw_odds and isinstance(raw_odds, dict):
        for event in raw_odds.get("events", []) or []:
            event_id = event.get("id")
            if not event_id:
                continue
            event_map[event_id] = {
                "home_team": _normalize_team_abbrev(event.get("home_team", "")),
                "away_team": _normalize_team_abbrev(event.get("away_team", "")),
                "game_time": event.get("commence_time") or event.get("game_time") or "",
            }

    # Seed props and players from odds snapshot props
    if raw_odds and isinstance(raw_odds, dict):
        for prop in raw_odds.get("props", []):
            player_name = canonicalize_player_name(prop.get("player", "Unknown Player"))
            player_id = make_player_id(player_name)
            roster_key = normalize_name_for_matching(player_name)
            player_team = roster_map.get(roster_key, "")
            players[player_id] = {
                "player_id": player_id,
                "player_name": player_name,
                "team_abbrev": player_team,
            }

            event_id = prop.get("event_id") or prop.get("game_id")
            event_info = event_map.get(event_id) if event_id else None
            home_team = _normalize_team_abbrev(prop.get("home_team", ""))
            away_team = _normalize_team_abbrev(prop.get("away_team", ""))
            game_time = prop.get("game_time") or prop.get("commence_time")
            if event_info:
                home_team = event_info.get("home_team") or home_team
                away_team = event_info.get("away_team") or away_team
                game_time = event_info.get("game_time") or game_time

            if player_team and home_team and away_team:
                if player_team not in (home_team, away_team):
                    continue

            prop_type = (prop.get("prop_type") or "").lower()
            line = prop.get("line")
            side = (prop.get("side") or "").lower()
            if prop_type and line is not None:
                prop_id = make_prop_key(player_id, prop_type, line, side)
                props[prop_id] = {
                    "prop_id": prop_id,
                    "player_id": player_id,
                    "player_name": player_name,
                    "prop_type": prop_type,
                    "line": line,
                    "odds": prop.get("odds"),
                    "side": side or None,
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "game_time": game_time,
                    "team_abbrev": player_team,
                }

    # Seed game data from the first odds event when available
    if raw_odds:
        event = None
        events = raw_odds.get("events") if isinstance(raw_odds, dict) else None
        if isinstance(events, list) and events:
            event = events[0]

        if event:
            home_team = _normalize_team_abbrev(event.get("home_team", "HOME"))
            away_team = _normalize_team_abbrev(event.get("away_team", "AWAY"))
            game_time = event.get("commence_time")
        else:
            home_team = _normalize_team_abbrev(raw_odds.get("home_team", "HOME"))
            away_team = _normalize_team_abbrev(raw_odds.get("away_team", "AWAY"))
            game_time = raw_odds.get("game_time")

        if not game_time:
            game_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        game_id = make_game_id(home_team, away_team, game_time)

        games[game_id] = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "game_time": game_time,
        }

        for team in (home_team, away_team):
            team_id = make_team_id(team)
            teams[team_id] = {
                "team_id": team_id,
                "team_abbrev": team,
            }

    normalized = {
        "players": list(players.values()),
        "teams": list(teams.values()),
        "games": list(games.values()),
        "props": list(props.values()),
        "injuries": list(injuries.values()),
        "prop_features": [],
        "picks": [],
    }

    for name, rows in normalized.items():
        validate_table(name, rows)

    return normalized
