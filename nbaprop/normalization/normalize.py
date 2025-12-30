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
from nbaprop.normalization.schema import validate_table


def normalize_raw_data(
    raw_odds: Dict,
    raw_player_logs: List[Dict],
    raw_injuries: Dict,
) -> Dict[str, List[Dict]]:
    players: Dict[str, Dict] = {}
    teams: Dict[str, Dict] = {}
    games: Dict[str, Dict] = {}
    props: Dict[str, Dict] = {}

    for row in raw_player_logs:
        player_name = canonicalize_player_name(row.get("player", "Unknown Player"))
        player_id = make_player_id(player_name)
        players[player_id] = {
            "player_id": player_id,
            "player_name": player_name,
        }

    # Seed props and players from odds snapshot props
    if raw_odds and isinstance(raw_odds, dict):
        for prop in raw_odds.get("props", []):
            player_name = canonicalize_player_name(prop.get("player", "Unknown Player"))
            player_id = make_player_id(player_name)
            players[player_id] = {
                "player_id": player_id,
                "player_name": player_name,
            }

            prop_type = (prop.get("prop_type") or "").lower()
            line = prop.get("line")
            side = (prop.get("side") or "").lower()
            if prop_type and line is not None:
                prop_id = make_prop_key(player_id, prop_type, line, side)
                props[prop_id] = {
                    "prop_id": prop_id,
                    "player_id": player_id,
                    "prop_type": prop_type,
                    "line": line,
                    "odds": prop.get("odds"),
                    "side": side or None,
                    "event_id": prop.get("event_id"),
                }

    # Seed game data from the first odds event when available
    if raw_odds:
        event = None
        events = raw_odds.get("events") if isinstance(raw_odds, dict) else None
        if isinstance(events, list) and events:
            event = events[0]

        if event:
            home_team = canonicalize_team_abbrev(event.get("home_team", "HOME"))
            away_team = canonicalize_team_abbrev(event.get("away_team", "AWAY"))
            game_time = event.get("commence_time")
        else:
            home_team = canonicalize_team_abbrev(raw_odds.get("home_team", "HOME"))
            away_team = canonicalize_team_abbrev(raw_odds.get("away_team", "AWAY"))
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
        "prop_features": [],
        "picks": [],
    }

    for name, rows in normalized.items():
        validate_table(name, rows)

    return normalized
