"""Normalize raw ingestion data into canonical tables."""

from typing import Dict, List, Tuple
import time

from nbaprop.normalization.ids import (
    make_player_id,
    make_team_id,
    make_game_id,
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

    # Stub game based on odds snapshot to seed tables
    if raw_odds:
        home_team = canonicalize_team_abbrev(raw_odds.get("home_team", "HOME"))
        away_team = canonicalize_team_abbrev(raw_odds.get("away_team", "AWAY"))
        game_time = raw_odds.get("game_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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
