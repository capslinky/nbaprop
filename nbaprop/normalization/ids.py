"""Canonical ID helpers."""

import hashlib


def canonicalize_player_name(name: str) -> str:
    return " ".join(name.strip().split())


def canonicalize_team_abbrev(team: str) -> str:
    return team.strip().upper()


def make_prop_key(player_id: str, prop_type: str, line: float) -> str:
    return f"{player_id}:{prop_type}:{line}"


def make_player_id(player_name: str) -> str:
    canonical = canonicalize_player_name(player_name).lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_team_id(team_abbrev: str) -> str:
    canonical = canonicalize_team_abbrev(team_abbrev).lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def make_game_id(home_team: str, away_team: str, game_time: str) -> str:
    payload = f"{home_team}:{away_team}:{game_time}".lower()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
