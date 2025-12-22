"""Canonical ID helpers."""


def canonicalize_player_name(name: str) -> str:
    return " ".join(name.strip().split())


def canonicalize_team_abbrev(team: str) -> str:
    return team.strip().upper()


def make_prop_key(player_id: str, prop_type: str, line: float) -> str:
    return f"{player_id}:{prop_type}:{line}"
