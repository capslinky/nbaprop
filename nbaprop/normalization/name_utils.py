"""Name normalization and roster-based mapping helpers."""

from pathlib import Path
from typing import Dict
import json
import os
import re
import unicodedata

_ROSTER_MAP: Dict[str, str] = {}
_ROSTER_LOADED = False
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}


def normalize_name_for_matching(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = text.replace("-", " ")
    parts = [part for part in text.split() if part not in _SUFFIXES]
    return " ".join(parts)


def _rosters_path() -> Path:
    override = os.environ.get("NBAPROP_ROSTERS_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "rosters_2025_26.json"


def load_roster_map() -> Dict[str, str]:
    """Return mapping from normalized player name to team abbrev."""
    global _ROSTER_LOADED
    if _ROSTER_LOADED:
        return _ROSTER_MAP

    path = _rosters_path()
    if not path.exists():
        _ROSTER_LOADED = True
        return _ROSTER_MAP

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _ROSTER_LOADED = True
        return _ROSTER_MAP

    for team in payload.values():
        team_abbrev = team.get("team_abbrev")
        for player in team.get("players", []):
            name = player.get("name")
            key = normalize_name_for_matching(name)
            if key and team_abbrev:
                _ROSTER_MAP.setdefault(key, team_abbrev)

    _ROSTER_LOADED = True
    return _ROSTER_MAP
