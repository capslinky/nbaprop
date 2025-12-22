"""Schema validation for normalized tables."""

from typing import Dict, List

SCHEMA_VERSION = "v2"


REQUIRED_FIELDS: Dict[str, List[str]] = {
    "players": ["player_id", "player_name"],
    "teams": ["team_id", "team_abbrev"],
    "games": ["game_id", "home_team", "away_team", "game_time"],
    "props": ["prop_id", "player_id", "prop_type", "line"],
    "prop_features": ["prop_id", "features"],
    "picks": ["prop_id", "edge", "confidence", "pick"],
}


class SchemaValidationError(ValueError):
    pass


def validate_table(name: str, rows: list) -> None:
    """Validate a normalized table payload."""
    required = REQUIRED_FIELDS.get(name)
    if not required:
        return
    for idx, row in enumerate(rows):
        missing = [field for field in required if field not in row]
        if missing:
            raise SchemaValidationError(
                f"{name} row {idx} missing fields: {', '.join(missing)}"
            )
