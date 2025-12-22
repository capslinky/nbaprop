"""Configuration for the rebuild."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict
import json
import os


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _coerce_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = _strip_quotes(value.strip())
    return data


def _load_config_data(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in payload.items()}
    return _parse_env_file(path)


@dataclass
class Config:
    odds_api_key: str
    odds_max_events: int
    odds_max_players: int
    nba_api_delay: float
    cache_dir: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            odds_api_key=os.environ.get("ODDS_API_KEY", ""),
            odds_max_events=_coerce_int(os.environ.get("ODDS_MAX_EVENTS"), 5),
            odds_max_players=_coerce_int(os.environ.get("ODDS_MAX_PLAYERS"), 25),
            nba_api_delay=_coerce_float(os.environ.get("NBA_API_DELAY"), 1.5),
            cache_dir=os.environ.get("NBAPROP_CACHE_DIR", ".cache"),
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        env_config = cls.from_env()
        if not config_path:
            return env_config

        file_data = _load_config_data(Path(config_path))
        return cls(
            odds_api_key=file_data.get("ODDS_API_KEY", env_config.odds_api_key),
            odds_max_events=_coerce_int(
                file_data.get("ODDS_MAX_EVENTS"),
                env_config.odds_max_events,
            ),
            odds_max_players=_coerce_int(
                file_data.get("ODDS_MAX_PLAYERS"),
                env_config.odds_max_players,
            ),
            nba_api_delay=_coerce_float(
                file_data.get("NBA_API_DELAY"),
                env_config.nba_api_delay,
            ),
            cache_dir=file_data.get("NBAPROP_CACHE_DIR", env_config.cache_dir),
        )

    def to_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}
