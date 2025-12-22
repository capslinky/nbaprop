"""Configuration for the rebuild."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    odds_api_key: str
    nba_api_delay: float
    cache_dir: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            odds_api_key=os.environ.get("ODDS_API_KEY", ""),
            nba_api_delay=float(os.environ.get("NBA_API_DELAY", "1.5")),
            cache_dir=os.environ.get("NBAPROP_CACHE_DIR", ".cache"),
        )
