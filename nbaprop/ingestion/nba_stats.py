"""NBA stats ingestion."""

from typing import List, Dict
import logging
import time

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "nba_stats"


def fetch_player_logs(players: List[str], cache: CacheStore, ttl_seconds: int = 300) -> List[Dict]:
    """Fetch raw player logs for a list of players."""
    cache_key = f"player_logs:{','.join(players)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    rows = []
    fetched_at = time.time()
    for player in players:
        rows.append({
            "player": player,
            "source": SOURCE_NAME,
            "fetched_at": fetched_at,
            "logs": [],
        })

    cache.set(cache_key, rows, ttl_seconds)
    logger.info("Fetched player logs (stub) for %d players.", len(players))
    return rows


def fetch_team_stats(cache: CacheStore, ttl_seconds: int = 600) -> List[Dict]:
    """Fetch raw team stats."""
    cache_key = "team_stats"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    rows = [{
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "teams": [],
    }]
    cache.set(cache_key, rows, ttl_seconds)
    logger.info("Fetched team stats (stub).")
    return rows
