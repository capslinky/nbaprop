"""NBA stats ingestion."""

from typing import List, Dict, Optional
import logging
import os
import time

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "nba_stats"


def fetch_player_logs(
    players: List[str],
    cache: CacheStore,
    ttl_seconds: int = 300,
    season: Optional[str] = None,
    base_delay: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Fetch raw player logs for a list of players."""
    cache_key = f"player_logs:{season or 'current'}:{','.join(players)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    cache_dir = cache_dir or os.environ.get("NBAPROP_CACHE_DIR")

    try:
        from data.fetchers.nba_fetcher import NBADataFetcher
        fetcher = NBADataFetcher(cache_dir=cache_dir, base_delay=base_delay)
    except Exception as exc:
        logger.warning("NBADataFetcher unavailable: %s", exc)
        rows = [{
            "player": player,
            "source": SOURCE_NAME,
            "fetched_at": time.time(),
            "season": season,
            "logs": [],
            "error": str(exc),
        } for player in players]
        cache.set(cache_key, rows, min(ttl_seconds, 30))
        return rows

    limiter = get_rate_limiter()
    rows = []
    fetched_at = time.time()

    for player in players:
        limiter.wait(SOURCE_NAME)
        logs = []
        error = None
        try:
            df = fetcher.get_player_game_logs(player, season=season)
            logs = df.to_dict(orient="records") if df is not None else []
        except Exception as exc:
            error = str(exc)
            logger.warning("Player log fetch failed for %s: %s", player, exc)

        row = {
            "player": player,
            "source": SOURCE_NAME,
            "fetched_at": fetched_at,
            "season": season,
            "logs": logs,
        }
        if error:
            row["error"] = error
        rows.append(row)

    cache.set(cache_key, rows, ttl_seconds)
    logger.info("Fetched player logs for %d players.", len(players))
    return rows


def fetch_team_stats(
    cache: CacheStore,
    ttl_seconds: int = 600,
    season: Optional[str] = None,
    base_delay: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Fetch raw team stats."""
    cache_key = f"team_stats:{season or 'current'}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    cache_dir = cache_dir or os.environ.get("NBAPROP_CACHE_DIR")

    try:
        from data.fetchers.nba_fetcher import NBADataFetcher
        fetcher = NBADataFetcher(cache_dir=cache_dir, base_delay=base_delay)
    except Exception as exc:
        logger.warning("NBADataFetcher unavailable: %s", exc)
        rows = [{
            "source": SOURCE_NAME,
            "fetched_at": time.time(),
            "season": season,
            "defense_ratings": [],
            "pace": [],
            "error": str(exc),
        }]
        cache.set(cache_key, rows, min(ttl_seconds, 30))
        return rows

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    defense_rows = []
    pace_rows = []
    error = None

    try:
        defense = fetcher.get_team_defense_ratings(season=season)
        defense_rows = defense.to_dict(orient="records") if defense is not None else []
    except Exception as exc:
        error = str(exc)
        logger.warning("Team defense fetch failed: %s", exc)

    try:
        pace = fetcher.get_team_pace(season=season)
        pace_rows = pace.to_dict(orient="records") if pace is not None else []
    except Exception as exc:
        error = str(exc)
        logger.warning("Team pace fetch failed: %s", exc)

    rows = [{
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "season": season,
        "defense_ratings": defense_rows,
        "pace": pace_rows,
    }]
    if error:
        rows[0]["error"] = error

    cache.set(cache_key, rows, ttl_seconds)
    logger.info("Fetched team stats with %d defense rows.", len(defense_rows))
    return rows
