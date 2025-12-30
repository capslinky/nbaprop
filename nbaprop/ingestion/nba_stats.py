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
    cache_only: bool = False,
) -> List[Dict]:
    """Fetch raw player logs for a list of players."""
    if not players:
        return []

    def _player_cache_key(player_name: str) -> str:
        return f"player_logs:{season or 'current'}:{player_name}"

    cached_rows: Dict[str, Dict] = {}
    missing: List[str] = []
    for player in players:
        cached = cache.get(_player_cache_key(player))
        if cached is not None:
            cached_rows[player] = cached
        else:
            missing.append(player)

    if not missing:
        return [cached_rows[player] for player in players]

    if cache_only:
        rows = []
        fetched_at = time.time()
        for player in missing:
            row = {
                "player": player,
                "source": SOURCE_NAME,
                "fetched_at": fetched_at,
                "season": season,
                "logs": [],
                "error": "cache_only",
            }
            rows.append(row)
            cache.set(_player_cache_key(player), row, min(ttl_seconds, 30))
        for row in rows:
            cached_rows[row["player"]] = row
        logger.warning("Cache-only mode: missing %d player logs.", len(missing))
        return [cached_rows[player] for player in players]

    cache_dir = cache_dir or os.environ.get("NBAPROP_CACHE_DIR")

    try:
        from data.fetchers.nba_fetcher import NBADataFetcher
        fetcher = NBADataFetcher(cache_dir=cache_dir, base_delay=base_delay)
    except ImportError as exc:
        logger.warning("NBADataFetcher unavailable: %s", exc)
        rows = []
        for player in players:
            row = {
                "player": player,
                "source": SOURCE_NAME,
                "fetched_at": time.time(),
                "season": season,
                "logs": [],
                "error": str(exc),
            }
            cache.set(_player_cache_key(player), row, min(ttl_seconds, 30))
            rows.append(row)
        return rows

    limiter = get_rate_limiter()
    rows = []
    fetched_at = time.time()

    for player in missing:
        limiter.wait(SOURCE_NAME)
        logs = []
        error = None
        try:
            df = fetcher.get_player_game_logs(player, season=season)
            logs = df.to_dict(orient="records") if df is not None else []
        except (ConnectionError, TimeoutError) as exc:
            error = f"network:{exc}"
            logger.warning("Network error fetching player logs for %s: %s", player, exc)
        except (KeyError, TypeError, ValueError) as exc:
            error = f"data:{exc}"
            logger.warning("Data error fetching player logs for %s: %s", player, exc)
        except Exception as exc:
            error = str(exc)
            logger.warning("Unexpected error fetching player logs for %s: %s", player, exc)

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
        cache.set(_player_cache_key(player), row, ttl_seconds)

    for row in rows:
        cached_rows[row["player"]] = row
    logger.info("Fetched player logs for %d players.", len(missing))
    return [cached_rows[player] for player in players]


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
    except ImportError as exc:
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
    except (ConnectionError, TimeoutError) as exc:
        error = f"network:{exc}"
        logger.warning("Network error fetching team defense: %s", exc)
    except (KeyError, TypeError, ValueError) as exc:
        error = f"data:{exc}"
        logger.warning("Data error fetching team defense: %s", exc)
    except Exception as exc:
        error = str(exc)
        logger.warning("Unexpected error fetching team defense: %s", exc)

    try:
        pace = fetcher.get_team_pace(season=season)
        pace_rows = pace.to_dict(orient="records") if pace is not None else []
    except (ConnectionError, TimeoutError) as exc:
        error = f"network:{exc}"
        logger.warning("Network error fetching team pace: %s", exc)
    except (KeyError, TypeError, ValueError) as exc:
        error = f"data:{exc}"
        logger.warning("Data error fetching team pace: %s", exc)
    except Exception as exc:
        error = str(exc)
        logger.warning("Unexpected error fetching team pace: %s", exc)

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
