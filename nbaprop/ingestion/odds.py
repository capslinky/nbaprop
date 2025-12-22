"""Odds ingestion."""

from typing import Dict
import logging
import time

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "odds_api"


def fetch_odds_snapshot(cache: CacheStore, ttl_seconds: int = 60) -> Dict:
    """Fetch a snapshot of current odds and props."""
    cache_key = "odds_snapshot"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    snapshot = {
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "events": [],
        "props": [],
    }
    cache.set(cache_key, snapshot, ttl_seconds)
    logger.info("Fetched odds snapshot (stub).")
    return snapshot
