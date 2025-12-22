"""Odds ingestion."""

from typing import Dict, Optional
import logging
import os
import time

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "odds_api"


def fetch_odds_snapshot(
    cache: CacheStore,
    ttl_seconds: int = 60,
    api_key: Optional[str] = None,
    max_events: int = 5,
) -> Dict:
    """Fetch a snapshot of current odds and props."""
    cache_key = f"odds_snapshot:{max_events}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    snapshot = {
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "events": [],
        "props": [],
    }

    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        snapshot["error"] = "missing_api_key"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Odds API key missing; returning empty snapshot.")
        return snapshot

    try:
        from data.fetchers.odds_fetcher import OddsAPIClient
    except Exception as exc:
        snapshot["error"] = f"odds_client_unavailable:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Odds API client unavailable: %s", exc)
        return snapshot

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    try:
        client = OddsAPIClient(api_key=api_key)
        events = client.get_events()
        if max_events and isinstance(events, list):
            events = events[:max_events]
        snapshot["events"] = events or []
        snapshot["remaining_requests"] = client.remaining_requests
        cache.set(cache_key, snapshot, ttl_seconds)
        logger.info("Fetched odds snapshot with %d events.", len(snapshot["events"]))
        return snapshot
    except Exception as exc:
        snapshot["error"] = f"fetch_failed:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Odds snapshot fetch failed: %s", exc)
        return snapshot
