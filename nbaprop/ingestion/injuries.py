"""Injury ingestion."""

from typing import Dict
import logging
import time

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "injury_report"


def fetch_injury_report(cache: CacheStore, ttl_seconds: int = 300) -> Dict:
    """Fetch the latest injury report."""
    cache_key = "injury_report"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    report = {
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "entries": [],
    }
    cache.set(cache_key, report, ttl_seconds)
    logger.info("Fetched injury report (stub).")
    return report
