"""Injury ingestion."""

from typing import Dict, Optional, Callable, List
import logging
import os
import time

import requests

from nbaprop.ops import get_rate_limiter
from nbaprop.storage import CacheStore

logger = logging.getLogger(__name__)

SOURCE_NAME = "injury_report"
PERPLEXITY_SOURCE = "perplexity"
PERPLEXITY_MODEL = "sonar"


def _build_perplexity_client(
    api_key: str,
    rate_limiter,
) -> Callable[[List[Dict[str, str]]], str]:
    def _perplexity(messages: List[Dict[str, str]]) -> str:
        rate_limiter.wait(PERPLEXITY_SOURCE)
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": PERPLEXITY_MODEL,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=20,
            )
        except Exception as exc:
            logger.warning("Perplexity request failed: %s", exc)
            return ""

        if response.status_code != 200:
            logger.warning("Perplexity API error: %s", response.status_code)
            return ""

        try:
            payload = response.json()
        except ValueError:
            logger.warning("Perplexity API returned non-JSON response")
            return ""

        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content", "") or ""

    return _perplexity


def _normalize_entries(raw_entries: List[Dict]) -> List[Dict]:
    normalized = []
    seen = set()

    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        player = entry.get("player") or entry.get("PLAYER_NAME") or entry.get("Player")
        team = entry.get("team") or entry.get("TEAM_ABBREVIATION") or entry.get("Team")
        status = entry.get("status") or entry.get("INJURY_STATUS") or entry.get("Status")
        injury = entry.get("injury") or entry.get("INJURY_DESCRIPTION") or entry.get("Injury") or ""
        source = entry.get("source") or entry.get("Source") or "Unknown"

        if not player:
            continue

        player_text = str(player).strip()
        team_text = str(team).strip() if team else ""
        if team_text and len(team_text) <= 4:
            team_text = team_text.upper()
        status_text = str(status).strip().upper() if status else ""
        injury_text = str(injury).strip()
        source_text = str(source).strip() if source else "Unknown"

        key = (player_text.lower(), team_text, status_text)
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "player": player_text,
            "team": team_text,
            "status": status_text,
            "injury": injury_text,
            "source": source_text,
        })

    return normalized


def fetch_injury_report(
    cache: CacheStore,
    ttl_seconds: int = 300,
    perplexity_api_key: Optional[str] = None,
) -> Dict:
    """Fetch the latest injury report."""
    cache_key = "injury_report"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    if perplexity_api_key is None:
        api_key = ""
    else:
        api_key = perplexity_api_key or os.environ.get("PERPLEXITY_API_KEY", "")
    perplexity_fn = None
    if api_key:
        perplexity_fn = _build_perplexity_client(api_key, limiter)

    error = None
    debug_info = None
    try:
        from data.fetchers.injury_tracker import InjuryTracker
        tracker = InjuryTracker(perplexity_fn=perplexity_fn)
        injuries_df = tracker.get_all_injuries()
        raw_entries = injuries_df.to_dict(orient="records") if injuries_df is not None else []
        entries = _normalize_entries(raw_entries)
        if hasattr(tracker, "get_debug_info"):
            debug_info = tracker.get_debug_info()
        elif hasattr(tracker, "_debug_info"):
            debug_info = dict(getattr(tracker, "_debug_info"))
    except Exception as exc:
        error = str(exc)
        logger.warning("Injury tracker unavailable: %s", exc)
        entries = []

    source_counts: Dict[str, int] = {}
    for entry in entries:
        src = entry.get("source") or "Unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    report = {
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "entries": entries,
        "sources": sorted(source_counts.keys()),
        "source_counts": source_counts,
        "perplexity_enabled": bool(perplexity_fn),
    }
    if error:
        report["error"] = error
    if debug_info:
        report["injury_debug"] = debug_info

    cache.set(cache_key, report, ttl_seconds)
    logger.info("Fetched injury report with %d entries.", len(entries))
    return report
