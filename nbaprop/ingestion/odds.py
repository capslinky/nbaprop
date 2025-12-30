"""Odds ingestion."""

from typing import Dict, Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo
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
    include_props: bool = True,
    target_date: Optional[str] = None,
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
) -> Dict:
    """Fetch a snapshot of current odds and props."""
    markets_key = ",".join(markets) if markets else "default"
    books_key = ",".join(bookmakers) if bookmakers else "all_books"
    cache_key = f"odds_snapshot:{max_events}:{target_date or 'all'}:{markets_key}:{books_key}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    snapshot = {
        "source": SOURCE_NAME,
        "fetched_at": time.time(),
        "events": [],
        "props": [],
        "players": [],
    }

    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        snapshot["error"] = "missing_api_key"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Odds API key missing; returning empty snapshot.")
        return snapshot

    try:
        from data.fetchers.odds_fetcher import OddsAPIClient
    except ImportError as exc:
        snapshot["error"] = f"odds_client_unavailable:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Odds API client unavailable: %s", exc)
        return snapshot

    limiter = get_rate_limiter()
    limiter.wait(SOURCE_NAME)

    try:
        client = OddsAPIClient(api_key=api_key)
        events = client.get_events()
        if isinstance(events, list) and target_date:
            events = _filter_events_by_date(events, target_date)
        if max_events and max_events > 0 and isinstance(events, list):
            events = events[:max_events]
        snapshot["events"] = events or []
        snapshot["remaining_requests"] = client.remaining_requests

        if include_props and snapshot["events"]:
            props_rows: List[Dict] = []
            player_set = set()
            for event in snapshot["events"]:
                event_id = event.get("id")
                if not event_id:
                    continue
                limiter.wait(SOURCE_NAME)
                try:
                    props_data = client.get_player_props(
                        event_id,
                        markets=markets,
                        bookmakers=bookmakers,
                    )
                    props_df = client.parse_player_props(props_data)
                    if props_df is None or props_df.empty:
                        continue
                    for row in props_df.to_dict(orient="records"):
                        player = row.get("player")
                        if player:
                            player_set.add(player)
                        props_rows.append({
                            "player": player,
                            "prop_type": row.get("prop_type"),
                            "line": row.get("line"),
                            "odds": row.get("odds"),
                            "side": row.get("side"),
                            "event_id": row.get("game_id"),
                            "home_team": row.get("home_team"),
                            "away_team": row.get("away_team"),
                            "game_time": row.get("commence_time"),
                        })
                except (ConnectionError, TimeoutError) as exc:
                    logger.warning("Network error fetching props for event %s: %s", event_id, exc)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Data error parsing props for event %s: %s", event_id, exc)
                except Exception as exc:
                    logger.warning("Unexpected error fetching props for event %s: %s", event_id, exc)
            snapshot["props"] = props_rows
            snapshot["players"] = sorted(player_set)

        cache.set(cache_key, snapshot, ttl_seconds)
        logger.info(
            "Fetched odds snapshot with %d events and %d players.",
            len(snapshot["events"]),
            len(snapshot["players"]),
        )
        return snapshot
    except (ConnectionError, TimeoutError) as exc:
        snapshot["error"] = f"network_error:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Network error fetching odds snapshot: %s", exc)
        return snapshot
    except (KeyError, TypeError, ValueError) as exc:
        snapshot["error"] = f"data_error:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Data error in odds snapshot: %s", exc)
        return snapshot
    except Exception as exc:
        snapshot["error"] = f"fetch_failed:{exc}"
        cache.set(cache_key, snapshot, min(ttl_seconds, 30))
        logger.warning("Unexpected error fetching odds snapshot: %s", exc)
        return snapshot


def _parse_event_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filter_events_by_date(events: List[Dict], target_date: str) -> List[Dict]:
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return events

    tz = ZoneInfo("America/New_York")
    filtered: List[Dict] = []
    for event in events:
        event_time = _parse_event_time(event.get("commence_time"))
        if not event_time:
            continue
        event_date = event_time.astimezone(tz).date()
        if event_date == target:
            filtered.append(event)
    return filtered
