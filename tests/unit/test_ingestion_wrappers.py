"""Unit tests for ingestion wrappers with mocks."""

import pandas as pd

from nbaprop.storage import MemoryCache
from nbaprop.ingestion import odds as odds_ingestion
from nbaprop.ingestion import nba_stats as nba_ingestion
from nbaprop.ingestion import injuries as injury_ingestion


class StubLimiter:
    def wait(self, source, interval=None):
        return 0.0


def test_fetch_odds_snapshot_missing_key(monkeypatch):
    cache = MemoryCache()
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(odds_ingestion, "get_rate_limiter", lambda: StubLimiter())

    snapshot = odds_ingestion.fetch_odds_snapshot(cache, ttl_seconds=60, api_key="")

    assert snapshot["error"] == "missing_api_key"
    assert snapshot["events"] == []


def test_fetch_odds_snapshot_with_client_and_cache(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(odds_ingestion, "get_rate_limiter", lambda: StubLimiter())

    call_counter = {"init": 0}

    class StubClient:
        def __init__(self, api_key):
            call_counter["init"] += 1
            self.remaining_requests = "10"

        def get_events(self):
            return [
                {"id": "event-1", "home_team": "AAA", "away_team": "BBB"},
                {"id": "event-2", "home_team": "CCC", "away_team": "DDD"},
            ]

        def get_player_props(self, event_id, markets=None, bookmakers=None):
            return []

        def parse_player_props(self, props_data):
            return pd.DataFrame()

    import data.fetchers.odds_fetcher as odds_fetcher
    monkeypatch.setattr(odds_fetcher, "OddsAPIClient", StubClient)

    snapshot = odds_ingestion.fetch_odds_snapshot(
        cache,
        ttl_seconds=60,
        api_key="test",
        max_events=1,
    )
    assert len(snapshot["events"]) == 1
    assert snapshot["remaining_requests"] == "10"
    assert call_counter["init"] == 1

    snapshot_cached = odds_ingestion.fetch_odds_snapshot(
        cache,
        ttl_seconds=60,
        api_key="test",
        max_events=1,
    )
    assert snapshot_cached == snapshot
    assert call_counter["init"] == 1


def test_fetch_player_logs_with_fetcher_and_cache(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(nba_ingestion, "get_rate_limiter", lambda: StubLimiter())

    class StubFetcher:
        call_count = 0

        def __init__(self, cache_dir=None, base_delay=None):
            self.cache_dir = cache_dir
            self.base_delay = base_delay

        def get_player_game_logs(self, player, season=None):
            StubFetcher.call_count += 1
            return pd.DataFrame([{"points": 10}, {"points": 12}])

        def get_team_defense_ratings(self, season=None):
            return pd.DataFrame([{"team_abbrev": "AAA", "def_rating": 110}])

        def get_team_pace(self, season=None):
            return pd.DataFrame([{"team_abbrev": "AAA", "pace": 100}])

    import data.fetchers.nba_fetcher as nba_fetcher
    monkeypatch.setattr(nba_fetcher, "NBADataFetcher", StubFetcher)

    rows = nba_ingestion.fetch_player_logs(
        ["Player One"],
        cache,
        ttl_seconds=60,
        season="2024-25",
        base_delay=0.0,
        cache_dir=".cache",
    )
    assert rows[0]["player"] == "Player One"
    assert rows[0]["logs"][0]["points"] == 10
    assert StubFetcher.call_count == 1

    rows_cached = nba_ingestion.fetch_player_logs(
        ["Player One"],
        cache,
        ttl_seconds=60,
        season="2024-25",
        base_delay=0.0,
        cache_dir=".cache",
    )
    assert rows_cached == rows
    assert StubFetcher.call_count == 1


def test_fetch_team_stats_with_fetcher(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(nba_ingestion, "get_rate_limiter", lambda: StubLimiter())

    class StubFetcher:
        def __init__(self, cache_dir=None, base_delay=None):
            self.cache_dir = cache_dir
            self.base_delay = base_delay

        def get_team_defense_ratings(self, season=None):
            return pd.DataFrame([{"team_abbrev": "AAA", "def_rating": 110}])

        def get_team_pace(self, season=None):
            return pd.DataFrame([{"team_abbrev": "AAA", "pace": 100}])

    import data.fetchers.nba_fetcher as nba_fetcher
    monkeypatch.setattr(nba_fetcher, "NBADataFetcher", StubFetcher)

    rows = nba_ingestion.fetch_team_stats(
        cache,
        ttl_seconds=60,
        season="2024-25",
        base_delay=0.0,
        cache_dir=".cache",
    )

    assert rows[0]["defense_ratings"][0]["team_abbrev"] == "AAA"
    assert rows[0]["pace"][0]["team_abbrev"] == "AAA"


def test_fetch_injury_report_with_tracker(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(injury_ingestion, "get_rate_limiter", lambda: StubLimiter())

    class StubTracker:
        def __init__(self, perplexity_fn=None):
            self.perplexity_fn = perplexity_fn

        def get_all_injuries(self):
            return pd.DataFrame([{
                "player": "Test Player",
                "team": "AAA",
                "status": "OUT",
                "injury": "Ankle",
                "source": "NBA Official Report",
            }])

    import data.fetchers.injury_tracker as injury_tracker
    monkeypatch.setattr(injury_tracker, "InjuryTracker", StubTracker)

    report = injury_ingestion.fetch_injury_report(cache, ttl_seconds=60, perplexity_api_key="")
    assert report["entries"][0]["player"] == "Test Player"
    assert report["source_counts"]["NBA Official Report"] == 1

    report_cached = injury_ingestion.fetch_injury_report(cache, ttl_seconds=60, perplexity_api_key="")
    assert report_cached == report
