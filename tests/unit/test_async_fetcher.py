"""Unit tests for async data fetching functionality."""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from data.fetchers.async_fetcher import (
    AsyncDataFetcher,
    PlayerContext,
    BatchAnalysisResult,
    analyze_props_parallel,
    run_parallel_analysis,
)
from tests.mocks import MockNBADataFetcher, MockInjuryTracker


class TestPlayerContext:
    """Tests for PlayerContext dataclass."""

    def test_player_context_creation(self):
        """Test PlayerContext can be created with defaults."""
        context = PlayerContext(
            player_name="Test Player",
            game_logs=pd.DataFrame()
        )

        assert context.player_name == "Test Player"
        assert context.game_logs.empty
        assert context.opponent is None
        assert context.defense_rating is None
        assert context.pace_factor is None
        assert context.injury_status is None
        assert context.teammate_boost == 1.0
        assert context.is_b2b is False
        assert context.fetch_time_ms == 0

    def test_player_context_with_all_fields(self):
        """Test PlayerContext with all fields populated."""
        logs = pd.DataFrame({'points': [25, 30, 28]})
        context = PlayerContext(
            player_name="Luka Doncic",
            game_logs=logs,
            opponent="LAL",
            defense_rating={'pts_factor': 1.05},
            pace_factor=1.02,
            injury_status={'status': 'ACTIVE'},
            teammate_boost=1.10,
            is_b2b=True,
            fetch_time_ms=150.5
        )

        assert context.player_name == "Luka Doncic"
        assert len(context.game_logs) == 3
        assert context.opponent == "LAL"
        assert context.defense_rating['pts_factor'] == 1.05
        assert context.pace_factor == 1.02
        assert context.teammate_boost == 1.10
        assert context.is_b2b is True


class TestBatchAnalysisResult:
    """Tests for BatchAnalysisResult dataclass."""

    def test_batch_result_creation(self):
        """Test BatchAnalysisResult creation."""
        context = PlayerContext(player_name="Test", game_logs=pd.DataFrame())
        result = BatchAnalysisResult(
            contexts=[context],
            total_fetch_time_ms=100.0,
            players_fetched=1,
            errors=[]
        )

        assert len(result.contexts) == 1
        assert result.total_fetch_time_ms == 100.0
        assert result.players_fetched == 1
        assert len(result.errors) == 0

    def test_batch_result_with_errors(self):
        """Test BatchAnalysisResult with errors."""
        result = BatchAnalysisResult(
            contexts=[],
            total_fetch_time_ms=50.0,
            players_fetched=0,
            errors=["Player not found", "API timeout"]
        )

        assert len(result.errors) == 2
        assert "Player not found" in result.errors


class TestAsyncDataFetcher:
    """Tests for AsyncDataFetcher class."""

    def test_fetcher_initialization(self):
        """Test AsyncDataFetcher initialization."""
        fetcher = AsyncDataFetcher(odds_api_key="test_key")

        assert fetcher.odds_api_key == "test_key"
        assert fetcher._nba_fetcher is None
        assert fetcher._injury_tracker is None
        assert fetcher._defense_cache is None
        assert fetcher._pace_cache is None

    def test_fetcher_with_dependencies(self):
        """Test AsyncDataFetcher with injected dependencies."""
        mock_nba = MockNBADataFetcher()
        mock_injuries = MockInjuryTracker()

        fetcher = AsyncDataFetcher(
            odds_api_key="test_key",
            nba_fetcher=mock_nba,
            injury_tracker=mock_injuries
        )

        assert fetcher._nba_fetcher is mock_nba
        assert fetcher._injury_tracker is mock_injuries

    def test_cache_validity_initially_false(self):
        """Test cache is invalid when newly created."""
        fetcher = AsyncDataFetcher()
        assert fetcher._defense_cache_valid() is False

    def test_extract_team_from_empty_logs(self):
        """Test team extraction from empty logs."""
        fetcher = AsyncDataFetcher()
        team = fetcher._extract_team_from_logs(pd.DataFrame())
        assert team is None

    def test_extract_team_from_logs(self):
        """Test team extraction from logs with team_abbrev."""
        fetcher = AsyncDataFetcher()
        logs = pd.DataFrame({
            'team_abbrev': ['DAL', 'DAL', 'DAL'],
            'points': [25, 30, 28]
        })
        team = fetcher._extract_team_from_logs(logs)
        assert team == 'DAL'


class TestAsyncFetchPlayerContext:
    """Tests for async fetch_player_context method."""

    @pytest.mark.asyncio
    async def test_fetch_player_context_with_mock(self):
        """Test fetching player context with mocked dependencies."""
        mock_nba = MockNBADataFetcher()
        mock_injuries = MockInjuryTracker()

        fetcher = AsyncDataFetcher(
            nba_fetcher=mock_nba,
            injury_tracker=mock_injuries
        )

        import aiohttp
        async with aiohttp.ClientSession() as session:
            context = await fetcher.fetch_player_context(
                session,
                player_name="Test Player",
                opponent="LAL",
                last_n_games=15
            )

        assert context.player_name == "Test Player"
        assert not context.game_logs.empty
        assert context.fetch_time_ms > 0

    @pytest.mark.asyncio
    async def test_fetch_player_context_without_dependencies(self):
        """Test fetching player context without dependencies."""
        fetcher = AsyncDataFetcher()

        import aiohttp
        async with aiohttp.ClientSession() as session:
            context = await fetcher.fetch_player_context(
                session,
                player_name="Test Player",
                last_n_games=15
            )

        # Without dependencies, should return empty context
        assert context.player_name == "Test Player"
        assert context.game_logs.empty


class TestAsyncFetchMultiPlayerContext:
    """Tests for async fetch_multi_player_context method."""

    @pytest.mark.asyncio
    async def test_fetch_multi_player_context(self):
        """Test fetching context for multiple players."""
        mock_nba = MockNBADataFetcher()
        mock_injuries = MockInjuryTracker()

        fetcher = AsyncDataFetcher(
            nba_fetcher=mock_nba,
            injury_tracker=mock_injuries
        )

        result = await fetcher.fetch_multi_player_context(
            players=["Player 1", "Player 2", "Player 3"],
            opponents=["LAL", "BOS", "MIA"],
            last_n_games=10
        )

        assert isinstance(result, BatchAnalysisResult)
        assert len(result.contexts) == 3
        assert result.players_fetched == 3
        assert result.total_fetch_time_ms > 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_fetch_multi_player_uneven_opponents(self):
        """Test fetch with fewer opponents than players."""
        mock_nba = MockNBADataFetcher()

        fetcher = AsyncDataFetcher(nba_fetcher=mock_nba)

        result = await fetcher.fetch_multi_player_context(
            players=["Player 1", "Player 2", "Player 3"],
            opponents=["LAL"],  # Only one opponent
            last_n_games=10
        )

        # Should handle gracefully - pad with None
        assert len(result.contexts) == 3

    @pytest.mark.asyncio
    async def test_fetch_multi_player_no_opponents(self):
        """Test fetch without opponents."""
        mock_nba = MockNBADataFetcher()

        fetcher = AsyncDataFetcher(nba_fetcher=mock_nba)

        result = await fetcher.fetch_multi_player_context(
            players=["Player 1", "Player 2"],
            opponents=None,
            last_n_games=10
        )

        assert len(result.contexts) == 2


class TestParallelAnalysis:
    """Tests for parallel analysis functions."""

    @pytest.mark.asyncio
    async def test_analyze_props_parallel(self):
        """Test parallel prop analysis."""
        from models import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        props = [
            {'player': 'Player 1', 'prop_type': 'points', 'line': 25.0},
            {'player': 'Player 2', 'prop_type': 'rebounds', 'line': 8.0},
            {'player': 'Player 3', 'prop_type': 'assists', 'line': 6.0},
        ]

        results = await analyze_props_parallel(props, model, max_concurrent=3)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'projection')
            assert hasattr(result, 'edge')

    def test_run_parallel_analysis_sync(self):
        """Test synchronous wrapper for parallel analysis."""
        from models import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        props = [
            {'player': 'Player 1', 'prop_type': 'points', 'line': 25.0},
            {'player': 'Player 2', 'prop_type': 'rebounds', 'line': 8.0},
        ]

        results = run_parallel_analysis(props, model)

        assert len(results) == 2


class TestAsyncOddsFetching:
    """Tests for async odds fetching (requires mocking)."""

    @pytest.mark.asyncio
    async def test_fetch_odds_events_no_key(self):
        """Test fetch events without API key returns empty list."""
        fetcher = AsyncDataFetcher(odds_api_key=None)
        events = await fetcher.fetch_odds_events()
        assert events == []

    @pytest.mark.asyncio
    async def test_fetch_player_props_no_key(self):
        """Test fetch props without API key returns empty dict."""
        fetcher = AsyncDataFetcher(odds_api_key=None)
        props = await fetcher.fetch_player_props("event_123")
        assert props == {}

    @pytest.mark.asyncio
    async def test_fetch_all_props_parallel_no_key(self):
        """Test parallel props fetch without API key."""
        fetcher = AsyncDataFetcher(odds_api_key=None)
        results = await fetcher.fetch_all_props_parallel(
            event_ids=["event_1", "event_2"]
        )
        assert results == {}

    @pytest.mark.asyncio
    async def test_fetch_all_props_parallel_empty_events(self):
        """Test parallel props fetch with empty event list."""
        fetcher = AsyncDataFetcher(odds_api_key="test_key")
        results = await fetcher.fetch_all_props_parallel(event_ids=[])
        assert results == {}
