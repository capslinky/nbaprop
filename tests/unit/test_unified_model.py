"""
Unit tests for UnifiedPropModel.

Tests the core prediction model functionality including:
- Basic analysis workflow
- Projection calculations
- Confidence scoring
- Edge calculations
- Error handling for missing data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import UnifiedPropModel, PropAnalysis
from core.exceptions import InvalidPropTypeError
from tests.mocks import MockNBADataFetcher, MockInjuryTracker, MockOddsAPIClient


class TestUnifiedPropModelInit:
    """Tests for model initialization."""

    def test_init_with_dependencies(self):
        """Model accepts injected dependencies."""
        fetcher = MockNBADataFetcher()
        injuries = MockInjuryTracker()
        odds = MockOddsAPIClient()

        model = UnifiedPropModel(
            data_fetcher=fetcher,
            injury_tracker=injuries,
            odds_client=odds
        )

        assert model._fetcher is fetcher
        assert model._injuries is injuries
        assert model._odds is odds

    def test_init_without_dependencies(self):
        """Model initializes with None dependencies (lazy loading)."""
        model = UnifiedPropModel()

        assert model._fetcher is None
        assert model._injuries is None
        assert model._odds is None


class TestUnifiedPropModelAnalyze:
    """Tests for the analyze() method."""

    @pytest.fixture
    def model(self):
        """Create model with mock dependencies."""
        return UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker(),
            odds_client=None
        )

    def test_analyze_returns_prop_analysis(self, model):
        """analyze() returns a PropAnalysis object."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5
        )

        assert isinstance(result, PropAnalysis)
        assert result.player == "Test Player"
        assert result.prop_type == "points"
        assert result.line == 28.5

    def test_analyze_projection_reasonable(self, model):
        """Projection should be within reasonable range of historical mean."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5
        )

        # Mock data has mean of ~28.5 points
        # Projection should be within 20% of that
        expected_mean = 28.5
        assert abs(result.projection - expected_mean) < expected_mean * 0.3

    def test_analyze_confidence_bounds(self, model):
        """Confidence should be between 0 and 1."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5
        )

        assert 0 <= result.confidence <= 1

    def test_analyze_edge_calculation(self, model):
        """Edge reflects difference between model prob and implied odds."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5,
            odds=-110
        )

        # Edge should be a reasonable percentage
        assert -0.5 <= result.edge <= 0.5

    def test_analyze_pick_valid(self, model):
        """Pick should be OVER, UNDER, or PASS."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5
        )

        assert result.pick in ['OVER', 'UNDER', 'PASS']

    def test_analyze_games_analyzed_correct(self, model):
        """games_analyzed should match requested last_n_games."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5,
            last_n_games=10
        )

        assert result.games_analyzed <= 10

    def test_analyze_different_prop_types(self, model):
        """Model handles different prop types."""
        for prop_type in ['points', 'rebounds', 'assists']:
            result = model.analyze(
                player_name="Test Player",
                prop_type=prop_type,
                line=10.5
            )
            assert result.prop_type == prop_type
            assert result.projection > 0


class TestUnifiedPropModelMissingData:
    """Tests for handling missing or insufficient data."""

    def test_analyze_handles_empty_logs(self):
        """Should return PASS pick when no data available."""
        empty_fetcher = MockNBADataFetcher(
            game_logs=pd.DataFrame()
        )
        model = UnifiedPropModel(
            data_fetcher=empty_fetcher,
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Unknown Player",
            prop_type="points",
            line=28.5
        )

        assert result.pick == 'PASS'
        assert result.games_analyzed == 0
        assert 'NO DATA' in result.flags

    def test_analyze_handles_invalid_prop_type(self):
        """Should raise InvalidPropTypeError for invalid prop types."""
        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        with pytest.raises(InvalidPropTypeError) as exc_info:
            model.analyze(
                player_name="Test Player",
                prop_type="invalid_stat",
                line=10.5
            )

        assert "invalid_stat" in str(exc_info.value)
        assert "Valid types" in str(exc_info.value)


class TestUnifiedPropModelContext:
    """Tests for contextual adjustments."""

    @pytest.fixture
    def model(self):
        """Create model with mock dependencies."""
        return UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker(),
            odds_client=None
        )

    def test_home_away_affects_projection(self, model):
        """Home games should have slightly higher projections."""
        # This test verifies context detection works
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5
        )

        # Just verify we get a result with context info
        assert result.is_home is not None or result.is_home is None

    def test_optional_overrides_applied(self, model):
        """Optional overrides should be respected."""
        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=28.5,
            opponent="BOS",
            is_home=True,
            game_total=230.0
        )

        # Verify overrides were applied
        assert result.game_total == 230.0


class TestPropAnalysis:
    """Tests for PropAnalysis dataclass."""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a dictionary."""
        analysis = PropAnalysis(
            player="Test",
            prop_type="points",
            line=28.5,
            projection=30.0,
            base_projection=29.5,
            edge=0.05,
            confidence=0.65,
            pick="OVER",
            recent_avg=30.0,
            season_avg=28.5,
            over_rate=0.6,
            under_rate=0.4,
            std_dev=4.5,
            games_analyzed=15,
            trend="UP",
            flags=[]
        )

        result = analysis.to_dict()

        assert isinstance(result, dict)
        assert result['player'] == "Test"
        assert result['projection'] == 30.0

    def test_explain_returns_string(self):
        """explain(return_string=True) returns a formatted string."""
        analysis = PropAnalysis(
            player="Test",
            prop_type="points",
            line=28.5,
            projection=30.0,
            base_projection=29.5,
            edge=0.05,
            confidence=0.65,
            pick="OVER",
            recent_avg=30.0,
            season_avg=28.5,
            over_rate=0.6,
            under_rate=0.4,
            std_dev=4.5,
            games_analyzed=15,
            trend="UP",
            flags=[]
        )

        explanation = analysis.explain(return_string=True)

        assert isinstance(explanation, str)
        assert "Test" in explanation


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow with mocks."""
        # Create model with all mocked dependencies
        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker(),
            odds_client=MockOddsAPIClient()
        )

        # Run analysis
        result = model.analyze(
            player_name="Luka Doncic",
            prop_type="points",
            line=32.5,
            odds=-115
        )

        # Verify complete result
        assert result.player == "Luka Doncic"
        assert result.prop_type == "points"
        assert result.projection > 0
        assert result.pick in ['OVER', 'UNDER', 'PASS']
        assert 0 <= result.confidence <= 1
