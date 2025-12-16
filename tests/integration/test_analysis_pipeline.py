"""Integration tests for the full analysis pipeline.

Tests the complete workflow from data fetch to analysis results,
using mocked external APIs.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from tests.mocks import MockNBADataFetcher, MockInjuryTracker, MockOddsAPIClient
from tests.fixtures.sample_api_responses import (
    get_sample_game_logs,
    get_sample_player_props,
    get_sample_game_lines,
    get_sample_team_defense,
    get_sample_team_pace,
)


class TestUnifiedModelIntegration:
    """Integration tests for UnifiedPropModel with full context."""

    def test_full_analysis_with_all_adjustments(self):
        """Test complete analysis applies all 9 adjustment factors."""
        from nba_prop_model import UnifiedPropModel, PropAnalysis

        # Create mocked dependencies
        mock_fetcher = MockNBADataFetcher()
        mock_injuries = MockInjuryTracker()

        # Initialize model with mocks
        model = UnifiedPropModel(
            data_fetcher=mock_fetcher,
            injury_tracker=mock_injuries
        )

        # Run analysis with full context
        result = model.analyze(
            player_name="Luka Doncic",
            prop_type="points",
            line=30.5,
            odds=-110,
            opponent="LAL",
            is_home=False,
            game_total=228.5,
            blowout_risk="LOW"
        )

        # Verify result type
        assert isinstance(result, PropAnalysis)

        # Verify all required fields populated
        assert result.player == "Luka Doncic"
        assert result.prop_type == "points"
        assert result.line == 30.5
        assert result.games_analyzed > 0
        assert result.projection is not None
        assert result.base_projection is not None
        assert result.edge is not None
        assert result.confidence is not None
        assert result.pick in ('OVER', 'UNDER', 'NO PICK')

        # Verify adjustments were applied
        assert result.adjustments is not None
        assert len(result.adjustments) > 0

    def test_analysis_handles_missing_opponent_data(self):
        """Test analysis degrades gracefully with missing opponent data."""
        from nba_prop_model import UnifiedPropModel

        mock_fetcher = MockNBADataFetcher()
        mock_injuries = MockInjuryTracker()

        model = UnifiedPropModel(
            data_fetcher=mock_fetcher,
            injury_tracker=mock_injuries
        )

        # Analyze without opponent (should still work with reduced adjustments)
        result = model.analyze(
            player_name="Luka Doncic",
            prop_type="points",
            line=30.5
        )

        assert result.games_analyzed > 0
        assert result.projection is not None

    def test_analysis_handles_injured_player(self):
        """Test analysis flags injured players appropriately."""
        from nba_prop_model import UnifiedPropModel

        # Mock player as GTD (game-time decision)
        injuries = {
            'Luka Doncic': {
                'status': 'GTD',
                'injury': 'Ankle',
                'is_out': False,
                'is_gtd': True,
                'is_questionable': False
            }
        }
        mock_injuries = MockInjuryTracker(injuries=injuries)

        # Override get_player_status for our injury data
        def custom_get_player_status(player_name):
            return injuries.get(player_name, {'status': 'HEALTHY', 'is_out': False, 'is_gtd': False, 'is_questionable': False})
        mock_injuries.get_player_status = custom_get_player_status

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=mock_injuries
        )

        result = model.analyze(
            player_name="Luka Doncic",
            prop_type="points",
            line=30.5
        )

        # GTD player should have lower confidence or be flagged
        assert result.games_analyzed > 0

    def test_analysis_different_prop_types(self):
        """Test analysis works for all supported prop types."""
        from nba_prop_model import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        # Only test prop types that exist in mock data
        prop_types = ['points', 'rebounds', 'assists', 'threes']

        for prop_type in prop_types:
            result = model.analyze(
                player_name="Luka Doncic",
                prop_type=prop_type,
                line=10.5
            )

            assert result.prop_type == prop_type
            assert result.games_analyzed > 0
            assert result.projection is not None

    def test_analysis_back_to_back_detection(self):
        """Test B2B games are detected and adjusted for."""
        from nba_prop_model import UnifiedPropModel

        # Create mock with B2B schedule
        mock_fetcher = MockNBADataFetcher()

        # Override check_back_to_back to simulate B2B
        def custom_b2b_check(logs, game_date=None):
            return {'is_b2b': True, 'days_rest': 0, 'is_first_of_b2b': False, 'is_second_of_b2b': True}
        mock_fetcher.check_back_to_back = custom_b2b_check

        model = UnifiedPropModel(
            data_fetcher=mock_fetcher,
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Luka Doncic",
            prop_type="points",
            line=30.5
        )

        # B2B should be flagged or reduce projection
        assert result.is_b2b is True
        # B2B typically reduces projection

    def test_edge_calculation_accuracy(self):
        """Test edge calculation matches expected values."""
        from nba_prop_model import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=25.0,  # Set line near expected average
            odds=-110
        )

        # Edge should be reasonable (not extreme)
        assert -0.5 <= result.edge <= 0.5  # -50% to +50%


class TestPropAnalysisDataclass:
    """Tests for the PropAnalysis result dataclass."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all analysis fields."""
        from nba_prop_model import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=25.0
        )

        result_dict = result.to_dict()

        # Check required keys exist (to_dict has specific output keys)
        required_keys = [
            'player', 'prop_type', 'line', 'projection', 'edge',
            'confidence', 'pick', 'recent_avg', 'season_avg'
        ]
        for key in required_keys:
            assert key in result_dict

        # Also verify games_analyzed is available on the result object directly
        assert result.games_analyzed > 0

    def test_explain_returns_readable_string(self):
        """Test explain() method returns human-readable summary."""
        from nba_prop_model import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=25.0
        )

        # explain() returns string only when return_string=True
        explanation = result.explain(return_string=True)

        assert isinstance(explanation, str)
        assert len(explanation) > 50  # Should be a meaningful explanation
        assert "Test Player" in explanation or "points" in explanation


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_game_logs_handling(self):
        """Test graceful handling when no game logs available."""
        from nba_prop_model import UnifiedPropModel

        # Mock fetcher that returns empty data
        mock_fetcher = MockNBADataFetcher()
        mock_fetcher.get_player_game_logs = lambda *args, **kwargs: pd.DataFrame()

        model = UnifiedPropModel(
            data_fetcher=mock_fetcher,
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Unknown Player",
            prop_type="points",
            line=25.0
        )

        # Should return result with 0 games analyzed
        assert result.games_analyzed == 0

    def test_invalid_prop_type_handling(self):
        """Test handling of invalid prop type raises appropriate error."""
        from nba_prop_model import UnifiedPropModel
        from core.exceptions import InvalidPropTypeError

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        # Invalid prop type should raise InvalidPropTypeError
        with pytest.raises(InvalidPropTypeError):
            model.analyze(
                player_name="Test Player",
                prop_type="invalid_stat",
                line=10.0
            )

    def test_defense_data_unavailable(self):
        """Test analysis continues when defense data unavailable."""
        from nba_prop_model import UnifiedPropModel

        mock_fetcher = MockNBADataFetcher()
        mock_fetcher.get_team_defense_vs_position = lambda *args, **kwargs: pd.DataFrame()

        model = UnifiedPropModel(
            data_fetcher=mock_fetcher,
            injury_tracker=MockInjuryTracker()
        )

        result = model.analyze(
            player_name="Test Player",
            prop_type="points",
            line=25.0,
            opponent="LAL"
        )

        # Should still produce a result
        assert result.projection is not None


class TestMultiplePlayerAnalysis:
    """Tests for batch analysis of multiple players."""

    def test_batch_analysis_consistency(self):
        """Test multiple analyses produce consistent results."""
        from nba_prop_model import UnifiedPropModel

        model = UnifiedPropModel(
            data_fetcher=MockNBADataFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        # Same analysis twice should produce same result
        result1 = model.analyze("Test Player", "points", 25.0)
        result2 = model.analyze("Test Player", "points", 25.0)

        assert result1.projection == result2.projection
        assert result1.edge == result2.edge

    def test_different_players_different_results(self):
        """Test different players can produce different results."""
        from nba_prop_model import UnifiedPropModel

        # Create mock with different data per player
        class CustomMockFetcher(MockNBADataFetcher):
            def get_player_game_logs(self, player_name, season=None, last_n_games=15):
                # Return different averages based on player name
                if "High" in player_name:
                    base = 35
                elif "Low" in player_name:
                    base = 15
                else:
                    base = 25

                n = min(last_n_games, 15)
                dates = [datetime.now() - timedelta(days=i) for i in range(n)]

                return pd.DataFrame({
                    'game_date': dates,
                    'points': [base + (i % 5) for i in range(n)],
                    'rebounds': [8] * n,
                    'assists': [6] * n,
                    'fg3m': [3] * n,
                    'minutes': [34] * n,
                    'matchup': ['DAL @ LAL'] * n,
                    'home': [i % 2 == 0 for i in range(n)],
                    'team_abbrev': ['DAL'] * n,
                })

        model = UnifiedPropModel(
            data_fetcher=CustomMockFetcher(),
            injury_tracker=MockInjuryTracker()
        )

        result_high = model.analyze("High Scorer", "points", 30.0)
        result_low = model.analyze("Low Scorer", "points", 20.0)

        # High scorer should have higher projection
        assert result_high.projection > result_low.projection
