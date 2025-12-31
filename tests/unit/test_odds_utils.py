"""
Unit tests for nbaprop/utils/odds.py

Tests all betting math utilities including:
- Odds conversions (American to decimal, implied probability)
- Vig removal
- EV and edge calculations
- Kelly criterion
- Confidence scoring
- Probability calculations
"""

import pytest
import numpy as np
import pandas as pd
from nbaprop.utils.odds import (
    american_to_decimal,
    american_to_implied_prob,
    remove_vig,
    calculate_breakeven_winrate,
    calculate_ev,
    calculate_edge,
    kelly_criterion,
    calculate_confidence,
    calculate_prob_over,
    calculate_confidence_interval,
)


class TestAmericanToDecimal:
    """Tests for american_to_decimal conversion."""

    def test_negative_odds(self):
        """Standard favorite odds."""
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)
        assert american_to_decimal(-200) == pytest.approx(1.5, rel=0.01)
        assert american_to_decimal(-150) == pytest.approx(1.667, rel=0.01)

    def test_positive_odds(self):
        """Standard underdog odds."""
        assert american_to_decimal(110) == pytest.approx(2.1, rel=0.01)
        assert american_to_decimal(200) == pytest.approx(3.0, rel=0.01)
        assert american_to_decimal(150) == pytest.approx(2.5, rel=0.01)

    def test_even_odds(self):
        """Even money (+100)."""
        assert american_to_decimal(100) == pytest.approx(2.0, rel=0.01)


class TestAmericanToImpliedProb:
    """Tests for american_to_implied_prob conversion."""

    def test_negative_odds(self):
        """-110 implies ~52.4% win rate."""
        prob = american_to_implied_prob(-110)
        assert 0.52 < prob < 0.53

    def test_positive_odds(self):
        """+110 implies ~47.6% win rate."""
        prob = american_to_implied_prob(110)
        assert 0.47 < prob < 0.48

    def test_heavy_favorite(self):
        """-200 implies ~66.7% win rate."""
        prob = american_to_implied_prob(-200)
        assert 0.66 < prob < 0.68

    def test_heavy_underdog(self):
        """+200 implies ~33.3% win rate."""
        prob = american_to_implied_prob(200)
        assert 0.32 < prob < 0.35

    def test_even_odds(self):
        """+100 implies exactly 50%."""
        assert american_to_implied_prob(100) == pytest.approx(0.5, rel=0.01)


class TestRemoveVig:
    """Tests for vig removal from probability pairs."""

    def test_standard_vig(self):
        """Standard -110/-110 line has ~4.5% vig."""
        over_implied = american_to_implied_prob(-110)  # ~0.524
        under_implied = american_to_implied_prob(-110)  # ~0.524
        true_over, true_under = remove_vig(over_implied, under_implied)

        # After removing vig, should sum to 1.0
        assert true_over + true_under == pytest.approx(1.0, rel=0.01)
        # Each should be ~50%
        assert true_over == pytest.approx(0.5, rel=0.01)

    def test_skewed_line(self):
        """Line with different odds on each side."""
        over_implied = american_to_implied_prob(-120)  # ~0.545
        under_implied = american_to_implied_prob(100)   # ~0.500
        true_over, true_under = remove_vig(over_implied, under_implied)

        # Should sum to 1.0
        assert true_over + true_under == pytest.approx(1.0, rel=0.01)
        # Over should have higher probability
        assert true_over > true_under


class TestCalculateBreakevenWinrate:
    """Tests for breakeven win rate calculation."""

    def test_standard_odds(self):
        """-110 requires ~52.4% to break even."""
        breakeven = calculate_breakeven_winrate(-110)
        assert 0.52 < breakeven < 0.53

    def test_plus_odds(self):
        """+150 requires ~40% to break even."""
        breakeven = calculate_breakeven_winrate(150)
        assert 0.38 < breakeven < 0.42


class TestCalculateEV:
    """Tests for expected value calculation."""

    def test_positive_ev(self):
        """Projection above line with over bet should have positive EV."""
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        result = calculate_ev(
            projection=32.0,
            line=28.5,
            std=history.std()
        )
        # EV over should be positive when projection > line
        assert result['ev_over'] > 0

    def test_negative_ev(self):
        """Projection below line with over bet should have negative EV."""
        history = pd.Series([22, 24, 26, 23, 25, 27, 24, 25, 23, 26])
        result = calculate_ev(
            projection=25.0,
            line=28.5,
            std=history.std()
        )
        # EV over should be negative when projection < line
        assert result['ev_over'] < 0

    def test_zero_std(self):
        """Should handle zero standard deviation."""
        result = calculate_ev(
            projection=30.0,
            line=28.5,
            std=0.0
        )
        # With projection > line and std=0, should be certain to go over
        assert result['prob_over'] == 1.0
        assert result['ev_over'] > 0


class TestCalculateEdge:
    """Tests for edge calculation."""

    def test_positive_edge(self):
        """55% win rate at -110 odds has positive edge."""
        edge = calculate_edge(0.55, -110)
        assert edge > 0

    def test_negative_edge(self):
        """45% win rate at -110 odds has negative edge."""
        edge = calculate_edge(0.45, -110)
        assert edge < 0

    def test_breakeven(self):
        """52.4% win rate at -110 odds is approximately breakeven."""
        edge = calculate_edge(0.524, -110)
        assert abs(edge) < 0.02  # Within 2%


class TestKellyCriterion:
    """Tests for Kelly criterion bet sizing."""

    def test_positive_edge_bet(self):
        """Positive edge should suggest a bet."""
        kelly = kelly_criterion(prob_win=0.55, odds=-110)
        assert kelly > 0

    def test_negative_edge_no_bet(self):
        """Negative edge should suggest no bet."""
        kelly = kelly_criterion(prob_win=0.45, odds=-110)
        assert kelly == 0

    def test_fractional_kelly(self):
        """Fractional Kelly should reduce bet size proportionally."""
        full_kelly = kelly_criterion(prob_win=0.60, odds=-110, fraction=1.0, max_bet_percent=1.0)
        quarter_kelly = kelly_criterion(prob_win=0.60, odds=-110, fraction=0.25, max_bet_percent=1.0)
        # Quarter Kelly should be ~25% of full Kelly (both uncapped)
        assert quarter_kelly == pytest.approx(full_kelly * 0.25, rel=0.05)

    def test_max_bet_cap(self):
        """Max bet should cap the recommendation."""
        kelly = kelly_criterion(prob_win=0.80, odds=-110, max_bet_percent=0.05)
        assert kelly <= 0.05


class TestCalculateConfidence:
    """Tests for confidence score calculation."""

    def test_high_confidence(self):
        """Consistent stats with projection near average should have high confidence."""
        history = pd.Series([29, 30, 31, 30, 29, 30, 31, 29, 30, 30])  # Low variance
        confidence = calculate_confidence(
            history=history,
            projection=30.0,
            min_samples=5
        )
        assert confidence > 0.5

    def test_low_confidence_high_variance(self):
        """High variance stats should have lower confidence."""
        history = pd.Series([15, 45, 20, 40, 25, 35, 18, 42, 22, 38])  # High variance
        confidence = calculate_confidence(
            history=history,
            projection=30.0,
            min_samples=5
        )
        # Should be lower than low variance case
        assert confidence < 0.7

    def test_small_sample_penalty(self):
        """Small sample should have lower confidence."""
        large_history = pd.Series([29, 30, 31, 30, 29, 30, 31, 29, 30, 30, 29, 30, 31, 30, 29])
        small_history = pd.Series([29, 30, 31, 30, 29])

        large_sample = calculate_confidence(
            history=large_history,
            projection=30.0,
            min_samples=5
        )
        small_sample = calculate_confidence(
            history=small_history,
            projection=30.0,
            min_samples=5
        )
        assert large_sample > small_sample

    def test_bounds(self):
        """Confidence should be between 0 and 1."""
        history = pd.Series([29, 30, 31, 30, 29, 30, 31, 29, 30, 30])
        confidence = calculate_confidence(
            history=history,
            projection=30.0,
            min_samples=5
        )
        assert 0 <= confidence <= 1


class TestCalculateProbOver:
    """Tests for probability over line calculation."""

    def test_empirical_method(self):
        """Empirical method counts historical hits."""
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        prob = calculate_prob_over(history, line=29.5, method='empirical')
        # 7 out of 10 are over 29.5
        assert prob == pytest.approx(0.7, rel=0.05)

    def test_normal_method(self):
        """Normal distribution method."""
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        prob = calculate_prob_over(history, line=29.5, method='normal')
        # Should be reasonably high since mean > line
        assert prob > 0.5

    def test_line_affects_prob(self):
        """Lower line should increase probability of going over."""
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        low_line = calculate_prob_over(history, line=25.0, method='normal')
        high_line = calculate_prob_over(history, line=35.0, method='normal')
        assert low_line > high_line


class TestCalculateConfidenceInterval:
    """Tests for confidence interval calculation."""

    def test_interval_contains_mean(self):
        """95% CI should contain the mean."""
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        mean = history.mean()
        lower, upper = calculate_confidence_interval(history)
        assert lower <= mean <= upper

    def test_wider_interval_smaller_sample(self):
        """Smaller samples should have wider intervals."""
        large = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])
        small = pd.Series([30, 32, 28])

        large_lower, large_upper = calculate_confidence_interval(large)
        small_lower, small_upper = calculate_confidence_interval(small)

        large_width = large_upper - large_lower
        small_width = small_upper - small_lower

        assert small_width > large_width


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_edge_calculation(self):
        """Test complete edge calculation workflow."""
        # Player averaging 30 pts, line is 28.5 at -110
        history = pd.Series([30, 32, 28, 35, 31, 29, 33, 27, 30, 31])

        # Calculate probability of going over
        prob_over = calculate_prob_over(history, line=28.5, method='normal')

        # Calculate edge vs -110 odds
        edge = calculate_edge(prob_over, -110)

        # Should have positive edge since mean (30.6) > line (28.5)
        assert edge > 0

    def test_kelly_sizing_workflow(self):
        """Test complete bet sizing workflow."""
        # High confidence play
        prob_win = 0.58
        odds = -110

        # Calculate edge first
        edge = calculate_edge(prob_win, odds)
        assert edge > 0.05  # At least 5% edge

        # Get Kelly recommendation
        kelly = kelly_criterion(prob_win, odds, fraction=0.25)
        assert 0 < kelly < 0.1  # Should be a reasonable bet size
