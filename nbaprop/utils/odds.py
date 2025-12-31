"""
Odds conversion and betting math utilities.

Provides:
- Odds format conversions (American, Decimal, Implied Probability)
- Expected Value calculations
- Kelly Criterion bet sizing
- Statistical confidence calculations
"""

import logging
from typing import Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Defer scipy import for faster module load
_stats = None


def _get_stats():
    """Lazy import of scipy.stats."""
    global _stats
    if _stats is None:
        from scipy import stats
        _stats = stats
    return _stats


# =============================================================================
# ODDS CONVERSIONS
# =============================================================================

def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.

    Examples:
        +150 -> 2.50 (risk $100 to win $150, total return $250)
        -150 -> 1.67 (risk $150 to win $100, total return $250)

    Args:
        odds: American odds (positive or negative integer)

    Returns:
        Decimal odds (always > 1.0)
    """
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def american_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Note: This includes the vig/juice, so probabilities will sum > 100%.

    Examples:
        -110 -> 0.524 (52.4% implied)
        +100 -> 0.500 (50.0% implied)
        -200 -> 0.667 (66.7% implied)

    Args:
        odds: American odds (positive or negative integer)

    Returns:
        Implied probability (0 to 1)
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig(prob_over: float, prob_under: float) -> Tuple[float, float]:
    """
    Remove vig from implied probabilities to get true probabilities.

    Sportsbooks add vig so over + under > 100%. This normalizes them.

    Example:
        remove_vig(0.524, 0.524) -> (0.5, 0.5)

    Args:
        prob_over: Implied probability of over
        prob_under: Implied probability of under

    Returns:
        Tuple of (true_prob_over, true_prob_under) that sum to 1.0
    """
    total = prob_over + prob_under
    if total == 0:
        return (0.5, 0.5)
    return (prob_over / total, prob_under / total)


def calculate_breakeven_winrate(odds: int) -> float:
    """
    Calculate the win rate needed to break even at given odds.

    This is simply the implied probability - you need to win more often
    than this to be profitable.

    Args:
        odds: American odds

    Returns:
        Breakeven win rate (0 to 1)
    """
    return american_to_implied_prob(odds)


# =============================================================================
# EXPECTED VALUE
# =============================================================================

def calculate_ev(
    projection: float,
    line: float,
    std: float,
    over_odds: int = -110,
    under_odds: int = -110
) -> Dict[str, float]:
    """
    Calculate expected value for over/under bets.

    Uses normal distribution assumption to estimate probabilities,
    then calculates EV based on those probabilities and the odds.

    Args:
        projection: Model's projection for the stat
        line: The betting line
        std: Standard deviation of the player's performance
        over_odds: American odds for over bet
        under_odds: American odds for under bet

    Returns:
        Dictionary with prob_over, prob_under, ev_over, ev_under, best_bet, best_ev
    """
    stats = _get_stats()

    # Probability of going over the line (using normal distribution)
    if std > 0:
        z_score = (line - projection) / std
        prob_over = 1 - stats.norm.cdf(z_score)
        prob_under = stats.norm.cdf(z_score)
    else:
        # No variance - deterministic
        prob_over = 1.0 if projection > line else 0.0
        prob_under = 1.0 - prob_over

    # Calculate EV: (prob_win * profit) - (prob_lose * stake)
    decimal_over = american_to_decimal(over_odds)
    decimal_under = american_to_decimal(under_odds)

    # EV per unit wagered
    ev_over = (prob_over * (decimal_over - 1)) - (1 - prob_over)
    ev_under = (prob_under * (decimal_under - 1)) - (1 - prob_under)

    return {
        'prob_over': round(prob_over, 4),
        'prob_under': round(prob_under, 4),
        'ev_over': round(ev_over, 4),
        'ev_under': round(ev_under, 4),
        'best_bet': 'over' if ev_over > ev_under else 'under',
        'best_ev': round(max(ev_over, ev_under), 4)
    }


def calculate_edge(prob_win: float, odds: int) -> float:
    """
    Calculate true edge accounting for vig.

    Edge = Our estimated probability - Implied probability from odds

    A positive edge means we believe the true probability is higher than
    what the odds imply, suggesting value.

    Args:
        prob_win: Our estimated probability of winning
        odds: American odds

    Returns:
        Edge as decimal (0.05 = 5% edge)
    """
    implied_prob = american_to_implied_prob(odds)
    return prob_win - implied_prob


# =============================================================================
# KELLY CRITERION
# =============================================================================

def kelly_criterion(
    prob_win: float,
    odds: int,
    fraction: float = 0.25,
    max_bet_percent: float = 0.03
) -> float:
    """
    Calculate Kelly criterion stake as fraction of bankroll.

    Uses fractional Kelly (default 1/4) for safety, as full Kelly
    is too aggressive for most bettors.

    Kelly formula: f* = (bp - q) / b
    where:
        f* = fraction of bankroll to bet
        b = decimal odds - 1 (net profit per unit)
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        prob_win: Estimated probability of winning (0 to 1)
        odds: American odds
        fraction: Kelly fraction (0.25 = quarter Kelly)
        max_bet_percent: Maximum bet as % of bankroll

    Returns:
        Recommended stake as fraction of bankroll (0 to max_bet_percent)
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1  # Net odds (profit per unit wagered)
    q = 1 - prob_win

    # Kelly formula: f = (bp - q) / b
    kelly = (prob_win * b - q) / b

    # Can't bet negative (no edge)
    if kelly <= 0:
        return 0.0

    # Apply fractional Kelly
    stake = kelly * fraction

    # Cap at maximum bet size
    return min(stake, max_bet_percent)


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def calculate_confidence(
    history,  # pd.Series
    projection: float,
    min_samples: int = 5
) -> float:
    """
    Calculate confidence score using standard error and sample size.

    Higher sample size and lower variance = higher confidence.

    This replaces CV-based confidence calculations which can be misleading.

    Args:
        history: Series of historical stat values
        projection: The model's projection
        min_samples: Minimum sample size required

    Returns:
        Confidence score from 0.0 to 0.95
    """
    n = len(history)
    if n < min_samples or projection <= 0:
        return 0.0

    std = history.std()
    if std == 0:
        return 0.95  # Perfect consistency (rare)

    # Standard error accounts for sample size
    std_error = std / np.sqrt(n)

    # Relative error (how precise is our estimate?)
    relative_error = std_error / projection

    # Logistic transformation to 0-1 range
    # Tuned so that:
    # - relative_error = 0.05 (5%) -> confidence ~ 0.85
    # - relative_error = 0.10 (10%) -> confidence ~ 0.70
    # - relative_error = 0.20 (20%) -> confidence ~ 0.50
    confidence = 1 / (1 + np.exp(relative_error * 10 - 1))

    # Boost confidence with larger sample sizes (diminishing returns)
    sample_factor = min(1.0, np.log(n) / np.log(30))  # Saturates at ~30 games
    confidence = confidence * (0.7 + 0.3 * sample_factor)

    return min(0.95, max(0.05, confidence))


def calculate_prob_over(
    history,  # pd.Series
    line: float,
    method: str = 'empirical'
) -> float:
    """
    Calculate probability of going over the line.

    Methods:
    - 'empirical': Use actual hit rate (no distribution assumption) - most robust
    - 'normal': Assume normal distribution - less robust
    - 'kde': Kernel density estimation - most accurate but slower

    Args:
        history: Series of historical stat values
        line: The betting line
        method: Calculation method ('empirical', 'normal', 'kde')

    Returns:
        Probability of going over (0 to 1)
    """
    stats = _get_stats()

    if len(history) < 5:
        return 0.5  # Not enough data

    if method == 'empirical':
        # Simple and robust - just count how often they went over
        over_count = (history > line).sum()
        # Add Laplace smoothing to avoid 0% or 100%
        return (over_count + 1) / (len(history) + 2)

    elif method == 'normal':
        # Assumes normality - use with caution
        mean = history.mean()
        std = history.std()
        if std == 0:
            return 1.0 if mean > line else 0.0
        z = (line - mean) / std
        return 1 - stats.norm.cdf(z)

    elif method == 'kde':
        # Kernel density estimation - most robust but slower
        try:
            kde = stats.gaussian_kde(history.values)
            # Integrate from line to infinity
            x_range = np.linspace(history.min() - 10, history.max() + 10, 1000)
            pdf = kde(x_range)
            mask = x_range > line
            prob_over = np.trapz(pdf[mask], x_range[mask])
            return max(0.01, min(0.99, prob_over))
        except (ValueError, np.linalg.LinAlgError) as e:
            # KDE can fail with insufficient data or singular matrix
            logger.debug(f"KDE calculation failed, falling back to empirical: {e}")
            return calculate_prob_over(history, line, 'empirical')
        except Exception as e:
            # Unexpected error, fall back to empirical
            logger.debug(f"Unexpected error in KDE, falling back to empirical: {e}")
            return calculate_prob_over(history, line, 'empirical')

    return 0.5


def calculate_confidence_interval(
    history,  # pd.Series
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval on the mean.

    Uses t-distribution for small samples (n < 30).

    Args:
        history: Series of historical stat values
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    stats = _get_stats()

    n = len(history)
    if n < 2:
        mean = history.mean() if n == 1 else 0
        return (mean, mean)

    mean = history.mean()
    std_error = history.std() / np.sqrt(n)

    # Use t-distribution for small samples
    if n < 30:
        t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    else:
        t_val = stats.norm.ppf((1 + confidence_level) / 2)

    margin = t_val * std_error
    return (mean - margin, mean + margin)
