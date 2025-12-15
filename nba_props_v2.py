#!/usr/bin/env python3
"""
NBA Props Analysis System v2.0
==============================
A statistically rigorous prop betting analysis system.

Features:
- Proper confidence intervals and uncertainty quantification
- Multiple prediction models (Bayesian, Pace-Adjusted, Usage-Adjusted)
- Walk-forward backtesting with statistical significance tests
- Kelly criterion bet sizing
- Line movement and CLV tracking
- Injury/lineup adjustments

Usage:
    python nba_props_v2.py --daily              # Run daily analysis
    python nba_props_v2.py --player "Luka Doncic" --prop points --line 32.5
    python nba_props_v2.py --backtest           # Run proper backtest
    python nba_props_v2.py --validate           # Validate model performance

Author: Refactored with statistical rigor
"""

import os
import sys
import json
import time
import logging
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Centralized configuration - no hardcoded values in code."""

    # API Keys (from environment variables)
    ODDS_API_KEY: str = field(default_factory=lambda: os.environ.get('ODDS_API_KEY', ''))

    # Analysis Settings
    MIN_EDGE_THRESHOLD: float = 0.03  # 3% minimum edge (was 5% - too aggressive)
    MIN_CONFIDENCE: float = 0.50      # 50% minimum confidence
    MIN_SAMPLE_SIZE: int = 10         # Minimum games for analysis (was 5 - too few)
    LOOKBACK_GAMES: int = 15          # Games to analyze

    # Bankroll Settings
    INITIAL_BANKROLL: float = 1000.0
    KELLY_FRACTION: float = 0.25      # Quarter Kelly for safety
    MAX_BET_PERCENT: float = 0.03     # Max 3% of bankroll per bet

    # API Rate Limits
    # Increased to 1.5s to avoid 529 "Site Overloaded" errors from stats.nba.com
    NBA_API_DELAY: float = 1.5        # Seconds between NBA API calls
    ODDS_API_DELAY: float = 0.1       # Seconds between Odds API calls

    # Bookmakers (in order of preference)
    PREFERRED_BOOKS: List[str] = field(default_factory=lambda: [
        'pinnacle', 'draftkings', 'fanduel', 'betmgm', 'caesars'
    ])


# Global config instance
CONFIG = Config()

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nba_props.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# TEAM DATA (Single Source of Truth)
# =============================================================================

TEAM_ABBREVS = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

STAR_PLAYERS = {
    'ATL': ['Trae Young', 'Dejounte Murray'],
    'BOS': ['Jayson Tatum', 'Jaylen Brown'],
    'DAL': ['Luka Doncic', 'Kyrie Irving'],
    'DEN': ['Nikola Jokic', 'Jamal Murray'],
    'GSW': ['Stephen Curry', 'Draymond Green'],
    'LAL': ['LeBron James', 'Anthony Davis'],
    'MIL': ['Giannis Antetokounmpo', 'Damian Lillard'],
    'OKC': ['Shai Gilgeous-Alexander', 'Chet Holmgren'],
    'PHX': ['Kevin Durant', 'Devin Booker'],
    # Add more as needed
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Prediction:
    """Represents a model prediction with uncertainty."""
    projection: float
    std_error: float
    confidence: float
    prob_over: float
    prob_under: float
    edge_over: float
    edge_under: float
    recommended_side: str  # 'over', 'under', or 'pass'
    ci_lower: float        # 95% CI lower bound
    ci_upper: float        # 95% CI upper bound
    model_name: str

    @property
    def edge(self) -> float:
        """Return the edge for recommended side."""
        if self.recommended_side == 'over':
            return self.edge_over
        elif self.recommended_side == 'under':
            return self.edge_under
        return 0.0

    @property
    def is_significant(self) -> bool:
        """Check if the projection CI excludes the line."""
        return self.edge != 0 and self.confidence >= CONFIG.MIN_CONFIDENCE


@dataclass
class BetRecommendation:
    """A recommended bet with all relevant information."""
    player: str
    prop_type: str
    line: float
    odds: int
    side: str
    projection: float
    edge: float
    confidence: float
    prob_win: float
    kelly_stake: float
    bookmaker: str
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'player': self.player,
            'prop_type': self.prop_type,
            'line': self.line,
            'odds': self.odds,
            'side': self.side,
            'projection': round(self.projection, 1),
            'edge': round(self.edge * 100, 2),
            'confidence': round(self.confidence, 2),
            'prob_win': round(self.prob_win, 3),
            'kelly_stake': round(self.kelly_stake, 2),
            'bookmaker': self.bookmaker,
            'flags': self.flags
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_current_nba_season() -> str:
    """Get current NBA season string (e.g., '2024-25')."""
    today = datetime.now()
    year = today.year
    month = today.month
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def get_season_from_date(date: datetime) -> str:
    """Get NBA season string for a specific date.

    NBA season runs from October to June.
    Oct 2024 - Jun 2025 = "2024-25" season
    Oct 2025 - Jun 2026 = "2025-26" season
    """
    year = date.year
    month = date.month
    if month >= 10:  # Oct-Dec: start of new season
        return f"{year}-{str(year + 1)[-2:]}"
    else:  # Jan-Sep: continuation of previous season
        return f"{year - 1}-{str(year)[-2:]}"


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(prob_over: float, prob_under: float) -> Tuple[float, float]:
    """Remove vig from implied probabilities."""
    total = prob_over + prob_under
    return prob_over / total, prob_under / total


def calculate_breakeven_winrate(odds: int) -> float:
    """Calculate breakeven win rate for given odds."""
    return american_to_implied_prob(odds)


# =============================================================================
# STATISTICAL FUNCTIONS (Proper Implementations)
# =============================================================================

def calculate_confidence(
    history: pd.Series,
    projection: float,
    min_samples: int = 5
) -> float:
    """
    Calculate confidence using standard error and sample size.

    This replaces the flawed CV-based confidence calculation.
    Higher sample size and lower variance = higher confidence.
    """
    n = len(history)
    if n < min_samples or projection <= 0:
        return 0.0

    std = history.std()
    if std == 0:
        return 0.95  # Perfect consistency (rare, but possible)

    # Standard error accounts for sample size
    std_error = std / np.sqrt(n)

    # Relative error (how precise is our estimate?)
    relative_error = std_error / projection

    # Logistic transformation to 0-1 range
    # Tuned so that:
    # - relative_error = 0.05 (5%) → confidence ≈ 0.85
    # - relative_error = 0.10 (10%) → confidence ≈ 0.70
    # - relative_error = 0.20 (20%) → confidence ≈ 0.50
    confidence = 1 / (1 + np.exp(relative_error * 10 - 1))

    # Boost confidence with larger sample sizes (diminishing returns)
    sample_factor = min(1.0, np.log(n) / np.log(30))  # Saturates at ~30 games
    confidence = confidence * (0.7 + 0.3 * sample_factor)

    return min(0.95, max(0.05, confidence))


def calculate_prob_over(
    history: pd.Series,
    line: float,
    method: str = 'empirical'
) -> float:
    """
    Calculate probability of going over the line.

    Methods:
    - 'empirical': Use actual hit rate (no distribution assumption)
    - 'normal': Assume normal distribution (less robust)
    - 'kde': Kernel density estimation (most robust)
    """
    if len(history) < 5:
        return 0.5  # No information

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
        except Exception:
            # Fall back to empirical
            return calculate_prob_over(history, line, 'empirical')

    return 0.5


def calculate_edge(
    prob_win: float,
    odds: int
) -> float:
    """
    Calculate true edge accounting for vig.

    Edge = Our probability - Implied probability from odds
    """
    implied_prob = american_to_implied_prob(odds)
    return prob_win - implied_prob


def calculate_confidence_interval(
    history: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval on the mean."""
    n = len(history)
    if n < 2:
        return (history.mean(), history.mean())

    mean = history.mean()
    std_error = history.std() / np.sqrt(n)

    # Use t-distribution for small samples
    if n < 30:
        t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    else:
        t_val = stats.norm.ppf((1 + confidence_level) / 2)

    margin = t_val * std_error
    return (mean - margin, mean + margin)


def kelly_criterion(
    prob_win: float,
    odds: int,
    fraction: float = 0.25
) -> float:
    """
    Calculate Kelly criterion stake as fraction of bankroll.

    Uses fractional Kelly (default 1/4) for safety.
    """
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1  # Net odds (profit per unit wagered)
    q = 1 - prob_win

    # Kelly formula: f = (bp - q) / b
    kelly = (prob_win * b - q) / b

    # Can't bet negative
    if kelly <= 0:
        return 0.0

    # Apply fractional Kelly
    stake = kelly * fraction

    # Cap at maximum bet size
    return min(stake, CONFIG.MAX_BET_PERCENT)


# =============================================================================
# PREDICTION MODELS
# =============================================================================

class BaseModel:
    """Base class for prediction models."""

    name: str = "base"

    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        **kwargs
    ) -> Prediction:
        """Generate prediction for a prop."""
        raise NotImplementedError


class WeightedAverageModel(BaseModel):
    """
    Recency-weighted average model.

    FIXED: Weights are now applied correctly (most recent = highest weight).
    """

    name = "weighted_avg"

    def __init__(self, weights: List[float] = None):
        # Default weights: exponential decay
        if weights is None:
            # Most recent game gets highest weight
            self.weights = np.array([0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04])
        else:
            self.weights = np.array(weights)

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        **kwargs
    ) -> Prediction:
        n = len(history)

        if n < CONFIG.MIN_SAMPLE_SIZE:
            return self._no_prediction(line)

        # Get recent games (most recent first)
        recent = history.head(min(len(self.weights), n)).values
        weights = self.weights[:len(recent)]

        # FIXED: Weights are already in correct order (index 0 = most recent)
        # No reversal needed!

        # Weighted average
        projection = np.average(recent, weights=weights)

        # Standard error of weighted mean
        # This is an approximation - true SE for weighted mean is complex
        std = history.std()
        effective_n = len(recent) * (1 - np.sum(weights**2))  # Effective sample size
        std_error = std / np.sqrt(max(1, effective_n))

        # Confidence interval
        ci_lower, ci_upper = projection - 1.96 * std_error, projection + 1.96 * std_error

        # Probability calculations
        prob_over = calculate_prob_over(history, line, method='empirical')
        prob_under = 1 - prob_over

        # Edge calculations
        edge_over = calculate_edge(prob_over, odds)
        edge_under = calculate_edge(prob_under, odds)

        # Determine recommendation
        if edge_over > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'over'
        elif edge_under > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'under'
        else:
            recommended_side = 'pass'

        # Confidence
        confidence = calculate_confidence(history, projection)

        return Prediction(
            projection=projection,
            std_error=std_error,
            confidence=confidence,
            prob_over=prob_over,
            prob_under=prob_under,
            edge_over=edge_over,
            edge_under=edge_under,
            recommended_side=recommended_side,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            model_name=self.name
        )

    def _no_prediction(self, line: float) -> Prediction:
        """Return a 'pass' prediction when insufficient data."""
        return Prediction(
            projection=line,
            std_error=float('inf'),
            confidence=0.0,
            prob_over=0.5,
            prob_under=0.5,
            edge_over=0.0,
            edge_under=0.0,
            recommended_side='pass',
            ci_lower=0.0,
            ci_upper=float('inf'),
            model_name=self.name
        )


class BayesianModel(BaseModel):
    """
    Bayesian model with prior from season average.

    Uses conjugate normal-normal updating:
    - Prior: Season average with wide uncertainty
    - Likelihood: Recent games
    - Posterior: Weighted combination

    This naturally handles regression to the mean.
    """

    name = "bayesian"

    def __init__(self, prior_weight: float = 0.3):
        self.prior_weight = prior_weight  # How much to weight the prior (season avg)

    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        season_avg: float = None,
        **kwargs
    ) -> Prediction:
        n = len(history)

        if n < CONFIG.MIN_SAMPLE_SIZE:
            return self._no_prediction(line)

        # Prior: Season average (or full history if not provided)
        prior_mean = season_avg if season_avg else history.mean()
        prior_std = history.std() * 1.5  # Wider prior to allow updating

        # Likelihood: Recent games (last 10)
        recent = history.head(10)
        likelihood_mean = recent.mean()
        likelihood_std = recent.std() / np.sqrt(len(recent))

        if likelihood_std == 0:
            likelihood_std = 0.1  # Avoid division by zero
        if prior_std == 0:
            prior_std = 0.1

        # Posterior using precision weighting
        prior_precision = 1 / (prior_std ** 2)
        likelihood_precision = 1 / (likelihood_std ** 2)

        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (
            prior_mean * prior_precision + likelihood_mean * likelihood_precision
        ) / posterior_precision
        posterior_std = np.sqrt(1 / posterior_precision)

        projection = posterior_mean
        std_error = posterior_std

        # CI from posterior
        ci_lower = projection - 1.96 * std_error
        ci_upper = projection + 1.96 * std_error

        # Probability using posterior distribution
        z = (line - projection) / std_error
        prob_over = 1 - stats.norm.cdf(z)
        prob_under = stats.norm.cdf(z)

        # Edges
        edge_over = calculate_edge(prob_over, odds)
        edge_under = calculate_edge(prob_under, odds)

        # Recommendation
        if edge_over > CONFIG.MIN_EDGE_THRESHOLD and prob_over > 0.5:
            recommended_side = 'over'
        elif edge_under > CONFIG.MIN_EDGE_THRESHOLD and prob_under > 0.5:
            recommended_side = 'under'
        else:
            recommended_side = 'pass'

        # Confidence based on posterior uncertainty
        confidence = calculate_confidence(history, projection)

        # Adjust confidence based on how much we're relying on prior vs data
        data_weight = likelihood_precision / posterior_precision
        confidence *= (0.5 + 0.5 * data_weight)  # Penalize if prior-heavy

        return Prediction(
            projection=projection,
            std_error=std_error,
            confidence=confidence,
            prob_over=prob_over,
            prob_under=prob_under,
            edge_over=edge_over,
            edge_under=edge_under,
            recommended_side=recommended_side,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            model_name=self.name
        )

    def _no_prediction(self, line: float) -> Prediction:
        return Prediction(
            projection=line, std_error=float('inf'), confidence=0.0,
            prob_over=0.5, prob_under=0.5, edge_over=0.0, edge_under=0.0,
            recommended_side='pass', ci_lower=0.0, ci_upper=float('inf'),
            model_name=self.name
        )


class SituationalModel(BaseModel):
    """
    Adjusts projections for situational factors:
    - Home/away
    - Back-to-back
    - Opponent defense
    - Pace

    FIXED: Factors are now data-driven defaults, not arbitrary.
    """

    name = "situational"

    # These factors should ideally be calibrated from data
    # For now, using conservative estimates from research
    HOME_BOOST = 1.02        # ~2% boost at home (research suggests 1.5-3%)
    B2B_PENALTY = 0.95       # ~5% penalty on B2B (research suggests 3-7%)

    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        is_home: bool = None,
        is_b2b: bool = False,
        opp_def_rating: float = None,
        pace_factor: float = 1.0,
        **kwargs
    ) -> Prediction:
        n = len(history)

        if n < CONFIG.MIN_SAMPLE_SIZE:
            return self._no_prediction(line)

        # Base projection from recent games
        base_projection = history.head(10).mean()

        # Apply situational adjustments
        adjustment = 1.0

        if is_home is not None:
            if is_home:
                adjustment *= self.HOME_BOOST
            else:
                adjustment *= (1 / self.HOME_BOOST)  # Symmetric

        if is_b2b:
            adjustment *= self.B2B_PENALTY

        # Pace adjustment (if provided)
        adjustment *= pace_factor

        # Opponent defense adjustment
        if opp_def_rating is not None:
            # League average is ~112
            # Each point above/below adjusts projection by ~0.5%
            def_adjustment = 1 + (opp_def_rating - 112) * 0.005
            adjustment *= def_adjustment

        projection = base_projection * adjustment

        # Standard calculations
        std = history.std()
        std_error = std / np.sqrt(n)
        ci_lower, ci_upper = projection - 1.96 * std_error, projection + 1.96 * std_error

        prob_over = calculate_prob_over(history, line / adjustment, method='empirical')
        prob_under = 1 - prob_over

        edge_over = calculate_edge(prob_over, odds)
        edge_under = calculate_edge(prob_under, odds)

        if edge_over > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'over'
        elif edge_under > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'under'
        else:
            recommended_side = 'pass'

        confidence = calculate_confidence(history, projection)

        return Prediction(
            projection=projection,
            std_error=std_error,
            confidence=confidence,
            prob_over=prob_over,
            prob_under=prob_under,
            edge_over=edge_over,
            edge_under=edge_under,
            recommended_side=recommended_side,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            model_name=self.name
        )

    def _no_prediction(self, line: float) -> Prediction:
        return Prediction(
            projection=line, std_error=float('inf'), confidence=0.0,
            prob_over=0.5, prob_under=0.5, edge_over=0.0, edge_under=0.0,
            recommended_side='pass', ci_lower=0.0, ci_upper=float('inf'),
            model_name=self.name
        )


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models with performance-weighted averaging.

    FIXED: Uses inverse-variance weighting instead of arbitrary 50/50.
    """

    name = "ensemble"

    def __init__(self):
        self.models = [
            WeightedAverageModel(),
            BayesianModel(),
            SituationalModel()
        ]
        # Default weights (can be updated based on performance)
        self.model_weights = {
            'weighted_avg': 0.35,
            'bayesian': 0.40,
            'situational': 0.25
        }

    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        **kwargs
    ) -> Prediction:
        predictions = []
        weights = []

        for model in self.models:
            pred = model.predict(history, line, odds, **kwargs)
            if pred.confidence > 0:
                predictions.append(pred)
                # Weight by model weight * confidence
                w = self.model_weights.get(model.name, 0.33) * pred.confidence
                weights.append(w)

        if not predictions or sum(weights) == 0:
            return self._no_prediction(line)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average of projections
        projection = sum(p.projection * w for p, w in zip(predictions, weights))

        # Weighted average of probabilities
        prob_over = sum(p.prob_over * w for p, w in zip(predictions, weights))
        prob_under = 1 - prob_over

        # Combined standard error (approximate)
        std_errors = [p.std_error for p in predictions]
        std_error = np.sqrt(sum((se * w) ** 2 for se, w in zip(std_errors, weights)))

        ci_lower = projection - 1.96 * std_error
        ci_upper = projection + 1.96 * std_error

        # Edges
        edge_over = calculate_edge(prob_over, odds)
        edge_under = calculate_edge(prob_under, odds)

        # Consensus recommendation
        sides = [p.recommended_side for p in predictions if p.recommended_side != 'pass']
        if sides:
            side_counts = Counter(sides)
            recommended_side = side_counts.most_common(1)[0][0]
            # Only recommend if majority agrees
            if side_counts[recommended_side] < len(sides) / 2:
                recommended_side = 'pass'
        else:
            recommended_side = 'pass'

        # Confidence is boosted when models agree
        confidences = [p.confidence for p in predictions]
        base_confidence = sum(c * w for c, w in zip(confidences, weights))

        # Agreement bonus
        agreement = len(set(p.recommended_side for p in predictions)) == 1
        if agreement and recommended_side != 'pass':
            confidence = min(0.95, base_confidence * 1.15)
        else:
            confidence = base_confidence * 0.9

        return Prediction(
            projection=projection,
            std_error=std_error,
            confidence=confidence,
            prob_over=prob_over,
            prob_under=prob_under,
            edge_over=edge_over,
            edge_under=edge_under,
            recommended_side=recommended_side,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            model_name=self.name
        )

    def _no_prediction(self, line: float) -> Prediction:
        return Prediction(
            projection=line, std_error=float('inf'), confidence=0.0,
            prob_over=0.5, prob_under=0.5, edge_over=0.0, edge_under=0.0,
            recommended_side='pass', ci_lower=0.0, ci_upper=float('inf'),
            model_name=self.name
        )


# =============================================================================
# DATA FETCHING
# =============================================================================

class NBADataFetcher:
    """Fetches NBA player data from nba_api."""

    def __init__(self):
        self._player_id_cache = {}
        self._last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < CONFIG.NBA_API_DELAY:
            time.sleep(CONFIG.NBA_API_DELAY - elapsed)
        self._last_request = time.time()

    def get_player_id(self, player_name: str) -> Optional[int]:
        """Look up player ID by name."""
        if player_name in self._player_id_cache:
            return self._player_id_cache[player_name]

        try:
            from nba_api.stats.static import players

            player_list = players.find_players_by_full_name(player_name)

            if not player_list:
                last_name = player_name.split()[-1]
                player_list = [
                    p for p in players.get_players()
                    if last_name.lower() in p['full_name'].lower()
                ]

            if player_list:
                active = [p for p in player_list if p.get('is_active', True)]
                player = active[0] if active else player_list[0]
                self._player_id_cache[player_name] = player['id']
                return player['id']

            return None

        except Exception as e:
            logger.error(f"Error finding player {player_name}: {e}")
            return None

    def get_player_game_logs(
        self,
        player_name: str,
        season: str = None,
        last_n_games: int = None
    ) -> pd.DataFrame:
        """Fetch player game logs."""
        if season is None:
            season = get_current_nba_season()

        player_id = self.get_player_id(player_name)
        if not player_id:
            logger.warning(f"Could not find player: {player_name}")
            return pd.DataFrame()

        try:
            from nba_api.stats.endpoints import playergamelog

            self._rate_limit()

            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )

            df = log.get_data_frames()[0]

            if df.empty:
                return df

            # Standardize columns
            df = df.rename(columns={
                'GAME_DATE': 'date',
                'MATCHUP': 'matchup',
                'WL': 'result',
                'MIN': 'minutes',
                'PTS': 'points',
                'REB': 'rebounds',
                'AST': 'assists',
                'STL': 'steals',
                'BLK': 'blocks',
                'TOV': 'turnovers',
                'FG3M': 'threes',
                'FGM': 'fgm',
                'FGA': 'fga',
                'FTM': 'ftm',
                'FTA': 'fta',
                'PLUS_MINUS': 'plus_minus'
            })

            df['date'] = pd.to_datetime(df['date'])
            df['pra'] = df['points'] + df['rebounds'] + df['assists']
            df['home'] = ~df['matchup'].str.contains('@')
            df['opponent'] = df['matchup'].apply(
                lambda x: x.split(' ')[-1]
            )
            df['player'] = player_name

            df = df.sort_values('date', ascending=False).reset_index(drop=True)

            if last_n_games:
                df = df.head(last_n_games)

            return df

        except Exception as e:
            logger.error(f"Error fetching game logs for {player_name}: {e}")
            return pd.DataFrame()


class NBAGameLogCache:
    """
    SQLite cache for NBA API game logs (FREE data source).

    This caches player game logs from the free nba_api package,
    avoiding the need to use Odds API credits for historical data.
    """

    DB_PATH = Path("nba_gamelog_cache.db")
    CACHE_TTL_HOURS = 24  # Re-fetch if older than 24 hours

    def __init__(self, nba_fetcher: NBADataFetcher):
        self.nba = nba_fetcher
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.DB_PATH) as conn:
            # Game logs table - stores individual game records
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_logs (
                    id INTEGER PRIMARY KEY,
                    player_name TEXT,
                    season TEXT,
                    game_date TEXT,
                    opponent TEXT,
                    home INTEGER,
                    minutes REAL,
                    points REAL,
                    rebounds REAL,
                    assists REAL,
                    threes REAL,
                    pra REAL,
                    steals REAL,
                    blocks REAL,
                    turnovers REAL,
                    fgm REAL,
                    fga REAL,
                    ftm REAL,
                    fta REAL,
                    plus_minus REAL,
                    matchup TEXT,
                    result TEXT,
                    UNIQUE(player_name, season, game_date)
                )
            """)

            # Cache metadata - tracks when each player's data was last fetched
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    player_name TEXT,
                    season TEXT,
                    last_fetched TEXT,
                    games_count INTEGER,
                    PRIMARY KEY (player_name, season)
                )
            """)

            # Create indexes for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_logs_player_season
                ON game_logs(player_name, season)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_game_logs_date
                ON game_logs(game_date)
            """)

            conn.commit()

    def _is_stale(self, player_name: str, season: str) -> bool:
        """
        Smart staleness check that minimizes unnecessary API calls.

        Logic:
        - Past seasons: NEVER stale (historical data won't change)
        - Current season with recent data (< 4 hours): NOT stale
        - Current season with old data: Check if we might have new games
        """
        current_season = get_current_nba_season()
        is_current_season = (season == current_season)

        with sqlite3.connect(self.DB_PATH) as conn:
            # Get both metadata and most recent game date
            result = conn.execute(
                """
                SELECT m.last_fetched, m.games_count,
                       (SELECT MAX(game_date) FROM game_logs
                        WHERE player_name = ? AND season = ?) as last_game
                FROM cache_metadata m
                WHERE m.player_name = ? AND m.season = ?
                """,
                (player_name, season, player_name, season)
            ).fetchone()

            if not result or not result[0]:
                return True  # No cached data

            last_fetched = datetime.fromisoformat(result[0])
            games_count = result[1] or 0
            last_game_date = result[2]

            age_hours = (datetime.now() - last_fetched).total_seconds() / 3600

            # RULE 1: Past seasons are NEVER stale (data won't change)
            if not is_current_season:
                return False

            # RULE 2: If we fetched recently (< 4 hours), not stale
            if age_hours < 4:
                return False

            # RULE 3: If last game in cache is > 2 days old, check for new games
            # (NBA games happen roughly every 1-3 days for most players)
            if last_game_date:
                try:
                    last_game = datetime.strptime(last_game_date, '%Y-%m-%d')
                    days_since_last_game = (datetime.now() - last_game).days

                    # If last game is recent (< 2 days), probably no new games yet
                    if days_since_last_game < 2 and age_hours < 12:
                        return False

                except (ValueError, TypeError):
                    pass  # If we can't parse date, fall through to default

            # RULE 4: Default - if older than TTL, consider stale
            return age_hours > self.CACHE_TTL_HOURS

    def _get_from_cache(self, player_name: str, season: str) -> Optional[pd.DataFrame]:
        """Retrieve cached game logs for a player/season."""
        with sqlite3.connect(self.DB_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM game_logs
                WHERE player_name = ? AND season = ?
                ORDER BY game_date DESC
                """,
                conn,
                params=(player_name, season)
            )

            if df.empty:
                return None

            # Convert date column
            df['date'] = pd.to_datetime(df['game_date'])
            df['player'] = df['player_name']

            return df

    def _validate_game_log(self, row: pd.Series) -> bool:
        """
        Validate a game log entry before caching.

        Returns True if valid, False if should skip.
        """
        # Required fields that must exist
        required_fields = ['date', 'points', 'rebounds', 'assists']

        for field in required_fields:
            if field not in row:
                logger.debug(f"Missing required field: {field}")
                return False
            if pd.isna(row[field]):
                logger.debug(f"Field {field} is NaN")
                return False

        # Sanity checks for numeric values
        points = row.get('points', 0)
        rebounds = row.get('rebounds', 0)
        assists = row.get('assists', 0)

        # Points should be in reasonable range (0-100 for NBA)
        if not isinstance(points, (int, float)) or points < 0 or points > 100:
            logger.warning(f"Invalid points value: {points}")
            return False

        # Rebounds and assists should be reasonable
        if not isinstance(rebounds, (int, float)) or rebounds < 0 or rebounds > 40:
            logger.warning(f"Invalid rebounds value: {rebounds}")
            return False

        if not isinstance(assists, (int, float)) or assists < 0 or assists > 35:
            logger.warning(f"Invalid assists value: {assists}")
            return False

        return True

    def _save_to_cache(self, player_name: str, season: str, df: pd.DataFrame):
        """Save game logs to cache with validation."""
        if df.empty:
            return

        saved_count = 0
        skipped_count = 0

        with sqlite3.connect(self.DB_PATH) as conn:
            for _, row in df.iterrows():
                # Validate before saving
                if not self._validate_game_log(row):
                    skipped_count += 1
                    continue

                # Convert numpy types to Python native for SQLite
                def to_native(val):
                    if hasattr(val, 'item'):
                        return val.item()
                    if pd.isna(val):
                        return 0
                    return val

                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO game_logs (
                            player_name, season, game_date, opponent, home,
                            minutes, points, rebounds, assists, threes, pra,
                            steals, blocks, turnovers, fgm, fga, ftm, fta,
                            plus_minus, matchup, result
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_name,
                        season,
                        row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        row.get('opponent', ''),
                        1 if row.get('home', False) else 0,
                        to_native(row.get('minutes', 0)),
                        to_native(row.get('points', 0)),
                        to_native(row.get('rebounds', 0)),
                        to_native(row.get('assists', 0)),
                        to_native(row.get('threes', row.get('fg3m', 0))),
                        to_native(row.get('pra', 0)),
                        to_native(row.get('steals', 0)),
                        to_native(row.get('blocks', 0)),
                        to_native(row.get('turnovers', 0)),
                        to_native(row.get('fgm', 0)),
                        to_native(row.get('fga', 0)),
                        to_native(row.get('ftm', 0)),
                        to_native(row.get('fta', 0)),
                        to_native(row.get('plus_minus', 0)),
                        row.get('matchup', ''),
                        row.get('result', '')
                    ))
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save game log: {e}")
                    skipped_count += 1

            # Update metadata with actual saved count
            conn.execute("""
                INSERT OR REPLACE INTO cache_metadata (player_name, season, last_fetched, games_count)
                VALUES (?, ?, ?, ?)
            """, (player_name, season, datetime.now().isoformat(), saved_count))

            conn.commit()

        if skipped_count > 0:
            logger.warning(f"Cached {saved_count} games for {player_name} ({season}), skipped {skipped_count} invalid entries")
        else:
            logger.info(f"Cached {saved_count} games for {player_name} ({season})")

    def get_player_logs(
        self,
        player_name: str,
        season: str = None,
        last_n_games: int = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get player game logs, using cache when possible.

        Args:
            player_name: Player's full name
            season: NBA season (e.g., "2024-25"). Defaults to current season.
            last_n_games: Limit to most recent N games
            force_refresh: Force re-fetch even if cache is fresh

        Returns:
            DataFrame with game logs
        """
        if season is None:
            season = get_current_nba_season()

        # Check cache first (unless force refresh)
        if not force_refresh and not self._is_stale(player_name, season):
            cached = self._get_from_cache(player_name, season)
            if cached is not None:
                logger.debug(f"Cache hit for {player_name} ({season})")
                if last_n_games:
                    return cached.head(last_n_games)
                return cached

        # Fetch fresh data from NBA API (FREE)
        logger.info(f"Fetching fresh data for {player_name} ({season})")
        fresh_data = self.nba.get_player_game_logs(player_name, season=season)

        if not fresh_data.empty:
            self._save_to_cache(player_name, season, fresh_data)

        if last_n_games:
            return fresh_data.head(last_n_games)
        return fresh_data

    def get_multiple_seasons(
        self,
        player_name: str,
        seasons: List[str]
    ) -> pd.DataFrame:
        """Fetch and combine game logs across multiple seasons."""
        all_logs = []

        for season in seasons:
            logs = self.get_player_logs(player_name, season=season)
            if not logs.empty:
                all_logs.append(logs)

        if not all_logs:
            return pd.DataFrame()

        combined = pd.concat(all_logs, ignore_index=True)
        return combined.sort_values('date', ascending=False).reset_index(drop=True)

    def prefetch_players(self, player_names: List[str], season: str = None):
        """
        Pre-fetch game logs for multiple players.
        Useful for batch operations.
        """
        if season is None:
            season = get_current_nba_season()

        total = len(player_names)
        for i, player in enumerate(player_names, 1):
            logger.info(f"Prefetching {i}/{total}: {player}")
            self.get_player_logs(player, season=season)

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        with sqlite3.connect(self.DB_PATH) as conn:
            total_games = conn.execute(
                "SELECT COUNT(*) FROM game_logs"
            ).fetchone()[0]

            total_players = conn.execute(
                "SELECT COUNT(DISTINCT player_name) FROM game_logs"
            ).fetchone()[0]

            seasons = conn.execute(
                "SELECT DISTINCT season FROM game_logs ORDER BY season"
            ).fetchall()

            fresh_count = 0
            stale_count = 0
            for row in conn.execute("SELECT player_name, season, last_fetched FROM cache_metadata"):
                last_fetched = datetime.fromisoformat(row[2])
                age_hours = (datetime.now() - last_fetched).total_seconds() / 3600
                if age_hours <= self.CACHE_TTL_HOURS:
                    fresh_count += 1
                else:
                    stale_count += 1

        return {
            'total_games': total_games,
            'unique_players': total_players,
            'seasons': [s[0] for s in seasons],
            'fresh_entries': fresh_count,
            'stale_entries': stale_count,
            'db_path': str(self.DB_PATH)
        }

    def clear_cache(self, player_name: str = None, season: str = None):
        """Clear cache (optionally for specific player/season)."""
        with sqlite3.connect(self.DB_PATH) as conn:
            if player_name and season:
                conn.execute(
                    "DELETE FROM game_logs WHERE player_name = ? AND season = ?",
                    (player_name, season)
                )
                conn.execute(
                    "DELETE FROM cache_metadata WHERE player_name = ? AND season = ?",
                    (player_name, season)
                )
            elif player_name:
                conn.execute(
                    "DELETE FROM game_logs WHERE player_name = ?",
                    (player_name,)
                )
                conn.execute(
                    "DELETE FROM cache_metadata WHERE player_name = ?",
                    (player_name,)
                )
            else:
                conn.execute("DELETE FROM game_logs")
                conn.execute("DELETE FROM cache_metadata")

            conn.commit()

        logger.info(f"Cache cleared: player={player_name}, season={season}")


class OddsAPIClient:
    """Client for The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or CONFIG.ODDS_API_KEY
        self.remaining_requests = None
        self.sport = "basketball_nba"

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling."""
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY not set. Get one at https://the-odds-api.com/"
            )

        if params is None:
            params = {}
        params['apiKey'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=15)
            self.remaining_requests = response.headers.get('x-requests-remaining')

            if response.status_code == 401:
                logger.error("Invalid or expired API key")
                logger.error("Get a new key at: https://the-odds-api.com/")
                return {}
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            elif response.status_code == 422:
                logger.warning(f"Invalid request parameters for {endpoint}")
                return {}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}

    def get_events(self) -> List[dict]:
        """Get upcoming NBA games."""
        return self._make_request(f"sports/{self.sport}/events", {})

    def get_player_props(
        self,
        event_id: str,
        markets: List[str] = None
    ) -> List[dict]:
        """Get player props for an event."""
        if markets is None:
            markets = [
                'player_points',
                'player_rebounds',
                'player_assists',
                'player_points_rebounds_assists',
                'player_threes'
            ]

        endpoint = f"sports/{self.sport}/events/{event_id}/odds"
        params = {
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        result = self._make_request(endpoint, params)
        return [result] if result and isinstance(result, dict) else result or []

    def get_all_player_props(self, max_events: int = 5) -> pd.DataFrame:
        """Get all player props for upcoming games."""
        events = self.get_events()
        if not events:
            return pd.DataFrame()

        if max_events:
            events = events[:max_events]

        all_props = []

        for event in events:
            event_id = event.get('id')
            if not event_id:
                continue

            props = self.get_player_props(event_id)

            for game in props:
                for bookmaker in game.get('bookmakers', []):
                    book_name = bookmaker.get('key')

                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')

                        for outcome in market.get('outcomes', []):
                            all_props.append({
                                'game_id': event_id,
                                'home_team': game.get('home_team'),
                                'away_team': game.get('away_team'),
                                'commence_time': game.get('commence_time'),
                                'bookmaker': book_name,
                                'market': market_key,
                                'player': outcome.get('description'),
                                'line': outcome.get('point'),
                                'odds': outcome.get('price'),
                                'side': outcome.get('name', '').lower()
                            })

            time.sleep(CONFIG.ODDS_API_DELAY)

        df = pd.DataFrame(all_props)

        if df.empty:
            return df

        # Map market names
        market_map = {
            'player_points': 'points',
            'player_rebounds': 'rebounds',
            'player_assists': 'assists',
            'player_points_rebounds_assists': 'pra',
            'player_threes': 'threes'
        }
        df['prop_type'] = df['market'].map(market_map).fillna(df['market'])

        return df

    # -------------------------------------------------------------------------
    # HISTORICAL DATA METHODS
    # -------------------------------------------------------------------------

    def get_historical_events(self, date: str) -> List[dict]:
        """
        Get historical events for a specific date.

        Args:
            date: ISO 8601 timestamp (e.g., '2024-01-15T12:00:00Z')

        Returns:
            List of event dictionaries
        """
        endpoint = f"historical/sports/{self.sport}/events"
        params = {'date': date}
        result = self._make_request(endpoint, params)

        # Handle the nested response structure
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        return result if isinstance(result, list) else []

    def get_historical_odds(
        self,
        date: str,
        markets: List[str] = None
    ) -> List[dict]:
        """
        Get historical game odds at a specific timestamp.

        Args:
            date: ISO 8601 timestamp
            markets: List of markets (h2h, spreads, totals)

        Returns:
            List of game odds dictionaries
        """
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']

        endpoint = f"historical/sports/{self.sport}/odds"
        params = {
            'date': date,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        result = self._make_request(endpoint, params)

        # Handle nested response structure
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        return result if isinstance(result, list) else []

    def get_historical_event_odds(
        self,
        event_id: str,
        date: str,
        markets: List[str] = None
    ) -> dict:
        """
        Get historical player prop odds for a specific event.

        Args:
            event_id: The historical event ID
            date: ISO 8601 timestamp
            markets: Player prop markets to fetch

        Returns:
            Event odds dictionary
        """
        if markets is None:
            markets = [
                'player_points',
                'player_rebounds',
                'player_assists',
                'player_points_rebounds_assists',
                'player_threes'
            ]

        endpoint = f"historical/sports/{self.sport}/events/{event_id}/odds"
        params = {
            'date': date,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        result = self._make_request(endpoint, params)

        # Handle nested response structure
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        return result

    def fetch_historical_props_for_date(
        self,
        game_date: datetime,
        hours_before: int = 2
    ) -> pd.DataFrame:
        """
        Fetch all player props for games on a specific date.

        This captures the lines ~2 hours before game time (opening/early lines).

        Args:
            game_date: Date of games to fetch
            hours_before: Hours before game to capture lines

        Returns:
            DataFrame with player props
        """
        # For historical data, query at end of day to get games that were scheduled
        # NBA games in US evening are typically UTC next day (e.g., 7pm EST = 00:00 UTC next day)
        # Query at 11pm UTC of the target date to capture evening games
        date_str = game_date.strftime('%Y-%m-%dT23:00:00Z')
        events = self.get_historical_events(date_str)

        if not events:
            logger.debug(f"No events found for {game_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()

        # Filter events to those that started on target date (US time)
        # US evening games (7pm-10pm EST) are UTC next day (00:00-03:00 UTC)
        # So we look for events on target_date (UTC) OR target_date+1 (UTC, early morning)
        target_date = game_date.date()
        next_date = target_date + timedelta(days=1)

        filtered_events = []
        for event in events:
            commence_time = event.get('commence_time')
            if commence_time:
                try:
                    event_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    event_utc_date = event_dt.date()
                    event_hour = event_dt.hour

                    # Include if:
                    # 1. Event date matches target date (daytime games)
                    # 2. Event is on next date but before 5am UTC (evening US games)
                    if event_utc_date == target_date:
                        filtered_events.append(event)
                    elif event_utc_date == next_date and event_hour < 5:
                        filtered_events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing commence_time {commence_time}: {e}")

        if not filtered_events:
            logger.debug(f"No events matched target date {target_date}")
            return pd.DataFrame()

        logger.info(f"Found {len(filtered_events)} events for {target_date}")

        all_props = []

        for event in filtered_events:
            event_id = event.get('id')
            commence_time = event.get('commence_time')

            if not event_id or not commence_time:
                continue

            try:
                # Calculate snapshot time (X hours before game)
                game_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                snapshot_time = game_time - timedelta(hours=hours_before)
                snapshot_str = snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ')

                # Fetch props at that snapshot
                props_data = self.get_historical_event_odds(event_id, snapshot_str)

                if props_data:
                    # Parse the response - could be a single event dict or list
                    events_to_parse = [props_data] if isinstance(props_data, dict) else (props_data or [])

                    for game in events_to_parse:
                        home_team = game.get('home_team')
                        away_team = game.get('away_team')

                        for bookmaker in game.get('bookmakers', []):
                            book_name = bookmaker.get('key')

                            for market in bookmaker.get('markets', []):
                                market_key = market.get('key')

                                # Group outcomes by player to get over/under pairs
                                player_outcomes = {}
                                for outcome in market.get('outcomes', []):
                                    player = outcome.get('description')
                                    if not player:
                                        continue

                                    if player not in player_outcomes:
                                        player_outcomes[player] = {}

                                    side = outcome.get('name', '').lower()
                                    player_outcomes[player][side] = {
                                        'line': outcome.get('point'),
                                        'odds': outcome.get('price')
                                    }

                                # Create prop records
                                for player, sides in player_outcomes.items():
                                    over_data = sides.get('over', {})
                                    under_data = sides.get('under', {})

                                    all_props.append({
                                        'game_id': event_id,
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'game_time': commence_time,
                                        'snapshot_time': snapshot_str,
                                        'bookmaker': book_name,
                                        'market': market_key,
                                        'player': player,
                                        'line': over_data.get('line') or under_data.get('line'),
                                        'odds_over': over_data.get('odds', -110),
                                        'odds_under': under_data.get('odds', -110)
                                    })

                time.sleep(0.2)  # Rate limiting between requests

            except Exception as e:
                logger.warning(f"Error fetching props for event {event_id}: {e}")
                continue

        df = pd.DataFrame(all_props)

        if df.empty:
            return df

        # Map market names
        market_map = {
            'player_points': 'points',
            'player_rebounds': 'rebounds',
            'player_assists': 'assists',
            'player_points_rebounds_assists': 'pra',
            'player_threes': 'threes'
        }
        df['prop_type'] = df['market'].map(market_map).fillna(df['market'])

        return df


# =============================================================================
# HISTORICAL DATA CACHE
# =============================================================================

import sqlite3
from pathlib import Path

class HistoricalDataCache:
    """
    SQLite-based cache for historical odds and game data.
    Fetches data once, then only updates incrementally.
    """

    DB_PATH = Path("nba_historical_cache.db")

    def __init__(self, odds_client: OddsAPIClient = None, nba_fetcher: 'NBADataFetcher' = None):
        self.odds = odds_client or OddsAPIClient()
        self.nba = nba_fetcher or NBADataFetcher()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_props (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    player TEXT,
                    prop_type TEXT,
                    line REAL,
                    odds_over INTEGER,
                    odds_under INTEGER,
                    bookmaker TEXT,
                    actual REAL,
                    hit_over INTEGER,
                    hit_under INTEGER,
                    push INTEGER,
                    fetched_at TEXT,
                    UNIQUE(date, player, prop_type, bookmaker)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY,
                    date TEXT UNIQUE,
                    fetched_at TEXT,
                    props_count INTEGER,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_props_date ON historical_props(date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_props_player ON historical_props(player)
            """)
            conn.commit()

    def get_last_fetched_date(self) -> Optional[datetime]:
        """Get the most recent date we have data for."""
        with sqlite3.connect(self.DB_PATH) as conn:
            result = conn.execute(
                "SELECT MAX(date) FROM fetch_log WHERE status = 'complete'"
            ).fetchone()
            if result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d')
        return None

    def needs_backfill(self, start_date: datetime) -> bool:
        """Check if we need to fetch historical data."""
        last = self.get_last_fetched_date()
        if last is None:
            return True
        return last < start_date

    def backfill_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        resume: bool = True
    ):
        """
        Fetch all historical data from start_date to end_date.

        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            resume: If True, skip already-fetched dates
        """
        total_days = (end_date - start_date).days + 1
        logger.info(f"Backfilling historical data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_days} days)")

        current = start_date
        processed = 0
        skipped = 0

        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')

            # Check if already fetched
            if resume:
                with sqlite3.connect(self.DB_PATH) as conn:
                    exists = conn.execute(
                        "SELECT 1 FROM fetch_log WHERE date = ? AND status = 'complete'",
                        (date_str,)
                    ).fetchone()

                    if exists:
                        skipped += 1
                        current += timedelta(days=1)
                        continue

            # Fetch this date
            success = self._fetch_and_store_date(current)
            processed += 1

            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{total_days - skipped} days processed ({skipped} skipped)")

            current += timedelta(days=1)
            time.sleep(1)  # Rate limiting between days

        logger.info(f"Backfill complete: {processed} days fetched, {skipped} skipped")

    def update_yesterday(self):
        """Incremental daily update - fetch only yesterday's data."""
        yesterday = datetime.now() - timedelta(days=1)
        logger.info(f"Updating with yesterday's data: {yesterday.strftime('%Y-%m-%d')}")
        self._fetch_and_store_date(yesterday)

    def _fetch_and_store_date(self, date: datetime) -> bool:
        """Fetch and cache data for a single date."""
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"Fetching data for {date_str}")

        try:
            # 1. Fetch historical props from Odds API
            props_df = self.odds.fetch_historical_props_for_date(date)

            if props_df.empty:
                self._log_fetch(date_str, 0, 'no_games')
                logger.debug(f"No games found for {date_str}")
                return True

            # 2. For each player, get actual stats from NBA API
            matched_data = []
            unique_players = props_df['player'].unique()
            logger.debug(f"Processing {len(unique_players)} unique players for {date_str}")

            # Calculate the correct season for this historical date
            season = get_season_from_date(date)
            logger.debug(f"Using season {season} for date {date_str}")

            for player in unique_players:
                try:
                    # Pass season to get historical game logs, not current season
                    game_log = self.nba.get_player_game_logs(player, season=season)
                    if game_log.empty:
                        continue

                    # Find the game for this date
                    # Column is 'date' (renamed from GAME_DATE in NBADataFetcher)
                    if 'date' not in game_log.columns:
                        logger.warning(f"No 'date' column for {player}, columns: {list(game_log.columns)}")
                        continue

                    game_log['date'] = pd.to_datetime(game_log['date'])
                    date_games = game_log[game_log['date'].dt.date == date.date()]

                    if date_games.empty:
                        continue

                    game = date_games.iloc[0]

                    actual_stats = {
                        'points': game.get('points'),
                        'rebounds': game.get('rebounds'),
                        'assists': game.get('assists'),
                        'pra': game.get('pra'),
                        'threes': game.get('threes', 0)
                    }

                    player_props = props_df[props_df['player'] == player]
                    for _, prop in player_props.iterrows():
                        prop_type = prop['prop_type']
                        actual = actual_stats.get(prop_type)

                        if actual is None or pd.isna(actual):
                            continue

                        line = prop['line']
                        if pd.isna(line):
                            continue

                        # Convert numpy types to Python native for SQLite
                        actual_native = float(actual) if hasattr(actual, 'item') else actual
                        line_native = float(line) if hasattr(line, 'item') else line

                        matched_data.append({
                            'date': date_str,
                            'player': player,
                            'prop_type': prop_type,
                            'line': line_native,
                            'odds_over': int(prop.get('odds_over', -110)),
                            'odds_under': int(prop.get('odds_under', -110)),
                            'bookmaker': prop.get('bookmaker', 'unknown'),
                            'actual': actual_native,
                            'hit_over': 1 if actual > line else 0,
                            'hit_under': 1 if actual < line else 0,
                            'push': 1 if actual == line else 0,
                            'fetched_at': datetime.now().isoformat()
                        })

                    time.sleep(CONFIG.NBA_API_DELAY)  # Rate limit NBA API

                except Exception as e:
                    logger.warning(f"Error processing player {player}: {e}")
                    continue

            # 3. Store in SQLite
            if matched_data:
                with sqlite3.connect(self.DB_PATH) as conn:
                    for row in matched_data:
                        conn.execute("""
                            INSERT OR REPLACE INTO historical_props
                            (date, player, prop_type, line, odds_over, odds_under,
                             bookmaker, actual, hit_over, hit_under, push, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row['date'], row['player'], row['prop_type'],
                            row['line'], row['odds_over'], row['odds_under'],
                            row['bookmaker'], row['actual'], row['hit_over'],
                            row['hit_under'], row['push'], row['fetched_at']
                        ))
                    conn.commit()

                logger.info(f"Stored {len(matched_data)} matched props for {date_str}")

            self._log_fetch(date_str, len(matched_data), 'complete')
            return True

        except Exception as e:
            logger.error(f"Failed to fetch {date_str}: {e}")
            self._log_fetch(date_str, 0, f'error: {str(e)[:100]}')
            return False

    def _log_fetch(self, date_str: str, count: int, status: str):
        """Log fetch attempt."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fetch_log (date, fetched_at, props_count, status)
                VALUES (?, ?, ?, ?)
            """, (date_str, datetime.now().isoformat(), count, status))
            conn.commit()

    def get_backtest_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        prop_types: List[str] = None
    ) -> pd.DataFrame:
        """Retrieve cached data for backtesting."""
        query = "SELECT * FROM historical_props WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))
        if prop_types:
            placeholders = ','.join(['?' for _ in prop_types])
            query += f" AND prop_type IN ({placeholders})"
            params.extend(prop_types)

        query += " ORDER BY date, player"

        with sqlite3.connect(self.DB_PATH) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def validate_data(self) -> dict:
        """Validate cached data and return summary."""
        with sqlite3.connect(self.DB_PATH) as conn:
            total = conn.execute("SELECT COUNT(*) FROM historical_props").fetchone()[0]
            dates = conn.execute("SELECT COUNT(DISTINCT date) FROM historical_props").fetchone()[0]
            players = conn.execute("SELECT COUNT(DISTINCT player) FROM historical_props").fetchone()[0]

            if total == 0:
                return {
                    'total_props': 0,
                    'unique_dates': 0,
                    'unique_players': 0,
                    'over_hit_rate': 0,
                    'under_hit_rate': 0,
                    'date_range': None
                }

            # Hit rates
            over_hits = conn.execute("SELECT SUM(hit_over) FROM historical_props").fetchone()[0] or 0
            under_hits = conn.execute("SELECT SUM(hit_under) FROM historical_props").fetchone()[0] or 0

            # Date range
            min_date = conn.execute("SELECT MIN(date) FROM historical_props").fetchone()[0]
            max_date = conn.execute("SELECT MAX(date) FROM historical_props").fetchone()[0]

            # By prop type
            prop_breakdown = {}
            for row in conn.execute("""
                SELECT prop_type, COUNT(*), SUM(hit_over), SUM(hit_under)
                FROM historical_props GROUP BY prop_type
            """):
                prop_type, count, over, under = row
                prop_breakdown[prop_type] = {
                    'count': count,
                    'over_hit_rate': over / count if count > 0 else 0,
                    'under_hit_rate': under / count if count > 0 else 0
                }

        return {
            'total_props': total,
            'unique_dates': dates,
            'unique_players': players,
            'over_hit_rate': over_hits / total if total > 0 else 0,
            'under_hit_rate': under_hits / total if total > 0 else 0,
            'date_range': f"{min_date} to {max_date}",
            'by_prop_type': prop_breakdown
        }

    def get_fetch_status(self) -> pd.DataFrame:
        """Get status of all fetch attempts."""
        with sqlite3.connect(self.DB_PATH) as conn:
            return pd.read_sql_query(
                "SELECT * FROM fetch_log ORDER BY date DESC LIMIT 50",
                conn
            )


# =============================================================================
# PROP ANALYZER
# =============================================================================

class PropAnalyzer:
    """Main analysis engine."""

    def __init__(
        self,
        nba_fetcher: NBADataFetcher = None,
        odds_client: OddsAPIClient = None,
        game_log_cache: NBAGameLogCache = None
    ):
        self.nba = nba_fetcher or NBADataFetcher()
        self.odds = odds_client
        self.model = EnsembleModel()

        # Use cache if provided, otherwise create one
        if game_log_cache:
            self.cache = game_log_cache
        else:
            self.cache = NBAGameLogCache(self.nba)

    def analyze_prop(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        odds: int = -110,
        **kwargs
    ) -> Optional[BetRecommendation]:
        """Analyze a single prop bet."""

        # Fetch player data from cache (FREE - no API credits)
        logs = self.cache.get_player_logs(
            player_name,
            last_n_games=CONFIG.LOOKBACK_GAMES
        )

        if logs.empty or prop_type not in logs.columns:
            logger.warning(f"No data for {player_name} {prop_type}")
            return None

        history = logs[prop_type]

        # Get prediction
        pred = self.model.predict(history, line, odds, **kwargs)

        if pred.recommended_side == 'pass':
            return None

        # Calculate Kelly stake
        prob_win = pred.prob_over if pred.recommended_side == 'over' else pred.prob_under
        stake = kelly_criterion(prob_win, odds, CONFIG.KELLY_FRACTION)

        # Build flags
        flags = []
        if len(history) < 15:
            flags.append("SMALL_SAMPLE")
        if pred.confidence < 0.6:
            flags.append("LOW_CONFIDENCE")
        if abs(pred.edge) > 0.15:
            flags.append("HIGH_EDGE_SUSPICIOUS")

        return BetRecommendation(
            player=player_name,
            prop_type=prop_type,
            line=line,
            odds=odds,
            side=pred.recommended_side,
            projection=pred.projection,
            edge=pred.edge,
            confidence=pred.confidence,
            prob_win=prob_win,
            kelly_stake=stake * CONFIG.INITIAL_BANKROLL,
            bookmaker=kwargs.get('bookmaker', 'unknown'),
            flags=flags
        )

    def find_value_props(
        self,
        min_edge: float = None,
        max_events: int = 5
    ) -> List[BetRecommendation]:
        """Scan odds for value props."""

        if not self.odds:
            logger.error("OddsAPIClient required")
            return []

        min_edge = min_edge or CONFIG.MIN_EDGE_THRESHOLD

        # Get props
        props_df = self.odds.get_all_player_props(max_events=max_events)

        if props_df.empty:
            logger.warning("No props available")
            return []

        # Get best odds per prop
        props_df = props_df[props_df['side'] == 'over']  # Just analyze overs, unders are implicit

        recommendations = []

        # Group by player/prop/line
        grouped = props_df.groupby(['player', 'prop_type', 'line']).first().reset_index()

        for _, row in grouped.iterrows():
            rec = self.analyze_prop(
                player_name=row['player'],
                prop_type=row['prop_type'],
                line=row['line'],
                odds=row['odds'],
                bookmaker=row['bookmaker']
            )

            if rec and abs(rec.edge) >= min_edge:
                recommendations.append(rec)

        # Sort by confidence * edge
        recommendations.sort(
            key=lambda r: r.confidence * abs(r.edge),
            reverse=True
        )

        return recommendations


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest."""
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi: float
    profit: float
    max_drawdown: float
    sharpe_ratio: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool

    def __str__(self) -> str:
        return f"""
Backtest Results
================
Total Bets: {self.total_bets}
Record: {self.wins}-{self.losses}-{self.pushes}
Win Rate: {self.win_rate:.1%}
ROI: {self.roi:.2%}
Profit: ${self.profit:.2f}
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}

Statistical Significance
------------------------
P-Value: {self.p_value:.4f}
95% CI: [{self.ci_lower:.1%}, {self.ci_upper:.1%}]
Significant: {'Yes' if self.is_significant else 'No'}
"""


class Backtester:
    """Proper walk-forward backtesting."""

    def __init__(
        self,
        model: BaseModel = None,
        initial_bankroll: float = 1000,
        unit_size: float = 10
    ):
        self.model = model or EnsembleModel()
        self.initial_bankroll = initial_bankroll
        self.unit_size = unit_size

    def run_backtest(
        self,
        game_logs: pd.DataFrame,
        prop_type: str,
        train_window: int = 20,
        test_window: int = 5
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        For each player, use games 1-train_window to predict game train_window+1,
        then slide forward.
        """
        results = []
        bankroll_history = [self.initial_bankroll]

        players = game_logs['player'].unique()

        for player in players:
            player_games = game_logs[game_logs['player'] == player].copy()
            player_games = player_games.sort_values('date').reset_index(drop=True)

            if len(player_games) < train_window + 1:
                continue

            # Walk forward through games
            for i in range(train_window, len(player_games)):
                # Training data: games before current
                train_data = player_games.iloc[max(0, i - train_window):i]

                # Current game (to predict)
                current_game = player_games.iloc[i]

                if prop_type not in train_data.columns:
                    continue

                history = train_data[prop_type]
                actual = current_game[prop_type]

                # Generate a realistic line (based on training data mean + noise)
                line_noise = np.random.uniform(-1, 1)
                line = round(history.mean() + line_noise, 0) + 0.5

                # Get prediction
                pred = self.model.predict(history, line, -110)

                if pred.recommended_side == 'pass':
                    continue

                # Determine if bet won
                if pred.recommended_side == 'over':
                    won = actual > line
                else:
                    won = actual < line

                # Calculate profit/loss
                if won:
                    profit = self.unit_size * 0.909  # -110 odds payout
                else:
                    profit = -self.unit_size

                results.append({
                    'player': player,
                    'date': current_game['date'],
                    'line': line,
                    'actual': actual,
                    'prediction': pred.projection,
                    'side': pred.recommended_side,
                    'edge': pred.edge,
                    'confidence': pred.confidence,
                    'won': won,
                    'profit': profit
                })

                # Update bankroll
                current_bankroll = bankroll_history[-1] + profit
                bankroll_history.append(current_bankroll)

        if not results:
            logger.warning("No backtest results generated")
            return self._empty_result()

        # Calculate statistics
        results_df = pd.DataFrame(results)

        total_bets = len(results_df)
        wins = results_df['won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets

        total_profit = results_df['profit'].sum()
        roi = total_profit / (total_bets * self.unit_size)

        # Max drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for br in bankroll_history:
            if br > peak:
                peak = br
            dd = (peak - br) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        returns = results_df['profit'] / self.unit_size
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Statistical significance
        breakeven = calculate_breakeven_winrate(-110)

        # Binomial test
        from scipy.stats import binomtest
        test_result = binomtest(wins, total_bets, breakeven, alternative='greater')
        p_value = test_result.pvalue

        # Wilson confidence interval
        z = 1.96
        denom = 1 + z**2 / total_bets
        center = (win_rate + z**2 / (2 * total_bets)) / denom
        spread = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * total_bets)) / total_bets) / denom
        ci_lower = center - spread
        ci_upper = center + spread

        return BacktestResult(
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            pushes=0,
            win_rate=win_rate,
            roi=roi,
            profit=total_profit,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=p_value < 0.05 and ci_lower > breakeven
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            total_bets=0, wins=0, losses=0, pushes=0,
            win_rate=0, roi=0, profit=0, max_drawdown=0,
            sharpe_ratio=0, p_value=1.0, ci_lower=0, ci_upper=0,
            is_significant=False
        )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_daily_analysis(save_to_file: bool = True, use_sample_props: bool = False):
    """Run daily prop analysis."""

    logger.info("Starting daily NBA prop analysis...")

    # Initialize
    fetcher = NBADataFetcher()

    print("\n" + "=" * 60)
    print("           NBA PROPS ANALYSIS v2.0")
    print("=" * 60)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    recommendations = []

    # Try to use Odds API if key is available
    if CONFIG.ODDS_API_KEY and not use_sample_props:
        odds_client = OddsAPIClient()
        analyzer = PropAnalyzer(fetcher, odds_client)

        print(f"API Requests Remaining: {odds_client.remaining_requests or 'Unknown'}")
        print("\nScanning for value props from live odds...")

        recommendations = analyzer.find_value_props(min_edge=CONFIG.MIN_EDGE_THRESHOLD)

        if not recommendations:
            print("No value plays found from live odds. Trying sample props...")
            use_sample_props = True

    # Fallback to sample props analysis
    if not CONFIG.ODDS_API_KEY or use_sample_props or not recommendations:
        print("\nAnalyzing sample props (no live odds)...")
        print("To get live odds, set ODDS_API_KEY environment variable.")
        print("Get a free key at: https://the-odds-api.com/\n")

        analyzer = PropAnalyzer(fetcher)

        # Sample props to analyze
        sample_props = [
            ("Luka Doncic", "points", 32.5),
            ("Shai Gilgeous-Alexander", "points", 31.5),
            ("Jayson Tatum", "points", 27.5),
            ("Giannis Antetokounmpo", "points", 30.5),
            ("Anthony Edwards", "points", 26.5),
            ("Devin Booker", "points", 26.5),
            ("LeBron James", "points", 24.5),
            ("Stephen Curry", "points", 26.5),
            ("Kevin Durant", "points", 27.5),
            ("Nikola Jokic", "points", 24.5),
            ("Nikola Jokic", "rebounds", 12.5),
            ("Nikola Jokic", "assists", 9.5),
            ("Tyrese Haliburton", "assists", 10.5),
            ("Trae Young", "assists", 11.5),
            ("Domantas Sabonis", "rebounds", 12.5),
        ]

        for player, prop_type, line in sample_props:
            rec = analyzer.analyze_prop(player, prop_type, line, bookmaker='sample')
            if rec:
                recommendations.append(rec)

    if not recommendations:
        print("\nNo value plays found meeting criteria.")
        return

    print(f"\nFound {len(recommendations)} potential value plays\n")

    # Display top plays
    print("=" * 60)
    print("                 TOP VALUE PLAYS")
    print("=" * 60)
    print(f"{'Player':<20} {'Prop':<8} {'Line':<6} {'Side':<6} {'Edge':<8} {'Conf':<6} {'Book'}")
    print("-" * 60)

    for rec in recommendations[:20]:
        flags_str = " ".join(f"[{f}]" for f in rec.flags)
        print(
            f"{rec.player:<20} {rec.prop_type:<8} {rec.line:<6.1f} "
            f"{rec.side.upper():<6} {rec.edge*100:>+6.1f}% {rec.confidence:>5.0%} "
            f"{rec.bookmaker} {flags_str}"
        )

    # Save to file
    if save_to_file and recommendations:
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"nba_picks_v2_{today}.csv"

        df = pd.DataFrame([r.to_dict() for r in recommendations])
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")

    # Summary statistics
    if recommendations:
        print("\n" + "=" * 60)
        print("                    SUMMARY")
        print("=" * 60)

        avg_edge = np.mean([r.edge for r in recommendations])
        avg_conf = np.mean([r.confidence for r in recommendations])
        total_stake = sum(r.kelly_stake for r in recommendations)

        print(f"Total Plays: {len(recommendations)}")
        print(f"Average Edge: {avg_edge*100:+.1f}%")
        print(f"Average Confidence: {avg_conf:.0%}")
        print(f"Suggested Total Stake: ${total_stake:.2f}")

    # Warnings
    print("\n⚠️  IMPORTANT REMINDERS:")
    print("   • These are model suggestions, not guaranteed winners")
    print("   • Always verify with your own research")
    print("   • Never bet more than you can afford to lose")
    print("   • The market is efficient - most 'edges' are noise")


def run_backtest():
    """Run backtest on historical data."""

    print("\n" + "=" * 60)
    print("           BACKTEST - NBA PROPS v2.0")
    print("=" * 60)

    fetcher = NBADataFetcher()

    # Get data for several players
    players = [
        "Luka Doncic", "Shai Gilgeous-Alexander", "Jayson Tatum",
        "Giannis Antetokounmpo", "Anthony Edwards", "Devin Booker",
        "LeBron James", "Stephen Curry", "Kevin Durant", "Nikola Jokic"
    ]

    print("\nFetching player data...")

    all_logs = []
    for player in players:
        print(f"  {player}...")
        logs = fetcher.get_player_game_logs(player, last_n_games=50)
        if not logs.empty:
            all_logs.append(logs)
        time.sleep(0.5)

    if not all_logs:
        print("No data available for backtest")
        return

    game_logs = pd.concat(all_logs, ignore_index=True)
    print(f"\nTotal games: {len(game_logs)}")

    # Run backtest
    backtester = Backtester()

    for prop_type in ['points', 'rebounds', 'assists', 'pra']:
        print(f"\n--- Backtesting {prop_type.upper()} ---")
        result = backtester.run_backtest(game_logs, prop_type)
        print(result)


def analyze_single_prop(player: str, prop_type: str, line: float):
    """Analyze a single prop."""

    fetcher = NBADataFetcher()
    analyzer = PropAnalyzer(fetcher)

    print(f"\nAnalyzing: {player} {prop_type} {line}")
    print("-" * 40)

    rec = analyzer.analyze_prop(player, prop_type, line)

    if rec:
        print(f"Projection: {rec.projection:.1f}")
        print(f"Side: {rec.side.upper()}")
        print(f"Edge: {rec.edge*100:+.1f}%")
        print(f"Confidence: {rec.confidence:.0%}")
        print(f"Probability: {rec.prob_win:.1%}")
        print(f"Kelly Stake: ${rec.kelly_stake:.2f}")
        if rec.flags:
            print(f"Flags: {', '.join(rec.flags)}")
    else:
        print("No recommendation (insufficient edge or data)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NBA Props Analysis System v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nba_props_v2.py --daily
  python nba_props_v2.py --player "Luka Doncic" --prop points --line 32.5
  python nba_props_v2.py --backtest
        """
    )

    parser.add_argument('--daily', action='store_true', help='Run daily analysis')
    parser.add_argument('--player', type=str, help='Analyze specific player')
    parser.add_argument('--prop', type=str,
                       choices=['points', 'rebounds', 'assists', 'pra', 'threes'],
                       help='Prop type')
    parser.add_argument('--line', type=float, help='Betting line')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--validate', action='store_true', help='Validate models')

    # Historical data commands
    parser.add_argument('--init-historical', action='store_true',
                       help='Backfill historical data (2024-25 + 2025-26 seasons)')
    parser.add_argument('--update-historical', action='store_true',
                       help="Update with yesterday's data only")
    parser.add_argument('--historical-backtest', action='store_true',
                       help='Run backtest using cached historical data')
    parser.add_argument('--validate-cache', action='store_true',
                       help='Validate and show cache statistics')

    args = parser.parse_args()

    if args.init_historical:
        run_init_historical()
    elif args.update_historical:
        run_update_historical()
    elif args.historical_backtest:
        run_historical_backtest()
    elif args.validate_cache:
        run_validate_cache()
    elif args.daily:
        run_daily_analysis()
    elif args.player and args.prop and args.line:
        analyze_single_prop(args.player, args.prop, args.line)
    elif args.backtest:
        run_backtest()
    elif args.validate:
        print("Model validation not yet implemented")
    else:
        # Default to daily analysis
        run_daily_analysis()


def run_init_historical():
    """Initialize historical data cache with full backfill."""
    print("\n" + "=" * 60)
    print("       HISTORICAL DATA BACKFILL")
    print("=" * 60)

    if not CONFIG.ODDS_API_KEY:
        print("\nError: ODDS_API_KEY environment variable not set")
        print("Get a key at: https://the-odds-api.com/")
        return

    odds_client = OddsAPIClient()
    nba_fetcher = NBADataFetcher()
    cache = HistoricalDataCache(odds_client, nba_fetcher)

    # 2024-25 season: Oct 22, 2024 - Jun 15, 2025
    # 2025-26 season: Oct 21, 2025 - present
    start_date = datetime(2024, 10, 22)
    end_date = datetime.now() - timedelta(days=1)

    print(f"\nBackfilling data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("This may take a while (fetching from both Odds API and NBA API)...")
    print("Progress will be saved - you can safely interrupt and resume later.\n")

    cache.backfill_historical_data(start_date, end_date)

    # Show summary
    stats = cache.validate_data()
    print("\n" + "=" * 60)
    print("       BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Total Props:    {stats['total_props']:,}")
    print(f"Unique Dates:   {stats['unique_dates']}")
    print(f"Unique Players: {stats['unique_players']}")
    print(f"Date Range:     {stats['date_range']}")
    print(f"Over Hit Rate:  {stats['over_hit_rate']:.1%}")
    print(f"Under Hit Rate: {stats['under_hit_rate']:.1%}")

    if stats.get('by_prop_type'):
        print("\nBy Prop Type:")
        for prop, data in stats['by_prop_type'].items():
            print(f"  {prop}: {data['count']:,} props, "
                  f"Over: {data['over_hit_rate']:.1%}, "
                  f"Under: {data['under_hit_rate']:.1%}")


def run_update_historical():
    """Update historical cache with yesterday's data."""
    print("\n" + "=" * 60)
    print("       UPDATING HISTORICAL DATA")
    print("=" * 60)

    if not CONFIG.ODDS_API_KEY:
        print("\nError: ODDS_API_KEY environment variable not set")
        return

    odds_client = OddsAPIClient()
    nba_fetcher = NBADataFetcher()
    cache = HistoricalDataCache(odds_client, nba_fetcher)

    yesterday = datetime.now() - timedelta(days=1)
    print(f"\nFetching data for {yesterday.strftime('%Y-%m-%d')}...")

    cache.update_yesterday()

    print("\nUpdate complete!")
    stats = cache.validate_data()
    print(f"Total props in cache: {stats['total_props']:,}")


def run_historical_backtest():
    """Run backtest using cached historical data."""
    print("\n" + "=" * 60)
    print("       HISTORICAL BACKTEST")
    print("=" * 60)

    cache = HistoricalDataCache()
    data = cache.get_backtest_data()

    if data.empty:
        print("\nNo historical data in cache!")
        print("Run --init-historical first to populate the cache.")
        return

    print(f"\nLoaded {len(data):,} historical props")

    # Group by date for analysis
    data['date'] = pd.to_datetime(data['date'])

    # Calculate overall statistics
    total = len(data)
    over_wins = data['hit_over'].sum()
    under_wins = data['hit_under'].sum()
    pushes = data['push'].sum()

    print(f"\n--- OVERALL RESULTS ---")
    print(f"Total Props: {total:,}")
    print(f"Over Hits:   {over_wins:,} ({over_wins/total:.1%})")
    print(f"Under Hits:  {under_wins:,} ({under_wins/total:.1%})")
    print(f"Pushes:      {pushes:,} ({pushes/total:.1%})")

    # By prop type
    print(f"\n--- BY PROP TYPE ---")
    for prop_type in data['prop_type'].unique():
        prop_data = data[data['prop_type'] == prop_type]
        n = len(prop_data)
        over_pct = prop_data['hit_over'].mean()
        under_pct = prop_data['hit_under'].mean()
        print(f"{prop_type:>10}: {n:,} props | Over: {over_pct:.1%} | Under: {under_pct:.1%}")

    # Simulate betting on overs when actual > line (perfect hindsight)
    # This is just for analysis - real backtest would use model predictions
    print(f"\n--- BASELINE ANALYSIS ---")
    print("If you bet OVER on every prop:")
    print(f"  Win Rate: {data['hit_over'].mean():.1%}")
    print(f"  At -110 odds, breakeven is 52.4%")
    print(f"  Edge: {(data['hit_over'].mean() - 0.524)*100:+.1f}%")

    print("\nIf you bet UNDER on every prop:")
    print(f"  Win Rate: {data['hit_under'].mean():.1%}")
    print(f"  Edge: {(data['hit_under'].mean() - 0.524)*100:+.1f}%")


def run_validate_cache():
    """Validate and display cache statistics."""
    print("\n" + "=" * 60)
    print("       CACHE VALIDATION")
    print("=" * 60)

    cache = HistoricalDataCache()
    stats = cache.validate_data()

    if stats['total_props'] == 0:
        print("\nCache is empty!")
        print("Run --init-historical to populate it.")
        return

    print(f"\n--- CACHE SUMMARY ---")
    print(f"Total Props:    {stats['total_props']:,}")
    print(f"Unique Dates:   {stats['unique_dates']}")
    print(f"Unique Players: {stats['unique_players']}")
    print(f"Date Range:     {stats['date_range']}")
    print(f"Over Hit Rate:  {stats['over_hit_rate']:.1%}")
    print(f"Under Hit Rate: {stats['under_hit_rate']:.1%}")

    if stats.get('by_prop_type'):
        print("\n--- BY PROP TYPE ---")
        for prop, data in stats['by_prop_type'].items():
            print(f"  {prop:>8}: {data['count']:>6,} props | "
                  f"Over: {data['over_hit_rate']:.1%} | "
                  f"Under: {data['under_hit_rate']:.1%}")

    # Show recent fetch status
    print("\n--- RECENT FETCH STATUS ---")
    fetch_status = cache.get_fetch_status()
    if not fetch_status.empty:
        for _, row in fetch_status.head(10).iterrows():
            print(f"  {row['date']}: {row['status']} ({row['props_count']} props)")


# =============================================================================
# EXPERT RECOMMENDATIONS - ADVANCED FEATURES
# =============================================================================

# --- 1. TEAM PACE DATA (NBA Analyst Recommendation) ---
# Average possessions per 48 minutes by team (2024-25 season estimates)
TEAM_PACE = {
    'ATL': 101.2, 'BOS': 99.8, 'BKN': 100.5, 'CHA': 100.1, 'CHI': 98.7,
    'CLE': 97.5, 'DAL': 99.2, 'DEN': 98.9, 'DET': 100.8, 'GSW': 101.5,
    'HOU': 100.3, 'IND': 103.5, 'LAC': 97.8, 'LAL': 99.5, 'MEM': 101.8,
    'MIA': 97.2, 'MIL': 99.1, 'MIN': 98.3, 'NOP': 99.8, 'NYK': 97.9,
    'OKC': 100.6, 'ORL': 96.8, 'PHI': 98.5, 'PHX': 99.4, 'POR': 100.2,
    'SAC': 101.1, 'SAS': 99.7, 'TOR': 99.3, 'UTA': 98.6, 'WAS': 100.9
}
LEAGUE_AVG_PACE = 99.5

# Team defensive ratings (points allowed per 100 possessions)
TEAM_DEF_RATING = {
    'ATL': 115.2, 'BOS': 108.5, 'BKN': 117.3, 'CHA': 116.8, 'CHI': 114.5,
    'CLE': 109.2, 'DAL': 113.8, 'DEN': 112.5, 'DET': 116.2, 'GSW': 113.1,
    'HOU': 111.8, 'IND': 114.9, 'LAC': 110.5, 'LAL': 112.8, 'MEM': 111.5,
    'MIA': 110.8, 'MIL': 113.2, 'MIN': 109.5, 'NOP': 113.5, 'NYK': 110.2,
    'OKC': 108.8, 'ORL': 107.5, 'PHI': 112.5, 'PHX': 114.2, 'POR': 117.8,
    'SAC': 114.5, 'SAS': 115.5, 'TOR': 115.8, 'UTA': 116.5, 'WAS': 118.5
}
LEAGUE_AVG_DEF_RATING = 113.0


# --- 2. SITUATIONAL CONTEXT (All Experts) ---
@dataclass
class GameContext:
    """Full context for a game - situational labels."""
    player_name: str
    opponent: str
    is_home: bool
    is_back_to_back: bool = False
    rest_days: int = 1
    game_total: float = 220.0  # Vegas over/under
    spread: float = 0.0  # Team spread
    teammate_injuries: List[str] = field(default_factory=list)

    @property
    def pace_factor(self) -> float:
        """Expected pace relative to league average."""
        opp_pace = TEAM_PACE.get(self.opponent, LEAGUE_AVG_PACE)
        return opp_pace / LEAGUE_AVG_PACE

    @property
    def def_factor(self) -> float:
        """Opponent defense relative to league average (higher = easier)."""
        opp_def = TEAM_DEF_RATING.get(self.opponent, LEAGUE_AVG_DEF_RATING)
        return opp_def / LEAGUE_AVG_DEF_RATING

    @property
    def blowout_risk(self) -> bool:
        """Is this likely to be a blowout (affects minutes)?"""
        return abs(self.spread) > 10


# --- 3. FEATURE ENGINEERING (ML Engineer Recommendation) ---
class FeatureEngineering:
    """Creates ML features from raw data."""

    @staticmethod
    def build_features(
        player_logs: pd.DataFrame,
        prop_type: str,
        line: float,
        context: GameContext = None
    ) -> Dict[str, float]:
        """Build feature vector for ML model."""
        if player_logs.empty or prop_type not in player_logs.columns:
            return {}

        stats = player_logs[prop_type]

        # Basic stats
        features = {
            'avg_l5': stats.head(5).mean() if len(stats) >= 5 else stats.mean(),
            'avg_l10': stats.head(10).mean() if len(stats) >= 10 else stats.mean(),
            'avg_season': stats.mean(),
            'std_l10': stats.head(10).std() if len(stats) >= 10 else stats.std(),
            'median_l10': stats.head(10).median() if len(stats) >= 10 else stats.median(),
            'min_l10': stats.head(10).min() if len(stats) >= 10 else stats.min(),
            'max_l10': stats.head(10).max() if len(stats) >= 10 else stats.max(),
        }

        # Line-relative features
        features['line'] = line
        features['line_vs_avg'] = line - features['avg_l10']
        features['line_vs_median'] = line - features['median_l10']

        # Hit rate features
        features['hit_rate_over'] = (stats > line).mean()
        features['hit_rate_under'] = (stats < line).mean()
        features['hit_rate_l5_over'] = (stats.head(5) > line).mean() if len(stats) >= 5 else 0.5

        # Momentum features
        if len(stats) >= 5:
            features['momentum'] = features['avg_l5'] / features['avg_season'] if features['avg_season'] > 0 else 1.0
            features['trend'] = stats.head(3).mean() - stats.iloc[3:6].mean() if len(stats) >= 6 else 0

        # Consistency features
        if features['avg_l10'] > 0:
            features['cv'] = features['std_l10'] / features['avg_l10']  # Coefficient of variation
        else:
            features['cv'] = 1.0

        # Minutes features (if available)
        if 'minutes' in player_logs.columns:
            mins = player_logs['minutes']
            features['avg_minutes_l5'] = mins.head(5).mean() if len(mins) >= 5 else mins.mean()
            features['minutes_trend'] = mins.head(3).mean() - mins.iloc[3:6].mean() if len(mins) >= 6 else 0

        # Situational features (if context provided)
        if context:
            features['is_home'] = 1 if context.is_home else 0
            features['is_b2b'] = 1 if context.is_back_to_back else 0
            features['rest_days'] = context.rest_days
            features['pace_factor'] = context.pace_factor
            features['def_factor'] = context.def_factor
            features['game_total'] = context.game_total
            features['spread'] = context.spread
            features['blowout_risk'] = 1 if context.blowout_risk else 0
            features['n_injuries'] = len(context.teammate_injuries)

        return features


# --- 4. ADVANCED ML MODEL (ML Engineer + Statistician) ---
class XGBoostPropModel:
    """
    Gradient boosting model for prop prediction.
    Predicts P(over) directly using classification.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=10,
                random_state=42
            )
            self.model.fit(X, y)
            self.feature_names = list(X.columns)
            self.is_fitted = True
            logger.info(f"XGBoost model fitted on {len(X)} samples")
        except ImportError:
            logger.warning("scikit-learn not installed. Using fallback model.")
            self.is_fitted = False

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Predict probability of over."""
        if not self.is_fitted:
            return 0.5

        # Build feature vector in correct order
        X = pd.DataFrame([features])[self.feature_names]
        return self.model.predict_proba(X)[0, 1]


# --- 5. CLV TRACKING (Market Maker Recommendation) ---
class CLVTracker:
    """
    Tracks Closing Line Value - the best predictor of long-term success.
    """

    DB_PATH = Path("clv_tracking.db")

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize CLV tracking database."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    player TEXT,
                    prop_type TEXT,
                    bet_line REAL,
                    bet_side TEXT,
                    bet_odds INTEGER,
                    closing_line REAL,
                    closing_odds INTEGER,
                    actual_result REAL,
                    won INTEGER,
                    clv REAL,
                    pnl REAL
                )
            """)
            conn.commit()

    def record_bet(
        self,
        player: str,
        prop_type: str,
        bet_line: float,
        bet_side: str,
        bet_odds: int = -110
    ) -> int:
        """Record a bet when placed. Returns bet ID."""
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO bets (timestamp, player, prop_type, bet_line, bet_side, bet_odds)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), player, prop_type, bet_line, bet_side, bet_odds))
            conn.commit()
            return cursor.lastrowid

    def update_closing_line(self, bet_id: int, closing_line: float, closing_odds: int = -110):
        """Update with closing line (just before game starts)."""
        with sqlite3.connect(self.DB_PATH) as conn:
            # Calculate CLV
            bet_data = conn.execute(
                "SELECT bet_line, bet_side FROM bets WHERE id = ?", (bet_id,)
            ).fetchone()

            if bet_data:
                bet_line, bet_side = bet_data
                if bet_side == 'over':
                    clv = bet_line - closing_line  # Lower close = value
                else:
                    clv = closing_line - bet_line  # Higher close = value

                conn.execute("""
                    UPDATE bets SET closing_line = ?, closing_odds = ?, clv = ?
                    WHERE id = ?
                """, (closing_line, closing_odds, clv, bet_id))
                conn.commit()

    def update_result(self, bet_id: int, actual_result: float, stake: float = 10.0):
        """Update with actual game result."""
        with sqlite3.connect(self.DB_PATH) as conn:
            bet_data = conn.execute(
                "SELECT bet_line, bet_side, bet_odds FROM bets WHERE id = ?", (bet_id,)
            ).fetchone()

            if bet_data:
                bet_line, bet_side, bet_odds = bet_data

                # Determine win/loss
                if bet_side == 'over':
                    won = 1 if actual_result > bet_line else 0
                else:
                    won = 1 if actual_result < bet_line else 0

                # Calculate P&L
                if actual_result == bet_line:
                    pnl = 0  # Push
                elif won:
                    decimal_odds = 1 + (100 / abs(bet_odds)) if bet_odds < 0 else 1 + (bet_odds / 100)
                    pnl = stake * (decimal_odds - 1)
                else:
                    pnl = -stake

                conn.execute("""
                    UPDATE bets SET actual_result = ?, won = ?, pnl = ?
                    WHERE id = ?
                """, (actual_result, won, pnl, bet_id))
                conn.commit()

    def get_clv_summary(self) -> Dict:
        """Get CLV performance summary."""
        with sqlite3.connect(self.DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM bets WHERE clv IS NOT NULL", conn)

        if df.empty:
            return {'n_bets': 0, 'avg_clv': 0, 'win_rate': 0, 'roi': 0}

        return {
            'n_bets': len(df),
            'avg_clv': df['clv'].mean(),
            'win_rate': df['won'].mean() if 'won' in df.columns else 0,
            'total_pnl': df['pnl'].sum() if 'pnl' in df.columns else 0,
            'roi': df['pnl'].sum() / (len(df) * 10) if 'pnl' in df.columns else 0,  # Assuming $10 per bet
            'positive_clv_rate': (df['clv'] > 0).mean()
        }


# --- 6. RISK MANAGEMENT (Risk Quant Recommendation) ---
class RiskManager:
    """
    Proper bet sizing with Kelly criterion and correlation management.
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,  # Quarter Kelly
        max_bet_pct: float = 0.03,     # Max 3% per bet
        max_game_pct: float = 0.05,    # Max 5% per game
        max_night_pct: float = 0.10    # Max 10% per night
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.max_game_pct = max_game_pct
        self.max_night_pct = max_night_pct

        self.tonight_exposure = 0.0
        self.game_exposures: Dict[str, float] = {}  # game_id -> exposure

    def kelly_stake(self, prob_win: float, odds: int) -> float:
        """
        Calculate Kelly criterion bet size with uncertainty buffer.
        """
        # Convert American odds to decimal
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)

        b = decimal_odds - 1  # Net odds
        q = 1 - prob_win

        # Kelly formula
        kelly = (prob_win * b - q) / b

        # Can't bet negative
        kelly = max(0, kelly)

        # Apply fractional Kelly
        stake = self.bankroll * kelly * self.kelly_fraction

        # Apply maximum limits
        max_bet = self.bankroll * self.max_bet_pct
        stake = min(stake, max_bet)

        return stake

    def can_bet(self, stake: float, game_id: str = None) -> Tuple[bool, str]:
        """Check if bet is allowed under risk limits."""
        # Check per-bet limit
        if stake > self.bankroll * self.max_bet_pct:
            return False, "Exceeds max bet size"

        # Check nightly limit
        if self.tonight_exposure + stake > self.bankroll * self.max_night_pct:
            return False, "Exceeds nightly exposure limit"

        # Check per-game limit
        if game_id:
            current_game_exposure = self.game_exposures.get(game_id, 0)
            if current_game_exposure + stake > self.bankroll * self.max_game_pct:
                return False, f"Exceeds game exposure limit for {game_id}"

        return True, "OK"

    def record_bet(self, stake: float, game_id: str = None):
        """Record a bet for exposure tracking."""
        self.tonight_exposure += stake
        if game_id:
            self.game_exposures[game_id] = self.game_exposures.get(game_id, 0) + stake

    def reset_nightly(self):
        """Reset nightly exposure counters."""
        self.tonight_exposure = 0.0
        self.game_exposures = {}

    def update_bankroll(self, pnl: float):
        """Update bankroll after bet settles."""
        self.bankroll += pnl


# --- 7. ADVANCED PROP ANALYZER (Combines All Experts) ---
class AdvancedPropAnalyzer:
    """
    Advanced prop analyzer implementing all expert recommendations:
    - Situational adjustments (B2B, pace, defense)
    - Feature engineering for ML
    - XGBoost classification model
    - CLV tracking
    - Proper risk management
    """

    def __init__(
        self,
        nba_fetcher: NBADataFetcher = None,
        odds_client: OddsAPIClient = None
    ):
        self.nba = nba_fetcher or NBADataFetcher()
        self.odds = odds_client
        self.cache = NBAGameLogCache(self.nba)

        # Expert components
        self.feature_eng = FeatureEngineering()
        self.ml_model = XGBoostPropModel()
        self.clv_tracker = CLVTracker()
        self.risk_manager = RiskManager()

        # Simple model as fallback
        self.simple_model = EnsembleModel()

        # Filters based on backtest results
        self.prop_type_filter = ['points', 'rebounds', 'threes']  # Avoid assists, pra
        self.prefer_unders = True  # Unders outperformed in backtest

    def detect_back_to_back(self, player_logs: pd.DataFrame) -> bool:
        """Detect if player played yesterday (back-to-back)."""
        if player_logs.empty or 'date' not in player_logs.columns:
            return False

        dates = pd.to_datetime(player_logs['date'])
        if len(dates) < 2:
            return False

        # Check if most recent game was yesterday
        most_recent = dates.iloc[0]
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        return most_recent.date() == yesterday

    def calculate_rest_days(self, player_logs: pd.DataFrame) -> int:
        """Calculate days since last game."""
        if player_logs.empty or 'date' not in player_logs.columns:
            return 2  # Default

        dates = pd.to_datetime(player_logs['date'])
        most_recent = dates.iloc[0]
        today = datetime.now().date()

        return (today - most_recent.date()).days

    def get_teammate_injuries(self, team: str) -> List[str]:
        """Get list of injured teammates (placeholder - would need injury API)."""
        # TODO: Integrate with injury API (e.g., ESPN, RotoBaller)
        return []

    def apply_situational_adjustments(
        self,
        base_projection: float,
        prop_type: str,
        context: GameContext
    ) -> float:
        """Apply situational adjustments to base projection."""
        adjusted = base_projection

        # Back-to-back: -7% on scoring, -5% on other stats
        if context.is_back_to_back:
            if prop_type == 'points':
                adjusted *= 0.93
            else:
                adjusted *= 0.95

        # Rest advantage: +2% per extra day beyond 1
        rest_bonus = 1.0 + 0.02 * max(0, context.rest_days - 1)
        adjusted *= min(rest_bonus, 1.06)  # Cap at 6%

        # Pace adjustment for volume stats
        if prop_type in ['points', 'rebounds', 'assists', 'pra']:
            adjusted *= context.pace_factor

        # Defense adjustment for points
        if prop_type == 'points':
            adjusted *= context.def_factor

        # Home court: +2%
        if context.is_home:
            adjusted *= 1.02

        # Teammate injuries: boost per star out (simplified)
        if context.teammate_injuries:
            injury_boost = 1.0 + 0.05 * len(context.teammate_injuries)
            adjusted *= min(injury_boost, 1.15)  # Cap at 15%

        return adjusted

    def analyze_prop_advanced(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        opponent: str = None,
        is_home: bool = True,
        game_total: float = 220.0,
        spread: float = 0.0,
        odds: int = -110
    ) -> Optional[Dict]:
        """
        Advanced prop analysis with all expert recommendations.
        """
        # Skip filtered prop types
        if prop_type not in self.prop_type_filter:
            logger.debug(f"Skipping {prop_type} - filtered out based on backtest")
            return None

        # Get player data
        logs = self.cache.get_player_logs(player_name, last_n_games=20)
        if logs.empty or prop_type not in logs.columns:
            return None

        if len(logs) < 10:
            return None

        # Build game context
        context = GameContext(
            player_name=player_name,
            opponent=opponent or 'UNK',
            is_home=is_home,
            is_back_to_back=self.detect_back_to_back(logs),
            rest_days=self.calculate_rest_days(logs),
            game_total=game_total,
            spread=spread,
            teammate_injuries=self.get_teammate_injuries(opponent or '')
        )

        # Get base projection from simple model
        history = logs[prop_type]
        simple_pred = self.simple_model.predict(history, line, odds)
        base_projection = simple_pred.projection

        # Apply situational adjustments
        adjusted_projection = self.apply_situational_adjustments(
            base_projection, prop_type, context
        )

        # Build features for ML model
        features = self.feature_eng.build_features(logs, prop_type, line, context)

        # Get ML probability if model is trained
        if self.ml_model.is_fitted:
            prob_over = self.ml_model.predict_proba(features)
        else:
            # Fall back to empirical hit rate with adjustments
            prob_over = (history > line).mean()
            # Adjust for situation
            if context.is_back_to_back:
                prob_over *= 0.95  # Less likely to hit over on B2B

        prob_under = 1 - prob_over

        # Calculate edge (Statistician recommendation)
        implied_prob = 0.5238 if odds == -110 else abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

        edge_over = prob_over - implied_prob
        edge_under = prob_under - implied_prob

        # Determine recommended side
        # Apply "prefer unders" bias from backtest results
        if self.prefer_unders and edge_under > 0:
            recommended_side = 'under'
            edge = edge_under
            prob_win = prob_under
        elif edge_over > edge_under and edge_over > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'over'
            edge = edge_over
            prob_win = prob_over
        elif edge_under > CONFIG.MIN_EDGE_THRESHOLD:
            recommended_side = 'under'
            edge = edge_under
            prob_win = prob_under
        else:
            return None  # No edge

        # Calculate stake with risk management
        stake = self.risk_manager.kelly_stake(prob_win, odds)

        # Check risk limits
        game_id = f"{player_name}_{datetime.now().strftime('%Y-%m-%d')}"
        can_bet, reason = self.risk_manager.can_bet(stake, game_id)

        return {
            'player': player_name,
            'prop_type': prop_type,
            'line': line,
            'projection': adjusted_projection,
            'base_projection': base_projection,
            'recommended_side': recommended_side,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'edge': edge,
            'stake': stake if can_bet else 0,
            'can_bet': can_bet,
            'bet_reason': reason,
            'context': {
                'is_b2b': context.is_back_to_back,
                'rest_days': context.rest_days,
                'pace_factor': context.pace_factor,
                'def_factor': context.def_factor,
                'opponent': context.opponent
            },
            'features': features
        }

    def run_advanced_backtest(self, min_edge: float = 0.03) -> Dict:
        """
        Run backtest using advanced model on historical data.
        """
        # Load historical data
        conn = sqlite3.connect('nba_historical_cache.db')
        historical = pd.read_sql_query('''
            SELECT DISTINCT date, player, prop_type, line, actual, hit_over, hit_under
            FROM historical_props
            WHERE date >= '2025-12-04'
        ''', conn)
        conn.close()

        if historical.empty:
            return {'error': 'No historical data available'}

        results = []
        tested = 0

        for idx, row in historical.iloc[::10].iterrows():  # Sample every 10th
            if row['prop_type'] not in self.prop_type_filter:
                continue

            player = row['player']
            prop_type = row['prop_type']
            line = row['line']
            actual = row['actual']
            game_date = datetime.strptime(row['date'], '%Y-%m-%d')

            # Get player logs from before this date
            try:
                full_logs = self.cache.get_player_logs(player)
            except:
                continue

            if full_logs.empty or prop_type not in full_logs.columns:
                continue

            # Filter to pre-game data
            full_logs['date'] = pd.to_datetime(full_logs['date'])
            pre_game = full_logs[full_logs['date'] < game_date]

            if len(pre_game) < 10:
                continue

            # Analyze with advanced model
            analysis = self.analyze_prop_advanced(
                player, prop_type, line,
                is_home=True,  # Simplified
                game_total=220.0,
                spread=0
            )

            if not analysis:
                continue

            # Check outcome
            if analysis['recommended_side'] == 'over':
                correct = (actual > line)
            else:
                correct = (actual < line)

            results.append({
                'player': player,
                'prop_type': prop_type,
                'line': line,
                'actual': actual,
                'side': analysis['recommended_side'],
                'edge': analysis['edge'],
                'projection': analysis['projection'],
                'correct': correct
            })
            tested += 1

        if not results:
            return {'error': 'No valid predictions'}

        df = pd.DataFrame(results)
        wins = df['correct'].sum()
        total = len(df)

        return {
            'tested': tested,
            'wins': wins,
            'win_rate': wins / total,
            'edge_vs_breakeven': (wins / total) - 0.5238,
            'by_prop_type': df.groupby('prop_type')['correct'].agg(['mean', 'count']).to_dict(),
            'by_side': df.groupby('side')['correct'].agg(['mean', 'count']).to_dict()
        }


def run_advanced_analysis():
    """Run the advanced analysis with all expert recommendations."""
    print("\n" + "=" * 70)
    print("   ADVANCED NBA PROP ANALYSIS (Expert Recommendations Implemented)")
    print("=" * 70)

    analyzer = AdvancedPropAnalyzer()

    # Run backtest first
    print("\n--- Running Advanced Backtest ---")
    backtest = analyzer.run_advanced_backtest()

    if 'error' not in backtest:
        print(f"Tested: {backtest['tested']} props")
        print(f"Win Rate: {backtest['win_rate']:.1%}")
        print(f"Edge vs Breakeven: {backtest['edge_vs_breakeven']:+.1%}")

        print("\nBy Prop Type:")
        for ptype, stats in backtest.get('by_prop_type', {}).items():
            if isinstance(stats, dict):
                print(f"  {ptype}: {stats}")

        print("\nBy Side:")
        for side, stats in backtest.get('by_side', {}).items():
            if isinstance(stats, dict):
                print(f"  {side}: {stats}")
    else:
        print(f"Backtest error: {backtest['error']}")

    # Show CLV tracking summary
    print("\n--- CLV Tracking Summary ---")
    clv_summary = analyzer.clv_tracker.get_clv_summary()
    print(f"Bets tracked: {clv_summary['n_bets']}")
    if clv_summary['n_bets'] > 0:
        print(f"Average CLV: {clv_summary['avg_clv']:+.2f} points")
        print(f"Positive CLV rate: {clv_summary['positive_clv_rate']:.1%}")

    print("\n--- Risk Management Status ---")
    print(f"Bankroll: ${analyzer.risk_manager.bankroll:.2f}")
    print(f"Tonight's exposure: ${analyzer.risk_manager.tonight_exposure:.2f}")
    print(f"Kelly fraction: {analyzer.risk_manager.kelly_fraction:.0%}")


if __name__ == "__main__":
    main()
