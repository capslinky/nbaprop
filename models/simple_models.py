"""
Simple prediction models for NBA prop analysis.

Contains baseline models: WeightedAverageModel, SituationalModel, MedianModel.
"""

import numpy as np
import pandas as pd

from core.config import CONFIG
from models.prediction import Prediction


class WeightedAverageModel:
    """
    Weighted average model with recency bias.
    Simple but effective baseline model.
    """

    def __init__(self, weights: list = None):
        # Default: more weight on recent games
        self.weights = weights or [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.03]
        self.name = "WeightedAverage"

    def predict(self, history: pd.Series, line: float) -> Prediction:
        """Predict based on weighted recent performance."""
        recent = history.tail(len(self.weights)).values

        if len(recent) < 5:
            return Prediction(np.nan, 0, 0, 'pass', self.name)

        # Apply weights (reversed so most recent gets highest weight)
        weights = self.weights[:len(recent)][::-1]
        weights = np.array(weights) / sum(weights)

        projection = np.average(recent, weights=weights)
        std = np.std(recent)

        # Calculate edge
        edge = (projection - line) / line if line > 0 else 0

        # Confidence based on consistency
        cv = std / projection if projection > 0 else 1
        confidence = max(0, min(1, 1 - cv))

        # Determine recommendation
        if abs(edge) < CONFIG.MIN_EDGE_THRESHOLD:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'

        return Prediction(round(projection, 1), round(confidence, 3),
                         round(edge, 4), side, self.name)


class SituationalModel:
    """
    Model that adjusts projections based on situational factors:
    - Home/away
    - Back-to-back games
    - Opponent defensive rating
    """

    def __init__(self):
        self.name = "Situational"
        self.base_model = WeightedAverageModel()

        # Situational adjustment factors from config
        self.home_boost = CONFIG.HOME_BOOST
        self.away_penalty = CONFIG.AWAY_PENALTY
        self.b2b_penalty = CONFIG.B2B_PENALTY
        self.def_rtg_factor = 0.008  # Per point above/below 112

    def predict(self, history: pd.DataFrame, current_game: dict,
                prop_type: str, line: float) -> Prediction:
        """Predict with situational adjustments."""

        base_pred = self.base_model.predict(history[prop_type], line)

        if np.isnan(base_pred.projection):
            return base_pred

        # Apply adjustments
        projection = base_pred.projection

        if current_game.get('home', False):
            projection *= self.home_boost
        else:
            projection *= self.away_penalty

        if current_game.get('b2b', False):
            projection *= self.b2b_penalty

        # Opponent defense adjustment
        opp_def = current_game.get('opp_def_rtg', 112)
        def_adjustment = 1 + (opp_def - 112) * self.def_rtg_factor
        projection *= def_adjustment

        edge = (projection - line) / line if line > 0 else 0

        if abs(edge) < CONFIG.MIN_EDGE_THRESHOLD:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'

        return Prediction(round(projection, 1), base_pred.confidence,
                         round(edge, 4), side, self.name)


class MedianModel:
    """
    Uses median instead of mean - more robust to outliers.
    Good for volatile players.
    """

    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.name = "Median"

    def predict(self, history: pd.Series, line: float) -> Prediction:
        recent = history.tail(self.lookback)

        if len(recent) < 5:
            return Prediction(np.nan, 0, 0, 'pass', self.name)

        projection = recent.median()
        std = recent.std()

        edge = (projection - line) / line if line > 0 else 0

        cv = std / projection if projection > 0 else 1
        confidence = max(0, min(1, 1 - cv))

        if abs(edge) < CONFIG.MIN_EDGE_THRESHOLD:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'

        return Prediction(round(projection, 1), round(confidence, 3),
                         round(edge, 4), side, self.name)
