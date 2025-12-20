"""
Models module - Prediction models for NBA prop analysis.

This module provides:
- Prediction: Simple prediction container (projection, confidence, edge, side)
- PropAnalysis: Comprehensive analysis result with full context
- Various model implementations for generating predictions

Available Models:
- UnifiedPropModel: Production model with 12 contextual adjustments
- WeightedAverageModel: Recency-weighted baseline
- SituationalModel: Adjusts for home/away, B2B, opponent
- MedianModel: Outlier-resistant median-based
- EnsembleModel: Combines multiple models
- SmartModel: Advanced analysis with trend and hit rate

Usage:
    from models import UnifiedPropModel, Prediction, PropAnalysis

    model = UnifiedPropModel()
    analysis = model.analyze("Luka Doncic", "points", 32.5)
    print(f"Projection: {analysis.projection}, Edge: {analysis.edge:.1%}")
"""

# Import from submodules
from models.prediction import Prediction
from models.prop_analysis import PropAnalysis
from models.simple_models import WeightedAverageModel, SituationalModel, MedianModel
from models.ensemble import EnsembleModel, SmartModel
from models.unified import UnifiedPropModel
from models.backtesting import BetResult, Backtester
from models.generators import (
    generate_player_season_data,
    generate_sample_dataset,
    generate_prop_lines,
)

# Import from core for utilities
from core.odds_utils import (
    calculate_ev,
    calculate_edge,
    calculate_confidence,
    kelly_criterion,
)

__all__ = [
    # Data classes
    'Prediction',
    'PropAnalysis',

    # Models
    'WeightedAverageModel',
    'SituationalModel',
    'MedianModel',
    'EnsembleModel',
    'SmartModel',
    'UnifiedPropModel',

    # Backtesting
    'BetResult',
    'Backtester',

    # Generators
    'generate_player_season_data',
    'generate_sample_dataset',
    'generate_prop_lines',

    # Utilities (convenience re-exports)
    'calculate_ev',
    'calculate_edge',
    'calculate_confidence',
    'kelly_criterion',
]
