"""
Models module - Prediction models for NBA prop analysis.

This module provides:
- Prediction: Simple prediction container (projection, confidence, edge, side)
- PropAnalysis: Comprehensive analysis result with full context
- Various model implementations for generating predictions

Available Models:
- UnifiedPropModel: Production model with 9 contextual adjustments
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

# Re-export from nba_prop_model.py
# This maintains backward compatibility while establishing the new module structure
from nba_prop_model import (
    # Data classes (output formats)
    Prediction,
    PropAnalysis,

    # Models
    WeightedAverageModel,
    SituationalModel,
    MedianModel,
    EnsembleModel,
    SmartModel,
    UnifiedPropModel,

    # Backtesting
    BetResult,
    Backtester,
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

    # Utilities (convenience re-exports)
    'calculate_ev',
    'calculate_edge',
    'calculate_confidence',
    'kelly_criterion',
]
