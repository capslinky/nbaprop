"""
Prediction dataclass for model outputs.

Simple container for model predictions with projection, confidence, edge, and recommendation.
"""

from dataclasses import dataclass


@dataclass
class Prediction:
    """Container for model predictions."""
    projection: float
    confidence: float  # 0-1 scale
    edge: float  # Expected edge over the line
    recommended_side: str  # 'over', 'under', or 'pass'
    model_name: str
