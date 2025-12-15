"""
Base model interface for NBA prop prediction models.

This module defines the standard interface that all prediction models should implement.
Used for creating consistent model behavior and enabling easy model swapping.

Usage:
    from models.base import BaseModel, StandardPrediction

    class MyModel(BaseModel):
        name = "my_model"

        def predict(self, history, line, odds=-110, **kwargs):
            # Implement prediction logic
            return StandardPrediction(...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd


@dataclass
class StandardPrediction:
    """
    Standardized prediction output format.

    All values use consistent scales:
    - edge: Always decimal (0.05 = 5% edge)
    - confidence: Always 0-1 scale
    - probabilities: Always 0-1 scale
    - recommended_side: Always lowercase ('over', 'under', 'pass')
    """
    # Core prediction
    projection: float
    edge: float  # Decimal (0.05 = 5%)
    confidence: float  # 0-1 scale
    recommended_side: str  # 'over', 'under', 'pass'

    # Statistical context
    std_error: float = 0.0
    prob_over: float = 0.5
    prob_under: float = 0.5

    # Confidence interval (95%)
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    # Metadata
    model_name: str = "unknown"
    flags: List[str] = field(default_factory=list)

    @property
    def edge_percent(self) -> float:
        """Return edge as percentage for display."""
        return self.edge * 100

    @property
    def confidence_percent(self) -> float:
        """Return confidence as percentage for display."""
        return self.confidence * 100

    def to_display_dict(self) -> dict:
        """
        Convert to display format with user-friendly values.
        Percentages, uppercase pick, etc.
        """
        return {
            'Projection': round(self.projection, 1),
            'Edge': f"{self.edge_percent:+.1f}%",
            'Confidence': f"{self.confidence_percent:.0f}%",
            'Pick': self.recommended_side.upper(),
            'Model': self.model_name,
        }


class BaseModel(ABC):
    """
    Abstract base class for prediction models.

    All prediction models should inherit from this class and implement
    the predict() method with the standard signature.
    """

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        history: pd.Series,
        line: float,
        odds: int = -110,
        **kwargs
    ) -> StandardPrediction:
        """
        Generate a prediction for a prop.

        Args:
            history: Series of recent stat values (most recent last)
            line: The betting line
            odds: American odds (default -110)
            **kwargs: Additional context (opponent, is_home, is_b2b, etc.)

        Returns:
            StandardPrediction with projection, edge, confidence, and recommendation
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
