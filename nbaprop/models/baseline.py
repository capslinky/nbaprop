"""Baseline model stub."""

from typing import List, Dict

from nbaprop.models.scoring import score_prop


class BaselineModel:
    def __init__(self, min_edge: float = 0.03, min_confidence: float = 0.4) -> None:
        self._min_edge = min_edge
        self._min_confidence = min_confidence

    def fit(self, rows: List[Dict]) -> "BaselineModel":
        return self

    def predict(self, rows: List[Dict]) -> List[Dict]:
        return [
            score_prop(row, min_edge=self._min_edge, min_confidence=self._min_confidence)
            for row in rows
        ]
