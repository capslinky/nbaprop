"""Baseline model stub."""

from typing import List, Dict

from nbaprop.models.scoring import score_prop


class BaselineModel:
    def fit(self, rows: List[Dict]) -> "BaselineModel":
        return self

    def predict(self, rows: List[Dict]) -> List[Dict]:
        return [score_prop(row) for row in rows]
