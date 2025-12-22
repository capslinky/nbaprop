"""Baseline model stub."""

from typing import List, Dict


class BaselineModel:
    def fit(self, rows: List[Dict]) -> "BaselineModel":
        return self

    def predict(self, rows: List[Dict]) -> List[Dict]:
        raise NotImplementedError
