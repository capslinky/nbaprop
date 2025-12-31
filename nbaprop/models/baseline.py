"""Baseline model stub."""

from typing import List, Dict, Optional

from nbaprop.models.scoring import score_prop


class BaselineModel:
    def __init__(
        self,
        min_edge: float = 0.03,
        min_confidence: float = 0.4,
        prop_type_min_edge: Optional[Dict[str, float]] = None,
        prop_type_min_confidence: Optional[Dict[str, float]] = None,
        excluded_prop_types: Optional[List[str]] = None,
        injury_risk_edge_multiplier: float = 1.0,
        injury_risk_confidence_multiplier: float = 1.0,
        market_blend: float = 0.7,
        calibration: Optional[Dict] = None,
        odds_min: Optional[int] = None,
        odds_max: Optional[int] = None,
        odds_confidence_min: Optional[int] = None,
        odds_confidence_max: Optional[int] = None,
        confidence_cap_outside_band: Optional[float] = 0.65,
    ) -> None:
        self._min_edge = min_edge
        self._min_confidence = min_confidence
        self._prop_type_min_edge = prop_type_min_edge or {}
        self._prop_type_min_confidence = prop_type_min_confidence or {}
        self._excluded_prop_types = excluded_prop_types or []
        self._injury_risk_edge_multiplier = injury_risk_edge_multiplier
        self._injury_risk_confidence_multiplier = injury_risk_confidence_multiplier
        self._market_blend = market_blend
        self._calibration = calibration
        self._odds_min = odds_min
        self._odds_max = odds_max
        self._odds_confidence_min = odds_confidence_min
        self._odds_confidence_max = odds_confidence_max
        self._confidence_cap_outside_band = confidence_cap_outside_band

    def fit(self, rows: List[Dict]) -> "BaselineModel":
        return self

    def predict(self, rows: List[Dict]) -> List[Dict]:
        return [
            score_prop(
                row,
                min_edge=self._min_edge,
                min_confidence=self._min_confidence,
                prop_type_min_edge=self._prop_type_min_edge,
                prop_type_min_confidence=self._prop_type_min_confidence,
                excluded_prop_types=self._excluded_prop_types,
                injury_risk_edge_multiplier=self._injury_risk_edge_multiplier,
                injury_risk_confidence_multiplier=self._injury_risk_confidence_multiplier,
                market_blend=self._market_blend,
                calibration=self._calibration,
                odds_min=self._odds_min,
                odds_max=self._odds_max,
                odds_confidence_min=self._odds_confidence_min,
                odds_confidence_max=self._odds_confidence_max,
                confidence_cap_outside_band=self._confidence_cap_outside_band,
            )
            for row in rows
        ]
