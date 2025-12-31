"""Calibration helpers for model probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import json


@dataclass
class CalibrationParams:
    slope: float
    intercept: float
    samples: int
    mean_prob: float
    mean_outcome: float


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _outcome(row: Dict) -> Optional[int]:
    result = row.get("result")
    if result == "WIN":
        return 1
    if result == "LOSS":
        return 0
    return None


def _probability(row: Dict, key: str) -> Optional[float]:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 1:
        return None
    return value


def _normalize_prop_type(prop_type: Optional[str]) -> str:
    prop = (prop_type or "").lower()
    if prop in ("pts", "points"):
        return "points"
    if prop in ("reb", "rebounds"):
        return "rebounds"
    if prop in ("ast", "assists"):
        return "assists"
    if prop in ("threes", "3pt", "fg3m"):
        return "threes"
    if prop in ("pra", "pts_reb_ast"):
        return "pra"
    if prop in ("stl", "steals"):
        return "steals"
    if prop in ("blk", "blocks"):
        return "blocks"
    if prop in ("tov", "turnovers"):
        return "turnovers"
    return prop or "unknown"


def _fit_linear(probs: Iterable[float], outcomes: Iterable[int]) -> Optional[CalibrationParams]:
    probs_list = list(probs)
    outcomes_list = list(outcomes)
    if not probs_list or len(probs_list) != len(outcomes_list):
        return None
    n = len(probs_list)
    mean_prob = sum(probs_list) / n
    mean_outcome = sum(outcomes_list) / n
    var_prob = sum((p - mean_prob) ** 2 for p in probs_list)
    if var_prob <= 1e-6:
        slope = 1.0
    else:
        cov = sum((p - mean_prob) * (y - mean_outcome) for p, y in zip(probs_list, outcomes_list))
        slope = cov / var_prob
    slope = _clamp(slope, 0.3, 1.7)
    intercept = mean_outcome - slope * mean_prob
    intercept = max(-0.3, min(0.3, intercept))
    return CalibrationParams(
        slope=round(slope, 6),
        intercept=round(intercept, 6),
        samples=n,
        mean_prob=round(mean_prob, 6),
        mean_outcome=round(mean_outcome, 6),
    )


def fit_linear_calibration(
    rows: Iterable[Dict],
    prob_key: str = "raw_model_probability",
    min_samples: int = 30,
    prop_key: str = "prop_type",
) -> Dict:
    """Fit a linear calibration per prop type, with a global fallback."""
    grouped_probs: Dict[str, list] = {}
    grouped_outcomes: Dict[str, list] = {}
    all_probs: list = []
    all_outcomes: list = []

    for row in rows:
        outcome = _outcome(row)
        prob = _probability(row, prob_key)
        if outcome is None or prob is None:
            continue
        prop_type = _normalize_prop_type(row.get(prop_key))
        grouped_probs.setdefault(prop_type, []).append(prob)
        grouped_outcomes.setdefault(prop_type, []).append(outcome)
        all_probs.append(prob)
        all_outcomes.append(outcome)

    global_params = _fit_linear(all_probs, all_outcomes)
    by_prop_type: Dict[str, Dict] = {}
    for prop_type, probs in grouped_probs.items():
        outcomes = grouped_outcomes.get(prop_type, [])
        if len(probs) < min_samples:
            continue
        params = _fit_linear(probs, outcomes)
        if params:
            by_prop_type[prop_type] = params.__dict__

    calibration = {
        "generated_at": datetime.utcnow().isoformat(),
        "prob_key": prob_key,
        "min_samples": min_samples,
        "global": global_params.__dict__ if global_params else None,
        "by_prop_type": by_prop_type,
    }
    return calibration


def apply_calibration(
    prob: float,
    prop_type: Optional[str],
    calibration: Optional[Dict],
) -> Tuple[float, Optional[Dict]]:
    """Apply calibration if available, returning (prob, metadata)."""
    if calibration is None:
        return prob, None
    prop_key = _normalize_prop_type(prop_type)
    params = None
    source = None
    by_prop = calibration.get("by_prop_type") if isinstance(calibration, dict) else None
    if by_prop and prop_key in by_prop:
        params = by_prop.get(prop_key)
        source = prop_key
    if not params:
        params = calibration.get("global") if isinstance(calibration, dict) else None
        source = "global"
    if not params:
        return prob, None
    slope = params.get("slope", 1.0)
    intercept = params.get("intercept", 0.0)
    calibrated = _clamp((slope * prob) + intercept)
    metadata = {
        "source": source,
        "slope": slope,
        "intercept": intercept,
    }
    return calibrated, metadata


def resolve_calibration_path(
    calibration_path: Optional[str],
    cache_dir: Optional[str],
    repo_root: Optional[Path] = None,
) -> Path:
    if calibration_path:
        path = Path(calibration_path)
    else:
        base = Path(cache_dir or ".cache")
        path = base / "calibration" / "calibration_params.json"
    if not path.is_absolute() and repo_root:
        path = repo_root / path
    return path


def load_calibration(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_calibration(path: Path, calibration: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
