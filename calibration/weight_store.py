"""
Learned Weights Store
=====================
JSON persistence for learned calibration weights with fallback to CONFIG defaults.

Usage:
    from calibration.weight_store import LearnedWeightsStore

    store = LearnedWeightsStore()
    if store.load():
        home_boost = store.get_factor('HOME_BOOST', CONFIG.HOME_BOOST)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class FactorData:
    """Data for a single calibrated factor."""
    value: float
    default: float
    sample_size: int
    win_rate_active: Optional[float] = None
    win_rate_inactive: Optional[float] = None
    change_pct: float = 0.0
    quality: str = "unknown"  # high, medium, low, insufficient_data


@dataclass
class LearnedWeights:
    """Complete learned weights structure."""
    version: str = "1.0"
    calibrated_at: str = ""
    adjustment_factors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'calibrated_at': self.calibrated_at,
            'adjustment_factors': self.adjustment_factors,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LearnedWeights':
        """Create from dictionary."""
        return cls(
            version=data.get('version', '1.0'),
            calibrated_at=data.get('calibrated_at', ''),
            adjustment_factors=data.get('adjustment_factors', {}),
            metadata=data.get('metadata', {}),
        )


class LearnedWeightsStore:
    """
    JSON persistence for learned weights with fallback to CONFIG defaults.

    The store handles:
    - Loading weights from JSON file
    - Saving calibrated weights
    - Falling back to CONFIG defaults when weights are missing/invalid
    - Validating weight structure and age
    """

    DEFAULT_PATH = Path(__file__).parent.parent / "data" / "learned_weights.json"
    MAX_AGE_DAYS = 30  # Weights older than this trigger a warning

    def __init__(self, path: Path = None):
        """
        Initialize the weights store.

        Args:
            path: Optional custom path for weights file. Defaults to data/learned_weights.json
        """
        self.path = path or self.DEFAULT_PATH
        self._weights: Optional[LearnedWeights] = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load weights from JSON file.

        Returns:
            True if weights were loaded successfully, False otherwise
        """
        if not self.path.exists():
            logger.debug(f"No learned weights file at {self.path}")
            self._weights = None
            self._loaded = True
            return False

        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            self._weights = LearnedWeights.from_dict(data)
            self._loaded = True
            logger.info(f"Loaded learned weights from {self.path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in weights file: {e}")
            self._weights = None
            self._loaded = True
            return False
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            self._weights = None
            self._loaded = True
            return False

    def save(self, weights: LearnedWeights) -> bool:
        """
        Save weights to JSON file.

        Args:
            weights: LearnedWeights object to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Update calibration timestamp
            weights.calibrated_at = datetime.now().isoformat()

            with open(self.path, 'w') as f:
                json.dump(weights.to_dict(), f, indent=2)

            self._weights = weights
            logger.info(f"Saved learned weights to {self.path}")
            return True
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            return False

    def get_factor(self, factor_name: str, default: float) -> float:
        """
        Get a single factor value with fallback to default.

        Args:
            factor_name: Name of the factor (e.g., 'HOME_BOOST')
            default: Default value to return if factor not found

        Returns:
            The learned factor value, or default if not available
        """
        if not self._loaded:
            self.load()

        if self._weights is None:
            return default

        factor_data = self._weights.adjustment_factors.get(factor_name)
        if factor_data is None:
            return default

        # Check quality - don't use insufficient data weights
        quality = factor_data.get('quality', 'unknown')
        if quality == 'insufficient_data':
            return default

        value = factor_data.get('value')
        if value is None:
            return default

        return float(value)

    def get_all_factors(self) -> Dict[str, float]:
        """
        Get all adjustment factors as a dictionary.

        Returns:
            Dictionary mapping factor names to their values
        """
        if not self._loaded:
            self.load()

        if self._weights is None:
            return {}

        result = {}
        for name, data in self._weights.adjustment_factors.items():
            if data.get('quality') != 'insufficient_data':
                value = data.get('value')
                if value is not None:
                    result[name] = float(value)

        return result

    def get_factor_info(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full information about a factor including statistics.

        Args:
            factor_name: Name of the factor

        Returns:
            Dictionary with factor data, or None if not found
        """
        if not self._loaded:
            self.load()

        if self._weights is None:
            return None

        return self._weights.adjustment_factors.get(factor_name)

    def is_valid(self) -> bool:
        """
        Check if loaded weights are valid and not too old.

        Returns:
            True if weights are valid and usable
        """
        if not self._loaded:
            self.load()

        if self._weights is None:
            return False

        # Check version
        if self._weights.version != "1.0":
            logger.warning(f"Unknown weights version: {self._weights.version}")
            return False

        # Check age
        if self._weights.calibrated_at:
            try:
                calibrated = datetime.fromisoformat(self._weights.calibrated_at)
                age_days = (datetime.now() - calibrated).days
                if age_days > self.MAX_AGE_DAYS:
                    logger.warning(
                        f"Learned weights are {age_days} days old "
                        f"(max recommended: {self.MAX_AGE_DAYS})"
                    )
                    # Still valid, just old - return True but log warning
            except ValueError:
                pass

        # Check structure
        if not self._weights.adjustment_factors:
            logger.warning("No adjustment factors in weights file")
            return False

        return True

    def get_calibration_age_days(self) -> Optional[int]:
        """
        Get the age of the current calibration in days.

        Returns:
            Number of days since calibration, or None if no weights loaded
        """
        if not self._loaded:
            self.load()

        if self._weights is None or not self._weights.calibrated_at:
            return None

        try:
            calibrated = datetime.fromisoformat(self._weights.calibrated_at)
            return (datetime.now() - calibrated).days
        except ValueError:
            return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get calibration metadata.

        Returns:
            Dictionary with metadata about the calibration
        """
        if not self._loaded:
            self.load()

        if self._weights is None:
            return {}

        return self._weights.metadata

    def clear(self) -> bool:
        """
        Delete the weights file and reset to defaults.

        Returns:
            True if cleared successfully
        """
        try:
            if self.path.exists():
                self.path.unlink()
            self._weights = None
            self._loaded = False
            logger.info("Cleared learned weights")
            return True
        except Exception as e:
            logger.error(f"Error clearing weights: {e}")
            return False

    def __repr__(self) -> str:
        status = "loaded" if self._weights else "empty"
        age = self.get_calibration_age_days()
        age_str = f", {age} days old" if age is not None else ""
        return f"LearnedWeightsStore({status}{age_str})"
