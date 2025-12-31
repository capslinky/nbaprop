"""
Weight Optimizer
================
Calculates optimal adjustment weights with bounds and conservatism.

The optimizer:
1. Takes factor performance statistics from the analyzer
2. Calculates optimal values based on historical win rates
3. Applies conservatism (only uses fraction of calculated change)
4. Enforces bounds to prevent extreme values
5. Limits per-run changes to avoid wild swings

Usage:
    from calibration.optimizer import WeightOptimizer
    from calibration.analyzer import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer()
    result = analyzer.calculate_all_factor_stats()

    optimizer = WeightOptimizer()
    optimized = optimizer.optimize_all_factors(result)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from core.config import CONFIG
from calibration.analyzer import CalibrationResult, FactorStats
from calibration.weight_store import LearnedWeights, FactorData

logger = logging.getLogger(__name__)


@dataclass
class OptimizedFactor:
    """Result of optimizing a single factor."""
    factor_name: str
    current_value: float  # Current default from CONFIG
    optimal_value: float  # Calculated optimal (before conservatism)
    new_value: float  # Final value (after conservatism and bounds)
    change_pct: float  # Percentage change from current
    sample_size: int
    quality: str
    was_adjusted: bool  # Whether we made any change
    reason: str  # Why we did/didn't adjust


class WeightOptimizer:
    """
    Calculates optimal adjustment weights with safety guards.

    Safety features:
    - Minimum sample size required before adjusting
    - Conservative factor (only apply fraction of change)
    - Bounds to prevent extreme values
    - Maximum per-run change limit
    """

    def __init__(
        self,
        conservative_factor: float = None,
        max_change: float = None,
        bounds: Dict[str, Tuple[float, float]] = None,
        min_samples: int = None,
    ):
        """
        Initialize the optimizer.

        Args:
            conservative_factor: How much of calculated change to apply (0-1).
                                 Default from CONFIG.CALIBRATION_CONSERVATIVE_FACTOR
            max_change: Maximum change per factor per run.
                        Default from CONFIG.CALIBRATION_MAX_CHANGE
            bounds: Dictionary mapping factor names to (min, max) tuples.
                    Default from CONFIG.CALIBRATION_BOUNDS
            min_samples: Minimum samples required to adjust a factor.
                         Default from CONFIG.CALIBRATION_MIN_SAMPLES
        """
        self.conservative_factor = conservative_factor or CONFIG.CALIBRATION_CONSERVATIVE_FACTOR
        self.max_change = max_change or CONFIG.CALIBRATION_MAX_CHANGE
        self.bounds = bounds or CONFIG.CALIBRATION_BOUNDS
        self.min_samples = min_samples or CONFIG.CALIBRATION_MIN_SAMPLES

    def _get_bounds(self, factor_name: str) -> Tuple[float, float]:
        """Get bounds for a factor, with defaults if not specified."""
        if factor_name in self.bounds:
            return self.bounds[factor_name]

        # Default bounds based on factor type
        current = getattr(CONFIG, factor_name, 1.0)
        if current > 1.0:
            # Boost factor (like HOME_BOOST)
            return (1.0, 1.0 + (current - 1.0) * 2)
        elif current < 1.0:
            # Penalty factor (like B2B_PENALTY)
            return (1.0 - (1.0 - current) * 2, 1.0)
        else:
            # Weight factor
            return (0.1, 0.9)

    def _clamp_to_bounds(self, value: float, factor_name: str) -> float:
        """Clamp a value to the configured bounds."""
        min_val, max_val = self._get_bounds(factor_name)
        if value < min_val:
            logger.debug(f"{factor_name}: {value:.4f} clamped to min {min_val}")
            return min_val
        if value > max_val:
            logger.debug(f"{factor_name}: {value:.4f} clamped to max {max_val}")
            return max_val
        return value

    def _limit_change(self, current: float, new: float) -> float:
        """Limit the change to max_change percentage."""
        if current == 0:
            return new

        change_pct = abs(new - current) / abs(current)
        if change_pct > self.max_change:
            direction = 1 if new > current else -1
            limited = current * (1 + direction * self.max_change)
            logger.debug(
                f"Change limited from {new:.4f} to {limited:.4f} "
                f"(max {self.max_change*100:.1f}%)"
            )
            return limited
        return new

    def optimize_factor(
        self,
        factor_name: str,
        stats: FactorStats,
        current_value: float = None,
    ) -> OptimizedFactor:
        """
        Calculate optimal value for a single factor.

        Algorithm:
        1. Check if sample size is sufficient
        2. Calculate improvement ratio from win rates
        3. Determine optimal adjustment direction and magnitude
        4. Apply conservative factor (only use fraction of change)
        5. Clamp to bounds
        6. Limit to max per-run change

        Args:
            factor_name: Name of the factor
            stats: FactorStats from analyzer
            current_value: Current value. Defaults to CONFIG value.

        Returns:
            OptimizedFactor with the calculated values
        """
        if current_value is None:
            current_value = getattr(CONFIG, factor_name, 1.0)

        # Check sample size
        if stats.active_count < self.min_samples:
            return OptimizedFactor(
                factor_name=factor_name,
                current_value=current_value,
                optimal_value=current_value,
                new_value=current_value,
                change_pct=0.0,
                sample_size=stats.active_count,
                quality=stats.quality,
                was_adjusted=False,
                reason=f"Insufficient samples ({stats.active_count} < {self.min_samples})",
            )

        # Calculate optimal adjustment based on improvement
        # If active_win_rate > inactive_win_rate, the adjustment is helping
        if stats.active_win_rate is None or stats.inactive_win_rate is None:
            return OptimizedFactor(
                factor_name=factor_name,
                current_value=current_value,
                optimal_value=current_value,
                new_value=current_value,
                change_pct=0.0,
                sample_size=stats.active_count,
                quality=stats.quality,
                was_adjusted=False,
                reason="Missing win rate data",
            )

        # Calculate how effective the adjustment is
        improvement = stats.improvement  # active_win_rate - inactive_win_rate

        # Determine optimal value based on factor type
        optimal_value = self._calculate_optimal_value(
            factor_name, current_value, improvement, stats
        )

        # Apply conservative factor
        # Only move (conservative_factor) of the way from current to optimal
        conservative_value = current_value + (optimal_value - current_value) * self.conservative_factor

        # Clamp to bounds
        bounded_value = self._clamp_to_bounds(conservative_value, factor_name)

        # Limit per-run change
        final_value = self._limit_change(current_value, bounded_value)

        # Calculate change percentage
        change_pct = (final_value - current_value) / abs(current_value) if current_value != 0 else 0

        # Ensure native Python types (not numpy) for JSON serialization
        was_adjusted = bool(abs(change_pct) > 0.001)  # More than 0.1% change

        return OptimizedFactor(
            factor_name=factor_name,
            current_value=current_value,
            optimal_value=optimal_value,
            new_value=final_value,
            change_pct=change_pct,
            sample_size=stats.active_count,
            quality=stats.quality,
            was_adjusted=was_adjusted,
            reason="Calibrated" if was_adjusted else "No significant change needed",
        )

    def _calculate_optimal_value(
        self,
        factor_name: str,
        current_value: float,
        improvement: float,
        stats: FactorStats,
    ) -> float:
        """
        Calculate the optimal value for a factor based on historical performance.

        The logic varies by factor type:
        - Penalty factors (B2B, BLOWOUT): If picks with penalty applied win less,
          consider increasing the penalty (lower value)
        - Boost factors (HOME): If picks with boost applied win more,
          consider increasing the boost (higher value)
        """
        # Scale factor for adjustments (dampened to avoid over-fitting)
        # A 5% improvement in win rate suggests a ~1% adjustment
        adjustment_scale = 0.2

        if factor_name in ['B2B_PENALTY', 'BLOWOUT_HIGH_PENALTY', 'BLOWOUT_MEDIUM_PENALTY']:
            # Penalty factors: current_value < 1.0
            # If picks with penalty have LOWER win rate (negative improvement),
            # the penalty is working - we might need MORE penalty (lower value)
            # If improvement is positive, penalty might be too harsh
            if improvement < 0:
                # Penalty is working, but maybe not enough
                # Decrease the value (more penalty) proportionally
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            else:
                # Penalty might be too harsh, increase value (less penalty)
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            return optimal

        elif factor_name in ['HOME_BOOST']:
            # Boost factors: current_value > 1.0
            # If picks at home have HIGHER win rate (positive improvement),
            # boost is working - might need more boost
            if improvement > 0:
                # Boost is working, increase it
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            else:
                # Boost might be too high, decrease it
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            return optimal

        elif factor_name in ['AWAY_PENALTY']:
            # Similar to penalty factors
            # If away picks have lower win rate, penalty is correct
            if improvement < 0:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            else:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment
            return optimal

        elif factor_name in ['TOTAL_WEIGHT', 'TREND_MULTIPLIER']:
            # Weight factors: adjust magnitude based on correlation with wins
            # If high-total adjustments correlate with wins, increase weight
            if improvement > 0:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.1  # Smaller adjustments for weights
            else:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.1
            return optimal

        elif factor_name in ['USAGE_FACTOR_WEIGHT', 'TS_REGRESSION_WEIGHT']:
            # Weight factors for new v2.0 features
            # Positive improvement means factor is helping, increase weight
            if improvement > 0:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.1
            else:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.1
            return optimal

        elif factor_name in ['MAX_USAGE_ADJUSTMENT', 'MAX_SHOT_VOLUME_ADJUSTMENT', 'MAX_TS_ADJUSTMENT']:
            # Cap factors: adjust the maximum adjustment allowed
            # If factor is helping (positive improvement), allow more adjustment room
            if improvement > 0:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.05  # Very conservative changes to caps
            else:
                adjustment = improvement * adjustment_scale
                optimal = current_value + adjustment * 0.05
            return max(0.01, optimal)  # Never go below 1%

        else:
            # Default: no change
            return current_value

    def optimize_all_factors(
        self,
        calibration_result: CalibrationResult,
    ) -> Dict[str, OptimizedFactor]:
        """
        Optimize all factors from a calibration result.

        Args:
            calibration_result: Result from CalibrationAnalyzer

        Returns:
            Dictionary mapping factor names to OptimizedFactor
        """
        optimized = {}

        for factor_name, stats in calibration_result.factor_stats.items():
            optimized[factor_name] = self.optimize_factor(factor_name, stats)

        return optimized

    def create_learned_weights(
        self,
        optimized_factors: Dict[str, OptimizedFactor],
        calibration_result: CalibrationResult,
    ) -> LearnedWeights:
        """
        Create a LearnedWeights object from optimized factors.

        Args:
            optimized_factors: Result from optimize_all_factors()
            calibration_result: Original calibration result

        Returns:
            LearnedWeights ready to be saved
        """
        adjustment_factors = {}

        for factor_name, opt in optimized_factors.items():
            # Ensure all values are native Python types for JSON serialization
            adjustment_factors[factor_name] = {
                'value': float(opt.new_value),
                'default': float(opt.current_value),
                'change_pct': float(opt.change_pct * 100),  # Store as percentage
                'sample_size': int(opt.sample_size),
                'quality': str(opt.quality),
                'was_adjusted': bool(opt.was_adjusted),
                'reason': str(opt.reason),
            }

            # Add win rate data if available
            stats = calibration_result.factor_stats.get(factor_name)
            if stats:
                if stats.active_win_rate is not None:
                    adjustment_factors[factor_name]['win_rate_active'] = float(stats.active_win_rate)
                if stats.inactive_win_rate is not None:
                    adjustment_factors[factor_name]['win_rate_inactive'] = float(stats.inactive_win_rate)

        # Ensure metadata values are native Python types
        metadata = {
            'total_picks': int(calibration_result.total_picks),
            'picks_with_results': int(calibration_result.picks_with_results),
            'date_range': list(calibration_result.date_range),
            'overall_win_rate': float(calibration_result.overall_win_rate) if calibration_result.overall_win_rate else 0.0,
            'conservative_factor': float(self.conservative_factor),
            'max_change': float(self.max_change),
            'min_samples': int(self.min_samples),
        }

        return LearnedWeights(
            version="1.0",
            calibrated_at=datetime.now().isoformat(),
            adjustment_factors=adjustment_factors,
            metadata=metadata,
        )
