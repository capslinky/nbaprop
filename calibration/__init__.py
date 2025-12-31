"""
Calibration Module
==================
Model calibration learning loop for the NBA prop analysis system.

This module provides tools to:
- Analyze historical pick performance by adjustment factor
- Calculate optimal factor weights based on win rates
- Persist learned weights with fallback to CONFIG defaults
- Integrate with UnifiedPropModel for adaptive predictions

Usage:
    # Analyze historical performance
    from calibration import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer()
    result = analyzer.calculate_all_factor_stats(days=90)
    print(f"Overall win rate: {result.overall_win_rate:.1%}")

    # Optimize factors
    from calibration import WeightOptimizer

    optimizer = WeightOptimizer()
    optimized = optimizer.optimize_all_factors(result)
    for name, opt in optimized.items():
        if opt.was_adjusted:
            print(f"{name}: {opt.current_value:.3f} -> {opt.new_value:.3f}")

    # Save learned weights
    from calibration import LearnedWeightsStore

    weights = optimizer.create_learned_weights(optimized, result)
    store = LearnedWeightsStore()
    store.save(weights)

    # Load learned weights (used by UnifiedPropModel)
    store = LearnedWeightsStore()
    if store.load() and store.is_valid():
        home_boost = store.get_factor('HOME_BOOST', CONFIG.HOME_BOOST)

CLI:
    # Run full calibration
    python calibrate_model.py --calibrate

    # Analyze without saving (dry run)
    python calibrate_model.py --analyze

    # Show current weights
    python calibrate_model.py --show

    # Reset to defaults
    python calibrate_model.py --reset
"""

from calibration.weight_store import LearnedWeightsStore, LearnedWeights, FactorData
from calibration.analyzer import CalibrationAnalyzer, CalibrationResult, FactorStats
from calibration.optimizer import WeightOptimizer, OptimizedFactor

__all__ = [
    # Weight store
    'LearnedWeightsStore',
    'LearnedWeights',
    'FactorData',

    # Analyzer
    'CalibrationAnalyzer',
    'CalibrationResult',
    'FactorStats',

    # Optimizer
    'WeightOptimizer',
    'OptimizedFactor',
]
