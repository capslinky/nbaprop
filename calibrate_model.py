#!/usr/bin/env python3
"""
Model Calibration CLI
=====================

Calibrates UnifiedPropModel adjustment factors based on historical pick performance.

The calibration system:
1. Analyzes historical picks from clv_tracking.db
2. Calculates win rates when each adjustment factor is active vs inactive
3. Determines optimal factor values based on performance
4. Applies conservative changes (50% of calculated adjustment)
5. Saves learned weights to data/learned_weights.json

Usage:
    # Run full calibration (recommended: weekly)
    python calibrate_model.py --calibrate

    # Analyze without saving (dry run)
    python calibrate_model.py --analyze

    # Show current weights vs defaults
    python calibrate_model.py --show

    # Reset to defaults
    python calibrate_model.py --reset

    # Calibrate with custom settings
    python calibrate_model.py --calibrate --days 60 --min-samples 30

Examples:
    # Weekly calibration (run on Sundays)
    python calibrate_model.py --calibrate

    # Check if calibration is needed
    python calibrate_model.py --analyze

    # View current state
    python calibrate_model.py --show
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from core.config import CONFIG


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 40)


def run_analyze(days: int, min_samples: int = None):
    """Run analysis without saving results."""
    from calibration import CalibrationAnalyzer, WeightOptimizer

    print_header("MODEL CALIBRATION ANALYSIS")

    analyzer = CalibrationAnalyzer()

    # Check data availability
    summary = analyzer.get_data_summary()
    if not summary.get('exists'):
        print("\nNo pick data found. Run some picks first using:")
        print("  python pick_tracker.py --record <picks_file.csv>")
        return False

    print(f"\nData Summary:")
    print(f"  Total picks:         {summary['total_picks']}")
    print(f"  Picks with results:  {summary['picks_with_results']}")
    if summary.get('date_range', (None, None))[0]:
        print(f"  Date range:          {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"  Sufficient data:     {'Yes' if summary.get('sufficient_for_calibration') else 'No'}")

    if not summary.get('sufficient_for_calibration'):
        print(f"\nNeed at least {CONFIG.CALIBRATION_MIN_TOTAL_PICKS} picks with results for calibration.")
        print(f"Currently have {summary['picks_with_results']}.")
        return False

    # Run analysis
    min_samples = min_samples or CONFIG.CALIBRATION_MIN_SAMPLES
    resolved_days = None if days <= 0 else days
    result = analyzer.calculate_all_factor_stats(days=resolved_days)

    period_label = "all available data" if resolved_days is None else f"last {resolved_days} days"
    print(f"\nAnalysis Period: {period_label}")
    print(f"  Picks analyzed:      {result.picks_with_results}")
    print(f"  Overall win rate:    {result.overall_win_rate:.1%}")

    # Calculate optimizations
    optimizer = WeightOptimizer(min_samples=min_samples)
    optimized = optimizer.optimize_all_factors(result)

    print_section("FACTOR ANALYSIS")
    print(f"{'Factor':<25} {'Current':>8} {'Optimal':>8} {'New':>8} {'Change':>8} {'Samples':>8} {'Quality':<12}")
    print("-" * 90)

    for factor_name, opt in optimized.items():
        change_str = f"{opt.change_pct*100:+.1f}%" if opt.was_adjusted else "---"
        print(
            f"{factor_name:<25} "
            f"{opt.current_value:>8.3f} "
            f"{opt.optimal_value:>8.3f} "
            f"{opt.new_value:>8.3f} "
            f"{change_str:>8} "
            f"{opt.sample_size:>8} "
            f"{opt.quality:<12}"
        )

    print_section("RECOMMENDATIONS")
    adjusted_count = sum(1 for opt in optimized.values() if opt.was_adjusted)
    if adjusted_count > 0:
        print(f"  {adjusted_count} factor(s) would be adjusted.")
        print(f"  Conservative factor: {CONFIG.CALIBRATION_CONSERVATIVE_FACTOR:.0%}")
        print(f"  Max change per run:  {CONFIG.CALIBRATION_MAX_CHANGE:.0%}")
        print("\n  Run with --calibrate to apply changes.")
    else:
        print("  No factors need adjustment at this time.")

    return True


def run_calibrate(days: int, min_samples: int = None, dry_run: bool = False):
    """Run calibration and save results."""
    from calibration import CalibrationAnalyzer, WeightOptimizer, LearnedWeightsStore

    print_header("MODEL CALIBRATION")

    analyzer = CalibrationAnalyzer()

    # Check data
    if not analyzer.has_sufficient_data():
        print("\nInsufficient data for calibration.")
        print(f"Need at least {CONFIG.CALIBRATION_MIN_TOTAL_PICKS} picks with results.")
        return False

    # Run analysis
    min_samples = min_samples or CONFIG.CALIBRATION_MIN_SAMPLES
    resolved_days = None if days <= 0 else days
    result = analyzer.calculate_all_factor_stats(days=resolved_days)

    print(f"\nData Summary:")
    print(f"  Picks with results:  {result.picks_with_results}")
    print(f"  Date range:          {result.date_range[0]} to {result.date_range[1]}")
    print(f"  Overall win rate:    {result.overall_win_rate:.1%}")

    # Optimize
    optimizer = WeightOptimizer(min_samples=min_samples)
    optimized = optimizer.optimize_all_factors(result)

    print_section("CALIBRATION RESULTS")
    print(f"{'Factor':<25} {'Current':>8} {'New':>8} {'Change':>8} {'Reason':<30}")
    print("-" * 90)

    for factor_name, opt in optimized.items():
        change_str = f"{opt.change_pct*100:+.1f}%" if opt.was_adjusted else "---"
        reason = opt.reason[:28] + ".." if len(opt.reason) > 30 else opt.reason
        print(
            f"{factor_name:<25} "
            f"{opt.current_value:>8.3f} "
            f"{opt.new_value:>8.3f} "
            f"{change_str:>8} "
            f"{reason:<30}"
        )

    # Check if any changes
    adjusted_count = sum(1 for opt in optimized.values() if opt.was_adjusted)
    if adjusted_count == 0:
        print("\nNo factors were adjusted. Model is already well-calibrated.")
        return True

    # Save results
    if dry_run:
        print(f"\nDry run - {adjusted_count} factor(s) would be saved.")
        return True

    weights = optimizer.create_learned_weights(optimized, result)
    store = LearnedWeightsStore()

    if store.save(weights):
        print(f"\nSaved calibrated weights to: {store.path}")
        print(f"  {adjusted_count} factor(s) adjusted")
        print(f"  Conservative factor: {CONFIG.CALIBRATION_CONSERVATIVE_FACTOR:.0%}")
        return True
    else:
        print("\nError saving weights!")
        return False


def run_show():
    """Show current weights vs defaults."""
    from calibration import LearnedWeightsStore

    print_header("CURRENT WEIGHTS")

    store = LearnedWeightsStore()
    has_learned = store.load() and store.is_valid()

    if has_learned:
        age = store.get_calibration_age_days()
        metadata = store.get_metadata()
        print(f"\nLearned weights found:")
        print(f"  File: {store.path}")
        print(f"  Age:  {age} days")
        if metadata:
            print(f"  Based on: {metadata.get('picks_with_results', '?')} picks")
            print(f"  Win rate: {metadata.get('overall_win_rate', 0):.1%}")
    else:
        print("\nNo learned weights found. Using CONFIG defaults.")

    print_section("ADJUSTMENT FACTORS")
    print(f"{'Factor':<25} {'Default':>10} {'Current':>10} {'Change':>10}")
    print("-" * 60)

    factors = [
        'HOME_BOOST',
        'AWAY_PENALTY',
        'B2B_PENALTY',
        'BLOWOUT_HIGH_PENALTY',
        'BLOWOUT_MEDIUM_PENALTY',
        'TOTAL_WEIGHT',
        'TREND_MULTIPLIER',
    ]

    for factor_name in factors:
        default = getattr(CONFIG, factor_name, None)
        if default is None:
            continue

        if has_learned:
            current = store.get_factor(factor_name, default)
            change = (current - default) / abs(default) if default != 0 else 0
            change_str = f"{change*100:+.1f}%" if abs(change) > 0.001 else "---"
        else:
            current = default
            change_str = "---"

        print(f"{factor_name:<25} {default:>10.3f} {current:>10.3f} {change_str:>10}")

    if has_learned:
        print_section("FACTOR DETAILS")
        for factor_name in factors:
            info = store.get_factor_info(factor_name)
            if info and info.get('was_adjusted'):
                print(f"\n{factor_name}:")
                print(f"  Sample size:     {info.get('sample_size', '?')}")
                print(f"  Win rate active: {info.get('win_rate_active', 0):.1%}" if info.get('win_rate_active') else "")
                print(f"  Quality:         {info.get('quality', '?')}")

    return True


def run_reset():
    """Reset to CONFIG defaults by removing learned weights."""
    from calibration import LearnedWeightsStore

    print_header("RESET WEIGHTS")

    store = LearnedWeightsStore()

    if not store.path.exists():
        print("\nNo learned weights to reset. Already using defaults.")
        return True

    print(f"\nThis will delete: {store.path}")
    confirm = input("Are you sure? (y/N): ").strip().lower()

    if confirm != 'y':
        print("Cancelled.")
        return False

    if store.clear():
        print("Learned weights cleared. Model will use CONFIG defaults.")
        return True
    else:
        print("Error clearing weights!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='NBA Prop Model Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibrate_model.py --calibrate      # Run full calibration
  python calibrate_model.py --analyze        # Analyze without saving
  python calibrate_model.py --show           # Show current weights
  python calibrate_model.py --reset          # Reset to defaults
        """
    )

    # Actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--calibrate', action='store_true',
        help='Run calibration and save weights'
    )
    action_group.add_argument(
        '--analyze', action='store_true',
        help='Analyze data without saving (dry run)'
    )
    action_group.add_argument(
        '--show', action='store_true',
        help='Show current weights vs defaults'
    )
    action_group.add_argument(
        '--reset', action='store_true',
        help='Reset to CONFIG defaults'
    )

    # Options
    parser.add_argument(
        '--days', type=int, default=CONFIG.CALIBRATION_LOOKBACK_DAYS,
        help=(
            f'Days of history to analyze (default: {CONFIG.CALIBRATION_LOOKBACK_DAYS}, '
            'use 0 for all data)'
        )
    )
    parser.add_argument(
        '--min-samples', type=int, default=CONFIG.CALIBRATION_MIN_SAMPLES,
        help=f'Minimum samples per factor (default: {CONFIG.CALIBRATION_MIN_SAMPLES})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='With --calibrate: show what would be saved without saving'
    )

    args = parser.parse_args()

    # Run appropriate action
    success = False
    try:
        if args.calibrate:
            success = run_calibrate(args.days, args.min_samples, args.dry_run)
        elif args.analyze:
            success = run_analyze(args.days, args.min_samples)
        elif args.show:
            success = run_show()
        elif args.reset:
            success = run_reset()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
