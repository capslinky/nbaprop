"""
DEPRECATED: Import from 'models' module instead.

This file remains for backward compatibility only. All implementations
have been moved to the models/ package.

Example:
    # Old (deprecated):
    from nba_prop_model import UnifiedPropModel

    # New (preferred):
    from models import UnifiedPropModel
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'nba_prop_model' is deprecated. "
    "Import from 'models' module instead: from models import UnifiedPropModel",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from models for backward compatibility
from models import (
    # Data classes
    Prediction,
    PropAnalysis,

    # Models
    WeightedAverageModel,
    SituationalModel,
    MedianModel,
    EnsembleModel,
    SmartModel,
    UnifiedPropModel,

    # Backtesting
    BetResult,
    Backtester,

    # Generators
    generate_player_season_data,
    generate_sample_dataset,
    generate_prop_lines,
)

# Re-export utilities that were previously in this file
from core.constants import (
    TEAM_ABBREVIATIONS,
    normalize_team_abbrev,
    get_current_nba_season,
)
from core.odds_utils import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_ev,
)

# Backward compatibility alias
TEAM_ABBREV_MAP = TEAM_ABBREVIATIONS


def main():
    """Demo backtest - kept for backward compatibility."""
    print("NBA Prop Analysis & Backtesting System")
    print("="*50)

    # Generate sample data
    print("\n[1/5] Generating player game logs...")
    game_logs = generate_sample_dataset()
    print(f"  Generated {len(game_logs):,} game logs for {game_logs['player'].nunique()} players")

    # Generate prop lines
    print("\n[2/5] Generating historical prop lines...")
    props = generate_prop_lines(game_logs)
    print(f"  Generated {len(props):,} prop betting opportunities")

    # Initialize models
    print("\n[3/5] Initializing prediction models...")
    models = {
        'Weighted Average': WeightedAverageModel(),
        'Median': MedianModel(),
        'Ensemble': EnsembleModel(),
    }
    print(f"  Loaded {len(models)} models")

    # Run backtests
    print("\n[4/5] Running backtests...")
    all_results = {}

    for name, model in models.items():
        print(f"\n  Testing: {name}")
        backtester = Backtester(initial_bankroll=1000, unit_size=10)
        results = backtester.run_backtest(
            props, game_logs, model,
            min_edge=0.03,
            min_confidence=0.35
        )
        all_results[name] = {
            'backtester': backtester,
            'results': results,
            'metrics': backtester.get_metrics()
        }

        m = all_results[name]['metrics']
        print(f"    Bets: {m['total_bets']} | Win Rate: {m['win_rate']*100:.1f}% | ROI: {m['roi']*100:+.2f}%")

    # Detailed report for best model
    print("\n[5/5] Generating detailed report for best model...")

    best_model = max(all_results.keys(),
                     key=lambda x: all_results[x]['metrics'].get('roi', -999))

    print(f"\n  Best Performing Model: {best_model}")
    all_results[best_model]['backtester'].print_report()

    # Model comparison
    print("\n" + "="*60)
    print("              MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Bets':>8} {'Win%':>8} {'ROI':>10} {'Profit':>12}")
    print("-"*60)

    for name in models.keys():
        m = all_results[name]['metrics']
        print(f"{name:<20} {m['total_bets']:>8} {m['win_rate']*100:>7.1f}% {m['roi']*100:>9.2f}% ${m['total_profit']:>10.2f}")

    print("\n")

    return game_logs, props, all_results


if __name__ == "__main__":
    game_logs, props, results = main()
