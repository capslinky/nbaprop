#!/usr/bin/env python3
"""
NBA Prop System Validation Script
=================================
Comprehensive health check for all system components.

Run this before daily analysis to ensure everything is working.

Usage:
    python validate_system.py
    python validate_system.py --verbose
    python validate_system.py --test-player "Luka Doncic"
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_mark(success: bool) -> str:
    """Return colored check or X mark."""
    if success:
        return f"{Colors.GREEN}OK{Colors.END}"
    return f"{Colors.RED}FAIL{Colors.END}"


def warn_mark() -> str:
    """Return colored warning mark."""
    return f"{Colors.YELLOW}WARN{Colors.END}"


def print_header(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print("-" * 50)


def validate_imports() -> Tuple[bool, List[str]]:
    """Check that all required modules can be imported."""
    errors = []
    required_modules = [
        ('pandas', 'pip install pandas'),
        ('numpy', 'pip install numpy'),
        ('scipy', 'pip install scipy'),
        ('requests', 'pip install requests'),
    ]

    for module, install_cmd in required_modules:
        try:
            __import__(module)
        except ImportError:
            errors.append(f"{module} not found. Install with: {install_cmd}")

    # Check project modules
    try:
        from data import NBADataFetcher, OddsAPIClient, InjuryTracker
    except ImportError as e:
        errors.append(f"data module import error: {e}")

    try:
        from models import UnifiedPropModel, PropAnalysis
    except ImportError as e:
        errors.append(f"models module import error: {e}")

    try:
        from nba_quickstart import CONFIG
    except ImportError as e:
        errors.append(f"nba_quickstart import error: {e}")

    return len(errors) == 0, errors


def validate_nba_api(verbose: bool = False) -> Tuple[bool, str, float]:
    """Test NBA API connection."""
    try:
        from data import NBADataFetcher

        start = time.time()
        fetcher = NBADataFetcher()

        # Test with a known player
        logs = fetcher.get_player_game_logs("LeBron James", last_n_games=5)
        elapsed = time.time() - start

        if logs.empty:
            return False, "No data returned", elapsed

        if verbose:
            print(f"    Fetched {len(logs)} games for LeBron James")
            print(f"    Columns: {list(logs.columns)[:5]}...")

        return True, f"Connected ({elapsed:.1f}s)", elapsed

    except Exception as e:
        return False, str(e), 0


def validate_odds_api(verbose: bool = False) -> Tuple[bool, str, int]:
    """Test Odds API connection."""
    try:
        from nba_quickstart import CONFIG
        from data import OddsAPIClient

        api_key = CONFIG.get('ODDS_API_KEY') or os.environ.get('ODDS_API_KEY')

        if not api_key:
            return False, "No API key configured", 0

        client = OddsAPIClient(api_key=api_key)
        events = client.get_events()

        remaining = client.remaining_requests or 0

        if events is None:
            return False, "No events returned", remaining

        if verbose:
            print(f"    Found {len(events)} upcoming games")
            if events:
                print(f"    Example: {events[0].get('away_team')} @ {events[0].get('home_team')}")

        return True, f"Connected ({remaining} calls remaining)", remaining

    except Exception as e:
        return False, str(e), 0


def validate_defense_data(verbose: bool = False) -> Tuple[bool, str, int]:
    """Test defense data loading."""
    try:
        from data import NBADataFetcher

        fetcher = NBADataFetcher()
        defense_data = fetcher.get_team_defense_vs_position()

        if defense_data is None or defense_data.empty:
            return False, "No data returned", 0

        team_count = len(defense_data)

        if verbose:
            print(f"    Loaded {team_count} teams")
            print(f"    Columns: {list(defense_data.columns)[:5]}...")
            print(f"    Example: {defense_data.iloc[0]['team_abbrev']}")

        if team_count < 30:
            return False, f"Only {team_count} teams (expected 30)", team_count

        return True, f"Loaded ({team_count} teams)", team_count

    except Exception as e:
        return False, str(e), 0


def validate_pace_data(verbose: bool = False) -> Tuple[bool, str, int]:
    """Test pace data loading."""
    try:
        from data import NBADataFetcher

        fetcher = NBADataFetcher()
        pace_data = fetcher.get_team_pace()

        if pace_data is None or pace_data.empty:
            return False, "No data returned", 0

        team_count = len(pace_data)

        if verbose:
            print(f"    Loaded {team_count} teams")
            if 'pace_factor' in pace_data.columns:
                avg_pace = pace_data['pace_factor'].mean()
                print(f"    Avg pace factor: {avg_pace:.3f}")

        return True, f"Loaded ({team_count} teams)", team_count

    except Exception as e:
        return False, str(e), 0


def validate_injury_tracker(verbose: bool = False) -> Tuple[bool, str]:
    """Test injury tracker."""
    try:
        from data import InjuryTracker

        tracker = InjuryTracker()

        # Test player status
        status = tracker.get_player_status("LeBron James")

        if verbose:
            print(f"    LeBron James status: {status}")

        return True, "Initialized"

    except Exception as e:
        return False, str(e)


def validate_unified_model(test_player: str = "Luka Doncic",
                          verbose: bool = False) -> Tuple[bool, str, dict]:
    """Test UnifiedPropModel with full context analysis."""
    try:
        from models import UnifiedPropModel

        model = UnifiedPropModel()

        # Run analysis
        analysis = model.analyze(
            player_name=test_player,
            prop_type="points",
            line=32.5,
            last_n_games=15
        )

        if analysis.games_analyzed == 0:
            return False, f"No game data for {test_player}", {}

        # Collect context coverage info
        context = {
            'games_analyzed': analysis.games_analyzed,
            'opponent_detected': analysis.opponent is not None,
            'home_away_detected': analysis.is_home is not None,
            'b2b_checked': True,  # Always runs
            'defense_adjustment': analysis.adjustments.get('opp_defense', 0) != 0,
            'pace_adjustment': analysis.adjustments.get('pace', 0) != 0,
            'context_quality': analysis.context_quality,
            'warnings': analysis.warnings,
            'active_adjustments': sum(1 for v in analysis.adjustments.values() if v != 0),
        }

        if verbose:
            print(f"    Player: {test_player}")
            print(f"    Games analyzed: {context['games_analyzed']}")
            print(f"    Opponent: {analysis.opponent or 'Unknown'}")
            print(f"    Home/Away: {'Home' if analysis.is_home else 'Away' if analysis.is_home is False else 'Unknown'}")
            print(f"    B2B: {analysis.is_b2b}")
            print(f"    Matchup: {analysis.matchup_rating}")
            print(f"    Context Quality: {context['context_quality']}/100")
            print(f"    Active Adjustments: {context['active_adjustments']}/9")
            if analysis.warnings:
                print(f"    Warnings: {len(analysis.warnings)}")
                for w in analysis.warnings[:3]:
                    print(f"      - {w}")

        return True, f"Analyzed (quality: {context['context_quality']}/100)", context

    except Exception as e:
        import traceback
        if verbose:
            traceback.print_exc()
        return False, str(e), {}


def validate_pick_tracker(verbose: bool = False) -> Tuple[bool, str]:
    """Test pick tracker database."""
    try:
        from pick_tracker import PickTracker

        tracker = PickTracker()

        # Check if we can query
        pending = tracker.get_picks_awaiting_results()

        if verbose:
            print(f"    Pending picks: {len(pending)}")

        return True, f"Connected ({len(pending)} pending)"

    except ImportError:
        return False, "pick_tracker.py not found"
    except Exception as e:
        return False, str(e)


def run_validation(test_player: str = "Luka Doncic", verbose: bool = False):
    """Run full system validation."""
    print("\n" + "=" * 60)
    print("          NBA PROP SYSTEM VALIDATION")
    print("=" * 60)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_passed = True
    warnings = []

    # 1. Imports
    print_header("1. Module Imports")
    success, errors = validate_imports()
    print(f"  [{check_mark(success)}] Required modules")
    if not success:
        all_passed = False
        for err in errors:
            print(f"       {Colors.RED}{err}{Colors.END}")

    # 2. NBA API
    print_header("2. NBA Stats API")
    success, msg, elapsed = validate_nba_api(verbose)
    print(f"  [{check_mark(success)}] Connection: {msg}")
    if not success:
        all_passed = False

    # 3. Odds API
    print_header("3. Odds API")
    success, msg, remaining = validate_odds_api(verbose)
    if success:
        print(f"  [{check_mark(success)}] Connection: {msg}")
        remaining_int = int(remaining) if remaining else 0
        if remaining_int < 1000:
            warnings.append(f"Low API calls remaining: {remaining}")
            print(f"  [{warn_mark()}] Low calls remaining: {remaining}")
    else:
        print(f"  [{check_mark(success)}] Connection: {msg}")
        if "No API key" in msg:
            warnings.append("Odds API not configured")
        else:
            all_passed = False

    # 4. Defense Data
    print_header("4. Team Defense Data")
    success, msg, count = validate_defense_data(verbose)
    print(f"  [{check_mark(success)}] {msg}")
    if not success:
        all_passed = False

    # 5. Pace Data
    print_header("5. Team Pace Data")
    success, msg, count = validate_pace_data(verbose)
    print(f"  [{check_mark(success)}] {msg}")
    if not success:
        all_passed = False

    # 6. Injury Tracker
    print_header("6. Injury Tracker")
    success, msg = validate_injury_tracker(verbose)
    print(f"  [{check_mark(success)}] {msg}")
    if not success:
        warnings.append("Injury tracker issues")

    # 7. Unified Model Test
    print_header("7. UnifiedPropModel Test")
    success, msg, context = validate_unified_model(test_player, verbose)
    print(f"  [{check_mark(success)}] {msg}")
    if not success:
        all_passed = False
    else:
        # Check context quality
        quality = context.get('context_quality', 0)
        if quality < 50:
            print(f"  [{warn_mark()}] Low context quality: {quality}/100")
            warnings.append(f"Low context quality for test player: {quality}/100")

        active = context.get('active_adjustments', 0)
        if active < 3:
            print(f"  [{warn_mark()}] Few adjustments active: {active}/9")
            warnings.append(f"Only {active}/9 adjustments active")

        model_warnings = context.get('warnings', [])
        if model_warnings:
            for w in model_warnings[:3]:
                print(f"  [{warn_mark()}] {w}")

    # 8. Pick Tracker
    print_header("8. Pick Tracker Database")
    success, msg = validate_pick_tracker(verbose)
    print(f"  [{check_mark(success)}] {msg}")
    if not success:
        warnings.append("Pick tracker not available")

    # Summary
    print_header("SUMMARY")
    if all_passed and not warnings:
        print(f"  {Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.END}")
        print("  System is ready for analysis.")
    elif all_passed:
        print(f"  {Colors.YELLOW}{Colors.BOLD}Passed with warnings{Colors.END}")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  {Colors.RED}{Colors.BOLD}Some checks failed{Colors.END}")
        print("  Review errors above before running analysis.")

    print("\n" + "=" * 60 + "\n")

    return all_passed


def run_context_coverage_test(player: str = "Luka Doncic", prop_type: str = "points", line: float = 32.5):
    """
    Detailed context coverage test showing exactly what data was used.
    """
    from models import UnifiedPropModel

    print("\n" + "=" * 60)
    print("          CONTEXT COVERAGE TEST")
    print("=" * 60)
    print(f"  Player: {player}")
    print(f"  Prop: {prop_type} {line}")
    print()

    model = UnifiedPropModel()
    analysis = model.analyze(player, prop_type, line)

    # Call explain() method
    analysis.explain()

    # Additional quality assessment
    print("\nQUALITY ASSESSMENT")
    print("-" * 30)
    if analysis.is_high_quality():
        print(f"  {Colors.GREEN}HIGH QUALITY{Colors.END} - Pick can be trusted")
    else:
        print(f"  {Colors.YELLOW}LOW QUALITY{Colors.END} - Missing context, proceed with caution")

    return analysis


def main():
    parser = argparse.ArgumentParser(description='NBA Prop System Validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--test-player', type=str, default="Luka Doncic",
                       help='Player to use for model test')
    parser.add_argument('--context-test', action='store_true',
                       help='Run detailed context coverage test')
    parser.add_argument('--prop-type', type=str, default='points',
                       help='Prop type for context test')
    parser.add_argument('--line', type=float, default=32.5,
                       help='Line for context test')

    args = parser.parse_args()

    if args.context_test:
        run_context_coverage_test(args.test_player, args.prop_type, args.line)
    else:
        run_validation(args.test_player, args.verbose)


if __name__ == "__main__":
    main()
