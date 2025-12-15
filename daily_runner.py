#!/usr/bin/env python3
"""
NBA Prop Daily Runner - Automated Workflow
===========================================
Automates the entire daily prop analysis workflow:

1. PRE-GAME (run before games, e.g., 10 AM):
   - Validate system health
   - Run daily analysis
   - Auto-record picks to database
   - Generate report

2. POST-GAME (run after games, e.g., 11 PM):
   - Fetch actual results from NBA API
   - Record results to database
   - Generate accuracy report

Usage:
    # Full pre-game workflow
    python daily_runner.py --pre-game

    # Full post-game workflow
    python daily_runner.py --post-game

    # Just fetch and record results for a specific date
    python daily_runner.py --fetch-results --date 2024-12-10

    # Run everything (pre + post for testing)
    python daily_runner.py --full

    # Show what would run without executing
    python daily_runner.py --pre-game --dry-run

Cron Setup:
    # Add to crontab (crontab -e):
    # Pre-game at 10:00 AM daily
    0 10 * * * cd /path/to/nbaprop && python3 daily_runner.py --pre-game >> logs/daily.log 2>&1

    # Post-game at 11:30 PM daily
    30 23 * * * cd /path/to/nbaprop && python3 daily_runner.py --post-game >> logs/daily.log 2>&1
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DailyRunner:
    """Orchestrates the full daily prop analysis workflow."""

    def __init__(self, verbose: bool = True, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.log_lines = []
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        self.log_lines.append(formatted)
        if self.verbose:
            print(formatted, flush=True)

    def log_section(self, title: str):
        """Log a section header."""
        self.log("")
        self.log("=" * 60)
        self.log(f"  {title}")
        self.log("=" * 60)

    # =========================================================================
    # PRE-GAME WORKFLOW
    # =========================================================================

    def validate_system(self) -> Tuple[bool, dict]:
        """Run system validation and return status."""
        self.log_section("SYSTEM VALIDATION")

        if self.dry_run:
            self.log("Would validate system health", "DRY-RUN")
            return True, {}

        try:
            from validate_system import (
                validate_imports,
                validate_nba_api,
                validate_odds_api,
                validate_defense_data,
                validate_pace_data,
            )

            results = {}

            # Check imports
            success, errors = validate_imports()
            results['imports'] = success
            self.log(f"Module imports: {'OK' if success else 'FAIL'}")
            if not success:
                for err in errors:
                    self.log(f"  - {err}", "ERROR")

            # Check NBA API
            success, msg, elapsed = validate_nba_api()
            results['nba_api'] = success
            self.log(f"NBA API: {msg}")

            # Check Odds API
            success, msg, remaining = validate_odds_api()
            results['odds_api'] = success
            results['odds_remaining'] = remaining
            self.log(f"Odds API: {msg}")

            # Check defense data
            success, msg, count = validate_defense_data()
            results['defense_data'] = success
            self.log(f"Defense data: {msg}")

            # Check pace data
            success, msg, count = validate_pace_data()
            results['pace_data'] = success
            self.log(f"Pace data: {msg}")

            all_passed = all([
                results.get('imports', False),
                results.get('nba_api', False),
                results.get('defense_data', False),
            ])

            return all_passed, results

        except Exception as e:
            self.log(f"Validation error: {e}", "ERROR")
            return False, {'error': str(e)}

    def run_daily_analysis(self) -> Tuple[bool, Optional[str], int]:
        """
        Run daily analysis and save results.
        Returns (success, output_file, pick_count).
        """
        self.log_section("DAILY ANALYSIS")

        if self.dry_run:
            self.log("Would run daily analysis", "DRY-RUN")
            return True, None, 0

        try:
            # Use new modular imports
            from data import NBADataFetcher, OddsAPIClient
            from analysis import LivePropAnalyzer
            from core.config import CONFIG
            from datetime import datetime
            import pandas as pd

            # Initialize components
            self.log("Initializing data fetchers...")
            fetcher = NBADataFetcher()

            api_key = CONFIG.ODDS_API_KEY or os.environ.get('ODDS_API_KEY')
            if not api_key:
                self.log("No Odds API key configured", "ERROR")
                return False, None, 0

            odds_client = OddsAPIClient(api_key=api_key)
            analyzer = LivePropAnalyzer(nba_fetcher=fetcher, odds_client=odds_client)

            # Get today's date for filename
            today = datetime.now().strftime('%Y-%m-%d')
            excel_file = f"nba_daily_picks_{today}.xlsx"
            csv_file = f"nba_daily_picks_{today}.csv"

            # Run analysis
            self.log("Fetching live odds and analyzing props...")
            self.log(f"Min edge threshold: {CONFIG.MIN_EDGE_THRESHOLD*100}%")

            value_props = analyzer.find_value_props(min_edge=CONFIG.MIN_EDGE_THRESHOLD)

            if value_props is None or value_props.empty:
                self.log("No value plays found meeting criteria", "WARN")
                return True, None, 0

            pick_count = len(value_props)
            self.log(f"Found {pick_count} value plays")

            # Save to file
            try:
                value_props.to_excel(excel_file, index=False, sheet_name='Value Plays')
                output_file = excel_file
                self.log(f"Saved to: {excel_file}")
            except Exception as e:
                self.log(f"Excel save failed ({e}), using CSV", "WARN")
                value_props.to_csv(csv_file, index=False)
                output_file = csv_file
                self.log(f"Saved to: {csv_file}")

            # Store DataFrame for pick recording
            self._last_analysis = value_props
            self._last_date = today

            return True, output_file, pick_count

        except Exception as e:
            self.log(f"Analysis error: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return False, None, 0

    def record_picks(self, df=None, date_str: str = None) -> int:
        """
        Auto-record picks to database.
        Returns count of picks recorded.
        """
        self.log_section("RECORD PICKS")

        if self.dry_run:
            self.log("Would record picks to database", "DRY-RUN")
            return 0

        try:
            from pick_tracker import PickTracker

            # Use stored analysis or provided DataFrame
            if df is None:
                df = getattr(self, '_last_analysis', None)
            if date_str is None:
                date_str = getattr(self, '_last_date', datetime.now().strftime('%Y-%m-%d'))

            if df is None or df.empty:
                self.log("No picks to record", "WARN")
                return 0

            tracker = PickTracker()
            count = tracker.record_picks_from_df(df, date_str)

            self.log(f"Recorded {count} picks to database for {date_str}")
            return count

        except Exception as e:
            self.log(f"Error recording picks: {e}", "ERROR")
            return 0

    def run_pre_game(self) -> dict:
        """
        Full pre-game workflow:
        1. Validate system
        2. Run analysis
        3. Record picks
        """
        self.log_section("PRE-GAME WORKFLOW")
        self.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        results = {
            'validation': False,
            'analysis': False,
            'picks_recorded': 0,
            'output_file': None,
        }

        # Step 1: Validate
        valid, validation_results = self.validate_system()
        results['validation'] = valid
        results['validation_details'] = validation_results

        if not valid:
            self.log("System validation failed - aborting", "ERROR")
            return results

        # Step 2: Run analysis
        success, output_file, pick_count = self.run_daily_analysis()
        results['analysis'] = success
        results['output_file'] = output_file
        results['pick_count'] = pick_count

        if not success:
            self.log("Analysis failed", "ERROR")
            return results

        # Step 3: Record picks
        if pick_count > 0:
            recorded = self.record_picks()
            results['picks_recorded'] = recorded

        # Summary
        self.log_section("PRE-GAME SUMMARY")
        self.log(f"Validation: {'PASS' if results['validation'] else 'FAIL'}")
        self.log(f"Analysis: {'PASS' if results['analysis'] else 'FAIL'}")
        self.log(f"Picks found: {results.get('pick_count', 0)}")
        self.log(f"Picks recorded: {results['picks_recorded']}")
        if results['output_file']:
            self.log(f"Output file: {results['output_file']}")

        return results

    # =========================================================================
    # POST-GAME WORKFLOW
    # =========================================================================

    def fetch_results_from_api(self, date_str: str = None) -> Tuple[int, int]:
        """
        Fetch actual results from NBA API for pending picks.
        Returns (fetched_count, error_count).
        """
        self.log_section("FETCH RESULTS FROM NBA API")

        if date_str is None:
            # Default to yesterday (most games would have finished)
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        self.log(f"Fetching results for: {date_str}")

        if self.dry_run:
            self.log("Would fetch results from NBA API", "DRY-RUN")
            return 0, 0

        try:
            from pick_tracker import PickTracker
            from data import NBADataFetcher

            tracker = PickTracker()
            fetcher = NBADataFetcher()

            # Get pending picks
            pending = tracker.get_picks_awaiting_results(date_str)

            if pending.empty:
                self.log(f"No pending picks for {date_str}")
                return 0, 0

            self.log(f"Found {len(pending)} pending picks")

            # Group by player to minimize API calls
            players = pending['player'].unique()
            self.log(f"Fetching stats for {len(players)} players...")

            fetched = 0
            errors = 0
            player_cache = {}

            for player in players:
                try:
                    # Fetch recent game logs (last 3 games to find the right date)
                    logs = fetcher.get_player_game_logs(player, last_n_games=3)

                    if logs.empty:
                        self.log(f"  No data for {player}", "WARN")
                        errors += 1
                        continue

                    # Find the game from our target date
                    # Game logs have 'date' column in format like '2024-12-10'
                    if 'date' in logs.columns:
                        game_log = logs[logs['date'].astype(str).str.contains(date_str.replace('-', ''))]
                        if game_log.empty:
                            # Try matching on formatted date
                            logs['date_str'] = logs['date'].astype(str)
                            game_log = logs[logs['date_str'].str[:10] == date_str]

                    if game_log.empty:
                        # Use most recent game as fallback
                        game_log = logs.head(1)
                        self.log(f"  {player}: Using most recent game (date mismatch)", "WARN")

                    player_cache[player] = game_log.iloc[0].to_dict()

                    # Small delay to respect rate limits
                    time.sleep(0.3)

                except Exception as e:
                    self.log(f"  Error fetching {player}: {e}", "ERROR")
                    errors += 1

            # Now record results for each pending pick
            for _, pick in pending.iterrows():
                player = pick['player']
                prop_type = pick['prop_type']

                if player not in player_cache:
                    continue

                stats = player_cache[player]

                # Map prop type to stat column
                prop_map = {
                    'points': 'points',
                    'rebounds': 'rebounds',
                    'assists': 'assists',
                    'pra': 'pra',
                    'threes': 'fg3m',
                    'fg3m': 'fg3m',
                    'steals': 'steals',
                    'blocks': 'blocks',
                }

                stat_col = prop_map.get(prop_type.lower(), prop_type.lower())

                if stat_col in stats:
                    actual = float(stats[stat_col])
                    success = tracker.record_result(date_str, player, prop_type, actual)

                    if success:
                        # Determine hit/miss
                        hit = (pick['pick'] == 'OVER' and actual > pick['line']) or \
                              (pick['pick'] == 'UNDER' and actual < pick['line'])
                        result_str = "WIN" if hit else "LOSS" if actual != pick['line'] else "PUSH"

                        self.log(f"  {player} {prop_type}: {actual} ({result_str})")
                        fetched += 1
                    else:
                        errors += 1
                else:
                    self.log(f"  {player}: No {prop_type} stat found", "WARN")
                    errors += 1

            self.log(f"Fetched: {fetched}, Errors: {errors}")
            return fetched, errors

        except Exception as e:
            self.log(f"Error fetching results: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return 0, 0

    def generate_accuracy_report(self, days: int = 30) -> dict:
        """Generate and display accuracy report."""
        self.log_section("ACCURACY REPORT")

        if self.dry_run:
            self.log("Would generate accuracy report", "DRY-RUN")
            return {}

        try:
            from pick_tracker import PickTracker

            tracker = PickTracker()
            report = tracker.get_accuracy_report(days)

            overall = report.get('overall', {})
            if overall and overall.get('total', 0) > 0:
                win_rate = (overall.get('win_rate', 0) or 0) * 100
                self.log(f"Last {days} days:")
                self.log(f"  Total picks: {overall.get('total', 0)}")
                self.log(f"  Record: {overall.get('wins', 0)}-{overall.get('losses', 0)}")
                self.log(f"  Win rate: {win_rate:.1f}%")

                # Show by context quality
                by_quality = report.get('by_quality', [])
                if by_quality:
                    self.log("")
                    self.log("By Context Quality:")
                    for row in by_quality:
                        wr = (row.get('win_rate', 0) or 0) * 100
                        self.log(f"  {row['quality_tier']}: {wr:.1f}% ({row['total']} picks)")
            else:
                self.log("No results data available yet")

            return report

        except Exception as e:
            self.log(f"Error generating report: {e}", "ERROR")
            return {}

    def run_post_game(self, date_str: str = None) -> dict:
        """
        Full post-game workflow:
        1. Fetch results from NBA API
        2. Record results to database
        3. Generate accuracy report
        """
        self.log_section("POST-GAME WORKFLOW")

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        self.log(f"Processing results for: {date_str}")

        results = {
            'date': date_str,
            'fetched': 0,
            'errors': 0,
        }

        # Step 1: Fetch and record results
        fetched, errors = self.fetch_results_from_api(date_str)
        results['fetched'] = fetched
        results['errors'] = errors

        # Step 2: Generate report
        report = self.generate_accuracy_report()
        results['report'] = report

        # Summary
        self.log_section("POST-GAME SUMMARY")
        self.log(f"Results fetched: {results['fetched']}")
        self.log(f"Errors: {results['errors']}")

        return results

    # =========================================================================
    # NOTIFICATIONS (Optional)
    # =========================================================================

    def send_notification(self, title: str, message: str):
        """
        Send notification (stub - implement based on preferred service).
        Options: Email, Slack, Discord, Telegram, etc.
        """
        self.log(f"Notification: {title}")
        self.log(f"  {message}")

        # Example: Slack webhook (uncomment and configure)
        # import requests
        # webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        # if webhook_url:
        #     requests.post(webhook_url, json={'text': f"*{title}*\n{message}"})

        # Example: macOS notification (if running locally)
        # os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')


def main():
    parser = argparse.ArgumentParser(
        description='NBA Prop Daily Runner - Automated Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_runner.py --pre-game          # Run pre-game analysis
  python daily_runner.py --post-game         # Fetch results after games
  python daily_runner.py --post-game --date 2024-12-10  # Specific date
  python daily_runner.py --full              # Run both workflows
  python daily_runner.py --pre-game --dry-run  # Preview without executing
        """
    )

    parser.add_argument('--pre-game', action='store_true',
                       help='Run pre-game workflow (validate, analyze, record picks)')
    parser.add_argument('--post-game', action='store_true',
                       help='Run post-game workflow (fetch results, generate report)')
    parser.add_argument('--full', action='store_true',
                       help='Run both pre-game and post-game workflows')
    parser.add_argument('--fetch-results', action='store_true',
                       help='Just fetch and record results from NBA API')
    parser.add_argument('--accuracy', action='store_true',
                       help='Just show accuracy report')
    parser.add_argument('--date', type=str,
                       help='Date for results (YYYY-MM-DD), default: today for pre-game, yesterday for post-game')
    parser.add_argument('--days', type=int, default=30,
                       help='Days for accuracy report (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would run without executing')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    runner = DailyRunner(verbose=not args.quiet, dry_run=args.dry_run)

    if args.full:
        runner.run_pre_game()
        print()
        runner.run_post_game(args.date)

    elif args.pre_game:
        runner.run_pre_game()

    elif args.post_game:
        runner.run_post_game(args.date)

    elif args.fetch_results:
        runner.fetch_results_from_api(args.date)

    elif args.accuracy:
        runner.generate_accuracy_report(args.days)

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Before games:  python daily_runner.py --pre-game")
        print("  2. After games:   python daily_runner.py --post-game")
        print("=" * 60)


if __name__ == "__main__":
    main()
