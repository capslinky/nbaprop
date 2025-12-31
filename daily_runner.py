#!/usr/bin/env python3
"""
NBA Prop Daily Runner - Automated Workflow
===========================================
Automates the entire daily prop analysis workflow:

NOTE: This is the legacy pipeline. The source-of-truth pipeline is the v2 CLI
(`python -m nbaprop.cli run-daily`). Keep this script for reference only.

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

    # Real-time line monitoring with alerts
    python daily_runner.py --monitor

    # Monitor with custom settings
    python daily_runner.py --monitor --interval 60 --threshold 0.03 --duration 120

    # Monitor with Slack webhook
    python daily_runner.py --monitor --webhook https://hooks.slack.com/...

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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from core.logging_config import setup_logging, get_logger
from core.constants import get_current_nba_season

# Setup logging for daily runner
setup_logging(level="INFO")
_module_logger = get_logger(__name__)


class DailyRunner:
    """Orchestrates the full daily prop analysis workflow."""

    def __init__(self, verbose: bool = True, dry_run: bool = False, use_v2: bool = True):
        self.verbose = verbose
        self.dry_run = dry_run
        self.use_v2 = use_v2
        self.log_lines = []
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp using standard logging."""
        self.log_lines.append(f"[{level}] {message}")
        # Use standard logging
        log_method = getattr(_module_logger, level.lower(), _module_logger.info)
        log_method(message)

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

    def run_daily_analysis(self, bookmakers: list = None) -> Tuple[bool, Optional[str], int]:
        """
        Run daily analysis and save results.
        Returns (success, output_file, pick_count).

        Args:
            bookmakers: List of bookmaker keys to filter (e.g., ['fanduel']). None = all books.
        """
        if self.use_v2:
            return self._run_v2_daily_analysis(bookmakers=bookmakers)

        self.log_section("DAILY ANALYSIS (LEGACY)")

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

            # Setup news intelligence and injury tracking with Perplexity if available
            news_search_fn = None
            injury_tracker = None
            perplexity_key = CONFIG.PERPLEXITY_API_KEY or os.environ.get('PERPLEXITY_API_KEY')
            if CONFIG.NEWS_INTELLIGENCE_ENABLED and perplexity_key:
                from core.news_intelligence import create_perplexity_api_search
                news_search_fn = create_perplexity_api_search(perplexity_key)
                self.log("News intelligence: Perplexity API enabled")

                # Create InjuryTracker with Perplexity for real-time injury data
                from data import InjuryTracker

                def perplexity_injury_fn(messages):
                    """Wrapper to call Perplexity API for injury queries."""
                    try:
                        import requests
                        url = "https://api.perplexity.ai/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {perplexity_key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "sonar",
                            "messages": messages
                        }
                        response = requests.post(url, headers=headers, json=payload, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    except Exception as e:
                        self.log(f"Perplexity injury query error: {e}", "DEBUG")
                    return None

                injury_tracker = InjuryTracker(perplexity_fn=perplexity_injury_fn)
                self.log("Injury tracking: Perplexity real-time enabled")
            else:
                if CONFIG.NEWS_INTELLIGENCE_ENABLED:
                    self.log("News intelligence: Disabled (no PERPLEXITY_API_KEY)")
                else:
                    self.log("News intelligence: Disabled by config")
                self.log("Injury tracking: Using web scraping only")

            analyzer = LivePropAnalyzer(
                nba_fetcher=fetcher,
                odds_client=odds_client,
                injury_tracker=injury_tracker,
                news_search_fn=news_search_fn
            )

            # Get today's date for filename
            today = datetime.now().strftime('%Y-%m-%d')
            excel_file = f"nba_daily_picks_{today}.xlsx"
            csv_file = f"nba_daily_picks_{today}.csv"

            # Run analysis
            self.log("Fetching live odds and analyzing props...")
            self.log(f"Min edge threshold: {CONFIG.MIN_EDGE_THRESHOLD*100}%")
            if bookmakers:
                self.log(f"Bookmaker filter: {', '.join(bookmakers)}")

            value_props = analyzer.find_value_props(min_edge=CONFIG.MIN_EDGE_THRESHOLD, bookmakers=bookmakers)

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

    def _run_v2_daily_analysis(self, bookmakers: list = None) -> Tuple[bool, Optional[str], int]:
        """Run the v2 pipeline and hydrate legacy tracking outputs."""
        self.log_section("DAILY ANALYSIS (v2)")

        if self.dry_run:
            self.log("Would run v2 daily analysis", "DRY-RUN")
            return True, None, 0

        try:
            from nbaprop.cli import run_daily as run_daily_v2
            from nbaprop.config import Config as V2Config
            import pandas as pd

            if bookmakers:
                self.log("Bookmaker filter handled by v2 config (ODDS_BOOKMAKERS)")

            run_daily_v2()

            v2_config = V2Config.load()
            runs_dir = Path(v2_config.cache_dir) / "runs"
            manifest_path = self._find_latest_manifest(runs_dir)
            if manifest_path is None:
                self.log("No v2 manifest found after run", "ERROR")
                return False, None, 0

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            outputs = manifest.get("outputs", {})
            picks_path = outputs.get("picks_filtered_csv") or outputs.get("picks_csv")
            if not picks_path:
                self.log("No picks CSV found in v2 manifest", "ERROR")
                return False, None, 0

            picks_path = Path(picks_path)
            if not picks_path.is_absolute():
                picks_path = (Path(__file__).resolve().parents[0] / picks_path).resolve()

            if not picks_path.exists():
                self.log(f"v2 picks file missing: {picks_path}", "ERROR")
                return False, None, 0

            df = pd.read_csv(picks_path)
            if df.empty:
                self.log("v2 produced no picks", "WARN")
                return True, str(picks_path), 0

            normalized = self._normalize_v2_picks(df, v2_config)
            self._last_analysis = normalized
            self._last_date = v2_config.run_date or datetime.now().strftime('%Y-%m-%d')

            pdf_path = outputs.get("picks_pdf")
            if pdf_path:
                pdf_path = Path(pdf_path)
                if not pdf_path.is_absolute():
                    pdf_path = (Path(__file__).resolve().parents[0] / pdf_path).resolve()
                self._last_v2_pdf = str(pdf_path)

            return True, str(picks_path), len(normalized)
        except Exception as e:
            self.log(f"v2 analysis error: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return False, None, 0

    def _find_latest_manifest(self, runs_dir: Path) -> Optional[Path]:
        if not runs_dir.exists():
            return None
        manifests = sorted(
            runs_dir.glob("manifest_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return manifests[0] if manifests else None

    def _normalize_v2_picks(self, df: "pd.DataFrame", config: "V2Config") -> "pd.DataFrame":
        import pandas as pd

        default_book = (config.odds_bookmakers or ["fanduel"])[0]
        if "player_name" in df.columns:
            player_col = "player_name"
        elif "player" in df.columns:
            player_col = "player"
        else:
            player_col = None
        if "bookmaker" in df.columns:
            bookmaker_col = df["bookmaker"]
        else:
            bookmaker_col = [default_book] * len(df)
        normalized = pd.DataFrame({
            "player": df[player_col] if player_col else None,
            "prop_type": df.get("prop_type"),
            "line": df.get("line"),
            "pick": df.get("pick"),
            "edge": df.get("edge"),
            "confidence": df.get("confidence"),
            "projection": df.get("projection"),
            "context_quality": 50,
            "warnings": [[] for _ in range(len(df))],
            "adjustments": df.get("adjustments"),
            "flags": df.get("adjustment_notes"),
            "opponent": df.get("opponent_team"),
            "is_home": df.get("is_home"),
            "is_b2b": df.get("b2b"),
            "game_total": None,
            "matchup": "NEUTRAL",
            "bookmaker": bookmaker_col,
            "odds": df.get("odds"),
            "rest_days": df.get("rest_days"),
        })
        return normalized

    def generate_pdf_report(self, df=None, date_str: str = None) -> Optional[str]:
        """
        Generate PDF report from picks.
        Returns path to PDF file or None if failed.
        """
        self.log_section("GENERATE PDF REPORT")

        if self.dry_run:
            self.log("Would generate PDF report", "DRY-RUN")
            return None

        if self.use_v2:
            v2_pdf = getattr(self, '_last_v2_pdf', None)
            if v2_pdf:
                self.log(f"v2 PDF report: {v2_pdf}")
                return v2_pdf
            self.log("No v2 PDF report found", "WARN")
            return None

        try:
            from generate_report import PicksReportGenerator

            # Use stored analysis or provided DataFrame
            if df is None:
                df = getattr(self, '_last_analysis', None)
            if date_str is None:
                date_str = getattr(self, '_last_date', datetime.now().strftime('%Y-%m-%d'))

            if df is None or df.empty:
                self.log("No picks to generate report from", "WARN")
                return None

            # Generate PDF
            output_path = f"nba_picks_{date_str}.pdf"
            generator = PicksReportGenerator(df, date_str)
            result = generator.generate(output_path)

            self.log(f"PDF report saved to: {result}")
            return result

        except ImportError as e:
            self.log(f"PDF generation requires reportlab: {e}", "WARN")
            self.log("Install with: pip install reportlab", "WARN")
            return None
        except Exception as e:
            self.log(f"Error generating PDF: {e}", "ERROR")
            return None

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

    def run_pre_game(self, bookmakers: list = None) -> dict:
        """
        Full pre-game workflow:
        1. Validate system
        2. Run analysis
        3. Record picks
        4. Generate PDF report

        Args:
            bookmakers: List of bookmaker keys to filter (e.g., ['fanduel']). None = all books.
        """
        self.log_section("PRE-GAME WORKFLOW")
        self.log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        results = {
            'validation': False,
            'analysis': False,
            'picks_recorded': 0,
            'output_file': None,
            'pdf_file': None,
        }

        # Step 1: Validate
        valid, validation_results = self.validate_system()
        results['validation'] = valid
        results['validation_details'] = validation_results

        if not valid:
            self.log("System validation failed - aborting", "ERROR")
            return results

        # Step 2: Run analysis
        success, output_file, pick_count = self.run_daily_analysis(bookmakers=bookmakers)
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

        # Step 4: Generate PDF report
        if pick_count > 0:
            pdf_file = self.generate_pdf_report()
            results['pdf_file'] = pdf_file

        # Summary
        self.log_section("PRE-GAME SUMMARY")
        self.log(f"Validation: {'PASS' if results['validation'] else 'FAIL'}")
        self.log(f"Analysis: {'PASS' if results['analysis'] else 'FAIL'}")
        self.log(f"Picks found: {results.get('pick_count', 0)}")
        self.log(f"Picks recorded: {results['picks_recorded']}")
        if results['output_file']:
            self.log(f"Data file: {results['output_file']}")
        if results['pdf_file']:
            self.log(f"PDF report: {results['pdf_file']}")

        return results

    # =========================================================================
    # INJURY REPORT WORKFLOW
    # =========================================================================

    def generate_injury_report(self, perplexity_fn=None, output_dir: str = ".") -> Tuple[Optional[str], Optional[str]]:
        """
        Generate daily NBA injury report (PDF + CSV).

        Args:
            perplexity_fn: Optional Perplexity MCP function for real-time data
            output_dir: Directory to save reports

        Returns:
            Tuple of (pdf_path, csv_path) or (None, None) if failed
        """
        self.log_section("GENERATE INJURY REPORT")

        if self.dry_run:
            self.log("Would generate injury report", "DRY-RUN")
            return None, None

        try:
            from data import InjuryTracker
            from core.rosters import TEAM_ROSTERS, get_team_stars
            import pandas as pd
            from datetime import date

            today = date.today()
            date_str = today.strftime('%Y-%m-%d')

            # Initialize tracker with optional Perplexity
            tracker = InjuryTracker(perplexity_fn=perplexity_fn)

            self.log("Fetching injury data from all sources...")
            injuries = tracker.get_all_injuries(force_refresh=True)

            if injuries.empty:
                self.log("No injury data available", "WARN")
                return None, None

            self.log(f"Found {len(injuries)} injury records")

            # Enrich with star player info
            injuries = injuries.copy()
            if 'team' in injuries.columns:
                injuries['is_star'] = injuries.apply(
                    lambda row: row['player'] in get_team_stars(row.get('team', '')),
                    axis=1
                )
                injuries['impact'] = injuries['is_star'].apply(
                    lambda x: 'STAR' if x else 'ROTATION'
                )
            else:
                injuries['impact'] = 'UNKNOWN'

            # Sort by team and impact
            sort_cols = ['team', 'impact', 'status'] if 'team' in injuries.columns else ['status']
            injuries = injuries.sort_values(sort_cols)

            # Generate CSV
            csv_path = os.path.join(output_dir, f"nba_injury_report_{date_str}.csv")
            injuries.to_csv(csv_path, index=False)
            self.log(f"CSV saved to: {csv_path}")

            # Generate PDF
            pdf_path = os.path.join(output_dir, f"nba_injury_report_{date_str}.pdf")
            pdf_result = self._generate_injury_pdf(injuries, pdf_path, date_str)

            if pdf_result:
                self.log(f"PDF saved to: {pdf_path}")
            else:
                pdf_path = None
                self.log("PDF generation failed - CSV only", "WARN")

            # Summary statistics
            by_status = injuries['status'].value_counts().to_dict()
            self.log(f"By status: {by_status}")

            if 'is_star' in injuries.columns:
                star_count = injuries['is_star'].sum()
                self.log(f"Star players affected: {star_count}")

            return pdf_path, csv_path

        except Exception as e:
            self.log(f"Error generating injury report: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return None, None

    def _generate_injury_pdf(self, injuries: 'pd.DataFrame', output_path: str, date_str: str) -> bool:
        """
        Generate PDF injury report.

        Returns:
            True if successful, False otherwise
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(output_path, pagesize=letter,
                                   topMargin=0.5*inch, bottomMargin=0.5*inch)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=12,
                alignment=1  # Center
            )
            elements.append(Paragraph(f"NBA Injury Report - {date_str}", title_style))
            elements.append(Spacer(1, 12))

            # Summary
            total = len(injuries)
            out_count = len(injuries[injuries['status'].str.upper() == 'OUT'])
            gtd_count = len(injuries[injuries['status'].str.upper().isin(['GTD', 'GAME TIME DECISION'])])
            questionable_count = len(injuries[injuries['status'].str.upper() == 'QUESTIONABLE'])

            summary_text = f"Total: {total} | OUT: {out_count} | GTD: {gtd_count} | Questionable: {questionable_count}"
            elements.append(Paragraph(summary_text, styles['Normal']))
            elements.append(Spacer(1, 12))

            # Table data
            cols = ['player', 'team', 'status', 'injury', 'source']
            available_cols = [c for c in cols if c in injuries.columns]

            table_data = [['Player', 'Team', 'Status', 'Injury', 'Source'][:len(available_cols)]]
            for _, row in injuries.iterrows():
                table_data.append([str(row.get(c, ''))[:30] for c in available_cols])

            # Create table
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            # Highlight OUT players
            for i, row in enumerate(table_data[1:], start=1):
                if len(row) >= 3 and row[2].upper() == 'OUT':
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.Color(1, 0.9, 0.9)),
                    ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

            # Footer
            footer = Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Sources: Perplexity AI, NBA API, CBS Sports, Rotowire",
                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
            )
            elements.append(footer)

            doc.build(elements)
            return True

        except ImportError:
            self.log("reportlab not installed - skipping PDF generation", "WARN")
            return False
        except Exception as e:
            self.log(f"PDF generation error: {e}", "ERROR")
            return False

    # =========================================================================
    # POST-GAME WORKFLOW
    # =========================================================================

    def backfill_missing_results(self) -> Tuple[int, int]:
        """
        Automatically backfill results for all past dates with missing results.

        Returns:
            Tuple of (total_fetched, total_errors)
        """
        self.log_section("BACKFILL MISSING RESULTS")

        if self.dry_run:
            self.log("Would backfill missing results", "DRY-RUN")
            return 0, 0

        try:
            from pick_tracker import PickTracker

            tracker = PickTracker()
            missing_dates = tracker.get_dates_missing_results()

            if not missing_dates:
                self.log("No dates with missing results found")
                return 0, 0

            self.log(f"Found {len(missing_dates)} dates with missing results:")
            for date in missing_dates:
                self.log(f"  - {date}")

            total_fetched = 0
            total_errors = 0

            for date in missing_dates:
                self.log(f"\nBackfilling {date}...")
                fetched, errors = self.fetch_results_from_api(date)
                total_fetched += fetched
                total_errors += errors

            self.log(f"\nBackfill complete: {total_fetched} results fetched, {total_errors} errors")
            return total_fetched, total_errors

        except Exception as e:
            self.log(f"Backfill error: {e}", "ERROR")
            return 0, 0

    def generate_oversight_report(self, days: int = 7, output_path: str = None) -> Optional[str]:
        """
        Generate daily oversight report (PDF and console).

        Args:
            days: Number of days to include in report
            output_path: Optional custom output path for PDF

        Returns:
            Path to PDF file, or None if failed
        """
        self.log_section("OVERSIGHT REPORT")

        if self.dry_run:
            self.log("Would generate oversight report", "DRY-RUN")
            return None

        try:
            from pick_tracker import PickTracker
            from reports.oversight_report import OversightReportGenerator

            tracker = PickTracker()
            generator = OversightReportGenerator(tracker)

            # Print console report
            generator.print_console_report(days)

            # Generate PDF
            pdf_path = generator.generate_pdf(days, output_path)
            self.log(f"PDF oversight report saved to: {pdf_path}")

            return pdf_path

        except ImportError as e:
            self.log(f"Oversight report requires reportlab: {e}", "WARN")
            return None
        except Exception as e:
            self.log(f"Error generating oversight report: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return None

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

            # Parse target date for comparison
            import pandas as pd
            target_date = pd.to_datetime(date_str).date()

            for player in players:
                try:
                    # Fetch recent game logs (last 10 games to find the right date)
                    logs = fetcher.get_player_game_logs(player, last_n_games=10)

                    if logs.empty:
                        self.log(f"  No data for {player}", "WARN")
                        errors += 1
                        continue

                    # Find the game from our target date using proper datetime comparison
                    game_log = pd.DataFrame()
                    if 'date' in logs.columns:
                        # Convert date column to date objects for comparison
                        logs['game_date'] = pd.to_datetime(logs['date']).dt.date
                        game_log = logs[logs['game_date'] == target_date]

                    if game_log.empty:
                        # Player didn't play on this date - skip them (don't use wrong game data)
                        self.log(f"  {player}: No game on {date_str} (skipping)", "DEBUG")
                        continue

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

    def run_post_game(self, date_str: str = None, backfill: bool = True, oversight_days: int = 7) -> dict:
        """
        Full post-game workflow:
        1. Backfill any missing results from previous days (optional)
        2. Fetch results from NBA API for today/specified date
        3. Generate accuracy report
        4. Generate oversight report
        5. Run weekly calibration (Sundays)

        Args:
            date_str: Date to process (default: today)
            backfill: Whether to auto-backfill missing results (default: True)
            oversight_days: Number of days for oversight report (default: 7)
        """
        self.log_section("POST-GAME WORKFLOW")

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        self.log(f"Processing results for: {date_str}")

        results = {
            'date': date_str,
            'fetched': 0,
            'errors': 0,
            'backfill_fetched': 0,
            'backfill_errors': 0,
        }

        # Step 1: Backfill missing results (if enabled)
        if backfill:
            backfill_fetched, backfill_errors = self.backfill_missing_results()
            results['backfill_fetched'] = backfill_fetched
            results['backfill_errors'] = backfill_errors

        # Step 2: Fetch and record results for target date
        fetched, errors = self.fetch_results_from_api(date_str)
        results['fetched'] = fetched
        results['errors'] = errors

        # Step 3: Generate accuracy report
        report = self.generate_accuracy_report()
        results['report'] = report

        # Step 4: Generate oversight report
        oversight_pdf = self.generate_oversight_report(days=oversight_days)
        results['oversight_pdf'] = oversight_pdf

        # Step 5: Weekly calibration check (Sundays only)
        calibration_result = self.run_weekly_calibration_check()
        if calibration_result:
            results['calibration'] = calibration_result

        # Summary
        self.log_section("POST-GAME SUMMARY")
        if backfill:
            self.log(f"Backfill results: {results['backfill_fetched']} fetched, {results['backfill_errors']} errors")
        self.log(f"Today's results: {results['fetched']} fetched, {results['errors']} errors")
        if oversight_pdf:
            self.log(f"Oversight report: {oversight_pdf}")
        if calibration_result:
            self.log(f"Calibration: {calibration_result.get('status', 'skipped')}")

        return results

    def run_weekly_calibration_check(self) -> Optional[dict]:
        """
        Run model calibration on Sundays after sufficient data accumulates.

        Returns:
            Calibration result dict, or None if skipped
        """
        # Only run on Sundays
        if datetime.now().weekday() != 6:  # 0=Monday, 6=Sunday
            return None

        self.log_section("WEEKLY CALIBRATION CHECK")
        self.log("Running weekly model calibration (Sunday)...")

        try:
            from calibration import CalibrationAnalyzer, WeightOptimizer, LearnedWeightsStore
            from core.config import CONFIG

            analyzer = CalibrationAnalyzer()

            # Check if sufficient data
            if not analyzer.has_sufficient_data():
                self.log(f"Insufficient data for calibration (need {CONFIG.CALIBRATION_MIN_TOTAL_PICKS} picks)")
                return {'status': 'skipped', 'reason': 'insufficient_data'}

            # Run analysis
            result = analyzer.calculate_all_factor_stats()
            self.log(f"Analyzed {result.picks_with_results} picks, win rate: {result.overall_win_rate:.1%}")

            # Optimize
            optimizer = WeightOptimizer()
            optimized = optimizer.optimize_all_factors(result)

            # Count adjustments
            adjusted_count = sum(1 for opt in optimized.values() if opt.was_adjusted)

            if adjusted_count == 0:
                self.log("No factors need adjustment")
                return {'status': 'no_changes', 'picks_analyzed': result.picks_with_results}

            # Save if not dry run
            if self.dry_run:
                self.log(f"[DRY RUN] Would adjust {adjusted_count} factor(s)")
                return {'status': 'dry_run', 'would_adjust': adjusted_count}

            # Save weights
            weights = optimizer.create_learned_weights(optimized, result)
            store = LearnedWeightsStore()

            if store.save(weights):
                self.log(f"Calibration complete: {adjusted_count} factor(s) adjusted")
                for name, opt in optimized.items():
                    if opt.was_adjusted:
                        self.log(f"  {name}: {opt.current_value:.3f} -> {opt.new_value:.3f} ({opt.change_pct*100:+.1f}%)")
                return {
                    'status': 'success',
                    'factors_adjusted': adjusted_count,
                    'picks_analyzed': result.picks_with_results,
                }
            else:
                self.log("Error saving calibration weights", level="ERROR")
                return {'status': 'error', 'reason': 'save_failed'}

        except ImportError:
            self.log("Calibration module not available", level="WARNING")
            return {'status': 'skipped', 'reason': 'module_not_available'}
        except Exception as e:
            self.log(f"Calibration error: {e}", level="ERROR")
            return {'status': 'error', 'reason': str(e)}

    # =========================================================================
    # LINE MONITORING WORKFLOW
    # =========================================================================

    def run_line_monitor(
        self,
        interval_seconds: int = 300,
        alert_threshold: float = 0.05,
        duration_minutes: int = 0,
        console_alerts: bool = True,
        webhook_url: str = None
    ) -> dict:
        """
        Run continuous line monitoring with real-time alerts.

        Args:
            interval_seconds: Polling interval (default 5 minutes)
            alert_threshold: Minimum movement to trigger alert (default 5%)
            duration_minutes: How long to run (0 = indefinitely)
            console_alerts: Print alerts to console
            webhook_url: Optional webhook URL for Slack/Discord alerts

        Returns:
            Summary dict with stats
        """
        self.log_section("LINE MONITOR")
        self.log(f"Polling interval: {interval_seconds}s")
        self.log(f"Alert threshold: {alert_threshold*100}%")
        if duration_minutes > 0:
            self.log(f"Duration: {duration_minutes} minutes")
        else:
            self.log("Duration: Indefinite (Ctrl+C to stop)")

        if self.dry_run:
            self.log("Would start line monitoring", "DRY-RUN")
            return {'dry_run': True}

        try:
            from core.line_monitor import LineMonitor
            from core.alerts import AlertManager, ConsoleAlert, SlackWebhookAlert, WebhookAlert
            from data import OddsAPIClient
            from core.config import CONFIG
            import pandas as pd

            # Initialize components
            api_key = CONFIG.ODDS_API_KEY or os.environ.get('ODDS_API_KEY')
            if not api_key:
                self.log("No Odds API key configured", "ERROR")
                return {'error': 'No API key'}

            odds_client = OddsAPIClient(api_key=api_key)
            monitor = LineMonitor(alert_threshold=alert_threshold)

            # Setup alert handlers
            alert_manager = AlertManager()
            if console_alerts:
                alert_manager.add_handler(ConsoleAlert())

            if webhook_url:
                if 'slack' in webhook_url.lower():
                    alert_manager.add_handler(SlackWebhookAlert(webhook_url))
                else:
                    alert_manager.add_handler(WebhookAlert(webhook_url))

            self.log(f"Alert handlers: {alert_manager.handler_names}")

            # Stats tracking
            stats = {
                'polls': 0,
                'movements_detected': 0,
                'steam_moves': 0,
                'start_time': datetime.now(),
                'errors': 0,
            }

            end_time = None
            if duration_minutes > 0:
                end_time = datetime.now() + timedelta(minutes=duration_minutes)

            self.log("")
            self.log("Starting line monitor... (Ctrl+C to stop)")
            self.log("")

            try:
                while True:
                    # Check if we should stop
                    if end_time and datetime.now() >= end_time:
                        self.log("Duration reached, stopping monitor")
                        break

                    try:
                        # Fetch current odds
                        odds_df = self._fetch_all_current_odds(odds_client)
                        stats['polls'] += 1

                        if odds_df is None or odds_df.empty:
                            self.log("No odds data available", "WARN")
                        else:
                            # Check for movements
                            movements = monitor.check_for_alerts(odds_df)

                            if movements:
                                stats['movements_detected'] += len(movements)
                                for movement in movements:
                                    alert_manager.send_line_movement(
                                        movement,
                                        priority='high' if movement.is_significant else 'normal'
                                    )

                            # Check for steam moves (multi-book coordinated)
                            steam_moves = monitor.get_steam_moves(min_books=3)
                            if steam_moves:
                                if not hasattr(self, '_alerted_steam'):
                                    self._alerted_steam = set()
                                for steam in steam_moves:
                                    # Create key to track which steam moves we've alerted on
                                    steam_key = f"{steam['player']}|{steam['prop_type']}|{steam['direction']}"
                                    # Only alert if not already alerted
                                    if steam_key not in self._alerted_steam:
                                        stats['steam_moves'] += 1
                                        alert_manager.send_steam_move(steam)
                                        self._alerted_steam.add(steam_key)

                            # Periodic status
                            if stats['polls'] % 12 == 0:  # Every hour at 5min intervals
                                runtime = datetime.now() - stats['start_time']
                                self.log(
                                    f"[Status] Polls: {stats['polls']} | "
                                    f"Movements: {stats['movements_detected']} | "
                                    f"Steam: {stats['steam_moves']} | "
                                    f"Runtime: {runtime}"
                                )

                    except Exception as e:
                        stats['errors'] += 1
                        self.log(f"Poll error: {e}", "ERROR")

                    # Wait for next poll
                    time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.log("\nMonitor stopped by user")

            # Final summary
            runtime = datetime.now() - stats['start_time']
            stats['runtime_seconds'] = runtime.total_seconds()

            self.log_section("MONITOR SUMMARY")
            self.log(f"Runtime: {runtime}")
            self.log(f"Total polls: {stats['polls']}")
            self.log(f"Movements detected: {stats['movements_detected']}")
            self.log(f"Steam moves: {stats['steam_moves']}")
            self.log(f"Errors: {stats['errors']}")

            return stats

        except Exception as e:
            self.log(f"Monitor error: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return {'error': str(e)}

    def _fetch_all_current_odds(self, odds_client) -> 'pd.DataFrame':
        """
        Fetch all current player prop odds.
        Returns DataFrame with columns: player, prop_type, line, odds, bookmaker, event_id
        """
        import pandas as pd

        try:
            # Get today's events
            events = odds_client.get_todays_events()

            if not events:
                return pd.DataFrame()

            all_props = []
            for event in events:
                event_id = event.get('id')
                if not event_id:
                    continue

                # Fetch props for this event
                props_data = odds_client.get_player_props(event_id)

                if not props_data:
                    continue

                # Parse props into rows
                for prop in props_data:
                    player = prop.get('player', prop.get('description', ''))
                    prop_type = prop.get('market', prop.get('prop_type', ''))
                    line = prop.get('point', prop.get('line', 0))
                    odds = prop.get('price', prop.get('odds', -110))
                    bookmaker = prop.get('bookmaker', 'unknown')

                    all_props.append({
                        'player': player,
                        'prop_type': prop_type,
                        'line': line,
                        'odds': odds,
                        'bookmaker': bookmaker,
                        'event_id': event_id,
                    })

                # Rate limit between events
                time.sleep(0.5)

            return pd.DataFrame(all_props)

        except Exception as e:
            self.log(f"Error fetching odds: {e}", "WARN")
            return pd.DataFrame()

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
    from core.config import CONFIG

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
    parser.add_argument('--injury-report', action='store_true',
                       help='Generate daily NBA injury report (PDF + CSV)')
    parser.add_argument('--schedule', action='store_true',
                       help='Show NBA game schedule')
    parser.add_argument('--oversight-report', action='store_true',
                       help='Generate oversight report only (PDF + console)')
    parser.add_argument('--no-backfill', action='store_true',
                       help='Disable auto-backfill of missing results in post-game')
    parser.add_argument('--monitor', action='store_true',
                       help='Run continuous line monitoring with alerts')
    parser.add_argument('--date', type=str,
                       help='Date for results (YYYY-MM-DD), default: today for pre-game, yesterday for post-game')
    parser.add_argument('--days', type=int, default=7,
                       help='Days for accuracy/oversight report (default: 7)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitor polling interval in seconds (default: 300 = 5 min)')
    parser.add_argument('--duration', type=int, default=0,
                       help='Monitor duration in minutes (default: 0 = indefinite)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Line movement alert threshold as decimal (default: 0.05 = 5%%)')
    parser.add_argument('--webhook', type=str,
                       help='Webhook URL for Slack/Discord alerts')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would run without executing')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy analysis pipeline instead of v2')
    parser.add_argument('--book', type=str,
                       help='Filter to specific bookmaker (e.g., fanduel, draftkings, betmgm)')

    args = parser.parse_args()

    use_v2 = not args.legacy
    if use_v2 and args.book:
        os.environ["ODDS_BOOKMAKERS"] = args.book
        bookmakers = None
    else:
        # Parse bookmaker filter - use CONFIG.PREFERRED_BOOKS as default
        bookmakers = [args.book] if args.book else CONFIG.PREFERRED_BOOKS

    runner = DailyRunner(verbose=not args.quiet, dry_run=args.dry_run, use_v2=use_v2)

    # Log current date context for debugging
    from datetime import date
    current_season = get_current_nba_season()
    print(f"Running analysis for {date.today()} (Season: {current_season})")
    _module_logger.info(f"Session started: {date.today()} | Season: {current_season}")

    if args.full:
        runner.run_pre_game(bookmakers=bookmakers)
        print()
        runner.run_post_game(
            date_str=args.date,
            backfill=not args.no_backfill,
            oversight_days=args.days
        )

    elif args.pre_game:
        runner.run_pre_game(bookmakers=bookmakers)

    elif args.post_game:
        runner.run_post_game(
            date_str=args.date,
            backfill=not args.no_backfill,
            oversight_days=args.days
        )

    elif args.fetch_results:
        runner.fetch_results_from_api(args.date)

    elif args.accuracy:
        runner.generate_accuracy_report(args.days)

    elif args.oversight_report:
        runner.generate_oversight_report(days=args.days)

    elif args.injury_report:
        pdf_path, csv_path = runner.generate_injury_report()
        if csv_path:
            print(f"\nInjury report generated:")
            if pdf_path:
                print(f"  PDF: {pdf_path}")
            print(f"  CSV: {csv_path}")

    elif args.monitor:
        runner.run_line_monitor(
            interval_seconds=args.interval,
            alert_threshold=args.threshold,
            duration_minutes=args.duration,
            console_alerts=not args.quiet,
            webhook_url=args.webhook
        )

    elif args.schedule:
        from data.schedule_utils import print_schedule
        print_schedule(days=args.days, date_str=args.date)

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Before games:  python daily_runner.py --pre-game")
        print("  2. After games:   python daily_runner.py --post-game")
        print("  3. Schedule:      python daily_runner.py --schedule")
        print("  4. Oversight:     python daily_runner.py --oversight-report --days 7")
        print("  5. Line monitor:  python daily_runner.py --monitor")
        print("=" * 60)


if __name__ == "__main__":
    main()
