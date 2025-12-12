"""
NBA Prop Pick Tracker
=====================
SQLite-based system for tracking picks and comparing to actual results.

Usage:
    # Record picks
    python pick_tracker.py --record nba_picks_2024-12-12.csv

    # Enter results after games
    python pick_tracker.py --results

    # View accuracy report
    python pick_tracker.py --accuracy

    # Analyze which factors help/hurt
    python pick_tracker.py --analyze
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
import argparse


# Database path
DB_PATH = Path(__file__).parent / "clv_tracking.db"


@dataclass
class PickRecord:
    """A single pick to track."""
    date: str
    player: str
    prop_type: str
    line: float
    pick: str  # 'OVER' or 'UNDER'
    edge: float  # As percentage
    confidence: float  # As percentage
    projection: float
    context_quality: int
    warnings: str  # JSON list
    adjustments: str  # JSON dict
    flags: str  # JSON list
    opponent: Optional[str] = None
    is_home: Optional[bool] = None
    is_b2b: bool = False
    game_total: Optional[float] = None
    matchup_rating: str = 'NEUTRAL'
    bookmaker: Optional[str] = None
    odds: int = -110


@dataclass
class ResultRecord:
    """Actual game result."""
    date: str
    player: str
    prop_type: str
    actual: float
    hit: bool  # Did the pick win?


class PickTracker:
    """
    Track picks and results in SQLite database.
    Provides accuracy analysis by various factors.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    player TEXT NOT NULL,
                    prop_type TEXT NOT NULL,
                    line REAL NOT NULL,
                    pick TEXT NOT NULL,
                    edge REAL,
                    confidence REAL,
                    projection REAL,
                    context_quality INTEGER,
                    warnings TEXT,
                    adjustments TEXT,
                    flags TEXT,
                    opponent TEXT,
                    is_home INTEGER,
                    is_b2b INTEGER DEFAULT 0,
                    game_total REAL,
                    matchup_rating TEXT,
                    bookmaker TEXT,
                    odds INTEGER DEFAULT -110,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, player, prop_type, line)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    player TEXT NOT NULL,
                    prop_type TEXT NOT NULL,
                    actual REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, player, prop_type)
                )
            """)

            # Create view for joined data
            conn.execute("""
                CREATE VIEW IF NOT EXISTS picks_with_results AS
                SELECT
                    p.*,
                    r.actual,
                    CASE
                        WHEN p.pick = 'OVER' AND r.actual > p.line THEN 1
                        WHEN p.pick = 'UNDER' AND r.actual < p.line THEN 1
                        WHEN r.actual = p.line THEN NULL  -- Push
                        ELSE 0
                    END as hit
                FROM picks p
                LEFT JOIN results r ON
                    p.date = r.date AND
                    p.player = r.player AND
                    p.prop_type = r.prop_type
            """)

            conn.commit()

    def record_pick(self, pick: PickRecord) -> bool:
        """Record a single pick. Returns True if successful."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO picks
                    (date, player, prop_type, line, pick, edge, confidence,
                     projection, context_quality, warnings, adjustments, flags,
                     opponent, is_home, is_b2b, game_total, matchup_rating,
                     bookmaker, odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pick.date, pick.player, pick.prop_type, pick.line, pick.pick,
                    pick.edge, pick.confidence, pick.projection, pick.context_quality,
                    pick.warnings, pick.adjustments, pick.flags,
                    pick.opponent, 1 if pick.is_home else 0 if pick.is_home is False else None,
                    1 if pick.is_b2b else 0, pick.game_total, pick.matchup_rating,
                    pick.bookmaker, pick.odds
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error recording pick: {e}")
            return False

    def record_picks_from_df(self, df: pd.DataFrame, date_str: str = None) -> int:
        """
        Record picks from a DataFrame (e.g., from daily analysis output).
        Returns count of picks recorded.
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        count = 0
        for _, row in df.iterrows():
            # Skip PASS picks
            if row.get('pick', row.get('Pick', '')) == 'PASS':
                continue

            pick = PickRecord(
                date=date_str,
                player=row.get('player', row.get('Player', '')),
                prop_type=row.get('prop_type', row.get('Prop', '')).lower(),
                line=float(row.get('line', row.get('Line', 0))),
                pick=row.get('pick', row.get('Pick', '')),
                edge=float(row.get('edge', row.get('Edge', 0))),
                confidence=float(row.get('confidence', row.get('Conf', 0))),
                projection=float(row.get('projection', row.get('Proj', 0))),
                context_quality=int(row.get('context_quality', 50)),
                warnings=json.dumps(row.get('warnings', [])),
                adjustments=json.dumps(row.get('adjustments', {})),
                flags=json.dumps(row.get('flags', row.get('Flags', []))),
                opponent=row.get('opponent', None),
                is_home=row.get('is_home', None),
                is_b2b=bool(row.get('is_b2b', row.get('B2B', False))),
                game_total=row.get('game_total', row.get('Total', None)),
                matchup_rating=row.get('matchup', row.get('Matchup', 'NEUTRAL')),
                bookmaker=row.get('bookmaker', row.get('Book', None)),
                odds=int(row.get('odds', row.get('Over', -110))),
            )

            if self.record_pick(pick):
                count += 1

        return count

    def record_result(self, date_str: str, player: str, prop_type: str, actual: float) -> bool:
        """Record actual game result for a player prop."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO results (date, player, prop_type, actual)
                    VALUES (?, ?, ?, ?)
                """, (date_str, player, prop_type.lower(), actual))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error recording result: {e}")
            return False

    def get_picks_awaiting_results(self, date_str: str = None) -> pd.DataFrame:
        """Get picks that don't have results yet."""
        query = """
            SELECT p.date, p.player, p.prop_type, p.line, p.pick, p.projection
            FROM picks p
            LEFT JOIN results r ON
                p.date = r.date AND p.player = r.player AND p.prop_type = r.prop_type
            WHERE r.actual IS NULL
        """
        if date_str:
            query += f" AND p.date = '{date_str}'"

        query += " ORDER BY p.date DESC, p.player"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_accuracy_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive accuracy report.
        """
        cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        with sqlite3.connect(self.db_path) as conn:
            # Overall accuracy
            overall = pd.read_sql_query(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN hit = 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN hit IS NULL THEN 1 ELSE 0 END) as pushes,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
            """, conn)

            # By context quality
            by_quality = pd.read_sql_query(f"""
                SELECT
                    CASE
                        WHEN context_quality >= 70 THEN 'HIGH (70+)'
                        WHEN context_quality >= 50 THEN 'MEDIUM (50-69)'
                        ELSE 'LOW (<50)'
                    END as quality_tier,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY quality_tier
                ORDER BY quality_tier DESC
            """, conn)

            # By matchup rating
            by_matchup = pd.read_sql_query(f"""
                SELECT
                    matchup_rating,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY matchup_rating
            """, conn)

            # By B2B status
            by_b2b = pd.read_sql_query(f"""
                SELECT
                    CASE WHEN is_b2b = 1 THEN 'B2B' ELSE 'Not B2B' END as status,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY status
            """, conn)

            # By pick direction
            by_side = pd.read_sql_query(f"""
                SELECT
                    pick,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY pick
            """, conn)

            # By prop type
            by_prop = pd.read_sql_query(f"""
                SELECT
                    prop_type,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY prop_type
            """, conn)

            # By edge tier
            by_edge = pd.read_sql_query(f"""
                SELECT
                    CASE
                        WHEN ABS(edge) >= 15 THEN 'HIGH (15%+)'
                        WHEN ABS(edge) >= 10 THEN 'MEDIUM (10-15%)'
                        ELSE 'LOW (<10%)'
                    END as edge_tier,
                    COUNT(*) as total,
                    AVG(CASE WHEN hit IS NOT NULL THEN hit ELSE NULL END) as win_rate
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL
                GROUP BY edge_tier
                ORDER BY edge_tier DESC
            """, conn)

        return {
            'overall': overall.to_dict('records')[0] if len(overall) > 0 else {},
            'by_quality': by_quality.to_dict('records'),
            'by_matchup': by_matchup.to_dict('records'),
            'by_b2b': by_b2b.to_dict('records'),
            'by_side': by_side.to_dict('records'),
            'by_prop': by_prop.to_dict('records'),
            'by_edge': by_edge.to_dict('records'),
            'days': days,
        }

    def print_accuracy_report(self, days: int = 30):
        """Print formatted accuracy report."""
        report = self.get_accuracy_report(days)

        print("\n" + "=" * 60)
        print(f"          PICK ACCURACY REPORT (Last {days} Days)")
        print("=" * 60)

        overall = report['overall']
        if overall and overall.get('total', 0) > 0:
            win_rate = overall.get('win_rate', 0) or 0
            print(f"\nOVERALL")
            print("-" * 40)
            print(f"  Total Picks:   {overall.get('total', 0)}")
            print(f"  Record:        {overall.get('wins', 0)}-{overall.get('losses', 0)} ({overall.get('pushes', 0)} pushes)")
            print(f"  Win Rate:      {win_rate*100:.1f}%")
        else:
            print("\nNo picks with results found.")
            return

        print(f"\nBY CONTEXT QUALITY")
        print("-" * 40)
        for row in report['by_quality']:
            wr = (row.get('win_rate', 0) or 0) * 100
            print(f"  {row['quality_tier']:15} {row['total']:4} picks  {wr:5.1f}%")

        print(f"\nBY MATCHUP RATING")
        print("-" * 40)
        for row in report['by_matchup']:
            if row['matchup_rating']:
                wr = (row.get('win_rate', 0) or 0) * 100
                print(f"  {row['matchup_rating']:15} {row['total']:4} picks  {wr:5.1f}%")

        print(f"\nBY BACK-TO-BACK STATUS")
        print("-" * 40)
        for row in report['by_b2b']:
            wr = (row.get('win_rate', 0) or 0) * 100
            print(f"  {row['status']:15} {row['total']:4} picks  {wr:5.1f}%")

        print(f"\nBY PICK DIRECTION")
        print("-" * 40)
        for row in report['by_side']:
            wr = (row.get('win_rate', 0) or 0) * 100
            print(f"  {row['pick']:15} {row['total']:4} picks  {wr:5.1f}%")

        print(f"\nBY PROP TYPE")
        print("-" * 40)
        for row in report['by_prop']:
            wr = (row.get('win_rate', 0) or 0) * 100
            print(f"  {row['prop_type'].upper():15} {row['total']:4} picks  {wr:5.1f}%")

        print(f"\nBY EDGE SIZE")
        print("-" * 40)
        for row in report['by_edge']:
            wr = (row.get('win_rate', 0) or 0) * 100
            print(f"  {row['edge_tier']:15} {row['total']:4} picks  {wr:5.1f}%")

        print("\n" + "=" * 60 + "\n")

    def analyze_adjustment_impact(self, days: int = 30) -> pd.DataFrame:
        """
        Analyze which adjustments correlate with winning picks.
        """
        cutoff = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"""
                SELECT adjustments, hit
                FROM picks_with_results
                WHERE date >= '{cutoff}' AND actual IS NOT NULL AND hit IS NOT NULL
            """, conn)

        if df.empty:
            return pd.DataFrame()

        # Parse adjustments and analyze
        results = []
        for adj_name in ['opp_defense', 'location', 'b2b', 'pace', 'total', 'blowout', 'minutes', 'injury_boost']:
            # Get picks with this adjustment active vs not active
            active_wins = 0
            active_total = 0
            inactive_wins = 0
            inactive_total = 0

            for _, row in df.iterrows():
                try:
                    adjs = json.loads(row['adjustments']) if row['adjustments'] else {}
                    adj_val = adjs.get(adj_name, 0)

                    if adj_val != 0:  # Adjustment was active
                        active_total += 1
                        if row['hit']:
                            active_wins += 1
                    else:
                        inactive_total += 1
                        if row['hit']:
                            inactive_wins += 1
                except:
                    continue

            if active_total > 0 and inactive_total > 0:
                results.append({
                    'adjustment': adj_name,
                    'active_total': active_total,
                    'active_win_rate': active_wins / active_total if active_total > 0 else 0,
                    'inactive_total': inactive_total,
                    'inactive_win_rate': inactive_wins / inactive_total if inactive_total > 0 else 0,
                    'difference': (active_wins/active_total - inactive_wins/inactive_total) if active_total > 0 and inactive_total > 0 else 0
                })

        return pd.DataFrame(results).sort_values('difference', ascending=False)

    def enter_results_interactive(self, date_str: str = None):
        """Interactive mode to enter results for pending picks."""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        pending = self.get_picks_awaiting_results(date_str)

        if pending.empty:
            print(f"No pending picks for {date_str}")
            return

        print(f"\n{len(pending)} picks awaiting results for {date_str}")
        print("-" * 50)

        for _, row in pending.iterrows():
            print(f"\n{row['player']} - {row['prop_type'].upper()} {row['pick']} {row['line']}")
            print(f"  Projected: {row['projection']}")

            while True:
                actual_input = input("  Actual (or 's' to skip, 'q' to quit): ").strip()

                if actual_input.lower() == 'q':
                    return
                if actual_input.lower() == 's':
                    break

                try:
                    actual = float(actual_input)
                    self.record_result(row['date'], row['player'], row['prop_type'], actual)

                    # Determine if hit
                    if row['pick'] == 'OVER':
                        hit = actual > row['line']
                    else:
                        hit = actual < row['line']

                    hit_str = "WIN" if hit else "LOSS" if actual != row['line'] else "PUSH"
                    print(f"  Recorded: {actual} ({hit_str})")
                    break
                except ValueError:
                    print("  Invalid input. Enter a number, 's' to skip, or 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(description='NBA Prop Pick Tracker')
    parser.add_argument('--record', type=str, help='Record picks from CSV/Excel file')
    parser.add_argument('--results', action='store_true', help='Enter results interactively')
    parser.add_argument('--date', type=str, help='Date for results (YYYY-MM-DD)')
    parser.add_argument('--accuracy', action='store_true', help='Show accuracy report')
    parser.add_argument('--analyze', action='store_true', help='Analyze adjustment impact')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze (default 30)')
    parser.add_argument('--pending', action='store_true', help='Show picks awaiting results')

    args = parser.parse_args()

    tracker = PickTracker()

    if args.record:
        # Load picks from file
        file_path = Path(args.record)
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return

        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        count = tracker.record_picks_from_df(df, date_str)
        print(f"Recorded {count} picks from {file_path}")

    elif args.results:
        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        tracker.enter_results_interactive(date_str)

    elif args.accuracy:
        tracker.print_accuracy_report(args.days)

    elif args.analyze:
        impact = tracker.analyze_adjustment_impact(args.days)
        if impact.empty:
            print("Not enough data for analysis")
        else:
            print("\n" + "=" * 60)
            print("         ADJUSTMENT IMPACT ANALYSIS")
            print("=" * 60)
            print("\nWhich adjustments correlate with winning picks?\n")
            for _, row in impact.iterrows():
                diff = row['difference'] * 100
                direction = "+" if diff > 0 else ""
                print(f"  {row['adjustment']:15} Active: {row['active_win_rate']*100:.1f}% ({row['active_total']})  "
                      f"Inactive: {row['inactive_win_rate']*100:.1f}% ({row['inactive_total']})  "
                      f"Diff: {direction}{diff:.1f}%")
            print()

    elif args.pending:
        pending = tracker.get_picks_awaiting_results(args.date)
        if pending.empty:
            print("No pending picks")
        else:
            print(f"\n{len(pending)} picks awaiting results:")
            print(pending.to_string(index=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
