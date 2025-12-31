"""
Calibration Analyzer
====================
Statistical analysis of historical pick performance for model calibration.

Extends the pick_tracker analysis to calculate:
- Win rates when adjustments are active vs inactive
- Correlation between adjustment magnitude and outcomes
- Optimal factor recommendations

Usage:
    from calibration.analyzer import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer()
    results = analyzer.calculate_optimal_factors(days=90)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import numpy as np

from core.config import CONFIG

logger = logging.getLogger(__name__)


# Default database path (same as pick_tracker)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "clv_tracking.db"

# Adjustment factor names that we track and calibrate
CALIBRATABLE_FACTORS = [
    'HOME_BOOST',
    'AWAY_PENALTY',
    'B2B_PENALTY',
    'BLOWOUT_HIGH_PENALTY',
    'BLOWOUT_MEDIUM_PENALTY',
    'TOTAL_WEIGHT',
    'TREND_MULTIPLIER',
    # New v2.0 factors
    'USAGE_FACTOR_WEIGHT',
    'MAX_USAGE_ADJUSTMENT',
    'MAX_SHOT_VOLUME_ADJUSTMENT',
    'TS_REGRESSION_WEIGHT',
    'MAX_TS_ADJUSTMENT',
]

# Mapping from adjustment JSON keys to CONFIG factor names
ADJUSTMENT_TO_FACTOR = {
    'location': ['HOME_BOOST', 'AWAY_PENALTY'],
    'b2b': ['B2B_PENALTY'],
    'rest': ['B2B_PENALTY'],  # rest is new key for gradient rest days
    'blowout': ['BLOWOUT_HIGH_PENALTY', 'BLOWOUT_MEDIUM_PENALTY'],
    'total': ['TOTAL_WEIGHT'],
    # New v2.0 mappings
    'usage_rate': ['USAGE_FACTOR_WEIGHT', 'MAX_USAGE_ADJUSTMENT'],
    'shot_volume': ['MAX_SHOT_VOLUME_ADJUSTMENT'],
    'ts_regression': ['TS_REGRESSION_WEIGHT', 'MAX_TS_ADJUSTMENT'],
    # pace and opp_defense are dynamic, not directly calibratable
}


@dataclass
class FactorStats:
    """Statistics for a single adjustment factor."""
    factor_name: str
    active_count: int
    inactive_count: int
    active_win_rate: Optional[float]
    inactive_win_rate: Optional[float]
    improvement: float  # active_win_rate - inactive_win_rate
    sample_sufficient: bool
    quality: str  # high, medium, low, insufficient_data

    def to_dict(self) -> dict:
        return {
            'factor_name': self.factor_name,
            'active_count': self.active_count,
            'inactive_count': self.inactive_count,
            'active_win_rate': self.active_win_rate,
            'inactive_win_rate': self.inactive_win_rate,
            'improvement': self.improvement,
            'sample_sufficient': self.sample_sufficient,
            'quality': self.quality,
        }


@dataclass
class CalibrationResult:
    """Complete calibration analysis result."""
    total_picks: int
    picks_with_results: int
    date_range: Tuple[str, str]
    factor_stats: Dict[str, FactorStats]
    overall_win_rate: float
    has_sufficient_data: bool

    def to_dict(self) -> dict:
        return {
            'total_picks': self.total_picks,
            'picks_with_results': self.picks_with_results,
            'date_range': self.date_range,
            'factor_stats': {k: v.to_dict() for k, v in self.factor_stats.items()},
            'overall_win_rate': self.overall_win_rate,
            'has_sufficient_data': self.has_sufficient_data,
        }


class CalibrationAnalyzer:
    """
    Analyzes historical pick performance for model calibration.

    Extends pick_tracker.analyze_adjustment_impact() with:
    - More detailed statistics per factor
    - Confidence interval calculations
    - Factor quality ratings
    - Recommendations for optimal values
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize the analyzer.

        Args:
            db_path: Path to the SQLite database. Defaults to clv_tracking.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def has_sufficient_data(self) -> bool:
        """
        Check if there's enough data for calibration.

        Returns:
            True if minimum data requirements are met
        """
        if not self.db_path.exists():
            return False

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM picks p
                    JOIN results r ON p.date = r.date
                        AND p.player = r.player
                        AND p.prop_type = r.prop_type
                """)
                count = cursor.fetchone()[0]
                return count >= CONFIG.CALIBRATION_MIN_TOTAL_PICKS
        except sqlite3.Error:
            return False

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available data.

        Returns:
            Dictionary with data summary statistics
        """
        if not self.db_path.exists():
            return {'exists': False, 'total_picks': 0, 'picks_with_results': 0}

        try:
            with self._get_connection() as conn:
                # Total picks
                total = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]

                # Picks with results
                with_results = conn.execute("""
                    SELECT COUNT(*) FROM picks p
                    JOIN results r ON p.date = r.date
                        AND p.player = r.player
                        AND p.prop_type = r.prop_type
                """).fetchone()[0]

                # Date range
                dates = conn.execute("""
                    SELECT MIN(date), MAX(date) FROM picks
                """).fetchone()

                return {
                    'exists': True,
                    'total_picks': total,
                    'picks_with_results': with_results,
                    'date_range': (dates[0], dates[1]) if dates[0] else (None, None),
                    'sufficient_for_calibration': with_results >= CONFIG.CALIBRATION_MIN_TOTAL_PICKS,
                }
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return {'exists': False, 'error': str(e)}

    def get_picks_with_results(self, days: int = None) -> pd.DataFrame:
        """
        Get all picks with their results.

        Args:
            days: Number of days to look back. None for all data.

        Returns:
            DataFrame with picks and results
        """
        if not self.db_path.exists():
            return pd.DataFrame()

        query = """
            SELECT
                p.date,
                p.player,
                p.prop_type,
                p.line,
                p.pick,
                p.edge,
                p.confidence,
                p.projection,
                p.adjustments,
                p.is_home,
                p.is_b2b,
                p.matchup_rating,
                r.actual,
                CASE
                    WHEN p.pick = 'OVER' AND r.actual > p.line THEN 1
                    WHEN p.pick = 'UNDER' AND r.actual < p.line THEN 1
                    WHEN r.actual = p.line THEN NULL
                    ELSE 0
                END as hit
            FROM picks p
            JOIN results r ON p.date = r.date
                AND p.player = r.player
                AND p.prop_type = r.prop_type
        """

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            query += f" WHERE p.date >= '{cutoff}'"

        query += " ORDER BY p.date"

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return pd.DataFrame()

    def _parse_adjustments(self, adjustments_json: str) -> Dict[str, float]:
        """Parse adjustments JSON string."""
        if not adjustments_json:
            return {}
        try:
            return json.loads(adjustments_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def analyze_location_factor(self, df: pd.DataFrame) -> Dict[str, FactorStats]:
        """
        Analyze home/away adjustment performance.

        Returns:
            Dictionary with HOME_BOOST and AWAY_PENALTY stats
        """
        results = {}

        # Get picks where we know home/away status
        df_with_location = df[df['is_home'].notna()].copy()

        if len(df_with_location) < CONFIG.CALIBRATION_MIN_SAMPLES:
            # Insufficient data
            for factor in ['HOME_BOOST', 'AWAY_PENALTY']:
                results[factor] = FactorStats(
                    factor_name=factor,
                    active_count=0,
                    inactive_count=0,
                    active_win_rate=None,
                    inactive_win_rate=None,
                    improvement=0.0,
                    sample_sufficient=False,
                    quality='insufficient_data',
                )
            return results

        # Analyze home games
        home_picks = df_with_location[df_with_location['is_home'] == 1]
        away_picks = df_with_location[df_with_location['is_home'] == 0]

        home_hits = home_picks['hit'].dropna()
        away_hits = away_picks['hit'].dropna()

        home_win_rate = home_hits.mean() if len(home_hits) > 0 else None
        away_win_rate = away_hits.mean() if len(away_hits) > 0 else None

        # Overall baseline
        overall_win_rate = df_with_location['hit'].dropna().mean()

        # HOME_BOOST: compare home performance to overall
        home_improvement = (home_win_rate - overall_win_rate) if home_win_rate and overall_win_rate else 0
        results['HOME_BOOST'] = FactorStats(
            factor_name='HOME_BOOST',
            active_count=len(home_hits),
            inactive_count=len(away_hits),
            active_win_rate=home_win_rate,
            inactive_win_rate=away_win_rate,
            improvement=home_improvement,
            sample_sufficient=len(home_hits) >= CONFIG.CALIBRATION_MIN_SAMPLES,
            quality=self._get_quality(len(home_hits)),
        )

        # AWAY_PENALTY: compare away performance to overall
        away_improvement = (away_win_rate - overall_win_rate) if away_win_rate and overall_win_rate else 0
        results['AWAY_PENALTY'] = FactorStats(
            factor_name='AWAY_PENALTY',
            active_count=len(away_hits),
            inactive_count=len(home_hits),
            active_win_rate=away_win_rate,
            inactive_win_rate=home_win_rate,
            improvement=away_improvement,
            sample_sufficient=len(away_hits) >= CONFIG.CALIBRATION_MIN_SAMPLES,
            quality=self._get_quality(len(away_hits)),
        )

        return results

    def analyze_b2b_factor(self, df: pd.DataFrame) -> FactorStats:
        """
        Analyze back-to-back adjustment performance.

        Returns:
            FactorStats for B2B_PENALTY
        """
        df_with_b2b = df[df['is_b2b'].notna()].copy()

        b2b_picks = df_with_b2b[df_with_b2b['is_b2b'] == 1]
        non_b2b_picks = df_with_b2b[df_with_b2b['is_b2b'] == 0]

        b2b_hits = b2b_picks['hit'].dropna()
        non_b2b_hits = non_b2b_picks['hit'].dropna()

        b2b_win_rate = b2b_hits.mean() if len(b2b_hits) > 0 else None
        non_b2b_win_rate = non_b2b_hits.mean() if len(non_b2b_hits) > 0 else None

        improvement = 0.0
        if b2b_win_rate is not None and non_b2b_win_rate is not None:
            # Negative improvement means B2B hurts (as expected)
            improvement = b2b_win_rate - non_b2b_win_rate

        return FactorStats(
            factor_name='B2B_PENALTY',
            active_count=len(b2b_hits),
            inactive_count=len(non_b2b_hits),
            active_win_rate=b2b_win_rate,
            inactive_win_rate=non_b2b_win_rate,
            improvement=improvement,
            sample_sufficient=len(b2b_hits) >= CONFIG.CALIBRATION_MIN_SAMPLES,
            quality=self._get_quality(len(b2b_hits)),
        )

    def analyze_adjustment_factor(
        self, df: pd.DataFrame, adj_key: str, factor_name: str
    ) -> FactorStats:
        """
        Analyze a generic adjustment factor from the adjustments JSON.

        Args:
            df: DataFrame with picks and results
            adj_key: Key in the adjustments JSON (e.g., 'total', 'blowout')
            factor_name: CONFIG factor name (e.g., 'TOTAL_WEIGHT')

        Returns:
            FactorStats for the factor
        """
        # Parse adjustments and check if factor was active
        def is_active(adj_json):
            adjs = self._parse_adjustments(adj_json)
            if not adjs or not isinstance(adjs, dict):
                return False
            val = adjs.get(adj_key, 0)
            # Consider active if significantly different from 1.0 (neutral)
            return abs(val - 1.0) > 0.01 if val else False

        df_copy = df.copy()
        df_copy['factor_active'] = df_copy['adjustments'].apply(is_active)

        active_picks = df_copy[df_copy['factor_active'] == True]
        inactive_picks = df_copy[df_copy['factor_active'] == False]

        active_hits = active_picks['hit'].dropna()
        inactive_hits = inactive_picks['hit'].dropna()

        active_win_rate = active_hits.mean() if len(active_hits) > 0 else None
        inactive_win_rate = inactive_hits.mean() if len(inactive_hits) > 0 else None

        improvement = 0.0
        if active_win_rate is not None and inactive_win_rate is not None:
            improvement = active_win_rate - inactive_win_rate

        return FactorStats(
            factor_name=factor_name,
            active_count=len(active_hits),
            inactive_count=len(inactive_hits),
            active_win_rate=active_win_rate,
            inactive_win_rate=inactive_win_rate,
            improvement=improvement,
            sample_sufficient=len(active_hits) >= CONFIG.CALIBRATION_MIN_SAMPLES,
            quality=self._get_quality(len(active_hits)),
        )

    def _get_quality(self, sample_size: int) -> str:
        """Determine quality rating based on sample size."""
        if sample_size < CONFIG.CALIBRATION_MIN_SAMPLES:
            return 'insufficient_data'
        elif sample_size < CONFIG.CALIBRATION_MIN_SAMPLES * 2:
            return 'low'
        elif sample_size < CONFIG.CALIBRATION_MIN_SAMPLES * 4:
            return 'medium'
        else:
            return 'high'

    def calculate_all_factor_stats(self, days: int = None) -> CalibrationResult:
        """
        Calculate statistics for all calibratable factors.

        Args:
            days: Number of days to analyze. None for all data.

        Returns:
            CalibrationResult with all factor statistics
        """
        if days is None:
            days = CONFIG.CALIBRATION_LOOKBACK_DAYS
        elif days <= 0:
            days = None
        df = self.get_picks_with_results(days)

        if df.empty:
            return CalibrationResult(
                total_picks=0,
                picks_with_results=0,
                date_range=('', ''),
                factor_stats={},
                overall_win_rate=0.0,
                has_sufficient_data=False,
            )

        # Get date range
        date_range = (df['date'].min(), df['date'].max())

        # Overall win rate
        overall_win_rate = df['hit'].dropna().mean()

        # Analyze each factor
        factor_stats = {}

        # Location factors (HOME_BOOST, AWAY_PENALTY)
        location_stats = self.analyze_location_factor(df)
        factor_stats.update(location_stats)

        # B2B factor
        factor_stats['B2B_PENALTY'] = self.analyze_b2b_factor(df)

        # Blowout factors (analyzed together)
        blowout_stats = self.analyze_adjustment_factor(df, 'blowout', 'BLOWOUT_HIGH_PENALTY')
        factor_stats['BLOWOUT_HIGH_PENALTY'] = blowout_stats
        # Use same stats for medium (we can't easily distinguish high vs medium from data)
        factor_stats['BLOWOUT_MEDIUM_PENALTY'] = FactorStats(
            factor_name='BLOWOUT_MEDIUM_PENALTY',
            active_count=blowout_stats.active_count,
            inactive_count=blowout_stats.inactive_count,
            active_win_rate=blowout_stats.active_win_rate,
            inactive_win_rate=blowout_stats.inactive_win_rate,
            improvement=blowout_stats.improvement,
            sample_sufficient=blowout_stats.sample_sufficient,
            quality=blowout_stats.quality,
        )

        # Total weight factor
        factor_stats['TOTAL_WEIGHT'] = self.analyze_adjustment_factor(
            df, 'total', 'TOTAL_WEIGHT'
        )

        # Trend multiplier - analyze based on trend flags in the data
        # For now, mark as needing more sophisticated analysis
        factor_stats['TREND_MULTIPLIER'] = FactorStats(
            factor_name='TREND_MULTIPLIER',
            active_count=0,
            inactive_count=len(df),
            active_win_rate=None,
            inactive_win_rate=overall_win_rate,
            improvement=0.0,
            sample_sufficient=False,
            quality='insufficient_data',  # Needs more sophisticated tracking
        )

        # ==================== NEW v2.0 FACTORS ====================

        # Usage Rate Factor - analyze when usage_rate adjustment is active
        usage_stats = self.analyze_adjustment_factor(df, 'usage_rate', 'USAGE_FACTOR_WEIGHT')
        factor_stats['USAGE_FACTOR_WEIGHT'] = usage_stats
        # MAX_USAGE_ADJUSTMENT uses same data
        factor_stats['MAX_USAGE_ADJUSTMENT'] = FactorStats(
            factor_name='MAX_USAGE_ADJUSTMENT',
            active_count=usage_stats.active_count,
            inactive_count=usage_stats.inactive_count,
            active_win_rate=usage_stats.active_win_rate,
            inactive_win_rate=usage_stats.inactive_win_rate,
            improvement=usage_stats.improvement,
            sample_sufficient=usage_stats.sample_sufficient,
            quality=usage_stats.quality,
        )

        # Shot Volume Factor
        shot_vol_stats = self.analyze_adjustment_factor(df, 'shot_volume', 'MAX_SHOT_VOLUME_ADJUSTMENT')
        factor_stats['MAX_SHOT_VOLUME_ADJUSTMENT'] = shot_vol_stats

        # True Shooting Regression Factor
        ts_stats = self.analyze_adjustment_factor(df, 'ts_regression', 'TS_REGRESSION_WEIGHT')
        factor_stats['TS_REGRESSION_WEIGHT'] = ts_stats
        # MAX_TS_ADJUSTMENT uses same data
        factor_stats['MAX_TS_ADJUSTMENT'] = FactorStats(
            factor_name='MAX_TS_ADJUSTMENT',
            active_count=ts_stats.active_count,
            inactive_count=ts_stats.inactive_count,
            active_win_rate=ts_stats.active_win_rate,
            inactive_win_rate=ts_stats.inactive_win_rate,
            improvement=ts_stats.improvement,
            sample_sufficient=ts_stats.sample_sufficient,
            quality=ts_stats.quality,
        )

        # Check if we have sufficient data overall
        has_sufficient = len(df) >= CONFIG.CALIBRATION_MIN_TOTAL_PICKS

        return CalibrationResult(
            total_picks=len(df),
            picks_with_results=len(df[df['hit'].notna()]),
            date_range=date_range,
            factor_stats=factor_stats,
            overall_win_rate=overall_win_rate,
            has_sufficient_data=has_sufficient,
        )

    def get_factor_recommendations(
        self, days: int = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get recommendations for each factor based on historical performance.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary mapping factor names to recommendation data
        """
        result = self.calculate_all_factor_stats(days)

        recommendations = {}
        for factor_name, stats in result.factor_stats.items():
            default_value = getattr(CONFIG, factor_name, None)
            if default_value is None:
                continue

            recommendations[factor_name] = {
                'current_default': default_value,
                'sample_size': stats.active_count,
                'win_rate_active': stats.active_win_rate,
                'win_rate_inactive': stats.inactive_win_rate,
                'improvement': stats.improvement,
                'quality': stats.quality,
                'should_calibrate': stats.sample_sufficient and stats.quality != 'insufficient_data',
            }

        return recommendations
