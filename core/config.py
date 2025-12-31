"""
Centralized configuration for the NBA prop analysis system.

Consolidates configuration from:
- nba_quickstart.py CONFIG dict
- nba_props_v2.py Config dataclass

Usage:
    from core.config import CONFIG

    print(CONFIG.MIN_EDGE_THRESHOLD)  # 0.03
    print(CONFIG.PREFERRED_BOOKS)     # ['pinnacle', 'draftkings', ...]
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Config:
    """
    Centralized configuration - single source of truth for all settings.

    All values can be overridden via environment variables.
    """

    # =========================================================================
    # API Keys
    # =========================================================================

    # The Odds API (https://the-odds-api.com/)
    # Paid subscription: 20,000 requests/month
    ODDS_API_KEY: str = field(
        default_factory=lambda: os.environ.get('ODDS_API_KEY', '')
    )

    # Perplexity API (https://www.perplexity.ai/)
    # For real-time news intelligence
    PERPLEXITY_API_KEY: str = field(
        default_factory=lambda: os.environ.get('PERPLEXITY_API_KEY', '')
    )

    # =========================================================================
    # Analysis Settings
    # =========================================================================

    # Minimum edge required to recommend a bet (as decimal, 0.03 = 3%)
    MIN_EDGE_THRESHOLD: float = 0.03

    # Minimum confidence required (0-1 scale)
    MIN_CONFIDENCE: float = 0.40

    # Per-prop thresholds (loaded from data/prop_thresholds.json if present)
    PROP_THRESHOLDS_PATH: str = "data/prop_thresholds.json"
    PROP_TYPE_MIN_EDGE: Dict[str, float] = field(default_factory=lambda: {
        'points': 0.03,
        'rebounds': 0.03,
        'assists': 0.03,
        'pra': 0.03,
        'threes': 0.03,
    })
    PROP_TYPE_MIN_CONFIDENCE: Dict[str, float] = field(default_factory=lambda: {
        'points': 0.40,
        'rebounds': 0.40,
        'assists': 0.40,
        'pra': 0.40,
        'threes': 0.40,
    })

    # Excluded prop types (high variance)
    EXCLUDED_PROP_TYPES: List[str] = field(default_factory=lambda: [
        'steals',
        'blocks',
        'turnovers',
    ])

    # Picks selection
    TOP_PICKS_PER_GAME: int = 10
    PICK_RANKING_MODE: str = "edge_confidence"  # edge, edge_confidence

    # Injury risk penalties (applied to edge/confidence for GTD/QUESTIONABLE)
    INJURY_RISK_EDGE_MULTIPLIER: float = 0.85
    INJURY_RISK_CONFIDENCE_MULTIPLIER: float = 0.85

    # Minimum games required for reliable analysis
    MIN_SAMPLE_SIZE: int = 10

    # Number of recent games to analyze
    LOOKBACK_GAMES: int = 15

    # =========================================================================
    # Bankroll Settings
    # =========================================================================

    INITIAL_BANKROLL: float = 1000.0
    UNIT_SIZE: float = 10.0
    MAX_UNITS_PER_BET: int = 3

    # Kelly criterion settings
    KELLY_FRACTION: float = 0.25  # Quarter Kelly for safety
    MAX_BET_PERCENT: float = 0.03  # Max 3% of bankroll per bet

    # =========================================================================
    # API Rate Limits
    # =========================================================================

    # Seconds between NBA API calls (with jitter)
    # Increased to 1.5s to avoid 529 "Site Overloaded" errors from stats.nba.com
    NBA_API_DELAY: float = 1.5

    # Seconds between Odds API calls
    ODDS_API_DELAY: float = 0.1

    # =========================================================================
    # Sportsbooks
    # =========================================================================

    # Preferred sportsbooks (in order of preference for line shopping)
    PREFERRED_BOOKS: List[str] = field(
        default_factory=lambda: [
            'fanduel',
        ]
    )

    # Prop markets to request from Odds API (legacy pipeline)
    PROP_MARKETS: List[str] = field(default_factory=lambda: [
        'player_points',
        'player_rebounds',
        'player_assists',
        'player_points_rebounds_assists',
        'player_threes',
    ])

    # News intelligence toggle (avoid Perplexity by default)
    NEWS_INTELLIGENCE_ENABLED: bool = False

    # Correlation filter toggle (keep all picks by default)
    CORRELATION_FILTER_ENABLED: bool = False

    # =========================================================================
    # Adjustment Factors (for UnifiedPropModel)
    # =========================================================================

    # Home/away adjustment
    HOME_BOOST: float = 1.025  # +2.5% at home
    AWAY_PENALTY: float = 0.975  # -2.5% on road

    # Back-to-back penalty (legacy - use REST_DAY_FACTORS for granular control)
    B2B_PENALTY: float = 0.92  # -8% on back-to-backs

    # =========================================================================
    # Enhanced Rest Days (Gradient instead of binary B2B)
    # =========================================================================

    # Rest day factors by days since last game
    # Key = days rest, Value = multiplier
    REST_DAY_FACTORS: dict = field(default_factory=lambda: {
        0: 0.88,   # Same day (impossible but handle edge case)
        1: 0.92,   # Back-to-back (-8%)
        2: 1.00,   # Normal rest (neutral)
        3: 1.02,   # Well rested (+2%)
        4: 1.03,   # Very rested (+3%)
        5: 1.02,   # 4+ days may have some rust
        6: 1.01,   # Extended rest - slight rust factor
    })

    # =========================================================================
    # Usage Rate & Shot Volume Settings
    # =========================================================================

    # Usage rate factor settings
    LEAGUE_AVG_USG: float = 20.0  # League average usage rate %
    USAGE_FACTOR_WEIGHT: float = 0.3  # How much USG deviation affects projection
    MAX_USAGE_ADJUSTMENT: float = 0.05  # ±5% max for usage rate factor

    # Shot volume trend settings (FGA/minute trend)
    MAX_SHOT_VOLUME_ADJUSTMENT: float = 0.08  # ±8% max for shot volume trend
    SHOT_VOLUME_UP_THRESHOLD: float = 1.03  # >3% increase = trending up
    SHOT_VOLUME_DOWN_THRESHOLD: float = 0.97  # <-3% decrease = trending down

    # =========================================================================
    # True Shooting Efficiency Regression
    # =========================================================================

    # TS% regression to mean settings
    TS_REGRESSION_WEIGHT: float = 0.3  # How much to regress extreme efficiency
    TS_DEVIATION_THRESHOLD: float = 0.05  # 5% deviation triggers regression
    MAX_TS_ADJUSTMENT: float = 0.05  # ±5% max for TS regression factor

    # Maximum adjustment caps (to prevent extreme values)
    MAX_VS_TEAM_ADJUSTMENT: float = 0.10  # ±10% max for vs team history
    MAX_MINUTES_ADJUSTMENT: float = 0.10  # ±10% max for minutes trend
    MAX_INJURY_BOOST: float = 0.15  # +15% max for teammate injuries

    # Blowout risk adjustments
    BLOWOUT_HIGH_PENALTY: float = 0.95  # -5% for spreads >= 12
    BLOWOUT_MEDIUM_PENALTY: float = 0.98  # -2% for spreads 8-11

    # =========================================================================
    # Matchup Rating Thresholds (Defense Rankings)
    # =========================================================================

    # Rank thresholds for defense vs position matchup ratings
    SMASH_RANK: int = 5  # Rank 1-5 = SMASH spot
    GOOD_RANK: int = 10  # Rank 6-10 = GOOD matchup
    HARD_RANK: int = 21  # Rank 21-25 = HARD matchup
    TOUGH_RANK: int = 26  # Rank 26-30 = TOUGH defense

    # =========================================================================
    # Game Total & Pace Settings
    # =========================================================================

    LEAGUE_AVG_TOTAL: float = 225.0  # League average game total
    TOTAL_WEIGHT: float = 0.3  # Weight for game total adjustment on projections
    HIGH_TOTAL_THRESHOLD: float = 235.0  # Above this = HIGH total flag
    LOW_TOTAL_THRESHOLD: float = 215.0  # Below this = LOW total flag

    # Pace thresholds for combined team pace factor
    FAST_PACE_THRESHOLD: float = 1.03  # Above this = FAST PACE flag
    SLOW_PACE_THRESHOLD: float = 0.97  # Below this = SLOW PACE flag

    # =========================================================================
    # Projection Weighting
    # =========================================================================

    # Global projection bias (multiplier applied to all projections)
    # Based on calibration: OVER picks at 65.4%, UNDER at 37.8% → projections too low
    # 1.03 = 3% boost to all projections to correct systematic under-projection
    PROJECTION_BIAS: float = 1.03

    # Weight split for base projection calculation
    RECENT_WEIGHT: float = 0.60  # Weight for last 5 games
    OLDER_WEIGHT: float = 0.40  # Weight for games 6-15
    TREND_MULTIPLIER: float = 0.10  # Apply 10% of trend magnitude

    # Alternative weighting for scan_all_props (mean reversion blend)
    SCAN_RECENT_WEIGHT: float = 0.40  # 40% most recent 5
    SCAN_MID_WEIGHT: float = 0.35  # 35% mid-term games 6-10
    SCAN_SEASON_WEIGHT: float = 0.25  # 25% season average (regression)

    # =========================================================================
    # Trend Thresholds
    # =========================================================================

    TREND_HOT_THRESHOLD: float = 0.05  # +5% above older avg = HOT
    TREND_COLD_THRESHOLD: float = -0.05  # -5% below older avg = COLD

    # =========================================================================
    # Confidence Settings
    # =========================================================================

    MAX_CONFIDENCE: float = 0.95  # Cap confidence at 95%
    MIN_CONFIDENCE_FLOOR: float = 0.20  # Floor confidence at 20%

    # =========================================================================
    # Sample Size Minimums by Prop Type
    # =========================================================================

    MIN_SAMPLE_POINTS: int = 10  # Lower variance prop
    MIN_SAMPLE_REBOUNDS: int = 12
    MIN_SAMPLE_ASSISTS: int = 15  # Higher variance
    MIN_SAMPLE_PRA: int = 10  # Combined stat
    MIN_SAMPLE_THREES: int = 15  # Moderate variance
    MIN_SAMPLE_BLOCKS: int = 20  # High variance
    MIN_SAMPLE_STEALS: int = 20  # High variance
    MIN_SAMPLE_DEFAULT: int = 15  # Fallback for unknown props

    # =========================================================================
    # Valid Prop Types
    # =========================================================================

    VALID_PROP_TYPES: List[str] = field(
        default_factory=lambda: [
            'points', 'rebounds', 'assists', 'pra', 'threes',
            'steals', 'blocks', 'turnovers', 'fg3m'
        ]
    )

    # =========================================================================
    # Caching
    # =========================================================================

    # Cache TTL in seconds
    DEFENSE_CACHE_TTL: int = 3600  # 1 hour for team defense data
    PACE_CACHE_TTL: int = 3600  # 1 hour for pace data
    INJURY_CACHE_TTL: int = 1800  # 30 minutes for injury data
    GAME_LOG_CACHE_TTL: int = 3600  # 1 hour for player game logs

    # =========================================================================
    # Calibration Settings (Model Learning Loop)
    # =========================================================================

    # Minimum samples required before adjusting a factor
    CALIBRATION_MIN_SAMPLES: int = 50

    # Minimum total picks with results before calibration runs
    CALIBRATION_MIN_TOTAL_PICKS: int = 100

    # Conservatism factor (0-1): how much of calculated change to apply
    # 0.5 means only apply 50% of the optimal adjustment
    CALIBRATION_CONSERVATIVE_FACTOR: float = 0.5

    # Maximum allowed change from defaults per calibration run
    CALIBRATION_MAX_CHANGE: float = 0.03  # 3% max change per factor

    # Days of history to analyze for calibration
    CALIBRATION_LOOKBACK_DAYS: int = 90

    # Weight bounds to prevent extreme values
    # Format: (min_value, max_value)
    CALIBRATION_BOUNDS: dict = field(default_factory=lambda: {
        'HOME_BOOST': (1.00, 1.05),           # 0% to +5%
        'AWAY_PENALTY': (0.95, 1.00),         # -5% to 0%
        'B2B_PENALTY': (0.85, 0.95),          # -15% to -5%
        'BLOWOUT_HIGH_PENALTY': (0.90, 1.00), # -10% to 0%
        'BLOWOUT_MEDIUM_PENALTY': (0.95, 1.00),  # -5% to 0%
        'TOTAL_WEIGHT': (0.1, 0.5),           # 10% to 50%
        'TREND_MULTIPLIER': (0.05, 0.20),     # 5% to 20%
        # New factors
        'USAGE_FACTOR_WEIGHT': (0.1, 0.5),    # 10% to 50%
        'MAX_USAGE_ADJUSTMENT': (0.03, 0.10), # 3% to 10%
        'MAX_SHOT_VOLUME_ADJUSTMENT': (0.05, 0.12),  # 5% to 12%
        'TS_REGRESSION_WEIGHT': (0.2, 0.5),   # 20% to 50%
        'MAX_TS_ADJUSTMENT': (0.03, 0.08),    # 3% to 8%
    })

    def __post_init__(self) -> None:
        """Load optional per-prop thresholds from disk."""
        thresholds_path = Path(__file__).resolve().parents[1] / self.PROP_THRESHOLDS_PATH
        if not thresholds_path.exists():
            return
        try:
            payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        edge = payload.get("min_edge") if isinstance(payload, dict) else None
        conf = payload.get("min_confidence") if isinstance(payload, dict) else None
        if isinstance(edge, dict):
            self.PROP_TYPE_MIN_EDGE.update({
                str(k): float(v) for k, v in edge.items() if v is not None
            })
        if isinstance(conf, dict):
            self.PROP_TYPE_MIN_CONFIDENCE.update({
                str(k): float(v) for k, v in conf.items() if v is not None
            })

    def prop_thresholds_for(self, prop_type: str) -> Dict[str, float]:
        """Return per-prop thresholds with defaults."""
        key = (prop_type or "").lower()
        return {
            "min_edge": self.PROP_TYPE_MIN_EDGE.get(key, self.MIN_EDGE_THRESHOLD),
            "min_confidence": self.PROP_TYPE_MIN_CONFIDENCE.get(key, self.MIN_CONFIDENCE),
        }


    def get_min_sample_size(self, prop_type: str) -> int:
        """Get the minimum sample size for a given prop type.

        Args:
            prop_type: The prop type (points, rebounds, assists, etc.)

        Returns:
            Minimum number of games required for reliable analysis
        """
        sample_map = {
            'points': self.MIN_SAMPLE_POINTS,
            'rebounds': self.MIN_SAMPLE_REBOUNDS,
            'assists': self.MIN_SAMPLE_ASSISTS,
            'pra': self.MIN_SAMPLE_PRA,
            'threes': self.MIN_SAMPLE_THREES,
            'fg3m': self.MIN_SAMPLE_THREES,
            'blocks': self.MIN_SAMPLE_BLOCKS,
            'steals': self.MIN_SAMPLE_STEALS,
        }
        return sample_map.get(prop_type.lower(), self.MIN_SAMPLE_DEFAULT)


# Global singleton instance
CONFIG = Config()
