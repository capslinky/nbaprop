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

import os
from dataclasses import dataclass, field
from typing import List


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

    # =========================================================================
    # Analysis Settings
    # =========================================================================

    # Minimum edge required to recommend a bet (as decimal, 0.03 = 3%)
    MIN_EDGE_THRESHOLD: float = 0.03

    # Minimum confidence required (0-1 scale)
    MIN_CONFIDENCE: float = 0.40

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
            'pinnacle',
            'draftkings',
            'fanduel',
            'betmgm',
            'caesars',
        ]
    )

    # =========================================================================
    # Adjustment Factors (for UnifiedPropModel)
    # =========================================================================

    # Home/away adjustment
    HOME_BOOST: float = 1.025  # +2.5% at home
    AWAY_PENALTY: float = 0.975  # -2.5% on road

    # Back-to-back penalty
    B2B_PENALTY: float = 0.92  # -8% on back-to-backs

    # Maximum adjustment caps (to prevent extreme values)
    MAX_VS_TEAM_ADJUSTMENT: float = 0.10  # ±10% max for vs team history
    MAX_MINUTES_ADJUSTMENT: float = 0.10  # ±10% max for minutes trend
    MAX_INJURY_BOOST: float = 0.15  # +15% max for teammate injuries

    # =========================================================================
    # Caching
    # =========================================================================

    # Cache TTL in seconds
    DEFENSE_CACHE_TTL: int = 3600  # 1 hour for team defense data
    PACE_CACHE_TTL: int = 3600  # 1 hour for pace data
    INJURY_CACHE_TTL: int = 1800  # 30 minutes for injury data
    GAME_LOG_CACHE_TTL: int = 3600  # 1 hour for player game logs


# Global singleton instance
CONFIG = Config()
