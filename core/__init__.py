"""
Core module - Shared foundations for the NBA prop analysis system.

This module provides:
- Centralized configuration (config.py)
- Team mappings and constants (constants.py)
- Odds conversion utilities (odds_utils.py)
- Custom exceptions (exceptions.py)
"""

from .config import Config, CONFIG
from .constants import (
    TEAM_ABBREVIATIONS,
    STAR_PLAYERS,
    STAR_OUT_BOOST,
    normalize_team_abbrev,
    get_current_nba_season,
    get_season_from_date,
)
from .odds_utils import (
    american_to_decimal,
    american_to_implied_prob,
    remove_vig,
    calculate_breakeven_winrate,
    calculate_ev,
    kelly_criterion,
    calculate_edge,
    calculate_confidence,
    calculate_prob_over,
    calculate_confidence_interval,
)
from .exceptions import (
    NBAPropError,
    DataFetchError,
    PlayerNotFoundError,
    InsufficientDataError,
    OddsAPIError,
    RateLimitError,
)

__all__ = [
    # Config
    'Config',
    'CONFIG',
    # Constants
    'TEAM_ABBREVIATIONS',
    'STAR_PLAYERS',
    'STAR_OUT_BOOST',
    'normalize_team_abbrev',
    'get_current_nba_season',
    'get_season_from_date',
    # Odds utilities
    'american_to_decimal',
    'american_to_implied_prob',
    'remove_vig',
    'calculate_breakeven_winrate',
    'calculate_ev',
    'kelly_criterion',
    'calculate_edge',
    'calculate_confidence',
    'calculate_prob_over',
    'calculate_confidence_interval',
    # Exceptions
    'NBAPropError',
    'DataFetchError',
    'PlayerNotFoundError',
    'InsufficientDataError',
    'OddsAPIError',
    'RateLimitError',
]
