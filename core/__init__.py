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
from .logging_config import setup_logging, get_logger
from .news_intelligence import NewsIntelligence, NewsContext, create_perplexity_search
from .rosters import (
    PlayerRoster,
    TeamRoster,
    TEAM_ROSTERS,
    load_rosters_from_json,
    save_rosters_to_json,
    get_player_info,
    get_team_starters,
    get_team_stars,
    is_player_starter,
    get_player_avg_minutes,
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
    # Logging
    'setup_logging',
    'get_logger',
    # News Intelligence
    'NewsIntelligence',
    'NewsContext',
    'create_perplexity_search',
    # Rosters
    'PlayerRoster',
    'TeamRoster',
    'TEAM_ROSTERS',
    'load_rosters_from_json',
    'save_rosters_to_json',
    'get_player_info',
    'get_team_starters',
    'get_team_stars',
    'is_player_starter',
    'get_player_avg_minutes',
]
