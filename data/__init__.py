"""
Data module - Data fetching and integration layer.

This module provides access to:
- NBADataFetcher: Player game logs and team stats from NBA API
- OddsAPIClient: Live betting lines from The Odds API
- InjuryTracker: Player injury status and teammate boosts
- ResilientFetcher: Retry logic with exponential backoff

Re-exports from nba_integrations.py for backward compatibility.
Classes will be incrementally migrated to individual files.

Usage:
    from data import NBADataFetcher, OddsAPIClient, InjuryTracker

    fetcher = NBADataFetcher()
    logs = fetcher.get_player_game_logs("Luka Doncic")

    odds = OddsAPIClient(api_key="YOUR_KEY")
    props = odds.get_player_props()

    injuries = InjuryTracker()
    status = injuries.get_player_status("LeBron James")
"""

# Re-export from nba_integrations.py
# This maintains backward compatibility while establishing the new module structure
from nba_integrations import (
    # Data fetchers
    NBADataFetcher,
    ResilientFetcher,
    get_resilient_fetcher,

    # Odds integration
    OddsAPIClient,

    # Injury tracking
    InjuryTracker,

    # Helper functions
    get_current_nba_season,
)

# Also import from core for consolidated utilities
from core.constants import (
    TEAM_ABBREVIATIONS,
    STAR_PLAYERS,
    normalize_team_abbrev,
    get_season_from_date,
)

__all__ = [
    # Data fetchers
    'NBADataFetcher',
    'ResilientFetcher',
    'get_resilient_fetcher',

    # Odds
    'OddsAPIClient',

    # Injuries
    'InjuryTracker',

    # Helpers
    'get_current_nba_season',

    # From core (convenience re-exports)
    'TEAM_ABBREVIATIONS',
    'STAR_PLAYERS',
    'normalize_team_abbrev',
    'get_season_from_date',
]
