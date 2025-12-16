"""
Data module - Data fetching and integration layer.

This module provides access to:
- NBADataFetcher: Player game logs and team stats from NBA API
- OddsAPIClient: Live betting lines from The Odds API
- InjuryTracker: Player injury status and teammate boosts
- ResilientFetcher: Retry logic with exponential backoff
- AsyncDataFetcher: Async parallel data fetching for improved performance

Usage:
    from data import NBADataFetcher, OddsAPIClient, InjuryTracker

    fetcher = NBADataFetcher()
    logs = fetcher.get_player_game_logs("Luka Doncic")

    odds = OddsAPIClient(api_key="YOUR_KEY")
    props = odds.get_player_props()

    injuries = InjuryTracker()
    status = injuries.get_player_status("LeBron James")

Async Usage:
    import asyncio
    from data import AsyncDataFetcher, run_parallel_analysis

    # Option 1: Use convenience function
    results = run_parallel_analysis(props_list, model)

    # Option 2: Use async/await directly
    async def main():
        fetcher = AsyncDataFetcher(odds_api_key="YOUR_KEY")
        contexts = await fetcher.fetch_multi_player_context(players)
    asyncio.run(main())
"""

# Import from the new modular fetchers package
from data.fetchers import (
    # Sync data fetchers
    NBADataFetcher,
    ResilientFetcher,
    get_resilient_fetcher,

    # Odds integration
    OddsAPIClient,

    # Injury tracking
    InjuryTracker,

    # Async fetchers
    AsyncDataFetcher,
    PlayerContext,
    BatchAnalysisResult,
    analyze_props_parallel,
    run_parallel_analysis,
)

# Import helper function from core
from core.constants import get_current_nba_season

# Also import from core for consolidated utilities
from core.constants import (
    TEAM_ABBREVIATIONS,
    STAR_PLAYERS,
    normalize_team_abbrev,
    get_season_from_date,
)

__all__ = [
    # Sync data fetchers
    'NBADataFetcher',
    'ResilientFetcher',
    'get_resilient_fetcher',

    # Odds
    'OddsAPIClient',

    # Injuries
    'InjuryTracker',

    # Async fetchers
    'AsyncDataFetcher',
    'PlayerContext',
    'BatchAnalysisResult',
    'analyze_props_parallel',
    'run_parallel_analysis',

    # Helpers
    'get_current_nba_season',

    # From core (convenience re-exports)
    'TEAM_ABBREVIATIONS',
    'STAR_PLAYERS',
    'normalize_team_abbrev',
    'get_season_from_date',
]
