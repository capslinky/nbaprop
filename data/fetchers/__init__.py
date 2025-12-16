"""Data fetchers for NBA stats, odds, and injury data.

This package provides:
- ResilientFetcher: Retry logic and connection pooling
- NBADataFetcher: NBA stats API integration
- OddsAPIClient: Betting odds from The Odds API
- InjuryTracker: Player injury tracking and teammate boosts
- AsyncDataFetcher: Async parallel data fetching for improved performance
"""

from .resilient import ResilientFetcher, get_resilient_fetcher
from .nba_fetcher import NBADataFetcher
from .injury_tracker import InjuryTracker
from .odds_fetcher import OddsAPIClient
from .async_fetcher import (
    AsyncDataFetcher,
    PlayerContext,
    BatchAnalysisResult,
    analyze_props_parallel,
    run_parallel_analysis,
)

__all__ = [
    # Sync fetchers
    'ResilientFetcher',
    'get_resilient_fetcher',
    'NBADataFetcher',
    'InjuryTracker',
    'OddsAPIClient',
    # Async fetchers
    'AsyncDataFetcher',
    'PlayerContext',
    'BatchAnalysisResult',
    'analyze_props_parallel',
    'run_parallel_analysis',
]
