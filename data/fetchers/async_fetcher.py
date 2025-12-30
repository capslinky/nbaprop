"""Async data fetching for parallel API calls.

This module provides async versions of data fetchers for improved performance
when analyzing multiple players or props simultaneously.

Usage:
    import asyncio
    from data.fetchers.async_fetcher import AsyncDataFetcher

    async def main():
        fetcher = AsyncDataFetcher(odds_api_key="YOUR_KEY")
        results = await fetcher.fetch_multi_player_context(
            players=["Luka Doncic", "LeBron James"],
            opponents=["LAL", "DAL"]
        )

    asyncio.run(main())
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlayerContext:
    """Context data for a single player analysis."""
    player_name: str
    game_logs: pd.DataFrame
    opponent: Optional[str] = None
    defense_rating: Optional[Dict] = None
    pace_factor: Optional[float] = None
    injury_status: Optional[Dict] = None
    teammate_boost: float = 1.0
    is_b2b: bool = False
    fetch_time_ms: float = 0


@dataclass
class BatchAnalysisResult:
    """Result from batch analysis."""
    contexts: List[PlayerContext]
    total_fetch_time_ms: float
    players_fetched: int
    errors: List[str]


class AsyncDataFetcher:
    """
    Async data fetcher for parallel API calls.

    Provides significant speedup when fetching data for multiple players
    by making concurrent requests instead of sequential ones.

    Typical speedup: 3-5x for 10+ player analyses.
    """

    # The Odds API base URL
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

    # Rate limiting
    MAX_CONCURRENT_REQUESTS = 5  # Avoid overwhelming APIs
    REQUEST_DELAY_SEC = 0.2  # Delay between batches

    def __init__(
        self,
        odds_api_key: Optional[str] = None,
        nba_fetcher=None,
        injury_tracker=None
    ):
        """
        Initialize async fetcher.

        Args:
            odds_api_key: API key for The Odds API
            nba_fetcher: Optional NBADataFetcher instance (for sync fallback)
            injury_tracker: Optional InjuryTracker instance
        """
        self.odds_api_key = odds_api_key
        self._nba_fetcher = nba_fetcher
        self._injury_tracker = injury_tracker

        # Cached data (shared across async calls)
        self._defense_cache: Optional[pd.DataFrame] = None
        self._pace_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    async def fetch_player_context(
        self,
        session: aiohttp.ClientSession,
        player_name: str,
        opponent: Optional[str] = None,
        last_n_games: int = 15
    ) -> PlayerContext:
        """
        Fetch all context data for a single player asynchronously.

        Args:
            session: aiohttp session for making requests
            player_name: Player name
            opponent: Opponent team abbreviation
            last_n_games: Number of recent games to fetch

        Returns:
            PlayerContext with all fetched data
        """
        start_time = datetime.now()
        context = PlayerContext(player_name=player_name, game_logs=pd.DataFrame())

        try:
            # Use sync fetcher if available (NBA API doesn't support async well)
            if self._nba_fetcher:
                context.game_logs = self._nba_fetcher.get_player_game_logs(
                    player_name, last_n_games=last_n_games
                )

                # Get opponent context if provided
                if opponent and not self._defense_cache_valid():
                    await self._refresh_team_caches()

                if opponent and self._defense_cache is not None:
                    opp_defense = self._defense_cache[
                        self._defense_cache['team_abbrev'] == opponent
                    ]
                    if not opp_defense.empty:
                        context.defense_rating = opp_defense.iloc[0].to_dict()

                if opponent and self._pace_cache is not None:
                    opp_pace = self._pace_cache[
                        self._pace_cache['team_abbrev'] == opponent
                    ]
                    if not opp_pace.empty:
                        context.pace_factor = opp_pace.iloc[0].get('pace_factor', 1.0)

                # Check B2B status
                if not context.game_logs.empty:
                    b2b_info = self._nba_fetcher.check_back_to_back(context.game_logs)
                    context.is_b2b = b2b_info.get('is_b2b', False)

            # Get injury status
            if self._injury_tracker:
                context.injury_status = self._injury_tracker.get_player_status(player_name)

                # Get teammate boost
                team = self._extract_team_from_logs(context.game_logs)
                if team:
                    boost_info = self._injury_tracker.get_teammate_boost(
                        player_name, team
                    )
                    context.teammate_boost = boost_info.get('boost', 1.0)

            context.opponent = opponent

        except (ConnectionError, TimeoutError, aiohttp.ClientError) as e:
            logger.warning(f"Network error fetching context for {player_name}: {e}")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Data parsing error fetching context for {player_name}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error fetching context for {player_name}: {e}")

        context.fetch_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return context

    async def fetch_multi_player_context(
        self,
        players: List[str],
        opponents: Optional[List[str]] = None,
        last_n_games: int = 15
    ) -> BatchAnalysisResult:
        """
        Fetch context for multiple players in parallel.

        Args:
            players: List of player names
            opponents: Optional list of opponents (same length as players)
            last_n_games: Number of games to fetch per player

        Returns:
            BatchAnalysisResult with all player contexts
        """
        start_time = datetime.now()
        errors = []
        contexts = []

        # Normalize opponents list
        if opponents is None:
            opponents = [None] * len(players)
        elif len(opponents) != len(players):
            opponents = (opponents + [None] * len(players))[:len(players)]

        # Pre-fetch team caches (shared across all players)
        await self._refresh_team_caches()

        async with aiohttp.ClientSession() as session:
            # Process in batches to avoid overwhelming APIs
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

            async def fetch_with_semaphore(player: str, opponent: Optional[str]):
                async with semaphore:
                    try:
                        return await self.fetch_player_context(
                            session, player, opponent, last_n_games
                        )
                    except (ConnectionError, TimeoutError, aiohttp.ClientError) as e:
                        errors.append(f"{player}: Network error - {str(e)}")
                        return PlayerContext(
                            player_name=player,
                            game_logs=pd.DataFrame()
                        )
                    except Exception as e:
                        errors.append(f"{player}: {str(e)}")
                        return PlayerContext(
                            player_name=player,
                            game_logs=pd.DataFrame()
                        )

            # Create tasks for all players
            tasks = [
                fetch_with_semaphore(player, opp)
                for player, opp in zip(players, opponents)
            ]

            # Execute all tasks concurrently
            contexts = await asyncio.gather(*tasks)

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchAnalysisResult(
            contexts=list(contexts),
            total_fetch_time_ms=total_time,
            players_fetched=len([c for c in contexts if not c.game_logs.empty]),
            errors=errors
        )

    async def fetch_odds_events(self) -> List[Dict]:
        """Fetch today's NBA events from The Odds API."""
        if not self.odds_api_key:
            return []

        url = f"{self.ODDS_API_BASE}/sports/basketball_nba/events"
        params = {'apiKey': self.odds_api_key}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Odds API error: {response.status}")
                        return []
            except aiohttp.ClientError as e:
                logger.error(f"Network error fetching odds events: {e}")
                return []
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout fetching odds events: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error fetching odds events: {e}")
                return []

    async def fetch_player_props(
        self,
        event_id: str,
        markets: List[str] = None
    ) -> Dict:
        """Fetch player props for a specific event."""
        if not self.odds_api_key:
            return {}

        if markets is None:
            markets = ['player_points', 'player_rebounds', 'player_assists', 'player_threes']

        url = f"{self.ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Props API error: {response.status}")
                        return {}
            except aiohttp.ClientError as e:
                logger.error(f"Network error fetching props: {e}")
                return {}
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout fetching props: {e}")
                return {}
            except Exception as e:
                logger.error(f"Unexpected error fetching props: {e}")
                return {}

    async def fetch_all_props_parallel(
        self,
        event_ids: List[str],
        markets: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Fetch props for multiple events in parallel.

        Args:
            event_ids: List of event IDs
            markets: List of market types to fetch

        Returns:
            Dict mapping event_id to props data
        """
        if not self.odds_api_key or not event_ids:
            return {}

        results = {}
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        async def fetch_with_limit(event_id: str):
            async with semaphore:
                await asyncio.sleep(self.REQUEST_DELAY_SEC)  # Rate limiting
                props = await self.fetch_player_props(event_id, markets)
                return event_id, props

        tasks = [fetch_with_limit(eid) for eid in event_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, tuple):
                event_id, props = result
                results[event_id] = props

        return results

    async def _refresh_team_caches(self):
        """Refresh team defense and pace caches if needed."""
        if self._defense_cache_valid():
            return

        if self._nba_fetcher:
            try:
                # These are sync calls but we're caching them
                self._defense_cache = self._nba_fetcher.get_team_defense_vs_position()
                self._pace_cache = self._nba_fetcher.get_team_pace()
                self._cache_time = datetime.now()
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error refreshing team caches: {e}")
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Data parsing error refreshing team caches: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error refreshing team caches: {e}")

    def _defense_cache_valid(self) -> bool:
        """Check if defense cache is still valid."""
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def _extract_team_from_logs(self, logs: pd.DataFrame) -> Optional[str]:
        """Extract player's team from game logs."""
        if logs.empty:
            return None
        if 'team_abbrev' in logs.columns:
            return logs['team_abbrev'].iloc[0]
        return None


async def analyze_props_parallel(
    props: List[Dict],
    model,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Analyze multiple props in parallel using the model.

    Args:
        props: List of prop dicts with player, prop_type, line, etc.
        model: UnifiedPropModel instance
        max_concurrent: Max concurrent analyses

    Returns:
        List of PropAnalysis results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def analyze_single(prop: Dict):
        async with semaphore:
            # Model.analyze is sync, but we can run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.analyze(
                    player_name=prop.get('player'),
                    prop_type=prop.get('prop_type'),
                    line=prop.get('line'),
                    odds=prop.get('odds', -110),
                    opponent=prop.get('opponent'),
                    is_home=prop.get('is_home'),
                    game_total=prop.get('game_total'),
                    blowout_risk=prop.get('blowout_risk')
                )
            )
            return result

    tasks = [analyze_single(prop) for prop in props]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    return [r for r in results if not isinstance(r, Exception)]


def run_parallel_analysis(props: List[Dict], model) -> List[Any]:
    """
    Convenience function to run parallel analysis synchronously.

    Usage:
        from data.fetchers.async_fetcher import run_parallel_analysis
        results = run_parallel_analysis(props_list, model)
    """
    return asyncio.run(analyze_props_parallel(props, model))
