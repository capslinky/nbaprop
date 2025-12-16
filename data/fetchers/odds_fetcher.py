"""Betting odds API integration for NBA props.

This module provides the OddsAPIClient class for fetching live betting
lines and player props from The Odds API.
"""

import pandas as pd
from typing import List
import time
import logging
import requests

from core.exceptions import (
    NetworkError,
    AuthenticationError,
    RateLimitError,
    OddsAPIError,
)

logger = logging.getLogger(__name__)


class OddsAPIClient:
    """
    Client for The Odds API (https://the-odds-api.com/)
    Free tier: 500 requests/month

    To get an API key:
    1. Go to https://the-odds-api.com/
    2. Sign up for free account
    3. Copy your API key from dashboard
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.remaining_requests = None

        # Sport key for NBA
        self.sport = "basketball_nba"

        # Preferred bookmakers (in order of preference)
        self.preferred_books = [
            'draftkings',
            'fanduel',
            'betmgm',
            'caesars',
            'pointsbetus',
            'bovada'
        ]

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling.

        Raises:
            AuthenticationError: If API key is missing, invalid, or insufficient permissions
            RateLimitError: If rate limit exceeded (429)
            NetworkError: If network connection fails
            OddsAPIError: For other API errors
        """
        if not self.api_key:
            raise AuthenticationError(
                "OddsAPI",
                "API key required. Get one at https://the-odds-api.com/",
                status_code=None
            )

        if params is None:
            params = {}
        params['apiKey'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)

            # Track remaining requests
            self.remaining_requests = response.headers.get('x-requests-remaining')

            if response.status_code == 401:
                raise AuthenticationError(
                    "OddsAPI",
                    "Invalid API key",
                    status_code=401
                )
            elif response.status_code == 403:
                raise AuthenticationError(
                    "OddsAPI",
                    "Access forbidden - check API key permissions",
                    status_code=403
                )
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                raise RateLimitError(
                    "OddsAPI",
                    retry_after=int(retry_after) if retry_after else None
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            raise NetworkError("OddsAPI", "Request timeout", original_error=e)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError("OddsAPI", "Connection failed", original_error=e)
        except requests.exceptions.RequestException as e:
            # Re-raise our custom exceptions, wrap others
            if isinstance(e, (AuthenticationError, RateLimitError, NetworkError)):
                raise
            raise OddsAPIError(f"Request failed: {e}", status_code=None)

    def get_upcoming_games(self) -> List[dict]:
        """Get list of upcoming NBA games with odds."""
        endpoint = f"sports/{self.sport}/odds"
        params = {
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }

        return self._make_request(endpoint, params)

    def get_game_lines(self) -> pd.DataFrame:
        """
        Get game lines (totals and spreads) for all upcoming games.
        Returns DataFrame with game_id, teams, total, spread, etc.
        """
        games = self.get_upcoming_games()

        if not games:
            return pd.DataFrame()

        lines = []
        for game in games:
            game_id = game.get('id')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence = game.get('commence_time')

            # Initialize values
            total = None
            spread = None  # Home team spread

            for book in game.get('bookmakers', []):
                if book.get('key') not in self.preferred_books[:3]:
                    continue  # Skip non-preferred books

                for market in book.get('markets', []):
                    if market.get('key') == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over':
                                total = outcome.get('point')
                                break

                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == home_team:
                                spread = outcome.get('point')
                                break

                if total and spread:
                    break  # Got what we need

            # Determine if high/low total (league avg ~225)
            total_category = None
            if total:
                if total >= 235:
                    total_category = 'HIGH'
                elif total >= 225:
                    total_category = 'MEDIUM'
                else:
                    total_category = 'LOW'

            # Determine blowout risk from spread
            blowout_risk = None
            if spread is not None:
                abs_spread = abs(spread)
                if abs_spread >= 10:
                    blowout_risk = 'HIGH'
                elif abs_spread >= 6:
                    blowout_risk = 'MEDIUM'
                else:
                    blowout_risk = 'LOW'

            lines.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence,
                'total': total,
                'total_category': total_category,
                'home_spread': spread,
                'blowout_risk': blowout_risk,
                'favorite': home_team if spread and spread < 0 else away_team if spread else None
            })

        return pd.DataFrame(lines)

    def get_events(self) -> List[dict]:
        """Get list of upcoming NBA events/games."""
        endpoint = f"sports/{self.sport}/events"
        return self._make_request(endpoint, {})

    def get_player_props(self, event_id: str = None,
                         markets: List[str] = None) -> List[dict]:
        """
        Get player prop betting lines for a specific event.

        Args:
            event_id: Specific game ID (required for player props)
            markets: List of prop markets to fetch
                    Options: player_points, player_rebounds, player_assists,
                            player_threes, player_blocks, player_steals,
                            player_points_rebounds_assists, etc.
        """
        if markets is None:
            markets = [
                'player_points',
                'player_rebounds',
                'player_assists',
                'player_points_rebounds_assists',
                'player_threes'
            ]

        if not event_id:
            logger.warning("event_id required for player props. Use get_all_player_props() instead.")
            return []

        endpoint = f"sports/{self.sport}/events/{event_id}/odds"

        params = {
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        result = self._make_request(endpoint, params)
        # Wrap single event in list for consistent parsing
        return [result] if result and isinstance(result, dict) else result

    def get_all_player_props(self, markets: List[str] = None,
                             max_events: int = None) -> List[dict]:
        """
        Get player props for all upcoming games.

        Args:
            markets: List of prop markets to fetch
            max_events: Limit number of events to fetch (saves API calls)

        Returns:
            List of event data with player props
        """
        if markets is None:
            markets = [
                'player_points',
                'player_rebounds',
                'player_assists',
                'player_points_rebounds_assists',
                'player_threes'
            ]

        # First get all events
        events = self.get_events()
        if not events:
            return []

        if max_events:
            events = events[:max_events]

        all_props = []
        for event in events:
            event_id = event.get('id')
            if not event_id:
                continue

            props = self.get_player_props(event_id, markets)
            if props:
                all_props.extend(props)
            time.sleep(0.1)  # Small delay between requests

        return all_props

    def parse_player_props(self, raw_data: List[dict]) -> pd.DataFrame:
        """Parse raw API response into clean DataFrame."""
        props = []

        for game in raw_data:
            game_id = game.get('id')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')

            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('key')

                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')

                    for outcome in market.get('outcomes', []):
                        player_name = outcome.get('description')
                        line = outcome.get('point')
                        price = outcome.get('price')
                        side = outcome.get('name')  # 'Over' or 'Under'

                        props.append({
                            'game_id': game_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'bookmaker': book_name,
                            'market': market_key,
                            'player': player_name,
                            'line': line,
                            'odds': price,
                            'side': side.lower() if side else None
                        })

        df = pd.DataFrame(props)

        if df.empty:
            return df

        # Parse commence time
        df['commence_time'] = pd.to_datetime(df['commence_time'])

        # Map market names to friendly names
        market_map = {
            'player_points': 'points',
            'player_rebounds': 'rebounds',
            'player_assists': 'assists',
            'player_points_rebounds_assists': 'pra',
            'player_threes': 'threes',
            'player_blocks': 'blocks',
            'player_steals': 'steals'
        }
        df['prop_type'] = df['market'].map(market_map).fillna(df['market'])

        return df

    def get_best_odds(self, props_df: pd.DataFrame) -> pd.DataFrame:
        """Find best available odds across bookmakers for each prop."""
        if props_df.empty:
            return props_df

        # Group by player, prop type, line, and side
        best_odds = props_df.groupby(
            ['player', 'prop_type', 'line', 'side']
        ).apply(lambda x: x.loc[x['odds'].idxmax()]).reset_index(drop=True)

        return best_odds

    def check_remaining_requests(self) -> int:
        """Check how many API requests remain this month."""
        return int(self.remaining_requests) if self.remaining_requests else None
