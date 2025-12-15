"""
NBA Prop Analysis - Real Data Integrations
==========================================
Connects to:
1. NBA Stats API (nba_api) - Free, official NBA data
2. The Odds API - Live betting lines (free tier: 500 requests/month)

Usage:
    from nba_integrations import NBADataFetcher, OddsAPIClient, LivePropAnalyzer
    
    # Fetch player game logs
    fetcher = NBADataFetcher()
    logs = fetcher.get_player_game_logs("Luka Doncic", season="2024-25")
    
    # Get live odds (requires API key)
    odds_client = OddsAPIClient(api_key="YOUR_KEY")
    props = odds_client.get_player_props()
    
    # Run live analysis
    analyzer = LivePropAnalyzer(fetcher, odds_client)
    picks = analyzer.find_value_props(min_edge=0.05)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Tuple, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS FROM CORE MODULE (consolidated utilities)
# =============================================================================

from core.constants import (
    TEAM_ABBREVIATIONS,
    STAR_PLAYERS,
    STAR_OUT_BOOST,
    normalize_team_abbrev,
    get_current_nba_season as _core_get_current_nba_season,
)
from core.config import CONFIG as _CORE_CONFIG

# =============================================================================
# RESILIENT FETCHER - Retry Logic & Connection Pooling
# =============================================================================

class ResilientFetcher:
    """
    Utility class that wraps API calls with retry logic, exponential backoff,
    and connection pooling for improved reliability.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 15.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.consecutive_failures = 0

        # Create session with connection pooling
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=5,
            max_retries=0  # We handle retries ourselves
        )
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff: 1s, 2s, 4s, 8s...
        delay = self.base_delay * (2 ** attempt)
        # Add jitter (random 0-30% of delay)
        jitter = delay * random.uniform(0, 0.3)
        # Cap at max delay
        return min(delay + jitter, self.max_delay)

    def _should_retry(self, error: Exception) -> Tuple[bool, float]:
        """
        Determine if an error should trigger a retry and how long to wait.
        Returns (should_retry, delay_multiplier)
        """
        error_str = str(error).lower()

        # Rate limiting - back off aggressively
        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            return True, 3.0  # Triple the delay

        # 529 Site Overloaded - NBA API specific, back off very aggressively
        if '529' in error_str or 'site overloaded' in error_str or 'overloaded' in error_str:
            return True, 4.0  # Quadruple the delay for site overload

        # Connection errors - retry immediately
        if any(x in error_str for x in ['connection', 'timeout', 'timed out', 'reset', 'refused']):
            return True, 1.0

        # Server errors (5xx) - retry with normal backoff
        if any(x in error_str for x in ['500', '502', '503', '504', '529', 'server error', 'internal error']):
            return True, 1.5

        # Client errors (4xx except 429) - don't retry
        if any(x in error_str for x in ['400', '401', '403', '404', 'not found', 'forbidden']):
            return False, 0

        # JSON decode errors - might be transient, retry once
        if 'json' in error_str or 'decode' in error_str:
            return True, 1.0

        # Default: retry with normal backoff
        return True, 1.0

    def fetch_with_retry(
        self,
        fetch_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Execute a fetch function with automatic retry on failure.

        Args:
            fetch_func: The function to execute
            *args, **kwargs: Arguments to pass to fetch_func

        Returns:
            Tuple of (result, success). If all retries fail, result is None.
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = fetch_func(*args, **kwargs)
                self.consecutive_failures = 0  # Reset on success
                return result, True

            except Exception as e:
                last_error = e
                self.consecutive_failures += 1

                should_retry, delay_mult = self._should_retry(e)

                if not should_retry or attempt >= self.max_retries:
                    logger.warning(
                        f"Fetch failed after {attempt + 1} attempts: {e}"
                    )
                    break

                delay = self._calculate_delay(attempt) * delay_mult
                # Add extra delay if we've had many consecutive failures
                if self.consecutive_failures > 3:
                    delay *= 1.5

                logger.info(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        return None, False

    @property
    def session(self) -> requests.Session:
        """Get the shared session with connection pooling."""
        return self._session


# Global resilient fetcher instance
_resilient_fetcher = ResilientFetcher()


def get_resilient_fetcher() -> ResilientFetcher:
    """Get the global ResilientFetcher instance."""
    return _resilient_fetcher

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_nba_season() -> str:
    """
    Calculate the current NBA season string based on today's date.
    NBA seasons run from October to June.
    - Oct-Dec 2025 = "2025-26"
    - Jan-Sep 2026 = "2025-26"

    Note: Now delegates to core.constants.get_current_nba_season
    """
    return _core_get_current_nba_season()

# =============================================================================
# NBA STATS API INTEGRATION
# =============================================================================

class NBADataFetcher:
    """
    Fetches real NBA data using the nba_api package.
    Includes player game logs, team stats, and schedule data.

    Now includes:
    - Retry logic with exponential backoff
    - Connection pooling for better performance
    - Dynamic rate limiting based on API response
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self._player_id_cache = {}
        self._team_id_cache = {}

        # Rate limiting - NBA API can be sensitive to rapid requests (529 errors)
        self.base_delay = 1.5  # Base delay between requests (increased to avoid 529 errors)
        self.request_delay = self.base_delay
        self._last_request = 0
        self._consecutive_failures = 0

        # Get the global resilient fetcher for retry logic
        self._resilient = get_resilient_fetcher()

    def _rate_limit(self):
        """
        Enforce rate limiting between API calls with jitter and dynamic adjustment.
        Automatically backs off more when experiencing failures.
        """
        # Add jitter to avoid thundering herd (random 0-30% of delay)
        jitter = self.request_delay * random.uniform(0, 0.3)
        effective_delay = self.request_delay + jitter

        # Increase delay if we've had consecutive failures
        if self._consecutive_failures > 0:
            effective_delay *= (1 + 0.5 * min(self._consecutive_failures, 5))

        elapsed = time.time() - self._last_request
        if elapsed < effective_delay:
            time.sleep(effective_delay - elapsed)
        self._last_request = time.time()

    def _on_success(self):
        """Called after a successful API call to reset failure tracking."""
        self._consecutive_failures = 0
        self.request_delay = self.base_delay

    def _on_failure(self):
        """Called after a failed API call to increase backoff."""
        self._consecutive_failures += 1
        # Increase delay up to 3x the base
        self.request_delay = min(self.base_delay * 3,
                                  self.base_delay * (1.5 ** self._consecutive_failures))
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """Look up player ID by name."""
        if player_name in self._player_id_cache:
            return self._player_id_cache[player_name]
        
        try:
            from nba_api.stats.static import players
            
            # Find player (handles partial matches)
            player_list = players.find_players_by_full_name(player_name)
            
            if not player_list:
                # Try last name only
                last_name = player_name.split()[-1]
                player_list = [p for p in players.get_players() 
                              if last_name.lower() in p['full_name'].lower()]
            
            if player_list:
                # Get most recent/active player if multiple matches
                active_players = [p for p in player_list if p.get('is_active', True)]
                player = active_players[0] if active_players else player_list[0]
                self._player_id_cache[player_name] = player['id']
                return player['id']
            
            return None
            
        except Exception as e:
            print(f"Error finding player {player_name}: {e}")
            return None
    
    def get_player_game_logs(self, player_name: str, season: str = None,
                             last_n_games: int = None) -> pd.DataFrame:
        """
        Fetch player game logs for a season with automatic retry on failure.

        Args:
            player_name: Full player name (e.g., "Luka Doncic")
            season: NBA season format (e.g., "2025-26"). Defaults to current season.
            last_n_games: If set, only return last N games

        Returns:
            DataFrame with game logs including pts, reb, ast, etc.
        """
        if season is None:
            season = get_current_nba_season()

        player_id = self.get_player_id(player_name)
        if not player_id:
            logger.warning(f"Could not find player: {player_name}")
            return pd.DataFrame()

        def _fetch_logs():
            """Inner function to fetch logs - will be retried on failure."""
            from nba_api.stats.endpoints import playergamelog

            self._rate_limit()

            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )

            return log.get_data_frames()[0]

        # Use resilient fetcher with retry logic
        df, success = self._resilient.fetch_with_retry(_fetch_logs)

        if not success or df is None:
            self._on_failure()
            logger.warning(f"Failed to fetch game logs for {player_name} after retries")
            return pd.DataFrame()

        self._on_success()

        if df.empty:
            return df

        # Standardize column names
        df = df.rename(columns={
            'GAME_DATE': 'date',
            'MATCHUP': 'matchup',
            'WL': 'result',
            'MIN': 'minutes',
            'PTS': 'points',
            'REB': 'rebounds',
            'AST': 'assists',
            'STL': 'steals',
            'BLK': 'blocks',
            'TOV': 'turnovers',
            'FGM': 'fgm',
            'FGA': 'fga',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FTM': 'ftm',
            'FTA': 'fta',
            'PLUS_MINUS': 'plus_minus'
        })

        # Parse date
        df['date'] = pd.to_datetime(df['date'])

        # Add derived stats
        df['pra'] = df['points'] + df['rebounds'] + df['assists']
        df['pts_reb'] = df['points'] + df['rebounds']
        df['pts_ast'] = df['points'] + df['assists']
        df['reb_ast'] = df['rebounds'] + df['assists']
        df['fantasy'] = (df['points'] + df['rebounds'] * 1.2 +
                        df['assists'] * 1.5 + df['steals'] * 3 +
                        df['blocks'] * 3 - df['turnovers'])

        # Add home/away indicator
        df['home'] = ~df['matchup'].str.contains('@')

        # Extract opponent
        df['opponent'] = df['matchup'].apply(
            lambda x: x.split(' ')[-1] if '@' in x or 'vs.' in x else x.split(' ')[-1]
        )

        # Sort by date (most recent first for analysis)
        df = df.sort_values('date', ascending=False).reset_index(drop=True)

        if last_n_games:
            df = df.head(last_n_games)

        df['player'] = player_name

        return df
    
    def get_team_defense_ratings(self, season: str = None) -> pd.DataFrame:
        """Fetch team defensive ratings for matchup analysis."""
        if season is None:
            season = get_current_nba_season()

        try:
            from nba_api.stats.endpoints import leaguedashteamstats

            self._rate_limit()

            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )

            df = stats.get_data_frames()[0]

            # Extract relevant columns
            result = df[['TEAM_NAME', 'TEAM_ABBREVIATION', 'DEF_RATING']].copy()
            result.columns = ['team_name', 'team_abbrev', 'def_rating']

            return result

        except Exception as e:
            print(f"Error fetching team defense ratings: {e}")
            return pd.DataFrame()

    def get_team_defense_vs_position(self, season: str = None) -> pd.DataFrame:
        """
        Fetch how much each team allows by stat category.
        Returns points, rebounds, assists allowed per game vs league average.
        """
        if season is None:
            season = get_current_nba_season()

        try:
            from nba_api.stats.endpoints import leaguedashteamstats

            self._rate_limit()

            # Get opponent stats (what teams allow)
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Opponent',
                per_mode_detailed='PerGame'
            )

            df = stats.get_data_frames()[0]

            # Use imported TEAM_ABBREVIATIONS from core.constants
            # Filter to only entries where key is a full team name (contains space)
            team_abbrev_map = {k: v for k, v in TEAM_ABBREVIATIONS.items() if ' ' in k}

            # Filter to only NBA teams
            df = df[df['TEAM_NAME'].isin(team_abbrev_map.keys())].copy()

            # Calculate league averages (only from NBA teams)
            league_avg_pts = df['OPP_PTS'].mean()
            league_avg_reb = df['OPP_REB'].mean()
            league_avg_ast = df['OPP_AST'].mean()
            league_avg_fg3m = df['OPP_FG3M'].mean()

            result = pd.DataFrame({
                'team_name': df['TEAM_NAME'],
                'team_abbrev': df['TEAM_NAME'].map(team_abbrev_map),
                'pts_allowed': df['OPP_PTS'],
                'reb_allowed': df['OPP_REB'],
                'ast_allowed': df['OPP_AST'],
                'fg3m_allowed': df['OPP_FG3M'],
                # Relative to league average (>1 = allows more, <1 = allows less)
                'pts_factor': df['OPP_PTS'] / league_avg_pts,
                'reb_factor': df['OPP_REB'] / league_avg_reb,
                'ast_factor': df['OPP_AST'] / league_avg_ast,
                'threes_factor': df['OPP_FG3M'] / league_avg_fg3m,
            })

            # PRA factor is weighted average
            result['pra_factor'] = (result['pts_factor'] * 0.5 +
                                    result['reb_factor'] * 0.25 +
                                    result['ast_factor'] * 0.25)

            # Rank teams (1 = worst defense = allows most)
            result['pts_rank'] = result['pts_allowed'].rank(ascending=False).astype(int)
            result['reb_rank'] = result['reb_allowed'].rank(ascending=False).astype(int)
            result['ast_rank'] = result['ast_allowed'].rank(ascending=False).astype(int)
            result['threes_rank'] = result['fg3m_allowed'].rank(ascending=False).astype(int)

            return result.reset_index(drop=True)

        except Exception as e:
            print(f"Error fetching team defense vs position: {e}")
            return pd.DataFrame()

    def get_player_splits(self, player_name: str, season: str = None) -> dict:
        """
        Get player's home/away and other splits.
        Returns dict with home_avg, away_avg for each stat.
        """
        if season is None:
            season = get_current_nba_season()
        logs = self.get_player_game_logs(player_name, season, last_n_games=50)

        if logs.empty:
            return {}

        home_games = logs[logs['home'] == True]
        away_games = logs[logs['home'] == False]

        splits = {}
        for stat in ['points', 'rebounds', 'assists', 'pra', 'fg3m']:
            if stat in logs.columns:
                splits[stat] = {
                    'home_avg': round(home_games[stat].mean(), 1) if len(home_games) > 0 else None,
                    'away_avg': round(away_games[stat].mean(), 1) if len(away_games) > 0 else None,
                    'home_games': len(home_games),
                    'away_games': len(away_games),
                }
                # Calculate home/away factor
                if splits[stat]['home_avg'] and splits[stat]['away_avg']:
                    overall = logs[stat].mean()
                    splits[stat]['home_factor'] = round(splits[stat]['home_avg'] / overall, 3)
                    splits[stat]['away_factor'] = round(splits[stat]['away_avg'] / overall, 3)

        return splits
    
    def get_player_vs_team(self, player_name: str, team_abbrev: str,
                           seasons: List[str] = None) -> pd.DataFrame:
        """Get player's historical performance against a specific team."""
        if seasons is None:
            # Dynamically build last 3 seasons
            current = get_current_nba_season()
            year = int(current.split('-')[0])
            seasons = [
                f"{year}-{str(year + 1)[-2:]}",
                f"{year - 1}-{str(year)[-2:]}",
                f"{year - 2}-{str(year - 1)[-2:]}"
            ]
        
        all_logs = []
        
        for season in seasons:
            logs = self.get_player_game_logs(player_name, season)
            if not logs.empty:
                team_games = logs[logs['opponent'] == team_abbrev]
                all_logs.append(team_games)
        
        if all_logs:
            return pd.concat(all_logs, ignore_index=True)
        return pd.DataFrame()
    
    def get_schedule_today(self) -> pd.DataFrame:
        """Get today's NBA games."""
        try:
            from nba_api.stats.endpoints import scoreboardv2
            
            self._rate_limit()
            
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            games_df = scoreboard.get_data_frames()[0]
            
            if games_df.empty:
                return games_df
            
            return games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
                            'GAME_STATUS_TEXT']].copy()
            
        except Exception as e:
            print(f"Error fetching today's schedule: {e}")
            return pd.DataFrame()
    
    def get_multiple_players_logs(self, player_names: List[str],
                                   season: str = "2024-25",
                                   last_n_games: int = 15) -> pd.DataFrame:
        """Fetch game logs for multiple players."""
        all_logs = []

        for name in player_names:
            print(f"  Fetching: {name}...")
            logs = self.get_player_game_logs(name, season, last_n_games)
            if not logs.empty:
                all_logs.append(logs)
            time.sleep(0.5)  # Extra delay for bulk requests

        if all_logs:
            return pd.concat(all_logs, ignore_index=True)
        return pd.DataFrame()

    def get_team_pace(self, season: str = None) -> pd.DataFrame:
        """
        Fetch team pace ratings (possessions per game).
        Higher pace = more possessions = more counting stats.
        """
        if season is None:
            season = get_current_nba_season()

        try:
            from nba_api.stats.endpoints import leaguedashteamstats

            self._rate_limit()

            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )

            df = stats.get_data_frames()[0]

            # Map team names to abbreviations
            team_abbrev_map = {
                'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
                'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
                'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
                'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
                'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
                'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
                'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
                'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
                'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
                'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
                'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
            }

            # Filter to NBA teams only
            df = df[df['TEAM_NAME'].isin(team_abbrev_map.keys())].copy()

            # PACE is possessions per 48 minutes
            league_avg_pace = df['PACE'].mean()

            result = pd.DataFrame({
                'team_name': df['TEAM_NAME'],
                'team_abbrev': df['TEAM_NAME'].map(team_abbrev_map),
                'pace': df['PACE'],
                'pace_factor': df['PACE'] / league_avg_pace,  # >1 = faster than avg
                'pace_rank': df['PACE'].rank(ascending=False).astype(int),  # 1 = fastest
            })

            return result.reset_index(drop=True)

        except Exception as e:
            print(f"Error fetching team pace: {e}")
            return pd.DataFrame()

    def get_team_schedule(self, team_abbrev: str, season: str = None) -> pd.DataFrame:
        """Get a team's recent schedule to detect back-to-backs."""
        if season is None:
            season = get_current_nba_season()

        try:
            from nba_api.stats.endpoints import teamgamelog
            from nba_api.stats.static import teams

            # Get team ID
            nba_teams = teams.get_teams()
            team_info = next((t for t in nba_teams if t['abbreviation'] == team_abbrev), None)

            if not team_info:
                return pd.DataFrame()

            self._rate_limit()

            log = teamgamelog.TeamGameLog(
                team_id=team_info['id'],
                season=season
            )

            df = log.get_data_frames()[0]

            if df.empty:
                return df

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)

            return df[['GAME_DATE', 'MATCHUP', 'WL']].head(10)

        except Exception as e:
            print(f"Error fetching team schedule: {e}")
            return pd.DataFrame()

    def check_back_to_back(self, player_logs: pd.DataFrame, game_date: datetime = None) -> dict:
        """
        Check if player is on a back-to-back based on their game logs.
        Returns info about rest days and B2B status.
        """
        if player_logs.empty or 'date' not in player_logs.columns:
            return {'is_b2b': False, 'rest_days': None}

        if game_date is None:
            game_date = datetime.now()

        # Sort by date descending
        logs = player_logs.sort_values('date', ascending=False)

        # Get most recent game date
        last_game = logs.iloc[0]['date']
        if isinstance(last_game, str):
            last_game = pd.to_datetime(last_game)

        # Calculate days since last game
        days_rest = (game_date - last_game).days

        # Check if last two games were on consecutive days
        if len(logs) >= 2:
            second_last = logs.iloc[1]['date']
            if isinstance(second_last, str):
                second_last = pd.to_datetime(second_last)
            days_between = (last_game - second_last).days
            recent_b2b = days_between <= 1
        else:
            recent_b2b = False

        return {
            'is_b2b': days_rest <= 1,
            'rest_days': days_rest,
            'recent_b2b_played': recent_b2b,
            'last_game_date': last_game
        }

    def get_player_minutes_trend(self, player_logs: pd.DataFrame) -> dict:
        """Analyze player's recent minutes trend."""
        if player_logs.empty or 'minutes' not in player_logs.columns:
            return {}

        logs = player_logs.sort_values('date', ascending=False)

        # Handle minutes that might be strings like "32:15"
        def parse_minutes(m):
            if isinstance(m, str) and ':' in m:
                parts = m.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(m) if pd.notna(m) else 0

        mins = logs['minutes'].apply(parse_minutes)

        recent_5 = mins.head(5).mean()
        season_avg = mins.mean()
        min_val = mins.min()
        max_val = mins.max()

        return {
            'recent_avg': round(recent_5, 1),
            'season_avg': round(season_avg, 1),
            'min': round(min_val, 1),
            'max': round(max_val, 1),
            'trend': 'UP' if recent_5 > season_avg * 1.05 else 'DOWN' if recent_5 < season_avg * 0.95 else 'STABLE',
            'minutes_factor': round(recent_5 / season_avg, 3) if season_avg > 0 else 1.0
        }

    def get_player_vs_team_stats(self, player_name: str, opponent: str,
                                  prop_type: str = 'points') -> dict:
        """Get player's historical stats against a specific opponent."""
        vs_team = self.get_player_vs_team(player_name, opponent)

        if vs_team.empty or prop_type not in vs_team.columns:
            return {}

        # Also get overall average for comparison
        all_logs = self.get_player_game_logs(player_name, last_n_games=30)

        if all_logs.empty:
            return {}

        vs_avg = vs_team[prop_type].mean()
        overall_avg = all_logs[prop_type].mean()
        games_vs = len(vs_team)

        return {
            'vs_team_avg': round(vs_avg, 1),
            'overall_avg': round(overall_avg, 1),
            'games_vs_team': games_vs,
            'vs_factor': round(vs_avg / overall_avg, 3) if overall_avg > 0 else 1.0,
            'dominates': vs_avg > overall_avg * 1.1,  # 10%+ better vs this team
            'struggles': vs_avg < overall_avg * 0.9   # 10%+ worse vs this team
        }


# =============================================================================
# INJURY & LINEUP INTEGRATION
# =============================================================================

class InjuryTracker:
    """
    Tracks NBA injuries and lineup changes from multiple sources:
    1. NBA API (official injury reports)
    2. CBS Sports / Rotowire (web scraping)
    3. Manual overrides for late-breaking news

    Usage:
        tracker = InjuryTracker()
        injuries = tracker.get_all_injuries()
        status = tracker.get_player_status("LeBron James")
        boost = tracker.get_teammate_boost("Anthony Davis", "LAL")
    """

    # Use imported constants from core.constants (single source of truth)
    # These are class attributes for backward compatibility
    STAR_PLAYERS = STAR_PLAYERS  # From core.constants import at top of file
    STAR_OUT_BOOST = STAR_OUT_BOOST  # From core.constants import at top of file

    def __init__(self):
        self._injury_cache = {}
        self._cache_time = None
        self._cache_ttl = 1800  # 30 minutes cache
        self._manual_injuries = {}  # Manual overrides

    def get_injuries_from_nba_api(self) -> pd.DataFrame:
        """Fetch injury data from NBA API."""
        try:
            from nba_api.stats.endpoints import playerindex

            # Get all players with their current status
            time.sleep(0.6)  # Rate limit
            players = playerindex.PlayerIndex(season=get_current_nba_season())
            df = players.get_data_frames()[0]

            # Filter to players with injury info if available
            if 'INJURY_STATUS' in df.columns:
                injured = df[df['INJURY_STATUS'].notna() & (df['INJURY_STATUS'] != '')]
                return injured[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'INJURY_STATUS', 'INJURY_DESCRIPTION']].copy()

            return pd.DataFrame()

        except Exception as e:
            print(f"NBA API injury fetch error: {e}")
            return pd.DataFrame()

    def get_injuries_from_rotowire(self) -> pd.DataFrame:
        """
        Scrape injury data from Rotowire/CBS Sports.
        Falls back to CBS if Rotowire fails.
        """
        injuries = []

        # Try CBS Sports NBA injuries page
        try:
            url = "https://www.cbssports.com/nba/injuries/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                from html.parser import HTMLParser

                # Simple parsing - look for injury patterns
                content = response.text

                # Parse the page for injury info
                # CBS format typically has player name, team, status, injury
                # This is a simplified parser
                import re

                # Look for injury table patterns
                # Format: Player Name | Team | Status | Injury Type
                team_sections = re.findall(r'class="TeamName[^"]*"[^>]*>([^<]+)</a>', content)
                player_patterns = re.findall(
                    r'class="CellPlayerName[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>.*?'
                    r'class="[^"]*injury[^"]*"[^>]*>([^<]+)<',
                    content, re.DOTALL | re.IGNORECASE
                )

                for player, status in player_patterns:
                    injuries.append({
                        'player': player.strip(),
                        'status': status.strip().upper(),
                        'source': 'CBS Sports'
                    })

        except Exception as e:
            print(f"CBS Sports scrape error: {e}")

        # Try Rotowire as backup
        if not injuries:
            try:
                url = "https://www.rotowire.com/basketball/injury-report.php"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    import re
                    content = response.text

                    # Look for injury entries
                    # Rotowire format: Player - Team - Status - Injury
                    matches = re.findall(
                        r'<a[^>]*player[^>]*>([^<]+)</a>.*?'
                        r'<span[^>]*team[^>]*>([^<]+)</span>.*?'
                        r'<span[^>]*status[^>]*>([^<]+)</span>',
                        content, re.DOTALL | re.IGNORECASE
                    )

                    for player, team, status in matches:
                        injuries.append({
                            'player': player.strip(),
                            'team': team.strip(),
                            'status': status.strip().upper(),
                            'source': 'Rotowire'
                        })

            except Exception as e:
                print(f"Rotowire scrape error: {e}")

        return pd.DataFrame(injuries) if injuries else pd.DataFrame()

    def get_all_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get combined injury data from all sources.
        Caches results for 30 minutes.
        """
        now = datetime.now()

        # Check cache
        if not force_refresh and self._cache_time:
            if (now - self._cache_time).total_seconds() < self._cache_ttl:
                return self._injury_cache.get('all', pd.DataFrame())

        # Fetch from all sources
        all_injuries = []

        # 1. NBA API
        nba_injuries = self.get_injuries_from_nba_api()
        if not nba_injuries.empty:
            nba_injuries['source'] = 'NBA API'
            all_injuries.append(nba_injuries)

        # 2. Web scraping
        web_injuries = self.get_injuries_from_rotowire()
        if not web_injuries.empty:
            all_injuries.append(web_injuries)

        # 3. Manual overrides (always included)
        if self._manual_injuries:
            manual_df = pd.DataFrame(list(self._manual_injuries.values()))
            manual_df['source'] = 'Manual'
            all_injuries.append(manual_df)

        # Combine and deduplicate
        if all_injuries:
            combined = pd.concat(all_injuries, ignore_index=True)
            # Keep manual entries over others, then NBA API over web
            combined['priority'] = combined['source'].map({'Manual': 0, 'NBA API': 1, 'CBS Sports': 2, 'Rotowire': 3})
            combined = combined.sort_values('priority').drop_duplicates(subset=['player'], keep='first')
            combined = combined.drop(columns=['priority'])

            self._injury_cache['all'] = combined
            self._cache_time = now
            return combined

        return pd.DataFrame()

    def get_player_status(self, player_name: str) -> dict:
        """
        Get injury status for a specific player.

        Returns:
            dict with keys: status, is_out, is_gtd, is_questionable, source
            status: 'HEALTHY', 'OUT', 'GTD', 'QUESTIONABLE', 'PROBABLE', 'DOUBTFUL'
        """
        injuries = self.get_all_injuries()

        if injuries.empty:
            return {'status': 'HEALTHY', 'is_out': False, 'is_gtd': False,
                    'is_questionable': False, 'source': None}

        # Find player (case-insensitive partial match)
        player_lower = player_name.lower()
        mask = injuries['player'].str.lower().str.contains(player_lower, regex=False)

        if mask.any():
            row = injuries[mask].iloc[0]
            status = row.get('status', '').upper()

            return {
                'status': status,
                'is_out': status in ['OUT', 'O', 'DNP'],
                'is_gtd': status in ['GTD', 'GAME TIME DECISION', 'GAME-TIME DECISION'],
                'is_questionable': status in ['QUESTIONABLE', 'Q', 'DOUBTFUL', 'D'],
                'is_probable': status in ['PROBABLE', 'P'],
                'injury': row.get('injury', row.get('INJURY_DESCRIPTION', '')),
                'source': row.get('source', 'Unknown')
            }

        return {'status': 'HEALTHY', 'is_out': False, 'is_gtd': False,
                'is_questionable': False, 'source': None}

    def set_manual_injury(self, player_name: str, team: str, status: str, injury: str = ''):
        """
        Manually set a player's injury status (for late-breaking news).

        Args:
            player_name: Full player name
            team: Team abbreviation (e.g., 'LAL')
            status: 'OUT', 'GTD', 'QUESTIONABLE', 'PROBABLE', 'HEALTHY'
            injury: Optional injury description
        """
        self._manual_injuries[player_name] = {
            'player': player_name,
            'team': team,
            'status': status.upper(),
            'injury': injury
        }
        # Invalidate cache
        self._cache_time = None

    def clear_manual_injuries(self):
        """Clear all manual injury overrides."""
        self._manual_injuries = {}
        self._cache_time = None

    def get_team_injuries(self, team_abbrev: str) -> pd.DataFrame:
        """Get all injured players for a specific team."""
        injuries = self.get_all_injuries()

        if injuries.empty or 'team' not in injuries.columns:
            return pd.DataFrame()

        team_upper = team_abbrev.upper()
        return injuries[injuries['team'].str.upper() == team_upper]

    def get_stars_out(self, team_abbrev: str) -> List[str]:
        """Get list of star players who are OUT for a team."""
        team_upper = team_abbrev.upper()
        stars = self.STAR_PLAYERS.get(team_upper, [])

        stars_out = []
        for star in stars:
            status = self.get_player_status(star)
            if status['is_out']:
                stars_out.append(star)

        return stars_out

    def get_teammate_boost(self, player_name: str, team_abbrev: str,
                           prop_type: str = 'points') -> dict:
        """
        Calculate usage/production boost when star teammates are out.

        Returns:
            dict with keys: boost_factor, stars_out, reason
        """
        team_upper = team_abbrev.upper()
        stars_out = self.get_stars_out(team_upper)

        # Remove the player themselves from stars_out
        stars_out = [s for s in stars_out if s.lower() != player_name.lower()]

        if not stars_out:
            return {
                'boost_factor': 1.0,
                'stars_out': [],
                'reason': 'No star teammates out'
            }

        # Calculate boost based on number of stars out
        base_boost = self.STAR_OUT_BOOST.get(prop_type, 1.05)

        # Compound boost for multiple stars out (diminishing returns)
        total_boost = 1.0
        for i, star in enumerate(stars_out):
            # First star gives full boost, subsequent stars give 50% of boost
            multiplier = 1.0 if i == 0 else 0.5
            star_boost = (base_boost - 1.0) * multiplier
            total_boost += star_boost

        return {
            'boost_factor': round(total_boost, 3),
            'stars_out': stars_out,
            'reason': f"{len(stars_out)} star(s) OUT: {', '.join(stars_out)}"
        }

    def should_exclude_player(self, player_name: str) -> tuple:
        """
        Check if player should be excluded from analysis (OUT status).

        Returns:
            (should_exclude: bool, reason: str)
        """
        status = self.get_player_status(player_name)

        if status['is_out']:
            return True, f"Player is OUT ({status.get('injury', 'injury')})"

        return False, None

    def get_injury_adjustment(self, player_name: str, team_abbrev: str,
                               prop_type: str = 'points') -> dict:
        """
        Get complete injury-related adjustments for a player.

        Returns:
            dict with keys:
                - player_status: Player's own injury status
                - exclude: Whether to exclude from analysis
                - teammate_boost: Boost from stars being out
                - flags: List of warning/info flags
        """
        player_status = self.get_player_status(player_name)
        exclude, exclude_reason = self.should_exclude_player(player_name)
        teammate_boost = self.get_teammate_boost(player_name, team_abbrev, prop_type)

        flags = []

        if player_status['is_gtd']:
            flags.append('⚠️ GTD')
        elif player_status['is_questionable']:
            flags.append('❓ QUESTIONABLE')

        if teammate_boost['stars_out']:
            flags.append(f"📈 USAGE BOOST (+{(teammate_boost['boost_factor']-1)*100:.0f}%)")

        return {
            'player_status': player_status,
            'exclude': exclude,
            'exclude_reason': exclude_reason,
            'teammate_boost': teammate_boost,
            'total_injury_factor': teammate_boost['boost_factor'],
            'flags': flags
        }


# =============================================================================
# BALLDONTLIE API - FALLBACK DATA SOURCE
# =============================================================================

class BallDontLieFetcher:
    """
    Fallback data source using balldontlie.io API.
    Used when nba_api fails.

    API Docs: https://nba.balldontlie.io

    NOTE: The free tier only provides access to basic endpoints (teams, games).
    Stats and box_scores endpoints require a paid subscription (ALL-STAR or GOAT tier).
    If you need fallback stats, upgrade at: https://app.balldontlie.io

    Features (paid tier):
    - Player stats and game logs
    - Team information
    - Historical data back to 1979-80 season
    """

    BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.environ.get('BALLDONTLIE_API_KEY', 'a1f22400-27d9-4aa7-a015-39e4aaf54131')
        self._session = requests.Session()
        self._session.headers['Authorization'] = self.api_key
        self._player_cache = {}  # Cache player lookups
        self._resilient = get_resilient_fetcher()

        # Rate limiting
        self._last_request = 0
        self.request_delay = 0.5  # 500ms between requests

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.time()

    def get_player_id(self, player_name: str) -> Optional[int]:
        """Search for player by name and return their ID."""
        if player_name in self._player_cache:
            return self._player_cache[player_name]

        def _search():
            self._rate_limit()
            # Clean up player name for search
            search_name = player_name.replace("'", "").strip()
            resp = self._session.get(
                f"{self.BASE_URL}/players",
                params={"search": search_name},
                timeout=15
            )
            resp.raise_for_status()
            return resp.json()

        result, success = self._resilient.fetch_with_retry(_search)

        if not success or not result:
            logger.warning(f"BallDontLie: Could not find player {player_name}")
            return None

        data = result.get('data', [])
        if not data:
            # Try just last name
            last_name = player_name.split()[-1]

            def _search_lastname():
                self._rate_limit()
                resp = self._session.get(
                    f"{self.BASE_URL}/players",
                    params={"search": last_name},
                    timeout=15
                )
                resp.raise_for_status()
                return resp.json()

            result, success = self._resilient.fetch_with_retry(_search_lastname)
            if success and result:
                data = result.get('data', [])

        if data:
            # Find best match - prefer exact match or first active player
            for p in data:
                full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}"
                if full_name.lower() == player_name.lower():
                    self._player_cache[player_name] = p['id']
                    return p['id']
            # Take first result if no exact match
            self._player_cache[player_name] = data[0]['id']
            return data[0]['id']

        return None

    def _parse_minutes(self, min_str: str) -> float:
        """Parse minutes string like '32:45' to float 32.75."""
        if not min_str or min_str == '':
            return 0.0
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except (ValueError, IndexError):
            return 0.0

    def get_player_game_logs(self, player_name: str, season: str = None,
                             last_n_games: int = None) -> pd.DataFrame:
        """
        Fetch player game logs from BallDontLie API.

        Args:
            player_name: Full player name
            season: NBA season (e.g., "2024-25") - converted to year format
            last_n_games: Limit to last N games

        Returns:
            DataFrame with standardized columns matching NBADataFetcher format
        """
        player_id = self.get_player_id(player_name)
        if not player_id:
            return pd.DataFrame()

        # Convert season format "2024-25" to BDL year format (2024)
        if season:
            season_year = int(season.split('-')[0])
        else:
            current_season = get_current_nba_season()
            season_year = int(current_season.split('-')[0])

        def _fetch_stats():
            self._rate_limit()
            resp = self._session.get(
                f"{self.BASE_URL}/stats",
                params={
                    "player_ids[]": player_id,
                    "seasons[]": season_year,
                    "per_page": 100  # Max per page
                },
                timeout=15
            )
            resp.raise_for_status()
            return resp.json()

        result, success = self._resilient.fetch_with_retry(_fetch_stats)

        if not success or not result:
            logger.warning(f"BallDontLie: Failed to fetch stats for {player_name}")
            return pd.DataFrame()

        data = result.get('data', [])
        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        rows = []
        for game in data:
            game_info = game.get('game', {})
            team = game.get('team', {})

            # Determine home/away and opponent
            home_team_id = game_info.get('home_team_id')
            visitor_team_id = game_info.get('visitor_team_id')
            is_home = (team.get('id') == home_team_id)

            rows.append({
                'date': game_info.get('date', ''),
                'minutes': self._parse_minutes(game.get('min', '')),
                'points': game.get('pts', 0) or 0,
                'rebounds': game.get('reb', 0) or 0,
                'assists': game.get('ast', 0) or 0,
                'steals': game.get('stl', 0) or 0,
                'blocks': game.get('blk', 0) or 0,
                'turnovers': game.get('turnover', 0) or 0,
                'fgm': game.get('fgm', 0) or 0,
                'fga': game.get('fga', 0) or 0,
                'fg3m': game.get('fg3m', 0) or 0,
                'fg3a': game.get('fg3a', 0) or 0,
                'ftm': game.get('ftm', 0) or 0,
                'fta': game.get('fta', 0) or 0,
                'home': is_home,
                'player': player_name
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Parse date
        df['date'] = pd.to_datetime(df['date'])

        # Add derived stats (same as NBADataFetcher)
        df['pra'] = df['points'] + df['rebounds'] + df['assists']
        df['pts_reb'] = df['points'] + df['rebounds']
        df['pts_ast'] = df['points'] + df['assists']
        df['reb_ast'] = df['rebounds'] + df['assists']
        df['fantasy'] = (df['points'] + df['rebounds'] * 1.2 +
                        df['assists'] * 1.5 + df['steals'] * 3 +
                        df['blocks'] * 3 - df['turnovers'])

        # Sort by date (most recent first)
        df = df.sort_values('date', ascending=False).reset_index(drop=True)

        if last_n_games:
            df = df.head(last_n_games)

        return df


# =============================================================================
# HYBRID FETCHER - Orchestrates Multiple Data Sources
# =============================================================================

class HybridFetcher:
    """
    Orchestrates multiple data sources with automatic failover.

    Strategy:
    1. Try nba_api first (best data quality)
    2. On failure, fall back to balldontlie.io
    3. Log which source was used for debugging

    Usage:
        fetcher = HybridFetcher()
        logs = fetcher.get_player_game_logs("Luka Doncic")
    """

    def __init__(self):
        self.nba_fetcher = NBADataFetcher()
        self.bdl_fetcher = BallDontLieFetcher()
        self._last_source = None

    @property
    def last_source(self) -> str:
        """Returns which data source was used for the last fetch."""
        return self._last_source

    def get_player_game_logs(self, player_name: str, season: str = None,
                             last_n_games: int = None) -> pd.DataFrame:
        """
        Fetch player game logs with automatic failover.

        Tries nba_api first, falls back to balldontlie.io on failure.
        """
        # Try NBA API first
        logger.debug(f"Fetching {player_name} from nba_api...")
        df = self.nba_fetcher.get_player_game_logs(player_name, season, last_n_games)

        if not df.empty:
            self._last_source = 'nba_api'
            logger.debug(f"Successfully fetched {len(df)} games from nba_api")
            return df

        # Fallback to BallDontLie
        logger.info(f"nba_api failed for {player_name}, trying balldontlie.io...")
        df = self.bdl_fetcher.get_player_game_logs(player_name, season, last_n_games)

        if not df.empty:
            self._last_source = 'balldontlie'
            logger.info(f"Successfully fetched {len(df)} games from balldontlie.io")
            return df

        # Both failed
        self._last_source = None
        logger.warning(f"All data sources failed for {player_name}")
        return pd.DataFrame()

    def get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID, trying nba_api first then balldontlie."""
        player_id = self.nba_fetcher.get_player_id(player_name)
        if player_id:
            return player_id
        return self.bdl_fetcher.get_player_id(player_name)


# =============================================================================
# BETTING ODDS API INTEGRATION
# =============================================================================

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
        """Make API request with error handling."""
        if not self.api_key:
            raise ValueError("API key required. Get one at https://the-odds-api.com/")
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Track remaining requests
            self.remaining_requests = response.headers.get('x-requests-remaining')
            
            if response.status_code == 401:
                raise ValueError("Invalid API key")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded. Try again later.")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
    
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
            print("Warning: event_id required for player props. Use get_all_player_props() instead.")
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


# =============================================================================
# ALTERNATIVE FREE ODDS SOURCE
# =============================================================================

class FreeOddsSource:
    """
    Fallback odds source using publicly available data.
    Note: Less reliable than paid API but useful for testing.
    """
    
    def __init__(self):
        self.base_url = "https://www.bovada.lv/services/sports/event/v2/events/A/description"
    
    def get_nba_props(self) -> pd.DataFrame:
        """
        Attempt to scrape prop data from public sources.
        Note: This is for educational purposes - respect site ToS.
        """
        # Placeholder - in production you'd implement actual scraping
        # or use a different free data source
        print("Free odds source requires implementation specific to your needs.")
        print("Consider using The Odds API free tier (500 requests/month)")
        return pd.DataFrame()


# =============================================================================
# LIVE PROP ANALYZER (COMBINES EVERYTHING)
# =============================================================================

class LivePropAnalyzer:
    """
    Combines NBA stats and betting odds for live prop analysis.
    Uses UnifiedPropModel for consistent, context-aware predictions.
    """

    def __init__(self, nba_fetcher: NBADataFetcher = None,
                 odds_client: OddsAPIClient = None,
                 injury_tracker: InjuryTracker = None):
        self.nba = nba_fetcher or NBADataFetcher()
        self.odds = odds_client
        self.injuries = injury_tracker or InjuryTracker()

        # Import unified model
        from nba_prop_model import UnifiedPropModel

        # Initialize unified model with shared resources
        self.model = UnifiedPropModel(
            data_fetcher=self.nba,
            injury_tracker=self.injuries,
            odds_client=self.odds
        )

    def analyze_prop(self, player_name: str, prop_type: str,
                     line: float, odds: int = -110,
                     last_n_games: int = 15,
                     opponent: str = None,
                     is_home: bool = None,
                     game_total: float = None,
                     blowout_risk: str = None) -> dict:
        """
        Analyze a single prop bet using UnifiedPropModel with full context.

        Args:
            player_name: Player's full name
            prop_type: 'points', 'rebounds', 'assists', 'pra', 'threes'
            line: The betting line (e.g., 24.5)
            odds: American odds (default -110)
            last_n_games: Games to analyze
            opponent: Optional opponent team abbreviation
            is_home: Optional home/away indicator
            game_total: Optional Vegas over/under total
            blowout_risk: Optional 'HIGH', 'MEDIUM', 'LOW'

        Returns:
            Analysis dict with projection, edge, recommendation, and full context
        """
        # Use UnifiedPropModel for analysis
        analysis = self.model.analyze(
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            odds=odds,
            opponent=opponent,
            is_home=is_home,
            game_total=game_total,
            blowout_risk=blowout_risk,
            last_n_games=last_n_games
        )

        # Check for no data
        if analysis.games_analyzed == 0:
            return {'error': f'Could not fetch data for {player_name}'}

        # Convert PropAnalysis to dict format (backwards compatible)
        return {
            'player': analysis.player,
            'prop_type': analysis.prop_type,
            'line': analysis.line,
            'odds': odds,
            'sample_size': analysis.games_analyzed,
            'recent_avg': analysis.recent_avg,
            'recent_median': analysis.season_avg,  # Using season_avg as median proxy
            'recent_std': analysis.std_dev,
            'hit_rate_over': round(analysis.over_rate * 100, 1),
            'hit_rate_under': round(analysis.under_rate * 100, 1),
            'last_5_trend': '↑' if analysis.trend == 'HOT' else '↓' if analysis.trend == 'COLD' else '→',
            'last_5_avg': analysis.recent_avg,
            'projection': analysis.projection,
            'base_projection': analysis.base_projection,
            'recommended_side': analysis.pick.lower(),
            'avg_edge': round(analysis.edge * 100, 2),
            'confidence': round(analysis.confidence, 2),
            # New context fields
            'opponent': analysis.opponent,
            'is_home': analysis.is_home,
            'is_b2b': analysis.is_b2b,
            'matchup': analysis.matchup_rating,
            'opp_rank': analysis.opp_rank,
            'game_total': analysis.game_total,
            'blowout_risk': analysis.blowout_risk,
            'trend': analysis.trend,
            'flags': analysis.flags,
            'adjustments': analysis.adjustments,
            'total_adjustment': round(analysis.total_adjustment * 100, 1),
            'player_status': analysis.player_status,
            'teammate_boost': analysis.teammate_boost,
            'stars_out': analysis.stars_out,
        }

    def analyze_multiple_props(self, props: List[dict]) -> pd.DataFrame:
        """
        Analyze multiple props at once using UnifiedPropModel.

        Args:
            props: List of dicts with keys: player, prop_type, line, odds
        """
        results = []

        for prop in props:
            print(f"  Analyzing: {prop['player']} {prop['prop_type']} {prop['line']}...")

            analysis = self.analyze_prop(
                player_name=prop['player'],
                prop_type=prop['prop_type'],
                line=prop['line'],
                odds=prop.get('odds', -110),
                opponent=prop.get('opponent'),
                is_home=prop.get('is_home'),
                game_total=prop.get('game_total'),
                blowout_risk=prop.get('blowout_risk'),
            )

            if 'error' not in analysis:
                results.append({
                    'Player': analysis['player'],
                    'Prop': analysis['prop_type'],
                    'Line': analysis['line'],
                    'Proj': analysis['projection'],
                    'Avg': analysis['recent_avg'],
                    'Over%': analysis['hit_rate_over'],
                    'Under%': analysis['hit_rate_under'],
                    'Edge': analysis['avg_edge'],
                    'Conf': int(analysis['confidence'] * 100),
                    'Pick': analysis['recommended_side'].upper(),
                    'Trend': analysis['trend'],
                    'Matchup': analysis['matchup'],
                    'Flags': ', '.join(analysis['flags']) if analysis['flags'] else '',
                })

            time.sleep(0.5)  # Rate limiting

        return pd.DataFrame(results)

    def find_value_props(self, min_edge: float = 0.05, max_events: int = 5,
                         min_confidence: float = 0.4) -> pd.DataFrame:
        """
        Scan current odds for value props with FULL CONTEXTUAL ANALYSIS.

        Incorporates:
        - Opponent defense ratings
        - Home/away adjustments
        - Back-to-back detection
        - Pace factors
        - Minutes trends
        - Vig-adjusted edge calculation
        - Minimum sample sizes by prop type
        - Correlation filtering

        Args:
            min_edge: Minimum vig-adjusted edge threshold (default 5%)
            max_events: Max number of games to scan (saves API calls)
            min_confidence: Minimum confidence threshold (default 40%)
        """
        if not self.odds:
            print("OddsAPIClient required for live odds scanning", flush=True)
            return pd.DataFrame()

        # =================================================================
        # PHASE 1: DATA COLLECTION
        # =================================================================
        print("=" * 60, flush=True)
        print("PHASE 1: Fetching market data...", flush=True)

        raw_props = self.odds.get_all_player_props(max_events=max_events)
        props_df = self.odds.parse_player_props(raw_props)

        if props_df.empty:
            print("No props available", flush=True)
            return pd.DataFrame()

        # Get game context
        game_lines = self.odds.get_game_lines()

        # Get best odds per prop
        best_odds = self.odds.get_best_odds(props_df)

        # FIX #1: Properly pair OVER and UNDER odds instead of using .first()
        # Split into over and under, then merge to get both odds in same row
        overs = best_odds[best_odds['side'] == 'over'].copy()
        unders = best_odds[best_odds['side'] == 'under'].copy()

        # Rename odds columns to distinguish over vs under
        overs = overs.rename(columns={'odds': 'over_odds'})
        unders = unders.rename(columns={'odds': 'under_odds'})

        # Merge to get both odds in same row
        unique_props = overs.merge(
            unders[['player', 'prop_type', 'line', 'under_odds', 'bookmaker']].rename(
                columns={'bookmaker': 'under_bookmaker'}
            ),
            on=['player', 'prop_type', 'line'],
            how='outer'
        )

        # Fill missing odds with standard -110
        unique_props['over_odds'] = unique_props['over_odds'].fillna(-110).astype(int)
        unique_props['under_odds'] = unique_props['under_odds'].fillna(-110).astype(int)

        print(f"Found {len(unique_props)} unique props across {max_events} games", flush=True)

        # =================================================================
        # PHASE 2: CONTEXTUAL DATA LOADING
        # =================================================================
        print("\nPHASE 2: Loading contextual data...", flush=True)

        # Load defense ratings
        defense_ratings = self.nba.get_team_defense_ratings()
        if defense_ratings is not None:
            print(f"  ✓ Defense ratings: {len(defense_ratings)} teams", flush=True)
        else:
            print("  ✗ Defense ratings unavailable", flush=True)
            defense_ratings = pd.DataFrame()

        # Load pace data
        pace_data = self.nba.get_team_pace()
        if pace_data is not None:
            print(f"  ✓ Pace data: {len(pace_data)} teams", flush=True)
        else:
            print("  ✗ Pace data unavailable", flush=True)
            pace_data = pd.DataFrame()

        # =================================================================
        # PHASE 3: PLAYER DATA FETCHING (with extended history)
        # =================================================================
        print("\nPHASE 3: Fetching player data (30 games for statistical validity)...", flush=True)

        unique_players = unique_props['player'].unique()
        player_cache = {}
        player_context = {}  # Store contextual info per player

        for i, player in enumerate(unique_players):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Fetching {i+1}/{len(unique_players)}: {player}", flush=True)
            try:
                # Fetch MORE games for better sample size
                logs = self.nba.get_player_game_logs(player, last_n_games=30)
                if not logs.empty and len(logs) >= 5:
                    player_cache[player] = logs

                    # Get contextual data for this player
                    context = {}

                    # Home/Away splits
                    if 'home' in logs.columns:
                        home_games = logs[logs['home'] == True]
                        away_games = logs[logs['home'] == False]
                        context['home_games'] = len(home_games)
                        context['away_games'] = len(away_games)

                    # Back-to-back check
                    b2b_info = self.nba.check_back_to_back(logs)
                    context['is_b2b'] = b2b_info.get('is_b2b', False)
                    context['rest_days'] = b2b_info.get('rest_days', 2)

                    # Minutes trend
                    mins_info = self.nba.get_player_minutes_trend(logs)
                    context['minutes_trend'] = mins_info.get('trend', 'stable')
                    context['minutes_factor'] = mins_info.get('minutes_factor', 1.0)
                    context['recent_minutes'] = mins_info.get('last_5_avg', 30)

                    # Get player's team and opponent from most recent game
                    if 'matchup' in logs.columns and len(logs) > 0:
                        last_matchup = logs.iloc[0]['matchup'] if not logs.empty else ''
                        context['last_matchup'] = last_matchup

                    player_context[player] = context

            except Exception as e:
                pass

        print(f"  Cached {len(player_cache)} players with context", flush=True)

        # =================================================================
        # PHASE 4: CONTEXTUAL PROP ANALYSIS
        # =================================================================
        print("\nPHASE 4: Analyzing props with full context...", flush=True)

        value_props = []
        skipped = {'sample_size': 0, 'no_edge': 0, 'low_confidence': 0, 'correlation': 0}

        # Map prop types to column names
        prop_to_column = {
            'points': 'points',
            'rebounds': 'rebounds',
            'assists': 'assists',
            'pra': 'pra',
            'threes': 'fg3m',
            'blocks': 'blocks',
            'steals': 'steals',
        }

        # MINIMUM SAMPLE SIZES by prop type (for statistical validity)
        min_samples = {
            'points': 10,      # Lower variance
            'rebounds': 12,    # Medium variance
            'assists': 15,     # Higher variance
            'pra': 10,         # Aggregated, lower variance
            'threes': 20,      # VERY high variance - need more data
            'blocks': 20,      # High variance
            'steals': 20,      # High variance
        }

        # Defense factor mapping
        defense_stat_map = {
            'points': 'pts_factor',
            'rebounds': 'reb_factor',
            'assists': 'ast_factor',
            'threes': 'threes_factor',
            'pra': 'pts_factor',  # Use points as proxy
        }

        # Track correlated picks to filter later
        player_picks = {}  # player -> list of props picked

        for i, row in unique_props.iterrows():
            player = row['player']
            if player not in player_cache:
                continue

            logs = player_cache[player]
            context = player_context.get(player, {})
            prop_type = row['prop_type']
            stat_column = prop_to_column.get(prop_type, prop_type)

            if stat_column not in logs.columns:
                continue

            history = logs[stat_column]

            # MINIMUM SAMPLE SIZE CHECK
            min_required = min_samples.get(prop_type, 15)
            if len(history) < min_required:
                skipped['sample_size'] += 1
                continue

            try:
                # =============================================================
                # STEP 1: BASE PROJECTION (weighted average with mean reversion)
                # =============================================================
                recent_5 = history.head(5)  # Most recent 5
                recent_10 = history.head(10)
                season = history

                recent_avg = recent_5.mean()
                mid_avg = recent_10.mean()
                season_avg = season.mean()

                # Apply MEAN REVERSION - hot streaks regress, cold streaks recover
                # Weight: 40% recent, 35% mid-term, 25% season (regression to mean)
                base_projection = (recent_avg * 0.40) + (mid_avg * 0.35) + (season_avg * 0.25)

                # =============================================================
                # STEP 2: CONTEXTUAL ADJUSTMENTS
                # =============================================================
                adjustment_factors = []
                adjustment_notes = []

                # --- HOME/AWAY ADJUSTMENT ---
                if 'home' in logs.columns:
                    home_games = logs[logs['home'] == True]
                    away_games = logs[logs['home'] == False]

                    if len(home_games) >= 3 and len(away_games) >= 3:
                        home_avg = home_games[stat_column].mean()
                        away_avg = away_games[stat_column].mean()
                        overall = season_avg

                        # Determine if tonight is home or away (from matchup string)
                        # '@' in matchup means away game
                        is_home_tonight = True  # Default assumption
                        if 'event_info' in row and '@' in str(row.get('event_info', '')):
                            is_home_tonight = False

                        if is_home_tonight and home_avg > 0:
                            ha_factor = home_avg / overall if overall > 0 else 1.0
                            adjustment_factors.append(ha_factor)
                            if abs(ha_factor - 1.0) > 0.05:
                                adjustment_notes.append(f"Home: {ha_factor:.2f}x")
                        elif not is_home_tonight and away_avg > 0:
                            ha_factor = away_avg / overall if overall > 0 else 1.0
                            adjustment_factors.append(ha_factor)
                            if abs(ha_factor - 1.0) > 0.05:
                                adjustment_notes.append(f"Away: {ha_factor:.2f}x")

                # --- BACK-TO-BACK ADJUSTMENT ---
                if context.get('is_b2b', False):
                    b2b_factor = 0.93  # 7% reduction on back-to-backs
                    adjustment_factors.append(b2b_factor)
                    adjustment_notes.append("B2B: 0.93x")
                elif context.get('rest_days', 2) >= 3:
                    rest_factor = 1.03  # 3% boost with extra rest
                    adjustment_factors.append(rest_factor)
                    adjustment_notes.append("Rested: 1.03x")

                # --- MINUTES TREND ADJUSTMENT ---
                mins_factor = context.get('minutes_factor', 1.0)
                if mins_factor < 0.9 or mins_factor > 1.1:
                    # Only adjust if significant change
                    capped_factor = max(0.85, min(1.15, mins_factor))
                    adjustment_factors.append(capped_factor)
                    adjustment_notes.append(f"Mins: {capped_factor:.2f}x")

                # --- OPPONENT DEFENSE ADJUSTMENT ---
                # (would need opponent info from game data - placeholder)
                # For now, use neutral 1.0

                # Apply all adjustments
                final_projection = base_projection
                for factor in adjustment_factors:
                    final_projection *= factor

                # =============================================================
                # STEP 3: EDGE CALCULATION WITH VIG ADJUSTMENT
                # FIX #1: Use correct odds based on side (over_odds vs under_odds)
                # FIX #2: Conservative probability estimation
                # =============================================================
                line = row['line']
                over_odds = row.get('over_odds', -110)
                under_odds = row.get('under_odds', -110)

                # Raw edge (projection vs line)
                raw_edge = (final_projection - line) / line if line > 0 else 0

                # Historical hit rates
                historical_over = (history > line).mean()
                historical_under = (history < line).mean()

                # FIX #1: Calculate no-vig true probabilities from BOTH sides
                def american_to_implied(odds):
                    if odds < 0:
                        return abs(odds) / (abs(odds) + 100)
                    return 100 / (odds + 100)

                over_implied = american_to_implied(over_odds)
                under_implied = american_to_implied(under_odds)

                # Remove vig by normalizing
                total_implied = over_implied + under_implied
                true_over_prob = over_implied / total_implied
                true_under_prob = under_implied / total_implied

                # Determine pick direction based on raw edge
                pick_over = raw_edge > 0

                # FIX #2: Conservative probability estimation
                # Use market probability as anchor, adjust based on:
                # 1. Historical hit rate deviation from market
                # 2. Projection deviation from line
                # Cap total adjustment at ±15% from market
                proj_vs_line = (final_projection - line) / line if line > 0 else 0

                if pick_over:
                    # Use OVER odds for implied probability
                    implied_prob = over_implied
                    breakeven = over_implied
                    market_prob = true_over_prob

                    # Historical edge over market
                    hist_edge = historical_over - market_prob

                    # Projection edge (capped)
                    proj_adjustment = min(0.10, max(-0.10, proj_vs_line * 0.3))

                    # Combined adjustment (capped at ±15%)
                    total_adjustment = min(0.15, max(-0.15, hist_edge * 0.5 + proj_adjustment))
                    our_prob = min(0.85, max(0.15, market_prob + total_adjustment))
                else:
                    # Use UNDER odds for implied probability
                    implied_prob = under_implied
                    breakeven = under_implied
                    market_prob = true_under_prob

                    # Historical edge over market
                    hist_edge = historical_under - market_prob

                    # Projection edge (capped)
                    proj_adjustment = min(0.10, max(-0.10, abs(proj_vs_line) * 0.3))

                    # Combined adjustment (capped at ±15%)
                    total_adjustment = min(0.15, max(-0.15, hist_edge * 0.5 + proj_adjustment))
                    our_prob = min(0.85, max(0.15, market_prob + total_adjustment))

                # VIG-ADJUSTED EDGE = our probability - breakeven probability
                vig_adjusted_edge = (our_prob - breakeven) * 100

                # =============================================================
                # STEP 4: CONFIDENCE SCORE (multi-factor)
                # =============================================================
                std = history.std()
                cv = std / season_avg if season_avg > 0 else 1

                # Base confidence from consistency
                consistency_score = max(0, min(1, 1 - cv))

                # Sample size factor (more games = more confidence)
                sample_factor = min(1.0, len(history) / 25)

                # Agreement factor (do historical and projection agree?)
                if raw_edge > 0:
                    agreement = historical_over
                else:
                    agreement = historical_under

                # Combined confidence
                confidence = (consistency_score * 0.4) + (sample_factor * 0.3) + (agreement * 0.3)
                confidence = max(0.2, min(0.85, confidence))  # Cap at 85%

                # =============================================================
                # STEP 5: PICK DETERMINATION
                # =============================================================
                if abs(vig_adjusted_edge) < 3:  # Need at least 3% edge after vig
                    skipped['no_edge'] += 1
                    continue

                if confidence < min_confidence:
                    skipped['low_confidence'] += 1
                    continue

                pick = 'OVER' if raw_edge > 0 else 'UNDER'

                # =============================================================
                # STEP 6: RECORD VALUE PROP
                # =============================================================
                trend = 'HOT' if recent_avg > season_avg * 1.05 else 'COLD' if recent_avg < season_avg * 0.95 else 'NEUTRAL'

                # FIX #4: Filter out nan values from adjustment_notes
                clean_adjustments = [
                    note for note in adjustment_notes
                    if note and 'nan' not in str(note).lower()
                ]

                # Store the odds for the recommended side
                pick_odds = over_odds if pick == 'OVER' else under_odds

                # Build game matchup string
                home_team = row.get('home_team', '')
                away_team = row.get('away_team', '')
                game_matchup = f"{away_team} @ {home_team}" if home_team and away_team else 'Unknown'

                value_props.append({
                    'player': player,
                    'prop_type': prop_type,
                    'line': line,
                    'projection': round(final_projection, 1),
                    'raw_edge': round(raw_edge * 100, 1),
                    'avg_edge': round(vig_adjusted_edge, 1),  # VIG-ADJUSTED PROBABILITY EDGE
                    'confidence': round(confidence * 100, 0),
                    'recommended_side': pick,
                    'recent_avg': round(recent_avg, 1),
                    'season_avg': round(season_avg, 1),
                    'hit_rate_over': round(historical_over * 100, 0),
                    'hit_rate_under': round(historical_under * 100, 0),
                    'trend': trend,
                    'games_analyzed': len(history),
                    'bookmaker': row.get('bookmaker', 'unknown'),
                    'odds': pick_odds,  # Odds for the recommended side
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'implied_prob': round(implied_prob * 100, 1),  # Breakeven for recommended side
                    'market_prob': round(market_prob * 100, 1),  # No-vig probability
                    'our_prob': round(our_prob * 100, 1),
                    'adjustments': ' | '.join(clean_adjustments) if clean_adjustments else 'None',
                    'is_b2b': context.get('is_b2b', False),
                    'minutes_factor': round(context.get('minutes_factor', 1.0), 2),
                    'game': game_matchup,
                    'home_team': home_team,
                    'away_team': away_team,
                })

                # Track for correlation filtering
                if player not in player_picks:
                    player_picks[player] = []
                player_picks[player].append(prop_type)

            except Exception as e:
                continue

        print(f"\nSkipped: {skipped}", flush=True)

        # =================================================================
        # PHASE 5: CORRELATION FILTERING + ALT LINE DEDUPLICATION
        # =================================================================
        if value_props:
            df = pd.DataFrame(value_props)
            original_count = len(df)

            # Step 1: Remove highly correlated props (points vs pra, etc.)
            # If player has both 'points' and 'pra', keep only the higher edge one
            correlated_pairs = [('points', 'pra'), ('rebounds', 'pra'), ('assists', 'pra')]

            rows_to_drop = []
            for player in df['player'].unique():
                player_df = df[df['player'] == player]
                for prop1, prop2 in correlated_pairs:
                    p1 = player_df[player_df['prop_type'] == prop1]
                    p2 = player_df[player_df['prop_type'] == prop2]
                    if len(p1) > 0 and len(p2) > 0:
                        # Keep the one with higher vig-adjusted edge
                        if p1.iloc[0]['avg_edge'] > p2.iloc[0]['avg_edge']:
                            rows_to_drop.extend(p2.index.tolist())
                        else:
                            rows_to_drop.extend(p1.index.tolist())

            df = df.drop(index=list(set(rows_to_drop)))
            correlation_removed = original_count - len(df)

            # FIX #3: Alt Line Deduplication
            # Keep only ONE line per player per prop type (the one with highest edge)
            # This prevents picks like: Garland UNDER 23.5, 22.5, 21.5 all appearing
            before_alt_dedup = len(df)
            df = df.sort_values('avg_edge', ascending=False)
            df = df.drop_duplicates(subset=['player', 'prop_type'], keep='first')
            alt_lines_removed = before_alt_dedup - len(df)

            skipped['correlation'] = correlation_removed
            skipped['alt_lines'] = alt_lines_removed

            # Sort by vig-adjusted edge
            df = df.sort_values('avg_edge', ascending=False)

            print(f"\nPHASE 5: Filtering removed {correlation_removed} correlated + {alt_lines_removed} alt lines", flush=True)
            print(f"\n{'='*60}", flush=True)
            print(f"FINAL: {len(df)} value props (min {min_edge*100:.0f}% edge after vig)", flush=True)
            print(f"{'='*60}", flush=True)

            return df

        return pd.DataFrame()


# =============================================================================
# EXAMPLE USAGE & DEMO
# =============================================================================

def demo_nba_data():
    """Demonstrate NBA data fetching capabilities."""
    print("\n" + "="*60)
    print("        NBA DATA FETCHER DEMO")
    print("="*60)
    
    fetcher = NBADataFetcher()
    
    # Example players to analyze
    players = ["Luka Doncic", "Shai Gilgeous-Alexander", "Jayson Tatum"]
    
    for player in players:
        print(f"\n📊 {player}")
        print("-" * 40)
        
        logs = fetcher.get_player_game_logs(player, last_n_games=10)
        
        if not logs.empty:
            print(f"  Last 10 Games:")
            print(f"  Points:   {logs['points'].mean():.1f} avg | {logs['points'].median():.1f} med")
            print(f"  Rebounds: {logs['rebounds'].mean():.1f} avg | {logs['rebounds'].median():.1f} med")
            print(f"  Assists:  {logs['assists'].mean():.1f} avg | {logs['assists'].median():.1f} med")
            print(f"  PRA:      {logs['pra'].mean():.1f} avg | {logs['pra'].median():.1f} med")
        else:
            print("  Could not fetch data")
    
    return fetcher


def demo_prop_analysis():
    """Demonstrate prop analysis without needing API key."""
    print("\n" + "="*60)
    print("        PROP ANALYSIS DEMO")
    print("="*60)
    
    analyzer = LivePropAnalyzer()
    
    # Sample props to analyze
    sample_props = [
        {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
        {'player': 'Luka Doncic', 'prop_type': 'assists', 'line': 8.5},
        {'player': 'Shai Gilgeous-Alexander', 'prop_type': 'points', 'line': 30.5},
        {'player': 'Jayson Tatum', 'prop_type': 'pra', 'line': 39.5},
        {'player': 'Anthony Edwards', 'prop_type': 'points', 'line': 25.5},
    ]
    
    print("\nAnalyzing sample props...")
    results = analyzer.analyze_multiple_props(sample_props)
    
    if not results.empty:
        print("\n" + "="*60)
        print("                 ANALYSIS RESULTS")
        print("="*60)
        print(results.to_string(index=False))
        
        # Highlight value plays
        value_plays = results[abs(results['Edge']) >= 5]
        if not value_plays.empty:
            print("\n🎯 VALUE PLAYS (5%+ Edge):")
            print("-" * 40)
            for _, row in value_plays.iterrows():
                emoji = "🟢" if row['Pick'] != 'PASS' else "⚪"
                print(f"  {emoji} {row['Player']} {row['Prop'].upper()} {row['Pick']} {row['Line']} "
                      f"(Edge: {row['Edge']:+.1f}%)")
    
    return results


def main():
    """Run full demonstration."""
    print("🏀 NBA PROP ANALYSIS - LIVE DATA INTEGRATION")
    print("="*60)
    
    # Demo NBA data fetching
    fetcher = demo_nba_data()
    
    # Demo prop analysis
    results = demo_prop_analysis()
    
    # Instructions for odds API
    print("\n" + "="*60)
    print("        SETTING UP LIVE ODDS")
    print("="*60)
    print("""
To get live betting odds:

1. Sign up at https://the-odds-api.com/ (free)
2. Get your API key from the dashboard
3. Use it like this:

    from nba_integrations import OddsAPIClient, LivePropAnalyzer
    
    odds = OddsAPIClient(api_key="YOUR_API_KEY")
    analyzer = LivePropAnalyzer(odds_client=odds)
    
    # Find value props
    value_props = analyzer.find_value_props(min_edge=0.05)

Free tier includes 500 requests/month - plenty for daily analysis!
    """)
    
    return fetcher, results


if __name__ == "__main__":
    fetcher, results = main()
