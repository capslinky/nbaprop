"""NBA Stats API integration for fetching player and team data.

This module provides the NBADataFetcher class for fetching player game logs,
team defense ratings, pace factors, and other NBA statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import time
import random
import logging

from core.constants import (
    TEAM_ABBREVIATIONS,
    normalize_team_abbrev,
    get_current_nba_season,
)
from .resilient import get_resilient_fetcher

logger = logging.getLogger(__name__)


# =============================================================================
# STRING PARSING HELPERS
# =============================================================================

def _extract_last_name(player_name: str) -> str:
    """Extract searchable last name from player name.

    Handles:
    - "LeBron James" -> "James"
    - "Shaquille O'Neal" -> "O'Neal"
    - "Gary Payton Jr." -> "Payton" (skip suffix)
    - "Jaren Jackson Jr." -> "Jackson"

    Args:
        player_name: Full player name

    Returns:
        Last name suitable for searching
    """
    if not player_name:
        return ''

    suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}
    parts = player_name.split()

    # Remove suffixes from end
    while parts and parts[-1].lower() in suffixes:
        parts.pop()

    return parts[-1] if parts else player_name


def _extract_opponent(matchup: str) -> str:
    """Extract opponent team abbreviation from matchup string.

    Handles formats:
    - 'DAL vs. LAL' -> 'LAL'
    - 'DAL @ LAL' -> 'LAL'
    - 'DAL vs. Los Angeles Lakers' -> 'LAL' (via normalize)
    - 'LALvsBKN' -> '' (malformed, returns empty)

    Args:
        matchup: Matchup string from NBA API

    Returns:
        Normalized opponent team abbreviation, or empty string if parsing fails
    """
    if not matchup:
        return ''

    # Split on ' vs. ' or ' @ '
    if ' vs. ' in matchup:
        parts = matchup.split(' vs. ')
        opponent_part = parts[1].strip() if len(parts) > 1 else ''
    elif ' @ ' in matchup:
        parts = matchup.split(' @ ')
        opponent_part = parts[1].strip() if len(parts) > 1 else ''
    elif 'vs.' in matchup:
        # Handle 'DALvs.BKN' (no spaces)
        parts = matchup.split('vs.')
        opponent_part = parts[1].strip() if len(parts) > 1 else ''
    elif '@' in matchup:
        # Handle 'DAL@BKN' (no spaces)
        parts = matchup.split('@')
        opponent_part = parts[1].strip() if len(parts) > 1 else ''
    else:
        # Fallback - try last word
        opponent_part = matchup.split()[-1] if matchup.split() else ''

    # Normalize to standard abbreviation
    return normalize_team_abbrev(opponent_part)


def _parse_season_year(season: str) -> int:
    """Parse season string to starting year.

    Args:
        season: NBA season format "2024-25" or "2024"

    Returns:
        Starting year as integer

    Raises:
        ValueError: If season format is invalid
    """
    if not season:
        season = get_current_nba_season()

    # Handle "2024" format
    if season.isdigit() and len(season) == 4:
        return int(season)

    # Handle "2024-25" format
    if '-' in season:
        parts = season.split('-')
        if len(parts) == 2 and parts[0].isdigit():
            return int(parts[0])

    raise ValueError(f"Invalid season format: '{season}'. Expected '2024-25' or '2024'")


def _parse_minutes(min_str) -> float:
    """Parse minutes string to float with validation.

    Handles:
    - "32:45" -> 32.75
    - "32" -> 32.0
    - None/empty -> 0.0
    - Invalid -> 0.0 with warning

    Args:
        min_str: Minutes value (string, int, or float)

    Returns:
        Minutes as float, 0.0 if invalid
    """
    if min_str is None or min_str == '':
        return 0.0

    try:
        min_str = str(min_str).strip()

        if ':' in min_str:
            parts = min_str.split(':')
            if len(parts) != 2:
                logger.warning(f"Invalid minutes format: {min_str}")
                return 0.0

            minutes = float(parts[0])
            seconds = float(parts[1])

            # Bounds checking
            if not (0 <= minutes <= 60) or not (0 <= seconds < 60):
                logger.warning(f"Minutes out of bounds: {min_str}")
                return 0.0

            return minutes + seconds / 60

        result = float(min_str)
        if not (0 <= result <= 60):
            logger.warning(f"Minutes value out of bounds: {result}")
            return 0.0
        return result

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse minutes '{min_str}': {e}")
        return 0.0


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

    def _rate_limit(self) -> None:
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

    def _on_success(self) -> None:
        """Called after a successful API call to reset failure tracking."""
        self._consecutive_failures = 0
        self.request_delay = self.base_delay

    def _on_failure(self) -> None:
        """Called after a failed API call to increase backoff."""
        self._consecutive_failures += 1
        # Increase delay up to 3x the base
        self.request_delay = min(self.base_delay * 3,
                                  self.base_delay * (1.5 ** self._consecutive_failures))

    def reset_state(self) -> None:
        """Reset internal state for new analysis run.

        Call this between independent analysis runs to prevent
        exponential backoff from compounding across analyses.
        """
        self._consecutive_failures = 0
        self.request_delay = self.base_delay
        self._last_request = 0
        logger.debug("NBADataFetcher state reset")

    def get_player_id(self, player_name: str) -> Optional[int]:
        """Look up player ID by name."""
        if player_name in self._player_id_cache:
            return self._player_id_cache[player_name]

        try:
            from nba_api.stats.static import players

            # Find player (handles partial matches)
            player_list = players.find_players_by_full_name(player_name)

            if not player_list:
                # Try last name only (handles suffixes like Jr., Sr., II, etc.)
                last_name = _extract_last_name(player_name)
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
            logger.warning(f"Error finding player {player_name}: {e}")
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

        # Extract opponent (handles full team names and normalizes to abbreviation)
        df['opponent'] = df['matchup'].apply(_extract_opponent)

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

            # Filter to only entries where team name is a full NBA team name
            # (contains space and is in our mapping)
            team_abbrev_map = {k: v for k, v in TEAM_ABBREVIATIONS.items() if ' ' in k}
            df = df[df['TEAM_NAME'].isin(team_abbrev_map.keys())].copy()

            # Map team names to abbreviations (API doesn't return TEAM_ABBREVIATION)
            result = pd.DataFrame({
                'team_name': df['TEAM_NAME'],
                'team_abbrev': df['TEAM_NAME'].map(team_abbrev_map),
                'def_rating': df['DEF_RATING'],
            })

            return result.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Error fetching team defense ratings: {e}")
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
            logger.warning(f"Error fetching team defense vs position: {e}")
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
            year = _parse_season_year(current)
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
            logger.warning(f"Error fetching today's schedule: {e}")
            return pd.DataFrame()

    def get_multiple_players_logs(self, player_names: List[str],
                                   season: str = None,
                                   last_n_games: int = 15) -> pd.DataFrame:
        """Fetch game logs for multiple players."""
        if season is None:
            season = get_current_nba_season()

        all_logs = []

        for name in player_names:
            logger.info(f"  Fetching: {name}...")
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
            logger.warning(f"Error fetching team pace: {e}")
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
            logger.warning(f"Error fetching team schedule: {e}")
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

    def get_player_minutes_trend(self, player_logs: pd.DataFrame) -> Dict[str, Any]:
        """Analyze player's recent minutes trend."""
        if player_logs.empty or 'minutes' not in player_logs.columns:
            return {}

        logs = player_logs.sort_values('date', ascending=False)

        # Use robust minutes parser with bounds checking
        mins = logs['minutes'].apply(_parse_minutes)

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
