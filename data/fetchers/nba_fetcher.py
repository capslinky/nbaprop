"""NBA Stats API integration for fetching player and team data.

This module provides the NBADataFetcher class for fetching player game logs,
team defense ratings, pace factors, and other NBA statistics.
"""

import hashlib
import logging
import os
import pickle
import random
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from core.constants import (
    TEAM_ABBREVIATIONS,
    normalize_team_abbrev,
    get_current_nba_season,
)
from core.config import CONFIG
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

    def __init__(self, cache_dir: str = None,
                 base_delay: Optional[float] = None,
                 game_log_ttl: Optional[int] = None):
        if cache_dir is None:
            cache_dir = os.environ.get('NBAPROP_CACHE_DIR')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._player_id_cache = {}
        self._team_id_cache = {}

        # Session-level caches to avoid redundant API calls within a single run
        # Key: (player_name, season) -> DataFrame of game logs
        self._game_logs_cache = {}
        # Key: (player_name, opponent) -> dict of vs_team stats
        self._vs_team_cache = {}

        # Team-level caches to avoid repeated NBA API calls in a single run
        self._defense_ratings_cache = {'data': None, 'timestamp': None}
        self._defense_vs_position_cache = {'data': None, 'timestamp': None}
        self._pace_cache = {'data': None, 'timestamp': None}

        # Cache TTLs
        self._game_logs_cache_ttl = (
            game_log_ttl if game_log_ttl is not None else CONFIG.GAME_LOG_CACHE_TTL
        )
        self._defense_cache_ttl = CONFIG.DEFENSE_CACHE_TTL
        self._pace_cache_ttl = CONFIG.PACE_CACHE_TTL

        # Thread safety for rate limiting and caches
        self._rate_lock = threading.Lock()
        self._cache_lock = threading.RLock()
        self._inflight_lock = threading.Lock()
        self._inflight_game_logs = {}

        # Rate limiting - NBA API can be sensitive to rapid requests (529 errors)
        # Uses CONFIG.NBA_API_DELAY by default; dynamic backoff handles failures
        self.base_delay = base_delay if base_delay is not None else CONFIG.NBA_API_DELAY
        self.request_delay = self.base_delay
        self._last_request = 0
        self._consecutive_failures = 0

        # Get the global resilient fetcher for retry logic
        self._resilient = get_resilient_fetcher()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """
        Enforce rate limiting between API calls with jitter and dynamic adjustment.
        Automatically backs off more when experiencing failures.
        """
        with self._rate_lock:
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
        with self._rate_lock:
            self._consecutive_failures = 0
            self.request_delay = self.base_delay

    def _on_failure(self) -> None:
        """Called after a failed API call to increase backoff."""
        with self._rate_lock:
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
        # Clear session caches for fresh data
        with self._cache_lock:
            self._game_logs_cache.clear()
            self._vs_team_cache.clear()
            self._defense_ratings_cache['data'] = None
            self._defense_ratings_cache['timestamp'] = None
            self._defense_vs_position_cache['data'] = None
            self._defense_vs_position_cache['timestamp'] = None
            self._pace_cache['data'] = None
            self._pace_cache['timestamp'] = None
        with self._inflight_lock:
            self._inflight_game_logs.clear()
        logger.debug("NBADataFetcher state reset (caches cleared)")

    def _cache_entry_valid(self, entry: Optional[Dict[str, Any]], ttl: float) -> bool:
        """Return True if cache entry exists and is within TTL."""
        if not entry or entry.get('data') is None or entry.get('timestamp') is None:
            return False
        return (time.time() - entry['timestamp']) < ttl

    def _game_logs_cache_path(self, cache_key: tuple) -> Optional[Path]:
        """Generate a safe cache filename for a player/season key."""
        if not self.cache_dir:
            return None
        key_str = f"{cache_key[0]}_{cache_key[1]}"
        digest = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        safe_player = re.sub(r'[^a-z0-9]+', '_', cache_key[0]).strip('_')[:40]
        if not safe_player:
            safe_player = 'player'
        return self.cache_dir / f"game_logs_{safe_player}_{cache_key[1]}_{digest}.pkl"

    def _load_game_logs_disk_cache(self, cache_key: tuple) -> Optional[Dict[str, Any]]:
        """Load cached game logs from disk if valid."""
        path = self._game_logs_cache_path(cache_key)
        if not path or not path.exists():
            return None
        try:
            with path.open('rb') as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                return None
            if not self._cache_entry_valid(payload, self._game_logs_cache_ttl):
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
            return payload
        except Exception as e:
            logger.debug(f"Failed to load game log cache {path}: {e}")
            return None

    def _save_game_logs_disk_cache(self, cache_key: tuple, entry: Dict[str, Any]) -> None:
        """Persist game logs cache entry to disk."""
        path = self._game_logs_cache_path(cache_key)
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.debug(f"Failed to save game log cache {path}: {e}")

    def _get_cached_game_logs(self, cache_key: tuple,
                              last_n_games: Optional[int]) -> Optional[pd.DataFrame]:
        """Return cached game logs from memory/disk if available."""
        with self._cache_lock:
            entry = self._game_logs_cache.get(cache_key)
            if entry and self._cache_entry_valid(entry, self._game_logs_cache_ttl):
                cached_df = entry['data']
                return cached_df.head(last_n_games).copy() if last_n_games else cached_df.copy()
            if entry:
                self._game_logs_cache.pop(cache_key, None)

        disk_entry = self._load_game_logs_disk_cache(cache_key)
        if disk_entry:
            with self._cache_lock:
                self._game_logs_cache[cache_key] = disk_entry
            cached_df = disk_entry['data']
            return cached_df.head(last_n_games).copy() if last_n_games else cached_df.copy()

        return None

    def _set_game_logs_cache(self, cache_key: tuple, df: pd.DataFrame) -> None:
        """Store game logs in memory and disk cache with timestamp."""
        entry = {'data': df.copy(), 'timestamp': time.time()}
        with self._cache_lock:
            self._game_logs_cache[cache_key] = entry
        self._save_game_logs_disk_cache(cache_key, entry)

    def _get_cached_table(self, cache: Dict[str, Any], ttl: float) -> Optional[pd.DataFrame]:
        """Return cached table data if valid."""
        with self._cache_lock:
            if self._cache_entry_valid(cache, ttl):
                return cache['data'].copy()
        return None

    def _set_cached_table(self, cache: Dict[str, Any], df: pd.DataFrame) -> None:
        """Store cached table data with timestamp."""
        with self._cache_lock:
            cache['data'] = df.copy()
            cache['timestamp'] = time.time()

    def _acquire_inflight_game_logs(self, cache_key: tuple) -> tuple:
        """Return (event, is_leader) for in-flight game log fetches."""
        with self._inflight_lock:
            event = self._inflight_game_logs.get(cache_key)
            if event is None:
                event = threading.Event()
                self._inflight_game_logs[cache_key] = event
                return event, True
            return event, False

    def _release_inflight_game_logs(self, cache_key: tuple) -> None:
        """Release in-flight marker and notify waiting threads."""
        with self._inflight_lock:
            event = self._inflight_game_logs.pop(cache_key, None)
        if event:
            event.set()

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

        Uses session-level caching to avoid redundant API calls when the same
        player is analyzed for multiple prop types (points, rebounds, assists, etc).

        Args:
            player_name: Full player name (e.g., "Luka Doncic")
            season: NBA season format (e.g., "2025-26"). Defaults to current season.
            last_n_games: If set, only return last N games

        Returns:
            DataFrame with game logs including pts, reb, ast, etc.
        """
        if season is None:
            season = get_current_nba_season()

        # Check cache (memory + optional disk) first
        cache_key = (player_name.lower(), season)
        cached_df = self._get_cached_game_logs(cache_key, last_n_games)
        if cached_df is not None:
            return cached_df

        player_id = self.get_player_id(player_name)
        if not player_id:
            logger.warning(f"Could not find player: {player_name}")
            return pd.DataFrame()

        # Prevent duplicate fetches for the same player/season under concurrency
        while True:
            inflight_event, is_leader = self._acquire_inflight_game_logs(cache_key)
            if is_leader:
                break
            inflight_event.wait()
            cached_df = self._get_cached_game_logs(cache_key, last_n_games)
            if cached_df is not None:
                return cached_df

        try:
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

            df['player'] = player_name

            # Cache the full result (before slicing by last_n_games)
            self._set_game_logs_cache(cache_key, df)

            if last_n_games:
                df = df.head(last_n_games)

            return df
        finally:
            self._release_inflight_game_logs(cache_key)

    def get_team_defense_ratings(self, season: str = None) -> pd.DataFrame:
        """Fetch team defensive ratings for matchup analysis."""
        if season is None:
            season = get_current_nba_season()

        cached = self._get_cached_table(self._defense_ratings_cache, self._defense_cache_ttl)
        if cached is not None:
            return cached

        try:
            def _fetch_stats():
                from nba_api.stats.endpoints import leaguedashteamstats

                self._rate_limit()

                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame'
                )

                return stats.get_data_frames()[0]

            df, success = self._resilient.fetch_with_retry(_fetch_stats)

            if not success or df is None:
                self._on_failure()
                logger.warning("Failed to fetch team defense ratings after retries")
                return pd.DataFrame()

            self._on_success()

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

            result = result.reset_index(drop=True)
            self._set_cached_table(self._defense_ratings_cache, result)
            return result

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

        cached = self._get_cached_table(self._defense_vs_position_cache, self._defense_cache_ttl)
        if cached is not None:
            return cached

        try:
            def _fetch_stats():
                from nba_api.stats.endpoints import leaguedashteamstats

                self._rate_limit()

                # Get opponent stats (what teams allow)
                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Opponent',
                    per_mode_detailed='PerGame'
                )

                return stats.get_data_frames()[0]

            df, success = self._resilient.fetch_with_retry(_fetch_stats)

            if not success or df is None:
                self._on_failure()
                logger.warning("Failed to fetch team defense vs position after retries")
                return pd.DataFrame()

            self._on_success()

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

            result = result.reset_index(drop=True)
            self._set_cached_table(self._defense_vs_position_cache, result)
            return result

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
        """Get player's historical performance against a specific team.

        Uses current season only by default for speed (cached game logs).
        Pass explicit seasons list for historical analysis if needed.
        """
        if seasons is None:
            # Use current season only (fast - uses cached game logs)
            current = get_current_nba_season()
            seasons = [current]

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

        cached = self._get_cached_table(self._pace_cache, self._pace_cache_ttl)
        if cached is not None:
            return cached

        try:
            def _fetch_stats():
                from nba_api.stats.endpoints import leaguedashteamstats

                self._rate_limit()

                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame'
                )

                return stats.get_data_frames()[0]

            df, success = self._resilient.fetch_with_retry(_fetch_stats)

            if not success or df is None:
                self._on_failure()
                logger.warning("Failed to fetch team pace after retries")
                return pd.DataFrame()

            self._on_success()

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

            result = result.reset_index(drop=True)
            self._set_cached_table(self._pace_cache, result)
            return result

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
        """Get player's historical stats against a specific opponent.

        Uses session-level caching for the vs_team data (player+opponent combo).
        The prop_type-specific stats are computed from cached data.
        """
        # Check vs_team cache first (caches the raw vs_team DataFrame)
        cache_key = (player_name.lower(), opponent.upper())

        if cache_key in self._vs_team_cache:
            vs_team = self._vs_team_cache[cache_key]
        else:
            vs_team = self.get_player_vs_team(player_name, opponent)
            self._vs_team_cache[cache_key] = vs_team

        if vs_team.empty or prop_type not in vs_team.columns:
            return {}

        # Also get overall average for comparison (uses game logs cache)
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

    # =========================================================================
    # ADVANCED STATS (Usage Rate, True Shooting, etc.)
    # =========================================================================

    _advanced_stats_cache: Dict[str, pd.DataFrame] = {}
    _advanced_stats_timestamp: Dict[str, float] = {}
    _advanced_stats_lock = threading.Lock()
    _ADVANCED_STATS_TTL = 3600  # 1 hour cache

    def get_player_advanced_stats(self, season: str = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch league-wide advanced stats including USG%, TS%, AST%, EFG%.

        Uses NBA API leaguedashplayerstats endpoint with Advanced measure type.
        Results are cached for 1 hour to minimize API calls.

        Args:
            season: NBA season format (e.g., "2024-25"). Defaults to current.
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            DataFrame with columns: PLAYER_NAME, USG_PCT, TS_PCT, AST_PCT, EFG_PCT, MIN
        """
        if season is None:
            season = get_current_nba_season()

        cache_key = season
        with NBADataFetcher._advanced_stats_lock:
            cached_stats = NBADataFetcher._advanced_stats_cache.get(cache_key)
            cached_ts = NBADataFetcher._advanced_stats_timestamp.get(cache_key)

        cache_valid = (
            not force_refresh and
            cached_stats is not None and
            cached_ts is not None and
            (time.time() - cached_ts) < self._ADVANCED_STATS_TTL
        )

        if cache_valid:
            logger.debug("Using cached advanced stats")
            return cached_stats

        try:
            from nba_api.stats.endpoints import leaguedashplayerstats

            self._rate_limit()

            logger.info(f"Fetching player advanced stats for {season}...")

            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )

            df = stats.get_data_frames()[0]
            self._on_success()

            if df.empty:
                logger.warning("No advanced stats returned from NBA API")
                return pd.DataFrame()

            # Select relevant columns
            columns_to_keep = [
                'PLAYER_NAME', 'PLAYER_ID', 'TEAM_ABBREVIATION',
                'USG_PCT', 'TS_PCT', 'AST_PCT', 'EFG_PCT',
                'MIN', 'GP'  # Games played for sample size
            ]

            # Filter to columns that exist (API may vary)
            available_cols = [c for c in columns_to_keep if c in df.columns]
            result = df[available_cols].copy()

            # Rename for consistency
            result = result.rename(columns={
                'PLAYER_NAME': 'player_name',
                'PLAYER_ID': 'player_id',
                'TEAM_ABBREVIATION': 'team',
                'USG_PCT': 'usg_pct',
                'TS_PCT': 'ts_pct',
                'AST_PCT': 'ast_pct',
                'EFG_PCT': 'efg_pct',
                'MIN': 'minutes',
                'GP': 'games_played'
            })

            # Convert percentages (NBA API returns as decimals like 0.25 for 25%)
            # Some endpoints return 25.0, others return 0.25 - normalize to 0-100 scale
            for col in ['usg_pct', 'ts_pct', 'ast_pct', 'efg_pct']:
                if col in result.columns:
                    # If max is < 1, it's in decimal form - convert to percentage
                    if result[col].max() < 1:
                        result[col] = result[col] * 100

            # Cache the results
            with NBADataFetcher._advanced_stats_lock:
                NBADataFetcher._advanced_stats_cache[cache_key] = result
                NBADataFetcher._advanced_stats_timestamp[cache_key] = time.time()

            logger.info(f"Loaded advanced stats for {len(result)} players")
            return result

        except Exception as e:
            self._on_failure()
            logger.warning(f"Error fetching player advanced stats: {e}")
            return pd.DataFrame()

    def get_player_usage(self, player_name: str, season: str = None) -> dict:
        """
        Get usage rate and efficiency stats for a specific player.

        Args:
            player_name: Full player name
            season: NBA season format

        Returns:
            Dict with usg_pct, ts_pct, ast_pct, efg_pct, or empty if not found
        """
        advanced = self.get_player_advanced_stats(season)

        if advanced.empty:
            return {}

        # Find player (case-insensitive partial match)
        player_lower = player_name.lower()
        matches = advanced[advanced['player_name'].str.lower().str.contains(player_lower, na=False)]

        if matches.empty:
            # Try last name only
            last_name = _extract_last_name(player_name).lower()
            matches = advanced[advanced['player_name'].str.lower().str.contains(last_name, na=False)]

        if matches.empty:
            return {}

        # Take first match (most minutes if multiple)
        player = matches.sort_values('minutes', ascending=False).iloc[0]

        return {
            'usg_pct': round(player.get('usg_pct', 20.0), 1),
            'ts_pct': round(player.get('ts_pct', 55.0), 1),
            'ast_pct': round(player.get('ast_pct', 15.0), 1),
            'efg_pct': round(player.get('efg_pct', 50.0), 1),
            'minutes': round(player.get('minutes', 0), 1),
            'games_played': int(player.get('games_played', 0))
        }

    def calculate_usage_trend(self, game_logs: pd.DataFrame) -> dict:
        """
        Calculate shot volume/usage trend from game logs.

        Tracks FGA per minute to detect changes in shot volume independent
        of minutes played.

        Args:
            game_logs: DataFrame with fga, fta, minutes columns

        Returns:
            Dict with usage_factor, shot_volume_trend, recent/season usage
        """
        if game_logs.empty:
            return {'usage_factor': 1.0, 'trend': 'STABLE'}

        required_cols = ['fga', 'fta', 'minutes']
        if not all(col in game_logs.columns for col in required_cols):
            return {'usage_factor': 1.0, 'trend': 'STABLE'}

        logs = game_logs.sort_values('date', ascending=False) if 'date' in game_logs.columns else game_logs

        # Parse minutes safely
        mins = logs['minutes'].apply(_parse_minutes)

        # Filter out games with < 10 minutes (garbage time/injury)
        valid_mask = mins >= 10
        if valid_mask.sum() < 5:
            return {'usage_factor': 1.0, 'trend': 'STABLE'}

        logs_valid = logs[valid_mask].copy()
        mins_valid = mins[valid_mask]

        # Calculate usage score: (FGA + 0.44*FTA) / minutes
        # 0.44 is standard FTA possession weight
        usage_scores = (logs_valid['fga'] + 0.44 * logs_valid['fta']) / mins_valid

        recent_5 = usage_scores.head(5).mean()
        older_10 = usage_scores.iloc[5:15].mean() if len(usage_scores) > 5 else usage_scores.mean()

        if older_10 == 0 or np.isnan(older_10):
            return {'usage_factor': 1.0, 'trend': 'STABLE'}

        usage_trend = recent_5 / older_10
        usage_factor = max(0.92, min(1.08, usage_trend))  # ±8% cap

        return {
            'usage_factor': round(usage_factor, 3),
            'recent_usage': round(recent_5, 3),
            'season_usage': round(older_10, 3),
            'trend': 'UP' if usage_trend > 1.03 else 'DOWN' if usage_trend < 0.97 else 'STABLE',
            'raw_trend': round(usage_trend, 3)
        }

    def calculate_ts_efficiency(self, game_logs: pd.DataFrame) -> dict:
        """
        Calculate True Shooting % momentum factor.

        TS% = PTS / (2 * (FGA + 0.44 * FTA))

        BACKTEST VALIDATED (Dec 2025):
        - HOT shooters stay hot (62.5% OVER win rate)
        - COLD shooters stay cold (55.3% UNDER win rate)
        Uses MOMENTUM theory, not regression-to-mean.

        Args:
            game_logs: DataFrame with points, fga, fta columns

        Returns:
            Dict with ts_pct, ts_trend, ts_factor
        """
        if game_logs.empty:
            return {'ts_factor': 1.0, 'regression': 'NONE'}

        required_cols = ['points', 'fga', 'fta']
        if not all(col in game_logs.columns for col in required_cols):
            return {'ts_factor': 1.0, 'regression': 'NONE'}

        logs = game_logs.sort_values('date', ascending=False) if 'date' in game_logs.columns else game_logs

        def calc_ts(df):
            pts = df['points'].sum()
            fga = df['fga'].sum()
            fta = df['fta'].sum()
            denominator = 2 * (fga + 0.44 * fta)
            return pts / denominator if denominator > 0 else 0.5

        recent_ts = calc_ts(logs.head(5))
        season_ts = calc_ts(logs)

        if season_ts == 0:
            return {'ts_factor': 1.0, 'regression': 'NONE'}

        deviation = (recent_ts - season_ts) / season_ts

        # MOMENTUM logic (backtest validated) - hot stays hot, cold stays cold
        threshold = 0.05  # 5% deviation threshold
        weight = 0.3      # 30% momentum weight
        max_adj = 0.05    # ±5% cap

        if abs(deviation) > threshold:
            if deviation > 0:
                # Shooting HOT - momentum continues (62.5% OVER win rate)
                momentum_type = 'HOT'
                # Increase projection to favor OVER
                ts_factor = 1.0 + min(max_adj, deviation * weight)
            else:
                # Shooting COLD - slump continues (55.3% UNDER win rate)
                momentum_type = 'COLD'
                # Decrease projection to favor UNDER
                ts_factor = 1.0 - min(max_adj, abs(deviation) * weight)
        else:
            ts_factor = 1.0
            momentum_type = 'NONE'

        return {
            'ts_factor': round(ts_factor, 3),
            'recent_ts_pct': round(recent_ts * 100, 1),
            'season_ts_pct': round(season_ts * 100, 1),
            'deviation_pct': round(deviation * 100, 1),
            'regression': momentum_type
        }

    def get_historical_usage_boost(
        self,
        player_name: str,
        teammate_name: str,
        prop_type: str = 'points',
        min_games_without: int = 3
    ) -> dict:
        """
        Calculate historical usage/production boost when a specific teammate is out.

        This provides a more accurate boost than the flat STAR_OUT_BOOST by
        looking at actual historical games where the teammate didn't play.

        Args:
            player_name: Name of the player to analyze
            teammate_name: Name of the teammate who is/was out
            prop_type: The prop type to analyze ('points', 'rebounds', 'assists', etc.)
            min_games_without: Minimum games without teammate to calculate (default 3)

        Returns:
            dict with keys:
                - boost_factor: Historical production boost (1.0 = no change)
                - games_with_teammate: Number of games with teammate playing
                - games_without_teammate: Number of games without teammate
                - avg_with: Average production with teammate
                - avg_without: Average production without teammate
                - sufficient_data: Whether we have enough data
        """
        from core.constants import normalize_stat_key

        logger.debug(f"Calculating historical boost for {player_name} without {teammate_name}")

        # Get player's game logs
        player_logs = self.get_player_game_logs(player_name, last_n_games=30)
        if player_logs is None or player_logs.empty:
            return self._default_usage_boost("Player has no game logs")

        # Get teammate's game logs
        teammate_logs = self.get_player_game_logs(teammate_name, last_n_games=30)
        if teammate_logs is None or teammate_logs.empty:
            return self._default_usage_boost("Teammate has no game logs")

        # Map prop type to column name
        stat_col = normalize_stat_key(prop_type)
        if stat_col not in player_logs.columns:
            return self._default_usage_boost(f"Stat column '{stat_col}' not found")

        # Get the date column
        date_col = None
        for col in ['game_date', 'GAME_DATE', 'date']:
            if col in player_logs.columns:
                date_col = col
                break

        if date_col is None:
            return self._default_usage_boost("No date column found")

        # Build set of dates when teammate played
        teammate_date_col = None
        for col in ['game_date', 'GAME_DATE', 'date']:
            if col in teammate_logs.columns:
                teammate_date_col = col
                break

        if teammate_date_col is None:
            return self._default_usage_boost("Teammate logs missing date column")

        # Convert dates to comparable format
        try:
            player_logs[date_col] = pd.to_datetime(player_logs[date_col])
            teammate_logs[teammate_date_col] = pd.to_datetime(teammate_logs[teammate_date_col])
        except Exception as e:
            logger.warning(f"Date conversion error: {e}")
            return self._default_usage_boost("Date conversion error")

        # Get dates when teammate played (had non-zero minutes)
        minutes_col = None
        for col in ['min', 'minutes', 'MIN', 'MINUTES']:
            if col in teammate_logs.columns:
                minutes_col = col
                break

        if minutes_col:
            # Teammate played = had minutes
            teammate_played_dates = set(
                teammate_logs[teammate_logs[minutes_col] > 0][teammate_date_col].dt.date
            )
        else:
            # If no minutes column, assume teammate played in all their logged games
            teammate_played_dates = set(teammate_logs[teammate_date_col].dt.date)

        # Split player's games into with/without teammate
        player_logs['date_only'] = player_logs[date_col].dt.date
        player_logs['teammate_played'] = player_logs['date_only'].isin(teammate_played_dates)

        games_with = player_logs[player_logs['teammate_played']]
        games_without = player_logs[~player_logs['teammate_played']]

        # Check if we have enough games without teammate
        if len(games_without) < min_games_without:
            return self._default_usage_boost(
                f"Only {len(games_without)} games without {teammate_name} (need {min_games_without})"
            )

        if len(games_with) < 3:
            return self._default_usage_boost(f"Only {len(games_with)} games with {teammate_name}")

        # Calculate averages
        avg_with = games_with[stat_col].mean()
        avg_without = games_without[stat_col].mean()

        # Calculate boost factor
        if avg_with > 0:
            boost_factor = avg_without / avg_with
        else:
            boost_factor = 1.0

        # Cap the boost to reasonable bounds (0.8 to 1.3)
        boost_factor = max(0.8, min(1.3, boost_factor))

        logger.debug(
            f"Historical boost for {player_name} without {teammate_name}: "
            f"{boost_factor:.3f} ({avg_without:.1f} vs {avg_with:.1f})"
        )

        return {
            'boost_factor': round(boost_factor, 3),
            'games_with_teammate': len(games_with),
            'games_without_teammate': len(games_without),
            'avg_with': round(avg_with, 1),
            'avg_without': round(avg_without, 1),
            'sufficient_data': True,
            'teammate': teammate_name,
            'prop_type': prop_type
        }

    def _default_usage_boost(self, reason: str) -> dict:
        """Return default usage boost when calculation fails."""
        logger.debug(f"Using default boost: {reason}")
        return {
            'boost_factor': 1.0,
            'games_with_teammate': 0,
            'games_without_teammate': 0,
            'avg_with': 0,
            'avg_without': 0,
            'sufficient_data': False,
            'reason': reason
        }

    def get_combined_historical_boost(
        self,
        player_name: str,
        stars_out: list,
        prop_type: str = 'points'
    ) -> dict:
        """
        Calculate combined historical boost when multiple stars are out.

        Args:
            player_name: Name of the player to analyze
            stars_out: List of star teammate names who are out
            prop_type: The prop type to analyze

        Returns:
            dict with combined boost factor and breakdown per star
        """
        if not stars_out:
            return {
                'boost_factor': 1.0,
                'breakdown': [],
                'sufficient_data': True
            }

        # Remove the player themselves from the list
        stars_out = [s for s in stars_out if s.lower() != player_name.lower()]

        if not stars_out:
            return {
                'boost_factor': 1.0,
                'breakdown': [],
                'sufficient_data': True
            }

        breakdown = []
        combined_boost = 1.0
        any_sufficient = False

        for i, star in enumerate(stars_out):
            boost_info = self.get_historical_usage_boost(
                player_name, star, prop_type, min_games_without=3
            )

            if boost_info['sufficient_data']:
                any_sufficient = True
                # First star gets full weight, subsequent get diminishing weights
                weight = 1.0 if i == 0 else 0.5 ** i
                star_boost = boost_info['boost_factor']
                weighted_boost = 1.0 + (star_boost - 1.0) * weight
                combined_boost *= weighted_boost

                breakdown.append({
                    'star': star,
                    'boost': boost_info['boost_factor'],
                    'weighted_boost': round(weighted_boost, 3),
                    'games_without': boost_info['games_without_teammate'],
                    'avg_without': boost_info['avg_without'],
                    'avg_with': boost_info['avg_with']
                })
            else:
                # Fall back to default boost for this star
                breakdown.append({
                    'star': star,
                    'boost': 1.05,  # Default flat boost
                    'weighted_boost': 1.05 if i == 0 else 1.025,
                    'games_without': 0,
                    'reason': boost_info.get('reason', 'Insufficient data')
                })

        # Cap combined boost at 1.4 (40% max)
        combined_boost = min(1.4, combined_boost)

        return {
            'boost_factor': round(combined_boost, 3),
            'breakdown': breakdown,
            'sufficient_data': any_sufficient,
            'stars_out': stars_out
        }
