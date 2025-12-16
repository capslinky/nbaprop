"""Mock implementations for testing NBA prop analysis."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MockNBADataFetcher:
    """Mock data fetcher that returns fixture data for testing."""

    def __init__(self, game_logs=None, defense_data=None, pace_data=None):
        """Initialize with optional custom fixture data."""
        self._game_logs = game_logs
        self._defense_data = defense_data
        self._pace_data = pace_data

    def get_player_game_logs(self, player_name: str, season: str = None,
                             last_n_games: int = 15) -> pd.DataFrame:
        """Return mock game logs."""
        if self._game_logs is not None:
            return self._game_logs.head(last_n_games)

        # Default fixture data - ensure arrays match length
        base_points = [28, 32, 25, 30, 35, 22, 28, 31, 27, 29, 33, 26, 30, 28, 24]
        base_rebounds = [8, 10, 7, 9, 11, 6, 8, 9, 7, 8, 10, 7, 9, 8, 6]
        base_assists = [6, 8, 5, 7, 9, 4, 6, 7, 5, 6, 8, 5, 7, 6, 4]
        base_fg3m = [3, 4, 2, 3, 5, 1, 3, 4, 2, 3, 4, 2, 3, 3, 2]
        base_minutes = [34, 36, 32, 35, 38, 30, 34, 36, 33, 35, 37, 32, 35, 34, 31]

        # Extend or slice based on last_n_games
        n = min(last_n_games, 15)
        dates = [datetime.now() - timedelta(days=i) for i in range(n)]
        home_values = ([False, True] * ((n // 2) + 1))[:n]

        return pd.DataFrame({
            'game_date': dates,
            'points': base_points[:n],
            'rebounds': base_rebounds[:n],
            'assists': base_assists[:n],
            'fg3m': base_fg3m[:n],
            'minutes': base_minutes[:n],
            'matchup': ['DAL @ LAL'] * n,
            'home': home_values,
            'team_abbrev': ['DAL'] * n,
        })

    def get_team_defense_vs_position(self, season: str = None) -> pd.DataFrame:
        """Return mock team defense data."""
        if self._defense_data is not None:
            return self._defense_data

        # Default fixture - all teams average defense
        # Format matches real NBADataFetcher.get_team_defense_vs_position()
        teams = ['LAL', 'BOS', 'DAL', 'PHX', 'MIA', 'GSW', 'DEN', 'MIL']
        team_names = ['Los Angeles Lakers', 'Boston Celtics', 'Dallas Mavericks',
                      'Phoenix Suns', 'Miami Heat', 'Golden State Warriors',
                      'Denver Nuggets', 'Milwaukee Bucks']
        return pd.DataFrame({
            'team_name': team_names,
            'team_abbrev': teams,
            'pts_allowed': [112.0] * len(teams),
            'reb_allowed': [44.0] * len(teams),
            'ast_allowed': [25.0] * len(teams),
            'fg3m_allowed': [12.0] * len(teams),
            'pts_factor': [1.0] * len(teams),
            'reb_factor': [1.0] * len(teams),
            'ast_factor': [1.0] * len(teams),
            'threes_factor': [1.0] * len(teams),
            'pra_factor': [1.0] * len(teams),
            'pts_rank': [15] * len(teams),
            'reb_rank': [15] * len(teams),
            'ast_rank': [15] * len(teams),
            'threes_rank': [15] * len(teams),
        })

    def get_team_pace(self, season: str = None) -> pd.DataFrame:
        """Return mock team pace data."""
        if self._pace_data is not None:
            return self._pace_data

        # Default fixture - average pace
        # Format matches real NBADataFetcher.get_team_pace()
        teams = ['LAL', 'BOS', 'DAL', 'PHX', 'MIA', 'GSW', 'DEN', 'MIL']
        return pd.DataFrame({
            'team_abbrev': teams,
            'pace': [100.0] * len(teams),
            'pace_factor': [1.0] * len(teams),
        })

    def get_team_defense_ratings(self, season: str = None) -> pd.DataFrame:
        """Return mock team defense ratings."""
        teams = ['LAL', 'BOS', 'DAL', 'PHX', 'MIA', 'GSW', 'DEN', 'MIL']
        return pd.DataFrame({
            'TEAM_ABBREVIATION': teams,
            'DEF_RATING': [110.0] * len(teams),
        })

    def check_back_to_back(self, logs: pd.DataFrame, game_date=None) -> dict:
        """Return mock back-to-back status."""
        return {
            'is_b2b': False,
            'is_first_of_b2b': False,
            'is_second_of_b2b': False,
            'days_rest': 2
        }

    def get_team_schedule(self, team_abbrev: str, season: str = None) -> pd.DataFrame:
        """Return mock team schedule."""
        return pd.DataFrame({
            'GAME_DATE': [datetime.now() - timedelta(days=i) for i in range(10)],
            'MATCHUP': ['DAL @ LAL'] * 10,
            'WL': ['W', 'L'] * 5
        })

    def get_player_minutes_trend(self, logs: pd.DataFrame) -> dict:
        """Return mock minutes trend data."""
        return {
            'recent_avg': 35.0,
            'season_avg': 34.5,
            'trend': 'STABLE',
            'adjustment': 1.0
        }

    def get_player_vs_team(self, player_name: str, opponent: str) -> dict:
        """Return mock player vs team history."""
        return {
            'games': 3,
            'avg': 28.5,
            'adjustment': 1.0
        }


class MockInjuryTracker:
    """Mock injury tracker for testing."""

    def __init__(self, injuries=None):
        """Initialize with optional injury data."""
        self._injuries = injuries or {}

    def get_player_status(self, player_name: str) -> dict:
        """Return mock player injury status."""
        return self._injuries.get(player_name, {
            'status': 'ACTIVE',
            'injury': None,
            'return_date': None
        })

    def get_teammate_boosts(self, player_name: str, team: str = None) -> dict:
        """Return mock teammate boost data."""
        return {
            'boost': 1.0,
            'injured_teammates': [],
            'notes': []
        }

    def get_teammate_boost(self, player_name: str, team: str = None, prop_type: str = None) -> dict:
        """Return mock teammate boost for specific prop type."""
        return {
            'boost': 1.0,
            'injured_teammates': [],
            'notes': [],
            'boost_reason': None
        }

    def get_all_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return empty injury DataFrame."""
        return pd.DataFrame()


class MockOddsAPIClient:
    """Mock odds API client for testing."""

    def __init__(self, events=None, props=None):
        """Initialize with optional event/prop data."""
        self._events = events or []
        self._props = props or {}

    def get_events(self) -> list:
        """Return mock events."""
        return self._events

    def get_player_props(self, event_id: str, markets: list = None) -> dict:
        """Return mock player props."""
        return self._props.get(event_id, {})

    def parse_player_props(self, props_data: dict) -> pd.DataFrame:
        """Return empty props DataFrame."""
        return pd.DataFrame()

    def get_game_lines(self, team1: str = None, team2: str = None) -> dict:
        """Return mock game lines."""
        return {
            'total': 225.0,
            'spread': -5.5,
            'home_ml': -200,
            'away_ml': +170
        }
