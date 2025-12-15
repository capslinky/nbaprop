"""
Pytest configuration and shared fixtures for NBA prop analysis tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_game_logs():
    """Sample game log data for testing."""
    dates = [datetime.now() - timedelta(days=i) for i in range(15)]
    return pd.DataFrame({
        'game_date': dates,
        'points': [28, 32, 25, 30, 35, 22, 28, 31, 27, 29, 33, 26, 30, 28, 24],
        'rebounds': [8, 10, 7, 9, 11, 6, 8, 9, 7, 8, 10, 7, 9, 8, 6],
        'assists': [6, 8, 5, 7, 9, 4, 6, 7, 5, 6, 8, 5, 7, 6, 4],
        'fg3m': [3, 4, 2, 3, 5, 1, 3, 4, 2, 3, 4, 2, 3, 3, 2],
        'minutes': [34, 36, 32, 35, 38, 30, 34, 36, 33, 35, 37, 32, 35, 34, 31],
        'matchup': ['DAL @ LAL'] * 15,
        'home': [False, True, False, True, False, True, False, True, False, True,
                 False, True, False, True, False],
    })


@pytest.fixture
def sample_odds_data():
    """Sample odds data for testing."""
    return [
        {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5,
         'over_odds': -115, 'under_odds': -105},
        {'player': 'Luka Doncic', 'prop_type': 'rebounds', 'line': 8.5,
         'over_odds': -110, 'under_odds': -110},
        {'player': 'Jayson Tatum', 'prop_type': 'points', 'line': 28.5,
         'over_odds': -120, 'under_odds': +100},
    ]


@pytest.fixture
def sample_history():
    """Sample stat history as pandas Series."""
    return pd.Series([28, 32, 25, 30, 35, 22, 28, 31, 27, 29, 33, 26, 30, 28, 24])
