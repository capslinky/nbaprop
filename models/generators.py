"""
Data generation utilities for NBA prop analysis.

Functions for generating sample player data and prop lines for testing and demonstration.
"""

import pandas as pd
import numpy as np

from core.config import CONFIG
from core.constants import get_current_nba_season


def generate_player_season_data(player_name: str, team: str, position: str,
                                 base_stats: dict, games: int = 82,
                                 season: str = None) -> pd.DataFrame:
    """Generate realistic player game logs with natural variance."""
    if season is None:
        season = get_current_nba_season()

    np.random.seed(hash(player_name) % 2**32)

    # Calculate season start date dynamically
    season_start_year = int(season.split('-')[0])
    season_start = f'{season_start_year}-10-24'
    dates = pd.date_range(start=season_start, periods=games, freq='2D')

    data = []
    for i, date in enumerate(dates):
        # Add realistic game-to-game variance
        pts_var = np.random.normal(0, base_stats['pts'] * 0.25)
        reb_var = np.random.normal(0, base_stats['reb'] * 0.30)
        ast_var = np.random.normal(0, base_stats['ast'] * 0.35)

        # Situational factors
        is_home = np.random.random() > 0.5
        is_back_to_back = np.random.random() < 0.15
        opp_def_rating = np.random.uniform(105, 118)  # Opponent defensive rating

        # Adjust for situational factors
        home_boost = 1.03 if is_home else 0.97
        b2b_penalty = CONFIG.B2B_PENALTY if is_back_to_back else 1.0
        matchup_factor = 1 + (opp_def_rating - 112) / 200  # Easier vs worse defenses

        pts = max(0, round((base_stats['pts'] + pts_var) * home_boost * b2b_penalty * matchup_factor))
        reb = max(0, round((base_stats['reb'] + reb_var) * home_boost * b2b_penalty))
        ast = max(0, round((base_stats['ast'] + ast_var) * home_boost * b2b_penalty))

        # Minutes with variance
        mins = max(15, min(42, base_stats['mins'] + np.random.normal(0, 4)))

        opponents = ['LAL', 'BOS', 'MIA', 'PHI', 'DEN', 'MIL', 'PHX', 'GSW',
                     'NYK', 'CLE', 'SAC', 'MIN', 'OKC', 'DAL', 'NOP']

        data.append({
            'date': date,
            'player': player_name,
            'team': team,
            'position': position,
            'opponent': np.random.choice(opponents),
            'home': is_home,
            'b2b': is_back_to_back,
            'opp_def_rtg': round(opp_def_rating, 1),
            'minutes': round(mins, 1),
            'points': pts,
            'rebounds': reb,
            'assists': ast,
            'pra': pts + reb + ast,
            'season': season
        })

    return pd.DataFrame(data)


def generate_sample_dataset() -> pd.DataFrame:
    """Generate a full dataset of players for analysis."""

    players = [
        ('Luka Doncic', 'DAL', 'PG', {'pts': 33.9, 'reb': 9.2, 'ast': 9.8, 'mins': 37.5}),
        ('Shai Gilgeous-Alexander', 'OKC', 'PG', {'pts': 30.1, 'reb': 5.5, 'ast': 6.2, 'mins': 34.0}),
        ('Giannis Antetokounmpo', 'MIL', 'PF', {'pts': 30.4, 'reb': 11.5, 'ast': 6.5, 'mins': 35.2}),
        ('Jayson Tatum', 'BOS', 'SF', {'pts': 26.9, 'reb': 8.1, 'ast': 4.9, 'mins': 36.0}),
        ('Anthony Edwards', 'MIN', 'SG', {'pts': 25.9, 'reb': 5.4, 'ast': 5.1, 'mins': 35.1}),
        ('Kevin Durant', 'PHX', 'SF', {'pts': 27.1, 'reb': 6.6, 'ast': 5.0, 'mins': 37.2}),
        ('LeBron James', 'LAL', 'SF', {'pts': 25.7, 'reb': 7.3, 'ast': 8.3, 'mins': 35.3}),
        ('Nikola Jokic', 'DEN', 'C', {'pts': 26.4, 'reb': 12.4, 'ast': 9.0, 'mins': 34.6}),
        ('Tyrese Haliburton', 'IND', 'PG', {'pts': 20.1, 'reb': 3.9, 'ast': 10.9, 'mins': 32.2}),
        ('Donovan Mitchell', 'CLE', 'SG', {'pts': 26.6, 'reb': 5.1, 'ast': 6.1, 'mins': 35.0}),
        ('Devin Booker', 'PHX', 'SG', {'pts': 27.1, 'reb': 4.5, 'ast': 6.9, 'mins': 36.0}),
        ('Trae Young', 'ATL', 'PG', {'pts': 25.7, 'reb': 2.8, 'ast': 10.8, 'mins': 35.0}),
    ]

    all_data = []
    for name, team, pos, stats in players:
        player_df = generate_player_season_data(name, team, pos, stats)
        all_data.append(player_df)

    return pd.concat(all_data, ignore_index=True)


def generate_prop_lines(game_logs: pd.DataFrame, vig: float = 0.05) -> pd.DataFrame:
    """
    Generate realistic prop betting lines based on player averages.
    Lines are set with slight edge to the book (vig).
    """
    props = []

    for player in game_logs['player'].unique():
        player_games = game_logs[game_logs['player'] == player].copy()
        player_games = player_games.sort_values('date').reset_index(drop=True)

        for i in range(20, len(player_games)):  # Need 20 games of history
            current_game = player_games.iloc[i]
            history = player_games.iloc[max(0, i-15):i]  # Last 15 games

            for prop_type in ['points', 'rebounds', 'assists', 'pra']:
                recent_avg = history[prop_type].mean()
                recent_std = history[prop_type].std()

                # Books set lines slightly above average for overs (player tendencies)
                # Add some noise to simulate different book pricing
                line_noise = np.random.uniform(-1.5, 1.5)
                line = round(recent_avg + line_noise, 0) + 0.5

                # Standard juice is -110/-110 but varies
                over_odds = -110 + np.random.randint(-5, 6)
                under_odds = -110 + np.random.randint(-5, 6)

                actual = current_game[prop_type]

                props.append({
                    'date': current_game['date'],
                    'player': player,
                    'team': current_game['team'],
                    'opponent': current_game['opponent'],
                    'home': current_game['home'],
                    'b2b': current_game['b2b'],
                    'opp_def_rtg': current_game['opp_def_rtg'],
                    'prop_type': prop_type,
                    'line': line,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'actual': actual,
                    'over_hit': actual > line,
                    'under_hit': actual < line,
                    'push': actual == line,
                    'recent_avg': round(recent_avg, 1),
                    'recent_std': round(recent_std, 2)
                })

    return pd.DataFrame(props)
