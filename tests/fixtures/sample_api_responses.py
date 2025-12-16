"""Sample API responses for integration testing.

These fixtures represent realistic API responses from:
- NBA Stats API
- The Odds API
- Perplexity news search
"""

from datetime import datetime, timedelta


def get_sample_game_logs(num_games: int = 15):
    """Generate sample player game logs."""
    return [
        {
            'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
            'matchup': 'DAL @ LAL' if i % 2 == 0 else 'DAL vs. BOS',
            'points': 25 + (i % 10),
            'rebounds': 7 + (i % 4),
            'assists': 5 + (i % 3),
            'fg3m': 2 + (i % 3),
            'minutes': 32 + (i % 6),
            'home': i % 2 == 1,
            'opponent': 'LAL' if i % 2 == 0 else 'BOS',
        }
        for i in range(num_games)
    ]


def get_sample_odds_events():
    """Generate sample events from The Odds API."""
    return [
        {
            'id': 'event_001',
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': (datetime.now() + timedelta(hours=3)).isoformat(),
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Dallas Mavericks',
        },
        {
            'id': 'event_002',
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': (datetime.now() + timedelta(hours=5)).isoformat(),
            'home_team': 'Boston Celtics',
            'away_team': 'Phoenix Suns',
        },
    ]


def get_sample_player_props():
    """Generate sample player props from The Odds API."""
    return [
        {
            'id': 'event_001',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Dallas Mavericks',
            'commence_time': (datetime.now() + timedelta(hours=3)).isoformat(),
            'bookmakers': [
                {
                    'key': 'fanduel',
                    'title': 'FanDuel',
                    'markets': [
                        {
                            'key': 'player_points',
                            'outcomes': [
                                {'description': 'Luka Doncic', 'name': 'Over', 'point': 30.5, 'price': -115},
                                {'description': 'Luka Doncic', 'name': 'Under', 'point': 30.5, 'price': -105},
                                {'description': 'LeBron James', 'name': 'Over', 'point': 25.5, 'price': -110},
                                {'description': 'LeBron James', 'name': 'Under', 'point': 25.5, 'price': -110},
                            ]
                        },
                        {
                            'key': 'player_rebounds',
                            'outcomes': [
                                {'description': 'Luka Doncic', 'name': 'Over', 'point': 8.5, 'price': -110},
                                {'description': 'Luka Doncic', 'name': 'Under', 'point': 8.5, 'price': -110},
                            ]
                        },
                        {
                            'key': 'player_assists',
                            'outcomes': [
                                {'description': 'Luka Doncic', 'name': 'Over', 'point': 8.5, 'price': -120},
                                {'description': 'Luka Doncic', 'name': 'Under', 'point': 8.5, 'price': +100},
                            ]
                        },
                    ]
                },
                {
                    'key': 'draftkings',
                    'title': 'DraftKings',
                    'markets': [
                        {
                            'key': 'player_points',
                            'outcomes': [
                                {'description': 'Luka Doncic', 'name': 'Over', 'point': 30.5, 'price': -112},
                                {'description': 'Luka Doncic', 'name': 'Under', 'point': 30.5, 'price': -108},
                            ]
                        }
                    ]
                }
            ]
        }
    ]


def get_sample_game_lines():
    """Generate sample game lines (totals and spreads)."""
    return [
        {
            'game_id': 'event_001',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Dallas Mavericks',
            'commence_time': (datetime.now() + timedelta(hours=3)).isoformat(),
            'total': 228.5,
            'total_category': 'MEDIUM',
            'home_spread': -4.5,
            'blowout_risk': 'LOW',
            'favorite': 'Los Angeles Lakers',
        },
        {
            'game_id': 'event_002',
            'home_team': 'Boston Celtics',
            'away_team': 'Phoenix Suns',
            'commence_time': (datetime.now() + timedelta(hours=5)).isoformat(),
            'total': 222.0,
            'total_category': 'LOW',
            'home_spread': -8.5,
            'blowout_risk': 'MEDIUM',
            'favorite': 'Boston Celtics',
        },
    ]


def get_sample_injury_report():
    """Generate sample injury report."""
    return [
        {'player': 'Anthony Davis', 'team': 'LAL', 'status': 'OUT', 'injury': 'Knee'},
        {'player': 'Kyrie Irving', 'team': 'DAL', 'status': 'GTD', 'injury': 'Hamstring'},
        {'player': 'Kevin Durant', 'team': 'PHX', 'status': 'QUESTIONABLE', 'injury': 'Calf'},
    ]


def get_sample_team_defense():
    """Generate sample team defense vs position data."""
    return {
        'LAL': {'pts_factor': 1.05, 'reb_factor': 0.98, 'ast_factor': 1.02, 'pts_rank': 22, 'matchup_rating': 'GOOD'},
        'BOS': {'pts_factor': 0.92, 'reb_factor': 0.95, 'ast_factor': 0.90, 'pts_rank': 2, 'matchup_rating': 'TOUGH'},
        'DAL': {'pts_factor': 1.02, 'reb_factor': 1.00, 'ast_factor': 1.05, 'pts_rank': 18, 'matchup_rating': 'NEUTRAL'},
        'PHX': {'pts_factor': 1.08, 'reb_factor': 1.03, 'ast_factor': 1.10, 'pts_rank': 28, 'matchup_rating': 'SMASH'},
    }


def get_sample_team_pace():
    """Generate sample team pace data."""
    return {
        'LAL': {'pace': 101.5, 'pace_factor': 1.02},
        'BOS': {'pace': 99.2, 'pace_factor': 0.99},
        'DAL': {'pace': 100.8, 'pace_factor': 1.01},
        'PHX': {'pace': 103.0, 'pace_factor': 1.03},
        'IND': {'pace': 105.5, 'pace_factor': 1.06},
        'MIN': {'pace': 96.0, 'pace_factor': 0.96},
    }
