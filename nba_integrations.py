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
from typing import Optional, Dict, List
import requests
import time
import json

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_nba_season() -> str:
    """
    Calculate the current NBA season string based on today's date.
    NBA seasons run from October to June.
    - Oct-Dec 2025 = "2025-26"
    - Jan-Sep 2026 = "2025-26"
    """
    today = datetime.now()
    year = today.year
    month = today.month

    # If we're in Oct-Dec, season is current year to next year
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    # If we're in Jan-Sep, season started last year
    else:
        return f"{year - 1}-{str(year)[-2:]}"

# =============================================================================
# NBA STATS API INTEGRATION
# =============================================================================

class NBADataFetcher:
    """
    Fetches real NBA data using the nba_api package.
    Includes player game logs, team stats, and schedule data.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self._player_id_cache = {}
        self._team_id_cache = {}
        
        # Rate limiting - NBA API can be sensitive
        self.request_delay = 0.6  # seconds between requests
        self._last_request = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.time()
    
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
        Fetch player game logs for a season.

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
            print(f"Could not find player: {player_name}")
            return pd.DataFrame()
        
        try:
            from nba_api.stats.endpoints import playergamelog
            
            self._rate_limit()
            
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = log.get_data_frames()[0]
            
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
            
        except Exception as e:
            print(f"Error fetching game logs for {player_name}: {e}")
            return pd.DataFrame()
    
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

            # Map team names to abbreviations (only NBA teams)
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

    # Star players by team - their absence significantly impacts teammates
    STAR_PLAYERS = {
        'ATL': ['Trae Young', 'Dejounte Murray'],
        'BOS': ['Jayson Tatum', 'Jaylen Brown'],
        'BKN': ['Mikal Bridges', 'Cameron Johnson'],
        'CHA': ['LaMelo Ball', 'Brandon Miller'],
        'CHI': ['Zach LaVine', 'DeMar DeRozan', 'Coby White'],
        'CLE': ['Donovan Mitchell', 'Darius Garland', 'Evan Mobley'],
        'DAL': ['Luka Doncic', 'Kyrie Irving'],
        'DEN': ['Nikola Jokic', 'Jamal Murray'],
        'DET': ['Cade Cunningham', 'Jaden Ivey'],
        'GSW': ['Stephen Curry', 'Klay Thompson', 'Draymond Green'],
        'HOU': ['Jalen Green', 'Alperen Sengun', 'Fred VanVleet'],
        'IND': ['Tyrese Haliburton', 'Pascal Siakam', 'Myles Turner'],
        'LAC': ['Kawhi Leonard', 'Paul George', 'James Harden'],
        'LAL': ['LeBron James', 'Anthony Davis'],
        'MEM': ['Ja Morant', 'Desmond Bane', 'Jaren Jackson Jr.'],
        'MIA': ['Jimmy Butler', 'Bam Adebayo', 'Tyler Herro'],
        'MIL': ['Giannis Antetokounmpo', 'Damian Lillard', 'Khris Middleton'],
        'MIN': ['Anthony Edwards', 'Karl-Anthony Towns', 'Rudy Gobert'],
        'NOP': ['Zion Williamson', 'Brandon Ingram', 'CJ McCollum'],
        'NYK': ['Jalen Brunson', 'Julius Randle', 'RJ Barrett'],
        'OKC': ['Shai Gilgeous-Alexander', 'Chet Holmgren', 'Jalen Williams'],
        'ORL': ['Paolo Banchero', 'Franz Wagner', 'Jalen Suggs'],
        'PHI': ['Joel Embiid', 'Tyrese Maxey'],
        'PHX': ['Kevin Durant', 'Devin Booker', 'Bradley Beal'],
        'POR': ['Anfernee Simons', 'Jerami Grant', 'Scoot Henderson'],
        'SAC': ['De\'Aaron Fox', 'Domantas Sabonis', 'Keegan Murray'],
        'SAS': ['Victor Wembanyama', 'Devin Vassell', 'Keldon Johnson'],
        'TOR': ['Scottie Barnes', 'Pascal Siakam', 'OG Anunoby'],
        'UTA': ['Lauri Markkanen', 'Jordan Clarkson', 'Collin Sexton'],
        'WAS': ['Jordan Poole', 'Kyle Kuzma', 'Deni Avdija'],
    }

    # Usage boost when star is out (by stat type)
    STAR_OUT_BOOST = {
        'points': 1.08,    # 8% boost to scoring
        'assists': 1.06,   # 6% boost to assists
        'rebounds': 1.04,  # 4% boost to rebounds
        'pra': 1.07,       # 7% boost to PRA
        'threes': 1.05,    # 5% boost to threes
    }

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
                'player_points_rebounds_assists'
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
    """
    
    def __init__(self, nba_fetcher: NBADataFetcher = None,
                 odds_client: OddsAPIClient = None):
        self.nba = nba_fetcher or NBADataFetcher()
        self.odds = odds_client
        
        # Import prediction models
        import sys
        sys.path.insert(0, '/home/claude')
        from nba_prop_model import WeightedAverageModel, EnsembleModel
        
        self.models = {
            'weighted': WeightedAverageModel(),
            'ensemble': EnsembleModel()
        }
    
    def analyze_prop(self, player_name: str, prop_type: str, 
                     line: float, odds: int = -110,
                     last_n_games: int = 15) -> dict:
        """
        Analyze a single prop bet.
        
        Args:
            player_name: Player's full name
            prop_type: 'points', 'rebounds', 'assists', 'pra', etc.
            line: The betting line (e.g., 24.5)
            odds: American odds (default -110)
            last_n_games: Games to analyze
            
        Returns:
            Analysis dict with projection, edge, recommendation
        """
        # Fetch player data
        logs = self.nba.get_player_game_logs(player_name, last_n_games=last_n_games)
        
        if logs.empty or prop_type not in logs.columns:
            return {'error': f'Could not fetch data for {player_name}'}
        
        # Get predictions from models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(logs[prop_type], line)
            predictions[name] = {
                'projection': pred.projection,
                'confidence': pred.confidence,
                'edge': pred.edge,
                'side': pred.recommended_side
            }
        
        # Calculate stats
        recent_avg = logs[prop_type].mean()
        recent_median = logs[prop_type].median()
        recent_std = logs[prop_type].std()
        hit_rate_over = (logs[prop_type] > line).mean()
        hit_rate_under = (logs[prop_type] < line).mean()
        
        # Last 5 games trend
        last_5 = logs.head(5)[prop_type]
        trend = "↑" if last_5.iloc[0] > last_5.mean() else "↓"
        
        # Best recommendation (consensus)
        sides = [p['side'] for p in predictions.values() if p['side'] != 'pass']
        if sides:
            from collections import Counter
            recommended_side = Counter(sides).most_common(1)[0][0]
        else:
            recommended_side = 'pass'
        
        avg_edge = np.mean([p['edge'] for p in predictions.values()])
        
        return {
            'player': player_name,
            'prop_type': prop_type,
            'line': line,
            'odds': odds,
            'sample_size': len(logs),
            'recent_avg': round(recent_avg, 1),
            'recent_median': round(recent_median, 1),
            'recent_std': round(recent_std, 2),
            'hit_rate_over': round(hit_rate_over * 100, 1),
            'hit_rate_under': round(hit_rate_under * 100, 1),
            'last_5_trend': trend,
            'last_5_avg': round(last_5.mean(), 1),
            'model_predictions': predictions,
            'recommended_side': recommended_side,
            'avg_edge': round(avg_edge * 100, 2),
            'confidence': round(np.mean([p['confidence'] for p in predictions.values()]), 2)
        }
    
    def analyze_multiple_props(self, props: List[dict]) -> pd.DataFrame:
        """
        Analyze multiple props at once.
        
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
                odds=prop.get('odds', -110)
            )
            
            if 'error' not in analysis:
                results.append({
                    'Player': analysis['player'],
                    'Prop': analysis['prop_type'],
                    'Line': analysis['line'],
                    'Avg': analysis['recent_avg'],
                    'Median': analysis['recent_median'],
                    'Over%': analysis['hit_rate_over'],
                    'Under%': analysis['hit_rate_under'],
                    'Edge': analysis['avg_edge'],
                    'Conf': analysis['confidence'],
                    'Pick': analysis['recommended_side'].upper(),
                    'Trend': analysis['last_5_trend']
                })
            
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(results)
    
    def find_value_props(self, min_edge: float = 0.05, max_events: int = 5) -> pd.DataFrame:
        """
        Scan current odds for value props.
        Requires OddsAPIClient to be configured.

        Args:
            min_edge: Minimum edge threshold (default 5%)
            max_events: Max number of games to scan (saves API calls)
        """
        if not self.odds:
            print("OddsAPIClient required for live odds scanning")
            return pd.DataFrame()

        # Get current props from all events
        raw_props = self.odds.get_all_player_props(max_events=max_events)
        props_df = self.odds.parse_player_props(raw_props)
        
        if props_df.empty:
            print("No props available")
            return pd.DataFrame()
        
        # Get best odds
        best_odds = self.odds.get_best_odds(props_df)
        
        # Analyze each unique player/prop
        value_props = []
        
        unique_props = best_odds.groupby(['player', 'prop_type', 'line']).first().reset_index()
        
        for _, row in unique_props.iterrows():
            analysis = self.analyze_prop(
                player_name=row['player'],
                prop_type=row['prop_type'],
                line=row['line'],
                odds=row['odds']
            )
            
            if 'error' not in analysis and abs(analysis['avg_edge']) >= min_edge * 100:
                value_props.append({
                    **analysis,
                    'bookmaker': row['bookmaker']
                })
        
        return pd.DataFrame(value_props)


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
