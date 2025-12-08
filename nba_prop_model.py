"""
NBA Prop Betting Analysis & Backtesting System
================================================
A comprehensive model for analyzing NBA player props and backtesting betting strategies.

Key Features:
- Player performance modeling (points, rebounds, assists, PRA)
- Line value detection (expected value calculation)
- Historical backtesting with proper bankroll metrics
- Multiple modeling approaches (weighted averages, regression, situational)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TEAM ABBREVIATION NORMALIZER
# =============================================================================

# Standard NBA team abbreviations mapping - handles full names, partial names, and variants
TEAM_ABBREV_MAP = {
    # Full names -> Standard abbreviation
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL', 'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
    # Common variants and typos
    'Spurs': 'SAS', 'Clippers': 'LAC', 'Lakers': 'LAL', 'Warriors': 'GSW',
    'Pelicans': 'NOP', '76ers': 'PHI', 'Sixers': 'PHI', 'Blazers': 'POR',
    'Timberwolves': 'MIN', 'T-Wolves': 'MIN',
    # First 3 letter fallbacks that don't match standard abbrevs
    'SAN': 'SAS',  # San Antonio -> SAS
    'NEW': 'NOP',  # Could be New Orleans or New York - default to NOP
    'GOL': 'GSW',  # Golden State
    'LOS': 'LAL',  # Los Angeles - default to Lakers
    'POR': 'POR',  # Portland
    'PHO': 'PHX',  # Phoenix (PHO vs PHX)
    'BRO': 'BKN',  # Brooklyn
    # Standard abbreviations map to themselves
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
    'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
    'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
    'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHX', 'POR': 'POR',
    'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR', 'UTA': 'UTA', 'WAS': 'WAS',
}


def normalize_team_abbrev(team_input: str) -> str:
    """
    Normalize any team reference to standard 3-letter NBA abbreviation.

    Handles:
    - Full team names: "San Antonio Spurs" -> "SAS"
    - Partial names: "Spurs" -> "SAS"
    - Non-standard abbrevs: "SAN" -> "SAS", "PHO" -> "PHX"
    - Already standard: "SAS" -> "SAS"

    Args:
        team_input: Team name, abbreviation, or variant

    Returns:
        Standard 3-letter NBA abbreviation, or original if not found
    """
    if not team_input:
        return team_input

    team_str = str(team_input).strip()

    # Direct lookup
    if team_str in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[team_str]

    # Case-insensitive lookup
    team_upper = team_str.upper()
    if team_upper in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[team_upper]

    # Try matching by first 3 characters
    if len(team_str) >= 3:
        first_three = team_str[:3].upper()
        if first_three in TEAM_ABBREV_MAP:
            return TEAM_ABBREV_MAP[first_three]

    # If 3-letter code, check if it's already standard
    if len(team_str) == 3:
        return team_str.upper()

    return team_str


# =============================================================================
# DATA GENERATION (Simulating historical NBA data for demonstration)
# =============================================================================

def generate_player_season_data(player_name: str, team: str, position: str,
                                 base_stats: dict, games: int = 82,
                                 season: str = "2023-24") -> pd.DataFrame:
    """Generate realistic player game logs with natural variance."""
    np.random.seed(hash(player_name) % 2**32)
    
    dates = pd.date_range(start='2023-10-24', periods=games, freq='2D')
    
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
        b2b_penalty = 0.92 if is_back_to_back else 1.0
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


# =============================================================================
# PROP LINE GENERATION (Simulating sportsbook lines)
# =============================================================================

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


# =============================================================================
# PREDICTION MODELS
# =============================================================================

@dataclass
class Prediction:
    """Container for model predictions."""
    projection: float
    confidence: float  # 0-1 scale
    edge: float  # Expected edge over the line
    recommended_side: str  # 'over', 'under', or 'pass'
    model_name: str


class WeightedAverageModel:
    """
    Weighted average model with recency bias.
    Simple but effective baseline model.
    """
    
    def __init__(self, weights: list = None):
        # Default: more weight on recent games
        self.weights = weights or [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.03]
        self.name = "WeightedAverage"
    
    def predict(self, history: pd.Series, line: float) -> Prediction:
        """Predict based on weighted recent performance."""
        recent = history.tail(len(self.weights)).values
        
        if len(recent) < 5:
            return Prediction(np.nan, 0, 0, 'pass', self.name)
        
        # Apply weights (reversed so most recent gets highest weight)
        weights = self.weights[:len(recent)][::-1]
        weights = np.array(weights) / sum(weights)
        
        projection = np.average(recent, weights=weights)
        std = np.std(recent)
        
        # Calculate edge
        edge = (projection - line) / line if line > 0 else 0
        
        # Confidence based on consistency
        cv = std / projection if projection > 0 else 1
        confidence = max(0, min(1, 1 - cv))
        
        # Determine recommendation
        if abs(edge) < 0.03:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'
        
        return Prediction(round(projection, 1), round(confidence, 3), 
                         round(edge, 4), side, self.name)


class SituationalModel:
    """
    Model that adjusts projections based on situational factors:
    - Home/away
    - Back-to-back games
    - Opponent defensive rating
    """
    
    def __init__(self):
        self.name = "Situational"
        self.base_model = WeightedAverageModel()
        
        # Situational adjustment factors (derived from historical analysis)
        self.home_boost = 1.025
        self.away_penalty = 0.975
        self.b2b_penalty = 0.93
        self.def_rtg_factor = 0.008  # Per point above/below 112
    
    def predict(self, history: pd.DataFrame, current_game: dict, 
                prop_type: str, line: float) -> Prediction:
        """Predict with situational adjustments."""
        
        base_pred = self.base_model.predict(history[prop_type], line)
        
        if base_pred.projection is np.nan:
            return base_pred
        
        # Apply adjustments
        projection = base_pred.projection
        
        if current_game.get('home', False):
            projection *= self.home_boost
        else:
            projection *= self.away_penalty
        
        if current_game.get('b2b', False):
            projection *= self.b2b_penalty
        
        # Opponent defense adjustment
        opp_def = current_game.get('opp_def_rtg', 112)
        def_adjustment = 1 + (opp_def - 112) * self.def_rtg_factor
        projection *= def_adjustment
        
        edge = (projection - line) / line if line > 0 else 0
        
        if abs(edge) < 0.03:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'
        
        return Prediction(round(projection, 1), base_pred.confidence,
                         round(edge, 4), side, self.name)


class MedianModel:
    """
    Uses median instead of mean - more robust to outliers.
    Good for volatile players.
    """
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.name = "Median"
    
    def predict(self, history: pd.Series, line: float) -> Prediction:
        recent = history.tail(self.lookback)
        
        if len(recent) < 5:
            return Prediction(np.nan, 0, 0, 'pass', self.name)
        
        projection = recent.median()
        std = recent.std()
        
        edge = (projection - line) / line if line > 0 else 0
        
        cv = std / projection if projection > 0 else 1
        confidence = max(0, min(1, 1 - cv))
        
        if abs(edge) < 0.03:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'
        
        return Prediction(round(projection, 1), round(confidence, 3),
                         round(edge, 4), side, self.name)


class EnsembleModel:
    """
    Combines multiple models with weighted voting.
    """

    def __init__(self):
        self.name = "Ensemble"
        self.weighted_model = WeightedAverageModel()
        self.median_model = MedianModel()
        self.weights = {'weighted': 0.5, 'median': 0.5}

    def predict(self, history: pd.Series, line: float) -> Prediction:
        pred1 = self.weighted_model.predict(history, line)
        pred2 = self.median_model.predict(history, line)

        if np.isnan(pred1.projection) or np.isnan(pred2.projection):
            return Prediction(np.nan, 0, 0, 'pass', self.name)

        projection = (pred1.projection * self.weights['weighted'] +
                     pred2.projection * self.weights['median'])
        confidence = (pred1.confidence * self.weights['weighted'] +
                     pred2.confidence * self.weights['median'])

        edge = (projection - line) / line if line > 0 else 0

        if abs(edge) < 0.03:
            side = 'pass'
        elif edge > 0:
            side = 'over'
        else:
            side = 'under'

        return Prediction(round(projection, 1), round(confidence, 3),
                         round(edge, 4), side, self.name)


class SmartModel:
    """
    Advanced model that considers:
    - Recent form (last 5 games weighted 60%, games 6-15 weighted 40%)
    - Consistency (coefficient of variation)
    - Historical hit rate against the line
    - Trend direction (hot/cold streak)
    - Expected value based on odds
    """

    def __init__(self):
        self.name = "Smart"

    def predict(self, history: pd.Series, line: float, odds: int = -110) -> Prediction:
        """
        Generate prediction with comprehensive analysis.

        Args:
            history: Series of stat values (most recent last)
            line: The betting line
            odds: American odds (default -110)
        """
        if len(history) < 5:
            return Prediction(np.nan, 0, 0, 'pass', self.name)

        # Split into recent (last 5) and older games
        recent = history.tail(5)
        older = history.tail(15).head(10) if len(history) >= 10 else history.head(len(history) - 5)

        # Weighted projection: 60% recent, 40% older
        recent_avg = recent.mean()
        older_avg = older.mean() if len(older) > 0 else recent_avg
        projection = (recent_avg * 0.6) + (older_avg * 0.4)

        # Trend adjustment: if recent > older, player is hot
        trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        trend_adjustment = trend * 0.1  # 10% of the trend
        projection *= (1 + trend_adjustment)

        # Calculate consistency (lower CV = more consistent = higher confidence)
        full_std = history.tail(15).std()
        full_avg = history.tail(15).mean()
        cv = full_std / full_avg if full_avg > 0 else 1

        # Historical hit rates
        over_hits = (history > line).sum()
        under_hits = (history < line).sum()
        total_games = len(history)
        over_rate = over_hits / total_games
        under_rate = under_hits / total_games

        # Raw edge based on projection
        raw_edge = (projection - line) / line if line > 0 else 0

        # Adjust edge based on hit rate (blend projection edge with historical edge)
        if raw_edge > 0:
            # For overs: boost edge if hit rate is high
            hit_rate_edge = (over_rate - 0.5) * 2  # Convert 50-100% to 0-100%
            adjusted_edge = (raw_edge * 0.6) + (hit_rate_edge * 0.4)
        else:
            # For unders: boost edge if hit rate is high
            hit_rate_edge = (under_rate - 0.5) * 2
            adjusted_edge = (raw_edge * 0.6) + (-hit_rate_edge * 0.4)

        # Calculate expected value
        implied_prob = american_to_implied_prob(odds)
        if adjusted_edge > 0:
            our_prob = over_rate
        else:
            our_prob = under_rate

        # EV = (win_prob * payout) - (lose_prob * stake)
        decimal_odds = american_to_decimal(odds)
        ev = (our_prob * (decimal_odds - 1)) - (1 - our_prob)

        # Confidence based on:
        # - Consistency (30%)
        # - Sample size (20%)
        # - Hit rate clarity (30%)
        # - Trend strength (20%)
        consistency_score = max(0, 1 - cv)  # 0-1, higher is better
        sample_score = min(1, len(history) / 15)  # Max at 15 games
        hit_clarity = abs(over_rate - 0.5) * 2  # 0-1, how clear is the edge
        trend_score = min(1, abs(trend) * 5)  # Stronger trend = more confident

        confidence = (
            consistency_score * 0.3 +
            sample_score * 0.2 +
            hit_clarity * 0.3 +
            trend_score * 0.2
        )

        # Determine recommendation
        # Require higher edge threshold for lower confidence
        min_edge_required = 0.03 + (1 - confidence) * 0.02  # 3-5% depending on confidence

        if abs(adjusted_edge) < min_edge_required:
            side = 'pass'
        elif adjusted_edge > 0:
            side = 'over'
        else:
            side = 'under'

        return Prediction(
            projection=round(projection, 1),
            confidence=round(confidence, 3),
            edge=round(adjusted_edge, 4),
            recommended_side=side,
            model_name=self.name
        )

    def analyze(self, history: pd.Series, line: float, odds: int = -110) -> dict:
        """
        Return detailed analysis breakdown.
        """
        if len(history) < 5:
            return {'error': 'Not enough data'}

        pred = self.predict(history, line, odds)

        recent = history.tail(5)
        older = history.tail(15).head(10) if len(history) >= 10 else pd.Series([])

        over_rate = (history > line).mean() * 100
        under_rate = (history < line).mean() * 100

        trend = 'HOT' if recent.mean() > older.mean() else 'COLD' if len(older) > 0 else 'N/A'

        return {
            'projection': pred.projection,
            'edge': round(pred.edge * 100, 1),
            'confidence': round(pred.confidence * 100, 0),
            'pick': pred.recommended_side.upper(),
            'recent_avg': round(recent.mean(), 1),
            'season_avg': round(history.mean(), 1),
            'over_rate': round(over_rate, 0),
            'under_rate': round(under_rate, 0),
            'trend': trend,
            'std_dev': round(history.std(), 1),
            'games_analyzed': len(history)
        }

    def analyze_with_context(self, history: pd.Series, line: float, odds: int = -110,
                             prop_type: str = 'points', opponent: str = None,
                             is_home: bool = None, defense_data: pd.DataFrame = None,
                             home_away_factor: float = None,
                             # New context parameters
                             is_b2b: bool = False,
                             pace_data: pd.DataFrame = None,
                             player_team: str = None,
                             game_total: float = None,
                             blowout_risk: str = None,
                             vs_team_factor: float = None,
                             minutes_factor: float = None,
                             # Injury parameters
                             teammate_injury_boost: float = None,
                             player_injury_status: str = None,
                             stars_out: List[str] = None) -> dict:
        """
        Enhanced analysis with full situational context.

        Args:
            history: Series of stat values
            line: Betting line
            odds: American odds
            prop_type: 'points', 'rebounds', 'assists', 'pra', 'threes'
            opponent: Team abbreviation (e.g., 'LAL', 'BOS')
            is_home: True if home game, False if away
            defense_data: DataFrame from get_team_defense_vs_position()
            home_away_factor: Player's home/away multiplier for this stat
            is_b2b: True if player is on back-to-back
            pace_data: DataFrame from get_team_pace()
            player_team: Player's team abbreviation
            game_total: Vegas over/under total
            blowout_risk: 'HIGH', 'MEDIUM', 'LOW' from spread
            vs_team_factor: Player's historical performance vs opponent
            minutes_factor: Recent minutes trend factor
        """
        if len(history) < 5:
            return {'error': 'Not enough data'}

        # Normalize team abbreviations to standard format
        if opponent:
            opponent = normalize_team_abbrev(opponent)
        if player_team:
            player_team = normalize_team_abbrev(player_team)

        # Start with base prediction
        pred = self.predict(history, line, odds)
        base_projection = pred.projection

        # Initialize all adjustments
        adjustments = {
            'opp_defense': 1.0,
            'location': 1.0,
            'b2b': 1.0,
            'pace': 1.0,
            'total': 1.0,
            'blowout': 1.0,
            'vs_team': 1.0,
            'minutes': 1.0,
            'injury_boost': 1.0  # Teammate injury usage boost
        }
        opp_rank = None
        matchup_rating = 'NEUTRAL'
        flags = []  # Warning/boost flags

        # 0. INJURY STATUS FLAGS
        if player_injury_status:
            if player_injury_status.upper() == 'GTD':
                flags.append('⚠️ GTD')
            elif player_injury_status.upper() in ['QUESTIONABLE', 'DOUBTFUL']:
                flags.append('❓ QUESTIONABLE')

        # 1. OPPONENT DEFENSE ADJUSTMENT
        if opponent and defense_data is not None and not defense_data.empty:
            opp_row = defense_data[defense_data['team_abbrev'] == opponent]
            if not opp_row.empty:
                factor_map = {
                    'points': 'pts_factor',
                    'rebounds': 'reb_factor',
                    'assists': 'ast_factor',
                    'pra': 'pra_factor',
                    'threes': 'threes_factor'
                }
                rank_map = {
                    'points': 'pts_rank',
                    'rebounds': 'reb_rank',
                    'assists': 'ast_rank',
                    'threes': 'threes_rank'
                }

                factor_col = factor_map.get(prop_type, 'pts_factor')
                rank_col = rank_map.get(prop_type, 'pts_rank')

                adjustments['opp_defense'] = float(opp_row[factor_col].values[0])
                opp_rank = int(opp_row[rank_col].values[0])

                if opp_rank <= 5:
                    matchup_rating = 'SMASH'
                    flags.append('💥 SMASH SPOT')
                elif opp_rank <= 10:
                    matchup_rating = 'GOOD'
                elif opp_rank >= 26:
                    matchup_rating = 'TOUGH'
                    flags.append('🛡️ TOUGH DEF')
                elif opp_rank >= 21:
                    matchup_rating = 'HARD'

        # 2. HOME/AWAY ADJUSTMENT
        if is_home is not None:
            if home_away_factor:
                adjustments['location'] = home_away_factor
            else:
                adjustments['location'] = 1.025 if is_home else 0.975

        # 3. BACK-TO-BACK PENALTY (-8%)
        if is_b2b:
            adjustments['b2b'] = 0.92
            flags.append('⚠️ B2B')

        # 4. PACE ADJUSTMENT
        if pace_data is not None and not pace_data.empty and player_team and opponent:
            player_pace = pace_data[pace_data['team_abbrev'] == player_team]
            opp_pace = pace_data[pace_data['team_abbrev'] == opponent]

            if not player_pace.empty and not opp_pace.empty:
                # Average pace factor of both teams
                combined_pace = (float(player_pace['pace_factor'].values[0]) +
                                float(opp_pace['pace_factor'].values[0])) / 2
                adjustments['pace'] = combined_pace

                if combined_pace >= 1.03:
                    flags.append('🏃 FAST PACE')
                elif combined_pace <= 0.97:
                    flags.append('🐢 SLOW PACE')

        # 5. VEGAS TOTAL ADJUSTMENT (for scoring props)
        if game_total and prop_type in ['points', 'pra', 'threes']:
            # League avg total ~225
            total_factor = game_total / 225
            # Apply partial adjustment (don't over-weight)
            adjustments['total'] = 1 + (total_factor - 1) * 0.3

            if game_total >= 235:
                flags.append('📈 HIGH TOTAL')
            elif game_total <= 215:
                flags.append('📉 LOW TOTAL')

        # 6. BLOWOUT RISK (reduces minutes for starters)
        if blowout_risk == 'HIGH':
            adjustments['blowout'] = 0.95  # -5% for blowout risk
            flags.append('💨 BLOWOUT RISK')
        elif blowout_risk == 'MEDIUM':
            adjustments['blowout'] = 0.98

        # 7. PLAYER VS TEAM HISTORY
        if vs_team_factor and vs_team_factor != 1.0:
            # Cap the adjustment at +/- 10%
            adjustments['vs_team'] = max(0.9, min(1.1, vs_team_factor))
            if vs_team_factor >= 1.1:
                flags.append('🎯 OWNS THIS TEAM')
            elif vs_team_factor <= 0.9:
                flags.append('😰 STRUGGLES VS')

        # 8. MINUTES TREND
        if minutes_factor and minutes_factor != 1.0:
            adjustments['minutes'] = max(0.9, min(1.1, minutes_factor))
            if minutes_factor >= 1.05:
                flags.append('⏰ MINS UP')
            elif minutes_factor <= 0.95:
                flags.append('⏰ MINS DOWN')

        # 9. TEAMMATE INJURY BOOST (when stars are out, boost usage)
        if teammate_injury_boost and teammate_injury_boost > 1.0:
            adjustments['injury_boost'] = min(1.15, teammate_injury_boost)  # Cap at 15%
            boost_pct = round((teammate_injury_boost - 1) * 100)
            if stars_out:
                flags.append(f"📈 +{boost_pct}% ({', '.join(stars_out[:2])} OUT)")
            else:
                flags.append(f"📈 USAGE BOOST +{boost_pct}%")

        # CALCULATE FINAL ADJUSTED PROJECTION
        total_adjustment = 1.0
        for adj in adjustments.values():
            total_adjustment *= adj

        adjusted_projection = base_projection * total_adjustment

        # Recalculate edge
        adjusted_edge = (adjusted_projection - line) / line if line > 0 else 0

        # Stats
        recent = history.tail(5)
        older = history.tail(15).head(10) if len(history) >= 10 else pd.Series([])
        over_rate = (history > line).mean() * 100
        under_rate = (history < line).mean() * 100
        trend = 'HOT' if len(older) > 0 and recent.mean() > older.mean() else 'COLD' if len(older) > 0 else 'N/A'

        # CONFIDENCE SCORING
        base_confidence = pred.confidence
        confidence_adjustments = 0

        # Boost confidence for clear situations
        if matchup_rating in ['SMASH', 'TOUGH']:
            confidence_adjustments += 0.1
        elif matchup_rating in ['GOOD', 'HARD']:
            confidence_adjustments += 0.05

        # Reduce confidence for uncertainty
        if is_b2b:
            confidence_adjustments -= 0.05
        if blowout_risk == 'HIGH':
            confidence_adjustments -= 0.05

        adjusted_confidence = max(0.2, min(1.0, base_confidence + confidence_adjustments))

        # DETERMINE RECOMMENDATION
        min_edge_required = 0.03 + (1 - adjusted_confidence) * 0.02

        if abs(adjusted_edge) < min_edge_required:
            pick = 'PASS'
        elif adjusted_edge > 0:
            pick = 'OVER'
        else:
            pick = 'UNDER'

        # Calculate individual adjustment percentages for display
        adj_breakdown = {k: round((v - 1) * 100, 1) for k, v in adjustments.items()}

        return {
            'projection': round(adjusted_projection, 1),
            'base_projection': round(base_projection, 1),
            'edge': round(adjusted_edge * 100, 1),
            'confidence': round(adjusted_confidence * 100, 0),
            'pick': pick,
            'recent_avg': round(recent.mean(), 1),
            'season_avg': round(history.mean(), 1),
            'over_rate': round(over_rate, 0),
            'under_rate': round(under_rate, 0),
            'trend': trend,
            'std_dev': round(history.std(), 1),
            'games_analyzed': len(history),
            # Context info
            'opp_adjustment': adj_breakdown['opp_defense'],
            'location_adjustment': adj_breakdown['location'],
            'opp_rank': opp_rank,
            'matchup': matchup_rating,
            'is_home': is_home,
            'is_b2b': is_b2b,
            # New fields
            'total_adjustment': round((total_adjustment - 1) * 100, 1),
            'adjustments': adj_breakdown,
            'flags': flags,
            'game_total': game_total,
            'blowout_risk': blowout_risk
        }


# =============================================================================
# EXPECTED VALUE CALCULATION
# =============================================================================

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def calculate_ev(projection: float, line: float, std: float, 
                 over_odds: int, under_odds: int) -> dict:
    """
    Calculate expected value for over/under bets.
    Uses normal distribution assumption.
    """
    from scipy import stats
    
    # Probability of going over the line
    z_score = (line - projection) / std if std > 0 else 0
    prob_over = 1 - stats.norm.cdf(z_score)
    prob_under = stats.norm.cdf(z_score)
    
    # Calculate EV
    decimal_over = american_to_decimal(over_odds)
    decimal_under = american_to_decimal(under_odds)
    
    ev_over = (prob_over * (decimal_over - 1)) - (1 - prob_over)
    ev_under = (prob_under * (decimal_under - 1)) - (1 - prob_under)
    
    return {
        'prob_over': round(prob_over, 4),
        'prob_under': round(prob_under, 4),
        'ev_over': round(ev_over, 4),
        'ev_under': round(ev_under, 4),
        'best_bet': 'over' if ev_over > ev_under else 'under',
        'best_ev': max(ev_over, ev_under)
    }


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class BetResult:
    """Individual bet outcome."""
    date: datetime
    player: str
    prop_type: str
    side: str
    line: float
    odds: int
    projection: float
    edge: float
    actual: float
    won: bool
    profit: float
    units: float


class Backtester:
    """
    Backtesting engine for prop betting strategies.
    Tracks performance metrics and manages simulated bankroll.
    """
    
    def __init__(self, initial_bankroll: float = 1000, unit_size: float = 10):
        self.initial_bankroll = initial_bankroll
        self.unit_size = unit_size
        self.results: list[BetResult] = []
        self.bankroll_history = [initial_bankroll]
    
    def run_backtest(self, props_df: pd.DataFrame, game_logs: pd.DataFrame,
                     model, min_edge: float = 0.03, 
                     min_confidence: float = 0.4) -> pd.DataFrame:
        """
        Run backtest on historical prop data.
        
        Args:
            props_df: DataFrame with prop lines and results
            game_logs: DataFrame with player game logs
            model: Prediction model to use
            min_edge: Minimum edge required to bet
            min_confidence: Minimum confidence to bet
        """
        
        self.results = []
        bankroll = self.initial_bankroll
        self.bankroll_history = [bankroll]
        
        # Sort by date
        props_df = props_df.sort_values('date')
        
        for _, prop in props_df.iterrows():
            # Get player history before this game
            player_history = game_logs[
                (game_logs['player'] == prop['player']) & 
                (game_logs['date'] < prop['date'])
            ].sort_values('date')
            
            if len(player_history) < 10:
                continue
            
            # Get prediction
            if hasattr(model, 'predict') and model.name == 'Situational':
                game_context = {
                    'home': prop['home'],
                    'b2b': prop['b2b'],
                    'opp_def_rtg': prop['opp_def_rtg']
                }
                pred = model.predict(player_history, game_context, 
                                    prop['prop_type'], prop['line'])
            else:
                pred = model.predict(player_history[prop['prop_type']], prop['line'])
            
            # Check if we should bet
            if pred.recommended_side == 'pass':
                continue
            if abs(pred.edge) < min_edge:
                continue
            if pred.confidence < min_confidence:
                continue
            
            # Place bet
            side = pred.recommended_side
            odds = prop['over_odds'] if side == 'over' else prop['under_odds']
            
            # Determine units based on edge (Kelly-lite)
            units = min(3, max(1, abs(pred.edge) * 20))
            stake = units * self.unit_size
            
            if stake > bankroll:
                continue  # Can't afford this bet
            
            # Determine outcome
            actual = prop['actual']
            if side == 'over':
                won = actual > prop['line']
            else:
                won = actual < prop['line']
            
            push = actual == prop['line']
            
            if push:
                profit = 0
            elif won:
                decimal_odds = american_to_decimal(odds)
                profit = stake * (decimal_odds - 1)
            else:
                profit = -stake
            
            bankroll += profit
            self.bankroll_history.append(bankroll)
            
            result = BetResult(
                date=prop['date'],
                player=prop['player'],
                prop_type=prop['prop_type'],
                side=side,
                line=prop['line'],
                odds=odds,
                projection=pred.projection,
                edge=pred.edge,
                actual=actual,
                won=won,
                profit=profit,
                units=units
            )
            self.results.append(result)
        
        return self._generate_results_df()
    
    def _generate_results_df(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'date': r.date,
                'player': r.player,
                'prop_type': r.prop_type,
                'side': r.side,
                'line': r.line,
                'odds': r.odds,
                'projection': r.projection,
                'edge': r.edge,
                'actual': r.actual,
                'won': r.won,
                'profit': r.profit,
                'units': r.units
            }
            for r in self.results
        ])
    
    def get_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if not self.results:
            return {'error': 'No results to analyze'}
        
        results_df = self._generate_results_df()
        
        total_bets = len(results_df)
        wins = results_df['won'].sum()
        losses = total_bets - wins
        
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_profit = results_df['profit'].sum()
        total_wagered = (results_df['units'] * self.unit_size).sum()
        
        roi = total_profit / total_wagered if total_wagered > 0 else 0
        
        # Calculate max drawdown
        peak = self.initial_bankroll
        max_drawdown = 0
        for bal in self.bankroll_history:
            if bal > peak:
                peak = bal
            drawdown = (peak - bal) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Profit by prop type
        profit_by_prop = results_df.groupby('prop_type')['profit'].sum().to_dict()
        winrate_by_prop = results_df.groupby('prop_type')['won'].mean().to_dict()
        
        # Profit by side
        profit_by_side = results_df.groupby('side')['profit'].sum().to_dict()
        
        # Streaks
        results_df['streak'] = (results_df['won'] != results_df['won'].shift()).cumsum()
        win_streaks = results_df[results_df['won']].groupby('streak').size()
        loss_streaks = results_df[~results_df['won']].groupby('streak').size()
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 4),
            'total_profit': round(total_profit, 2),
            'total_wagered': round(total_wagered, 2),
            'roi': round(roi, 4),
            'final_bankroll': round(self.bankroll_history[-1], 2),
            'max_drawdown': round(max_drawdown, 4),
            'profit_by_prop': profit_by_prop,
            'winrate_by_prop': winrate_by_prop,
            'profit_by_side': profit_by_side,
            'longest_win_streak': win_streaks.max() if len(win_streaks) > 0 else 0,
            'longest_loss_streak': loss_streaks.max() if len(loss_streaks) > 0 else 0,
            'avg_odds': round(results_df['odds'].mean(), 1),
            'avg_edge': round(results_df['edge'].mean(), 4)
        }
    
    def print_report(self):
        """Print formatted backtest report."""
        metrics = self.get_metrics()
        
        if 'error' in metrics:
            print(metrics['error'])
            return
        
        print("\n" + "="*60)
        print("           NBA PROP BETTING BACKTEST REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print("-"*40)
        print(f"  Total Bets:        {metrics['total_bets']:,}")
        print(f"  Record:            {metrics['wins']}-{metrics['losses']}")
        print(f"  Win Rate:          {metrics['win_rate']*100:.1f}%")
        print(f"  ROI:               {metrics['roi']*100:+.2f}%")
        print(f"  Total Profit:      ${metrics['total_profit']:+,.2f}")
        print(f"  Total Wagered:     ${metrics['total_wagered']:,.2f}")
        
        print(f"\n💰 BANKROLL")
        print("-"*40)
        print(f"  Starting:          ${self.initial_bankroll:,.2f}")
        print(f"  Ending:            ${metrics['final_bankroll']:,.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:.1f}%")
        
        print(f"\n📈 BY PROP TYPE")
        print("-"*40)
        for prop, profit in metrics['profit_by_prop'].items():
            wr = metrics['winrate_by_prop'].get(prop, 0)
            print(f"  {prop.upper():12} ${profit:+8.2f}  ({wr*100:.1f}% win)")
        
        print(f"\n🎯 BY SIDE")
        print("-"*40)
        for side, profit in metrics['profit_by_side'].items():
            print(f"  {side.upper():12} ${profit:+8.2f}")
        
        print(f"\n📉 STREAKS")
        print("-"*40)
        print(f"  Best Win Streak:   {metrics['longest_win_streak']}")
        print(f"  Worst Loss Streak: {metrics['longest_loss_streak']}")
        
        print(f"\n⚙️  BET CHARACTERISTICS")
        print("-"*40)
        print(f"  Avg Odds:          {metrics['avg_odds']:.0f}")
        print(f"  Avg Edge:          {metrics['avg_edge']*100:.2f}%")
        
        print("\n" + "="*60 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("🏀 NBA Prop Analysis & Backtesting System")
    print("="*50)
    
    # Generate sample data
    print("\n[1/5] Generating player game logs...")
    game_logs = generate_sample_dataset()
    print(f"  ✓ Generated {len(game_logs):,} game logs for {game_logs['player'].nunique()} players")
    
    # Generate prop lines
    print("\n[2/5] Generating historical prop lines...")
    props = generate_prop_lines(game_logs)
    print(f"  ✓ Generated {len(props):,} prop betting opportunities")
    
    # Initialize models
    print("\n[3/5] Initializing prediction models...")
    models = {
        'Weighted Average': WeightedAverageModel(),
        'Median': MedianModel(),
        'Ensemble': EnsembleModel(),
    }
    print(f"  ✓ Loaded {len(models)} models")
    
    # Run backtests
    print("\n[4/5] Running backtests...")
    all_results = {}
    
    for name, model in models.items():
        print(f"\n  Testing: {name}")
        backtester = Backtester(initial_bankroll=1000, unit_size=10)
        results = backtester.run_backtest(
            props, game_logs, model,
            min_edge=0.03,
            min_confidence=0.35
        )
        all_results[name] = {
            'backtester': backtester,
            'results': results,
            'metrics': backtester.get_metrics()
        }
        
        m = all_results[name]['metrics']
        print(f"    Bets: {m['total_bets']} | Win Rate: {m['win_rate']*100:.1f}% | ROI: {m['roi']*100:+.2f}%")
    
    # Detailed report for best model
    print("\n[5/5] Generating detailed report for best model...")
    
    best_model = max(all_results.keys(), 
                     key=lambda x: all_results[x]['metrics'].get('roi', -999))
    
    print(f"\n  🏆 Best Performing Model: {best_model}")
    all_results[best_model]['backtester'].print_report()
    
    # Model comparison
    print("\n" + "="*60)
    print("              MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Bets':>8} {'Win%':>8} {'ROI':>10} {'Profit':>12}")
    print("-"*60)
    
    for name in models.keys():
        m = all_results[name]['metrics']
        print(f"{name:<20} {m['total_bets']:>8} {m['win_rate']*100:>7.1f}% {m['roi']*100:>9.2f}% ${m['total_profit']:>10.2f}")
    
    print("\n")
    
    return game_logs, props, all_results


if __name__ == "__main__":
    game_logs, props, results = main()
