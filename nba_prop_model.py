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
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS FROM CORE MODULE (consolidated utilities)
# =============================================================================

# Import from core module - single source of truth
from core.constants import (
    TEAM_ABBREVIATIONS,
    normalize_team_abbrev,
    get_current_nba_season,
)
from core.odds_utils import (
    american_to_decimal,
    american_to_implied_prob,
    calculate_ev,
)

# Backward compatibility alias
TEAM_ABBREV_MAP = TEAM_ABBREVIATIONS


# =============================================================================
# DATA GENERATION (Simulating historical NBA data for demonstration)
# =============================================================================

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

                factor_col = factor_map.get(prop_type)
                rank_col = rank_map.get(prop_type)
                if factor_col is None:
                    logger.warning(f"Unknown prop_type '{prop_type}' for defense factor, using pts_factor")
                    factor_col = 'pts_factor'
                    rank_col = 'pts_rank'

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
# Note: american_to_implied_prob, american_to_decimal, calculate_ev
# are now imported from core.odds_utils at the top of this file
# =============================================================================


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
# UNIFIED PROP MODEL - Production-Grade Analysis
# =============================================================================

@dataclass
class PropAnalysis:
    """
    Comprehensive analysis result from UnifiedPropModel.
    Contains projection, recommendations, and full context breakdown.
    Includes validation, warnings, and explain() for debugging.
    """
    # Core predictions
    player: str
    prop_type: str
    line: float
    projection: float
    base_projection: float
    edge: float  # As decimal (0.05 = 5%)
    confidence: float  # 0-1 scale
    pick: str  # 'OVER', 'UNDER', 'PASS'

    # Historical stats
    recent_avg: float  # Last 5 games
    season_avg: float  # All games analyzed
    over_rate: float  # % of games over line
    under_rate: float  # % of games under line
    std_dev: float
    games_analyzed: int
    trend: str  # 'HOT', 'COLD', 'NEUTRAL'

    # Context used
    opponent: Optional[str] = None
    is_home: Optional[bool] = None
    is_b2b: bool = False
    game_total: Optional[float] = None
    blowout_risk: Optional[str] = None
    matchup_rating: str = 'NEUTRAL'
    opp_rank: Optional[int] = None

    # Adjustment breakdown
    adjustments: dict = field(default_factory=dict)
    total_adjustment: float = 0.0
    flags: List[str] = field(default_factory=list)

    # Injury context
    player_status: str = 'HEALTHY'
    teammate_boost: float = 1.0
    stars_out: List[str] = field(default_factory=list)

    # Validation & Quality (NEW)
    context_quality: int = 0  # 0-100 score
    warnings: List[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)  # What data supported each adjustment

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'player': self.player,
            'prop_type': self.prop_type,
            'line': self.line,
            'projection': self.projection,
            'edge': round(self.edge * 100, 1),
            'confidence': round(self.confidence * 100, 0),
            'pick': self.pick,
            'recent_avg': self.recent_avg,
            'season_avg': self.season_avg,
            'over_rate': round(self.over_rate * 100, 0),
            'under_rate': round(self.under_rate * 100, 0),
            'trend': self.trend,
            'matchup': self.matchup_rating,
            'opp_rank': self.opp_rank,
            'is_home': self.is_home,
            'is_b2b': self.is_b2b,
            'flags': self.flags,
            'total_adjustment': round(self.total_adjustment * 100, 1),
            'context_quality': self.context_quality,
            'warnings': self.warnings,
        }

    def explain(self, return_string: bool = False) -> Optional[str]:
        """
        Print detailed breakdown of the analysis for debugging.
        Shows exactly why this pick was made and what context was/wasn't applied.
        """
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"PICK ANALYSIS: {self.player} {self.prop_type.upper()} {self.line}")
        lines.append("=" * 60)

        # Raw Stats
        lines.append("")
        lines.append(f"RAW STATS ({self.games_analyzed} games)")
        lines.append("-" * 30)
        lines.append(f"  L5 Average:   {self.recent_avg:.1f}")
        lines.append(f"  L15 Average:  {self.season_avg:.1f}")
        lines.append(f"  Std Dev:      {self.std_dev:.1f}")
        lines.append(f"  Over Rate:    {self.over_rate*100:.0f}% ({int(self.over_rate * self.games_analyzed)}/{self.games_analyzed})")
        lines.append(f"  Under Rate:   {self.under_rate*100:.0f}% ({int(self.under_rate * self.games_analyzed)}/{self.games_analyzed})")
        lines.append(f"  Trend:        {self.trend}")

        # Base Projection
        lines.append("")
        lines.append(f"BASE PROJECTION: {self.base_projection:.1f}")
        lines.append(f"  (60% x {self.recent_avg:.1f}) + (40% x {self.season_avg:.1f}) + trend adj")

        # Adjustments
        lines.append("")
        lines.append("ADJUSTMENTS APPLIED")
        lines.append("-" * 30)

        adj_names = {
            'opp_defense': 'Opponent Defense',
            'location': 'Location (H/A)',
            'b2b': 'Back-to-Back',
            'pace': 'Pace Factor',
            'total': 'Game Total',
            'blowout': 'Blowout Risk',
            'vs_team': 'vs Team History',
            'minutes': 'Minutes Trend',
            'injury_boost': 'Injury Boost',
        }

        active_adj = 0
        for key, name in adj_names.items():
            adj_val = self.adjustments.get(key, 0)
            evidence = self.evidence.get(key, '')

            if adj_val != 0:
                sign = '+' if adj_val > 0 else ''
                lines.append(f"  {name:20s} {sign}{adj_val:+.1f}%  {evidence}")
                active_adj += 1
            else:
                lines.append(f"  {name:20s}   0.0%  (not applied)")

        lines.append("")
        lines.append(f"TOTAL ADJUSTMENT: {self.total_adjustment*100:+.1f}%")
        lines.append(f"Adjustments Active: {active_adj}/9")

        # Final Projection
        lines.append("")
        lines.append(f"FINAL PROJECTION: {self.projection:.1f}")
        lines.append(f"  Base ({self.base_projection:.1f}) x Total Adj ({1 + self.total_adjustment:.3f})")

        # Edge Calculation
        lines.append("")
        lines.append("EDGE CALCULATION")
        lines.append("-" * 30)
        lines.append(f"  Projection: {self.projection:.1f}")
        lines.append(f"  Line:       {self.line}")
        lines.append(f"  Edge:       {self.edge*100:+.1f}%")

        # Confidence
        lines.append("")
        lines.append(f"CONFIDENCE: {self.confidence*100:.0f}%")

        # Context Quality
        lines.append("")
        lines.append(f"CONTEXT QUALITY: {self.context_quality}/100")
        quality_desc = (
            "EXCELLENT" if self.context_quality >= 80 else
            "GOOD" if self.context_quality >= 60 else
            "FAIR" if self.context_quality >= 40 else
            "POOR"
        )
        lines.append(f"  Rating: {quality_desc}")

        # Pick
        lines.append("")
        if self.pick == 'PASS':
            lines.append(f"PICK: PASS (edge {self.edge*100:+.1f}% below threshold)")
        else:
            lines.append(f"PICK: {self.pick} {self.line}")
            lines.append(f"  Edge: {self.edge*100:+.1f}%, Confidence: {self.confidence*100:.0f}%")

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            lines.append("-" * 30)
            for w in self.warnings:
                lines.append(f"  * {w}")
        else:
            lines.append("")
            lines.append("WARNINGS: None")

        # Flags
        if self.flags:
            lines.append("")
            lines.append(f"FLAGS: {', '.join(self.flags)}")

        lines.append("")
        lines.append("=" * 60)

        output = '\n'.join(lines)

        if return_string:
            return output
        else:
            print(output)
            return None

    def is_high_quality(self) -> bool:
        """Returns True if context quality is sufficient for a confident pick."""
        return self.context_quality >= 50 and len(self.warnings) <= 2


class UnifiedPropModel:
    """
    Production-grade NBA prop prediction model.
    Combines all contextual factors into a single, consistent analysis.

    Features:
    - Automatic context detection (opponent, home/away, B2B)
    - 9 multiplicative adjustments (defense, pace, injuries, etc.)
    - Multi-factor confidence scoring
    - Cached context data for performance

    Usage:
        model = UnifiedPropModel()
        analysis = model.analyze("Luka Doncic", "points", 32.5)
        print(f"Pick: {analysis.pick} (Edge: {analysis.edge:.1%})")
    """

    # Adjustment constants
    HOME_BOOST = 1.025
    AWAY_PENALTY = 0.975
    B2B_PENALTY = 0.92
    BLOWOUT_HIGH_PENALTY = 0.95
    BLOWOUT_MED_PENALTY = 0.98
    LEAGUE_AVG_TOTAL = 225.0
    TOTAL_WEIGHT = 0.3  # How much game total affects projection

    # Matchup thresholds
    SMASH_RANK = 5
    GOOD_RANK = 10
    HARD_RANK = 21
    TOUGH_RANK = 26

    def __init__(self, data_fetcher=None, injury_tracker=None, odds_client=None):
        """
        Initialize with optional data sources.
        If not provided, will create default instances.
        """
        # Lazy imports to avoid circular dependencies
        self._fetcher = data_fetcher
        self._injuries = injury_tracker
        self._odds = odds_client

        # Cached context data with timestamps and type-specific TTLs from CONFIG
        from core.config import CONFIG
        self._defense_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.DEFENSE_CACHE_TTL
        }
        self._pace_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.PACE_CACHE_TTL
        }
        self._game_lines_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.GAME_LOG_CACHE_TTL
        }

    @property
    def fetcher(self) -> 'NBADataFetcher':
        """Lazy load data fetcher."""
        if self._fetcher is None:
            from nba_integrations import NBADataFetcher
            self._fetcher = NBADataFetcher()
        return self._fetcher

    @property
    def injuries(self) -> 'InjuryTracker':
        """Lazy load injury tracker."""
        if self._injuries is None:
            from nba_integrations import InjuryTracker
            self._injuries = InjuryTracker()
        return self._injuries

    @property
    def odds(self) -> Optional['OddsAPIClient']:
        """Return odds client (may be None)."""
        return self._odds

    def _is_cache_valid(self, cache: Dict[str, Any]) -> bool:
        """Check if cached data is still valid using cache-specific TTL."""
        if cache['data'] is None or cache['timestamp'] is None:
            return False
        from datetime import datetime
        from core.config import CONFIG
        ttl = cache.get('ttl', CONFIG.DEFENSE_CACHE_TTL)  # Default to defense TTL
        age = (datetime.now() - cache['timestamp']).total_seconds()
        return age < ttl

    def _get_defense_data(self) -> pd.DataFrame:
        """Get team defense data with caching."""
        if not self._is_cache_valid(self._defense_cache):
            from datetime import datetime
            self._defense_cache['data'] = self.fetcher.get_team_defense_vs_position()
            self._defense_cache['timestamp'] = datetime.now()
        return self._defense_cache['data']

    def _get_pace_data(self) -> pd.DataFrame:
        """Get team pace data with caching."""
        if not self._is_cache_valid(self._pace_cache):
            from datetime import datetime
            self._pace_cache['data'] = self.fetcher.get_team_pace()
            self._pace_cache['timestamp'] = datetime.now()
        return self._pace_cache['data']

    def _get_game_lines(self) -> pd.DataFrame:
        """Get game lines with caching (requires odds client)."""
        if self.odds is None:
            return pd.DataFrame()
        if not self._is_cache_valid(self._game_lines_cache):
            from datetime import datetime
            self._game_lines_cache['data'] = self.odds.get_game_lines()
            self._game_lines_cache['timestamp'] = datetime.now()
        return self._game_lines_cache['data']

    def _detect_context_from_logs(self, logs: pd.DataFrame) -> dict:
        """
        Auto-detect game context from player's recent game logs.
        Returns opponent, home/away, player team, and B2B status.
        """
        context = {
            'player_team': None,
            'opponent': None,
            'is_home': None,
            'is_b2b': False,
        }

        if logs.empty:
            return context

        # Get player's team from most recent matchup
        if 'matchup' in logs.columns:
            recent_matchup = logs.iloc[0]['matchup']
            if 'vs.' in recent_matchup:
                context['player_team'] = recent_matchup.split(' vs.')[0].strip()
            elif '@' in recent_matchup:
                context['player_team'] = recent_matchup.split(' @')[0].strip()

        # Detect B2B from game dates
        b2b_info = self.fetcher.check_back_to_back(logs)
        context['is_b2b'] = b2b_info.get('is_b2b', False)

        return context

    def _calculate_base_projection(self, history: pd.Series) -> tuple:
        """
        Calculate base projection using weighted recent/older split.
        Returns (projection, trend, recent_avg, older_avg).
        """
        if len(history) < 5:
            return history.mean(), 'NEUTRAL', history.mean(), history.mean()

        # Split into recent (last 5) and older (6-15)
        recent = history.tail(5)
        older = history.tail(15).head(10) if len(history) >= 10 else history.head(len(history) - 5)

        recent_avg = recent.mean()
        older_avg = older.mean() if len(older) > 0 else recent_avg

        # Weighted projection: 60% recent, 40% older
        projection = (recent_avg * 0.6) + (older_avg * 0.4)

        # Trend adjustment (10% of trend magnitude)
        if older_avg > 0:
            trend_pct = (recent_avg - older_avg) / older_avg
            projection *= (1 + trend_pct * 0.1)
            trend = 'HOT' if trend_pct > 0.05 else 'COLD' if trend_pct < -0.05 else 'NEUTRAL'
        else:
            trend = 'NEUTRAL'

        return projection, trend, recent_avg, older_avg

    def _calculate_adjustments(
        self,
        prop_type: str,
        opponent: str,
        player_team: str,
        is_home: bool,
        is_b2b: bool,
        game_total: float,
        blowout_risk: str,
        minutes_factor: float,
        teammate_boost: float,
        defense_data: pd.DataFrame,
        pace_data: pd.DataFrame,
    ) -> tuple:
        """
        Calculate all 9 adjustment factors.
        Returns (adjustments_dict, total_multiplier, flags, matchup_rating, opp_rank).
        """
        adjustments = {
            'opp_defense': 1.0,
            'location': 1.0,
            'b2b': 1.0,
            'pace': 1.0,
            'total': 1.0,
            'blowout': 1.0,
            'vs_team': 1.0,  # Placeholder - would need additional data
            'minutes': 1.0,
            'injury_boost': 1.0,
        }
        flags = []
        matchup_rating = 'NEUTRAL'
        opp_rank = None

        # 1. OPPONENT DEFENSE
        if opponent and defense_data is not None and not defense_data.empty:
            opponent_normalized = normalize_team_abbrev(opponent)
            opp_row = defense_data[defense_data['team_abbrev'] == opponent_normalized]
            if not opp_row.empty:
                factor_map = {
                    'points': 'pts_factor',
                    'rebounds': 'reb_factor',
                    'assists': 'ast_factor',
                    'pra': 'pra_factor',
                    'threes': 'threes_factor',
                    'fg3m': 'threes_factor',
                }
                rank_map = {
                    'points': 'pts_rank',
                    'rebounds': 'reb_rank',
                    'assists': 'ast_rank',
                    'threes': 'threes_rank',
                    'fg3m': 'threes_rank',
                }
                factor_col = factor_map.get(prop_type)
                rank_col = rank_map.get(prop_type)
                if factor_col is None:
                    logger.warning(f"Unknown prop_type '{prop_type}' for defense factor, using pts_factor")
                    factor_col = 'pts_factor'
                    rank_col = 'pts_rank'

                if factor_col in opp_row.columns:
                    adjustments['opp_defense'] = float(opp_row[factor_col].values[0])
                if rank_col in opp_row.columns:
                    opp_rank = int(opp_row[rank_col].values[0])

                    if opp_rank <= self.SMASH_RANK:
                        matchup_rating = 'SMASH'
                        flags.append('SMASH SPOT')
                    elif opp_rank <= self.GOOD_RANK:
                        matchup_rating = 'GOOD'
                    elif opp_rank >= self.TOUGH_RANK:
                        matchup_rating = 'TOUGH'
                        flags.append('TOUGH DEF')
                    elif opp_rank >= self.HARD_RANK:
                        matchup_rating = 'HARD'

        # 2. HOME/AWAY
        if is_home is not None:
            adjustments['location'] = self.HOME_BOOST if is_home else self.AWAY_PENALTY

        # 3. BACK-TO-BACK
        if is_b2b:
            adjustments['b2b'] = self.B2B_PENALTY
            flags.append('B2B')

        # 4. PACE
        if pace_data is not None and not pace_data.empty and player_team and opponent:
            player_team_normalized = normalize_team_abbrev(player_team)
            opponent_normalized = normalize_team_abbrev(opponent)
            player_pace = pace_data[pace_data['team_abbrev'] == player_team_normalized]
            opp_pace = pace_data[pace_data['team_abbrev'] == opponent_normalized]

            if not player_pace.empty and not opp_pace.empty:
                combined_pace = (
                    float(player_pace['pace_factor'].values[0]) +
                    float(opp_pace['pace_factor'].values[0])
                ) / 2
                adjustments['pace'] = combined_pace

                if combined_pace >= 1.03:
                    flags.append('FAST PACE')
                elif combined_pace <= 0.97:
                    flags.append('SLOW PACE')

        # 5. GAME TOTAL (for scoring props)
        if game_total and prop_type in ['points', 'pra', 'threes', 'fg3m']:
            total_factor = game_total / self.LEAGUE_AVG_TOTAL
            adjustments['total'] = 1 + (total_factor - 1) * self.TOTAL_WEIGHT

            if game_total >= 235:
                flags.append('HIGH TOTAL')
            elif game_total <= 215:
                flags.append('LOW TOTAL')

        # 6. BLOWOUT RISK
        if blowout_risk == 'HIGH':
            adjustments['blowout'] = self.BLOWOUT_HIGH_PENALTY
            flags.append('BLOWOUT RISK')
        elif blowout_risk == 'MEDIUM':
            adjustments['blowout'] = self.BLOWOUT_MED_PENALTY

        # 7. MINUTES TREND
        if minutes_factor and minutes_factor != 1.0:
            adjustments['minutes'] = max(0.9, min(1.1, minutes_factor))
            if minutes_factor >= 1.05:
                flags.append('MINS UP')
            elif minutes_factor <= 0.95:
                flags.append('MINS DOWN')

        # 8. TEAMMATE INJURY BOOST
        if teammate_boost and teammate_boost > 1.0:
            adjustments['injury_boost'] = min(1.15, teammate_boost)
            boost_pct = round((teammate_boost - 1) * 100)
            flags.append(f'USAGE +{boost_pct}%')

        # Calculate total multiplier
        total = 1.0
        for adj in adjustments.values():
            total *= adj

        return adjustments, total, flags, matchup_rating, opp_rank

    def _calculate_confidence(
        self,
        history: pd.Series,
        line: float,
        trend: str,
        matchup_rating: str,
        is_b2b: bool,
        blowout_risk: str,
        teammate_boost: float,
    ) -> float:
        """
        Calculate multi-factor confidence score (0-1).
        """
        if len(history) < 5:
            return 0.2

        # Base factors
        mean_val = history.mean()
        std_val = history.std()
        cv = std_val / mean_val if mean_val > 0 else 1

        # 1. Consistency (30%) - lower CV = higher confidence
        consistency_score = max(0, 1 - cv)

        # 2. Sample size (20%) - max at 15 games
        sample_score = min(1, len(history) / 15)

        # 3. Hit rate clarity (30%) - how clear is the over/under edge
        over_rate = (history > line).mean()
        hit_clarity = abs(over_rate - 0.5) * 2

        # 4. Trend strength (20%)
        recent = history.tail(5).mean()
        older = history.tail(15).head(10).mean() if len(history) >= 10 else mean_val
        trend_magnitude = abs(recent - older) / older if older > 0 else 0
        trend_score = min(1, trend_magnitude * 5)

        base_confidence = (
            consistency_score * 0.3 +
            sample_score * 0.2 +
            hit_clarity * 0.3 +
            trend_score * 0.2
        )

        # Adjustments
        if matchup_rating == 'SMASH':
            base_confidence += 0.10
        elif matchup_rating == 'GOOD':
            base_confidence += 0.05
        elif matchup_rating == 'TOUGH':
            base_confidence -= 0.05

        if is_b2b:
            base_confidence -= 0.05

        if blowout_risk == 'HIGH':
            base_confidence -= 0.05

        if teammate_boost and teammate_boost > 1.0:
            base_confidence += 0.05

        return max(0.2, min(0.95, base_confidence))

    def analyze(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        odds: int = -110,
        # Optional overrides (auto-detected if not provided)
        opponent: str = None,
        is_home: bool = None,
        game_total: float = None,
        blowout_risk: str = None,
        last_n_games: int = 15,
    ) -> PropAnalysis:
        """
        Analyze a player prop with full contextual adjustments.

        Args:
            player_name: Full player name (e.g., "Luka Doncic")
            prop_type: 'points', 'rebounds', 'assists', 'pra', 'threes'
            line: Betting line (e.g., 32.5)
            odds: American odds (default -110)
            opponent: Optional opponent team abbreviation (auto-detected)
            is_home: Optional home/away indicator (auto-detected)
            game_total: Optional Vegas over/under total
            blowout_risk: Optional 'HIGH', 'MEDIUM', 'LOW'
            last_n_games: Games to analyze (default 15)

        Returns:
            PropAnalysis with projection, edge, confidence, and full context

        Raises:
            InvalidPropTypeError: If prop_type is not valid
        """
        # Validate prop_type early
        from core.config import CONFIG
        from core.exceptions import InvalidPropTypeError

        prop_type_lower = prop_type.lower()
        if prop_type_lower not in CONFIG.VALID_PROP_TYPES:
            raise InvalidPropTypeError(prop_type)
        prop_type = prop_type_lower  # Normalize to lowercase

        # Fetch player game logs
        logs = self.fetcher.get_player_game_logs(player_name, last_n_games=last_n_games)

        if logs.empty or prop_type not in logs.columns:
            # Return a "no data" analysis
            return PropAnalysis(
                player=player_name,
                prop_type=prop_type,
                line=line,
                projection=0,
                base_projection=0,
                edge=0,
                confidence=0,
                pick='PASS',
                recent_avg=0,
                season_avg=0,
                over_rate=0,
                under_rate=0,
                std_dev=0,
                games_analyzed=0,
                trend='NEUTRAL',
                flags=['NO DATA'],
            )

        history = logs[prop_type]

        # Auto-detect context
        context = self._detect_context_from_logs(logs)
        player_team = context['player_team']
        if opponent is None:
            opponent = context.get('opponent')
        if is_home is None:
            is_home = context.get('is_home')
        is_b2b = context['is_b2b']

        # Load cached context data
        defense_data = self._get_defense_data()
        pace_data = self._get_pace_data()

        # Get game lines if odds client available
        if game_total is None or blowout_risk is None:
            game_lines = self._get_game_lines()
            # Would need to match game - simplified for now

        # Get injury info
        player_status_info = self.injuries.get_player_status(player_name)
        player_status = player_status_info.get('status', 'HEALTHY')

        teammate_boost_info = self.injuries.get_teammate_boost(
            player_name, player_team or '', prop_type
        ) if player_team else {'boost_factor': 1.0, 'stars_out': []}
        teammate_boost = teammate_boost_info.get('boost_factor', 1.0)
        stars_out = teammate_boost_info.get('stars_out', [])

        # Get minutes trend
        mins_info = self.fetcher.get_player_minutes_trend(logs)
        minutes_factor = mins_info.get('minutes_factor', 1.0) if mins_info else 1.0

        # Calculate base projection
        base_projection, trend, recent_avg, older_avg = self._calculate_base_projection(history)

        # Calculate all adjustments
        adjustments, total_adj, flags, matchup_rating, opp_rank = self._calculate_adjustments(
            prop_type=prop_type,
            opponent=opponent,
            player_team=player_team,
            is_home=is_home,
            is_b2b=is_b2b,
            game_total=game_total,
            blowout_risk=blowout_risk,
            minutes_factor=minutes_factor,
            teammate_boost=teammate_boost,
            defense_data=defense_data,
            pace_data=pace_data,
        )

        # Apply adjustments
        projection = base_projection * total_adj

        # Calculate stats
        season_avg = history.mean()
        std_dev = history.std()
        over_rate = (history > line).mean()
        under_rate = (history < line).mean()

        # Calculate edge
        edge = (projection - line) / line if line > 0 else 0

        # Calculate confidence
        confidence = self._calculate_confidence(
            history=history,
            line=line,
            trend=trend,
            matchup_rating=matchup_rating,
            is_b2b=is_b2b,
            blowout_risk=blowout_risk,
            teammate_boost=teammate_boost,
        )

        # Determine pick
        min_edge = 0.03 + (1 - confidence) * 0.02  # 3-5% based on confidence
        if abs(edge) < min_edge:
            pick = 'PASS'
        elif edge > 0:
            pick = 'OVER'
        else:
            pick = 'UNDER'

        # Add player status flag if not healthy
        if player_status != 'HEALTHY':
            flags.insert(0, player_status)

        # =====================================================================
        # VALIDATION: Calculate context quality, warnings, and evidence
        # =====================================================================
        warnings_list = []
        evidence_dict = {}
        context_quality_score = 0

        # === EVIDENCE COLLECTION (what data supported each adjustment) ===

        # 1. Opponent Defense Evidence
        if opponent and adjustments['opp_defense'] != 1.0:
            evidence_dict['opp_defense'] = f"vs {opponent} (rank {opp_rank})"
            context_quality_score += 15
        elif opponent:
            evidence_dict['opp_defense'] = f"vs {opponent} (no defense data)"
        else:
            evidence_dict['opp_defense'] = ""
            warnings_list.append("Opponent unknown - defense adjustment not applied")

        # 2. Location Evidence
        if is_home is not None:
            evidence_dict['location'] = "HOME" if is_home else "AWAY"
            context_quality_score += 10
        else:
            evidence_dict['location'] = ""
            warnings_list.append("Home/away unknown - location adjustment not applied")

        # 3. B2B Evidence
        if is_b2b:
            evidence_dict['b2b'] = "TRUE (back-to-back detected)"
            context_quality_score += 10
        else:
            evidence_dict['b2b'] = "FALSE"
            context_quality_score += 5  # Knowing it's NOT B2B is still useful

        # 4. Pace Evidence
        if adjustments['pace'] != 1.0:
            evidence_dict['pace'] = f"combined pace factor {adjustments['pace']:.3f}"
            context_quality_score += 10
        elif player_team and opponent:
            evidence_dict['pace'] = "teams not found in pace data"
        else:
            evidence_dict['pace'] = ""
            warnings_list.append("Team info missing - pace adjustment not applied")

        # 5. Game Total Evidence
        if game_total:
            evidence_dict['total'] = f"O/U {game_total}"
            context_quality_score += 10
        else:
            evidence_dict['total'] = ""
            # Only warn for scoring props where total matters
            if prop_type in ['points', 'pra', 'threes']:
                warnings_list.append("Game total unknown - total adjustment not applied")

        # 6. Blowout Risk Evidence
        if blowout_risk:
            evidence_dict['blowout'] = f"{blowout_risk} risk"
            context_quality_score += 5
        else:
            evidence_dict['blowout'] = ""

        # 7. Minutes Trend Evidence
        if minutes_factor and minutes_factor != 1.0:
            evidence_dict['minutes'] = f"factor {minutes_factor:.2f}"
            context_quality_score += 10
        else:
            evidence_dict['minutes'] = "stable"
            context_quality_score += 5

        # 8. Injury Boost Evidence
        if teammate_boost > 1.0:
            if stars_out:
                evidence_dict['injury_boost'] = f"+{(teammate_boost-1)*100:.0f}% ({', '.join(stars_out[:2])} out)"
            else:
                evidence_dict['injury_boost'] = f"+{(teammate_boost-1)*100:.0f}%"
            context_quality_score += 10
        else:
            evidence_dict['injury_boost'] = "no boost"

        # === DATA QUALITY CHECKS ===

        # Sample size check
        if len(history) < 10:
            warnings_list.append(f"Only {len(history)} games analyzed (recommend 15+)")
            context_quality_score -= 10
        elif len(history) >= 15:
            context_quality_score += 10  # Full sample bonus

        # Data freshness - check if defense/pace data loaded
        if defense_data is None or defense_data.empty:
            warnings_list.append("Defense data not loaded - matchup analysis unavailable")
            context_quality_score -= 15
        else:
            context_quality_score += 5  # Defense data loaded bonus

        if pace_data is None or pace_data.empty:
            warnings_list.append("Pace data not loaded - pace analysis unavailable")
            context_quality_score -= 5

        # Count active adjustments (non-1.0 values indicate context was applied)
        active_adjustments = sum(1 for v in adjustments.values() if round(v, 3) != 1.0)
        if active_adjustments < 2:
            warnings_list.append(f"Only {active_adjustments}/9 adjustments applied - limited context")

        # Add bonus for active adjustments
        context_quality_score += active_adjustments * 3

        # Cap context quality at 0-100
        context_quality_score = max(0, min(100, context_quality_score))

        return PropAnalysis(
            player=player_name,
            prop_type=prop_type,
            line=line,
            projection=round(projection, 1),
            base_projection=round(base_projection, 1),
            edge=edge,
            confidence=confidence,
            pick=pick,
            recent_avg=round(recent_avg, 1),
            season_avg=round(season_avg, 1),
            over_rate=over_rate,
            under_rate=under_rate,
            std_dev=round(std_dev, 2),
            games_analyzed=len(history),
            trend=trend,
            opponent=opponent,
            is_home=is_home,
            is_b2b=is_b2b,
            game_total=game_total,
            blowout_risk=blowout_risk,
            matchup_rating=matchup_rating,
            opp_rank=opp_rank,
            adjustments={k: round((v - 1) * 100, 1) for k, v in adjustments.items()},
            total_adjustment=total_adj - 1,
            flags=flags,
            player_status=player_status,
            teammate_boost=teammate_boost,
            stars_out=stars_out,
            # Validation fields
            context_quality=context_quality_score,
            warnings=warnings_list,
            evidence=evidence_dict,
        )

    def analyze_batch(
        self,
        props: List[dict],
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Analyze multiple props efficiently.

        Args:
            props: List of dicts with keys: player, prop_type, line, odds (optional)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            DataFrame with all analysis results
        """
        import time

        results = []
        total = len(props)

        for i, prop in enumerate(props):
            if progress_callback:
                progress_callback(i, total)

            analysis = self.analyze(
                player_name=prop['player'],
                prop_type=prop['prop_type'],
                line=prop['line'],
                odds=prop.get('odds', -110),
                opponent=prop.get('opponent'),
                is_home=prop.get('is_home'),
                game_total=prop.get('game_total'),
                blowout_risk=prop.get('blowout_risk'),
            )

            results.append(analysis.to_dict())
            time.sleep(0.3)  # Rate limiting

        if progress_callback:
            progress_callback(total, total)

        return pd.DataFrame(results)


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
