"""
Ensemble prediction models for NBA prop analysis.

Contains advanced models: EnsembleModel, SmartModel.
"""

import numpy as np
import pandas as pd
from typing import List
import logging

from models.prediction import Prediction
from models.simple_models import WeightedAverageModel, MedianModel
from core.odds_utils import american_to_decimal, american_to_implied_prob
from core.constants import normalize_team_abbrev
from core.config import CONFIG

logger = logging.getLogger(__name__)


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
                flags.append('GTD')
            elif player_injury_status.upper() in ['QUESTIONABLE', 'DOUBTFUL']:
                flags.append('QUESTIONABLE')

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
                    flags.append('SMASH SPOT')
                elif opp_rank <= 10:
                    matchup_rating = 'GOOD'
                elif opp_rank >= 26:
                    matchup_rating = 'TOUGH'
                    flags.append('TOUGH DEF')
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
            flags.append('B2B')

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
                    flags.append('FAST PACE')
                elif combined_pace <= 0.97:
                    flags.append('SLOW PACE')

        # 5. VEGAS TOTAL ADJUSTMENT (for scoring props)
        if game_total and prop_type in ['points', 'pra', 'threes']:
            # League avg total ~225
            total_factor = game_total / 225
            # Apply partial adjustment (don't over-weight)
            adjustments['total'] = 1 + (total_factor - 1) * 0.3

            if game_total >= 235:
                flags.append('HIGH TOTAL')
            elif game_total <= 215:
                flags.append('LOW TOTAL')

        # 6. BLOWOUT RISK (reduces minutes for starters)
        if blowout_risk == 'HIGH':
            adjustments['blowout'] = 0.95  # -5% for blowout risk
            flags.append('BLOWOUT RISK')
        elif blowout_risk == 'MEDIUM':
            adjustments['blowout'] = 0.98

        # 7. PLAYER VS TEAM HISTORY
        if vs_team_factor and vs_team_factor != 1.0:
            # Cap the adjustment at +/- 10%
            adjustments['vs_team'] = max(0.9, min(1.1, vs_team_factor))
            if vs_team_factor >= 1.1:
                flags.append('OWNS THIS TEAM')
            elif vs_team_factor <= 0.9:
                flags.append('STRUGGLES VS')

        # 8. MINUTES TREND
        if minutes_factor and minutes_factor != 1.0:
            adjustments['minutes'] = max(0.9, min(1.1, minutes_factor))
            if minutes_factor >= 1.05:
                flags.append('MINS UP')
            elif minutes_factor <= 0.95:
                flags.append('MINS DOWN')

        # 9. TEAMMATE INJURY BOOST (when stars are out, boost usage)
        if teammate_injury_boost and teammate_injury_boost > 1.0:
            adjustments['injury_boost'] = min(1.15, teammate_injury_boost)  # Cap at 15%
            boost_pct = round((teammate_injury_boost - 1) * 100)
            if stars_out:
                flags.append(f"+{boost_pct}% ({', '.join(stars_out[:2])} OUT)")
            else:
                flags.append(f"USAGE BOOST +{boost_pct}%")

        # CALCULATE FINAL ADJUSTED PROJECTION
        total_adjustment = 1.0
        for adj in adjustments.values():
            total_adjustment *= adj

        adjusted_projection = base_projection * total_adjustment

        # Apply global projection bias (calibration correction)
        # PROJECTION_BIAS > 1.0 boosts projections to correct systematic under-projection
        adjusted_projection *= CONFIG.PROJECTION_BIAS

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
