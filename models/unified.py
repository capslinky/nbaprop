"""
UnifiedPropModel - Production-grade NBA prop prediction model.

Combines all contextual factors into a single, consistent analysis with 12 adjustments.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from core.config import CONFIG
from core.constants import normalize_team_abbrev
from core.exceptions import InvalidPropTypeError
from core.rosters import get_player_team
from models.prop_analysis import PropAnalysis

logger = logging.getLogger(__name__)


class UnifiedPropModel:
    """
    Production-grade NBA prop prediction model.
    Combines all contextual factors into a single, consistent analysis.

    Features:
    - Automatic context detection (opponent, home/away, B2B)
    - 12 multiplicative adjustments (defense, pace, injuries, etc.)
    - Multi-factor confidence scoring
    - Cached context data for performance

    Usage:
        model = UnifiedPropModel()
        analysis = model.analyze("Luka Doncic", "points", 32.5)
        print(f"Pick: {analysis.pick} (Edge: {analysis.edge:.1%})")
    """

    # Default adjustment constants (can be overridden by learned weights)
    _DEFAULT_HOME_BOOST = 1.025
    _DEFAULT_AWAY_PENALTY = 0.975
    _DEFAULT_B2B_PENALTY = 0.92
    _DEFAULT_BLOWOUT_HIGH_PENALTY = 0.95
    _DEFAULT_BLOWOUT_MED_PENALTY = 0.98
    _DEFAULT_LEAGUE_AVG_TOTAL = 225.0
    _DEFAULT_TOTAL_WEIGHT = 0.3  # How much game total affects projection

    # Matchup thresholds (not calibrated)
    SMASH_RANK = 5
    GOOD_RANK = 10
    HARD_RANK = 21
    TOUGH_RANK = 26

    def __init__(self, data_fetcher=None, injury_tracker=None, odds_client=None,
                 use_learned_weights: bool = True):
        """
        Initialize with optional data sources.
        If not provided, will create default instances.

        Args:
            data_fetcher: Optional NBADataFetcher instance
            injury_tracker: Optional InjuryTracker instance
            odds_client: Optional OddsAPIClient instance
            use_learned_weights: Whether to load and use learned calibration weights.
                                 Set to False to use CONFIG defaults only.
        """
        # Lazy imports to avoid circular dependencies
        self._fetcher = data_fetcher
        self._injuries = injury_tracker
        self._odds = odds_client

        # Learned weights store (for calibrated factors)
        self._weights_store = None
        self._use_learned_weights = use_learned_weights
        if use_learned_weights:
            self._weights_store = self._load_learned_weights()

        # Cached context data with timestamps and type-specific TTLs from CONFIG
        self._defense_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.DEFENSE_CACHE_TTL
        }
        self._pace_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.PACE_CACHE_TTL
        }
        self._game_lines_cache = {
            'data': None, 'timestamp': None, 'ttl': CONFIG.GAME_LOG_CACHE_TTL
        }

    def _load_learned_weights(self):
        """Load learned weights if available."""
        try:
            from calibration.weight_store import LearnedWeightsStore
            store = LearnedWeightsStore()
            if store.load() and store.is_valid():
                return store
        except ImportError:
            pass  # Calibration module not available
        except Exception:
            pass  # Any other error, fall back to defaults
        return None

    @property
    def HOME_BOOST(self) -> float:
        """Home court advantage factor."""
        default = CONFIG.HOME_BOOST
        if self._weights_store:
            return self._weights_store.get_factor('HOME_BOOST', default)
        return default

    @property
    def AWAY_PENALTY(self) -> float:
        """Road game penalty factor."""
        default = CONFIG.AWAY_PENALTY
        if self._weights_store:
            return self._weights_store.get_factor('AWAY_PENALTY', default)
        return default

    @property
    def B2B_PENALTY(self) -> float:
        """Back-to-back game penalty factor."""
        default = CONFIG.B2B_PENALTY
        if self._weights_store:
            return self._weights_store.get_factor('B2B_PENALTY', default)
        return default

    @property
    def BLOWOUT_HIGH_PENALTY(self) -> float:
        """High blowout risk penalty factor (spread >= 12)."""
        default = CONFIG.BLOWOUT_HIGH_PENALTY
        if self._weights_store:
            return self._weights_store.get_factor('BLOWOUT_HIGH_PENALTY', default)
        return default

    @property
    def BLOWOUT_MED_PENALTY(self) -> float:
        """Medium blowout risk penalty factor (spread 8-11)."""
        default = CONFIG.BLOWOUT_MEDIUM_PENALTY
        if self._weights_store:
            return self._weights_store.get_factor('BLOWOUT_MEDIUM_PENALTY', default)
        return default

    @property
    def LEAGUE_AVG_TOTAL(self) -> float:
        """League average game total for scaling."""
        return CONFIG.LEAGUE_AVG_TOTAL

    @property
    def TOTAL_WEIGHT(self) -> float:
        """Weight for game total adjustment on projections."""
        default = CONFIG.TOTAL_WEIGHT
        if self._weights_store:
            return self._weights_store.get_factor('TOTAL_WEIGHT', default)
        return default

    # =========================================================================
    # NEW FACTOR PROPERTIES: Rest Days, Usage Rate, Shot Volume, TS%
    # =========================================================================

    @property
    def REST_DAY_FACTORS(self) -> dict:
        """Rest day adjustment factors (gradient instead of binary B2B)."""
        return CONFIG.REST_DAY_FACTORS

    @property
    def LEAGUE_AVG_USG(self) -> float:
        """League average usage rate percentage."""
        return CONFIG.LEAGUE_AVG_USG

    @property
    def USAGE_FACTOR_WEIGHT(self) -> float:
        """Weight for usage rate adjustment."""
        default = CONFIG.USAGE_FACTOR_WEIGHT
        if self._weights_store:
            return self._weights_store.get_factor('USAGE_FACTOR_WEIGHT', default)
        return default

    @property
    def MAX_USAGE_ADJUSTMENT(self) -> float:
        """Maximum usage rate adjustment cap (+/- %)."""
        default = CONFIG.MAX_USAGE_ADJUSTMENT
        if self._weights_store:
            return self._weights_store.get_factor('MAX_USAGE_ADJUSTMENT', default)
        return default

    @property
    def fetcher(self) -> 'NBADataFetcher':
        """Lazy load data fetcher."""
        if self._fetcher is None:
            from data import NBADataFetcher
            self._fetcher = NBADataFetcher()
        return self._fetcher

    @property
    def injuries(self) -> 'InjuryTracker':
        """Lazy load injury tracker."""
        if self._injuries is None:
            from data import InjuryTracker
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
        ttl = cache.get('ttl', CONFIG.DEFENSE_CACHE_TTL)  # Default to defense TTL
        age = (datetime.now() - cache['timestamp']).total_seconds()
        return age < ttl

    def _get_defense_data(self) -> pd.DataFrame:
        """Get team defense data with caching."""
        if not self._is_cache_valid(self._defense_cache):
            self._defense_cache['data'] = self.fetcher.get_team_defense_vs_position()
            self._defense_cache['timestamp'] = datetime.now()
        return self._defense_cache['data']

    def _get_pace_data(self) -> pd.DataFrame:
        """Get team pace data with caching."""
        if not self._is_cache_valid(self._pace_cache):
            self._pace_cache['data'] = self.fetcher.get_team_pace()
            self._pace_cache['timestamp'] = datetime.now()
        return self._pace_cache['data']

    def _get_game_lines(self) -> pd.DataFrame:
        """Get game lines with caching (requires odds client)."""
        if self.odds is None:
            return pd.DataFrame()
        if not self._is_cache_valid(self._game_lines_cache):
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

        # Detect B2B from game dates and get rest days
        b2b_info = self.fetcher.check_back_to_back(logs)
        context['is_b2b'] = b2b_info.get('is_b2b', False)
        context['rest_days'] = b2b_info.get('rest_days')

        return context

    def _calculate_base_projection(self, history: pd.Series) -> tuple:
        """
        Calculate base projection using weighted recent/older split.
        Returns (projection, trend, recent_avg, older_avg).

        NOTE: history is sorted newest-first (index 0 = most recent game).
        Use .head() for recent games and .iloc[5:15] for older games.
        """
        if len(history) < 5:
            return history.mean(), 'NEUTRAL', history.mean(), history.mean()

        # Split into recent (first 5 = most recent) and older (6-15)
        # Logs are sorted newest-first, so .head(5) = last 5 games played
        recent = history.head(5)
        older = history.iloc[5:15] if len(history) >= 10 else history.iloc[5:]

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
        player_name: str,
        is_home: bool,
        is_b2b: bool,
        game_total: float,
        blowout_risk: str,
        minutes_factor: float,
        teammate_boost: float,
        defense_data: pd.DataFrame,
        pace_data: pd.DataFrame,
        # New parameters for enhanced factors
        rest_days: int = None,
        game_logs: pd.DataFrame = None,
    ) -> tuple:
        """
        Calculate all 12 adjustment factors.
        Returns (adjustments_dict, total_multiplier, flags, matchup_rating, opp_rank).

        Factors:
        1-9: Original factors (defense, location, B2B/rest, pace, total, blowout, vs_team, minutes, injury)
        10: Usage rate (from NBA API advanced stats)
        11: Shot volume trend (FGA/minute trend)
        12: TS% efficiency regression
        """
        adjustments = {
            'opp_defense': 1.0,
            'location': 1.0,
            'rest': 1.0,  # Enhanced from b2b to full rest days gradient
            'pace': 1.0,
            'total': 1.0,
            'blowout': 1.0,
            'vs_team': 1.0,
            'minutes': 1.0,
            'injury_boost': 1.0,
            # New factors
            'usage_rate': 1.0,
            'shot_volume': 1.0,
            'ts_regression': 1.0,
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

        # 3. REST DAYS (Enhanced gradient instead of binary B2B)
        if rest_days is not None:
            # Use gradient rest day factors from config
            rest_factors = getattr(self, 'REST_DAY_FACTORS', {
                0: 0.88, 1: 0.92, 2: 1.00, 3: 1.02, 4: 1.03, 5: 1.02, 6: 1.01
            })
            # Clamp to 0-6 range, use 6 for anything higher
            clamped_days = min(6, max(0, rest_days))
            adjustments['rest'] = rest_factors.get(clamped_days, 1.0)

            if rest_days == 1:
                flags.append('B2B')
            elif rest_days >= 4:
                flags.append('WELL RESTED')
        elif is_b2b:
            # Fallback to legacy B2B if rest_days not provided
            adjustments['rest'] = self.B2B_PENALTY
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

        # 7. PLAYER VS TEAM HISTORY
        if player_name and opponent:
            vs_stats = self.fetcher.get_player_vs_team_stats(
                player_name, opponent, prop_type
            )
            if vs_stats and vs_stats.get('games_vs_team', 0) >= 3:
                vs_factor = vs_stats.get('vs_factor', 1.0)
                # Cap at +/-10%
                adjustments['vs_team'] = max(0.9, min(1.1, vs_factor))
                if vs_stats.get('dominates'):
                    flags.append(f'DOMINATES vs {opponent}')
                elif vs_stats.get('struggles'):
                    flags.append(f'STRUGGLES vs {opponent}')

        # 8. MINUTES TREND
        if minutes_factor and minutes_factor != 1.0:
            adjustments['minutes'] = max(0.9, min(1.1, minutes_factor))
            if minutes_factor >= 1.05:
                flags.append('MINS UP')
            elif minutes_factor <= 0.95:
                flags.append('MINS DOWN')

        # 9. TEAMMATE INJURY BOOST
        if teammate_boost and teammate_boost > 1.0:
            adjustments['injury_boost'] = min(1.15, teammate_boost)
            boost_pct = round((teammate_boost - 1) * 100)
            flags.append(f'INJ BOOST +{boost_pct}%')

        # =====================================================================
        # NEW FACTORS (10-12): Usage, Shot Volume, TS% Regression
        # =====================================================================

        # 10. USAGE RATE (from NBA API advanced stats)
        # Higher usage = more reliable projection for scoring props
        if player_name and prop_type in ['points', 'pra', 'threes', 'fg3m', 'pts_reb', 'pts_ast']:
            try:
                usage_stats = self.fetcher.get_player_usage(player_name)
                if usage_stats:
                    player_usg = usage_stats.get('usg_pct', 20.0)
                    league_avg = getattr(self, 'LEAGUE_AVG_USG', 20.0)
                    weight = getattr(self, 'USAGE_FACTOR_WEIGHT', 0.3)
                    max_adj = getattr(self, 'MAX_USAGE_ADJUSTMENT', 0.05)

                    # Higher usage = more reliable, slight boost
                    usg_deviation = (player_usg - league_avg) / 100
                    usg_factor = 1.0 + (usg_deviation * weight)
                    adjustments['usage_rate'] = max(1 - max_adj, min(1 + max_adj, usg_factor))

                    if player_usg >= 28:
                        flags.append('HIGH USG')
                    elif player_usg <= 15:
                        flags.append('LOW USG')
            except Exception as e:
                logger.debug(f"Could not fetch usage for {player_name}: {e}")

        # 11. SHOT VOLUME TREND (FGA/minute trend)
        # Catches role changes independent of minutes
        if game_logs is not None and not game_logs.empty and prop_type in ['points', 'pra', 'threes', 'fg3m']:
            try:
                vol_trend = self.fetcher.calculate_usage_trend(game_logs)
                if vol_trend:
                    adjustments['shot_volume'] = vol_trend.get('usage_factor', 1.0)
                    trend_dir = vol_trend.get('trend', 'STABLE')
                    if trend_dir == 'UP':
                        flags.append('VOL UP')
                    elif trend_dir == 'DOWN':
                        flags.append('VOL DOWN')
            except Exception as e:
                logger.debug(f"Could not calculate shot volume trend: {e}")

        # 12. TRUE SHOOTING EFFICIENCY REGRESSION
        # Regress extreme efficiency outliers toward mean
        if game_logs is not None and not game_logs.empty and prop_type in ['points', 'pra']:
            try:
                ts_info = self.fetcher.calculate_ts_efficiency(game_logs)
                if ts_info:
                    adjustments['ts_regression'] = ts_info.get('ts_factor', 1.0)
                    regression_type = ts_info.get('regression', 'NONE')
                    if regression_type == 'DOWN':
                        flags.append('TS REG DOWN')  # Shooting too hot, expect regression
                    elif regression_type == 'UP':
                        flags.append('TS REG UP')  # Shooting cold, expect bounce back
            except Exception as e:
                logger.debug(f"Could not calculate TS efficiency: {e}")

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
        vol_trend: str = None,  # 'UP', 'DOWN', or None
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
        # NOTE: history is sorted newest-first, use .head() for recent
        recent = history.head(5).mean()
        older = history.iloc[5:15].mean() if len(history) >= 10 else mean_val
        trend_magnitude = abs(recent - older) / older if older > 0 else 0
        trend_score = min(1, trend_magnitude * 5)

        # Weights calibrated from 12/19 analysis:
        # - Increased consistency weight (most predictive)
        # - Reduced trend weight (NEUTRAL outperforms HOT/COLD by 20pts)
        base_confidence = (
            consistency_score * 0.4 +  # Increased from 0.3
            sample_score * 0.2 +
            hit_clarity * 0.3 +
            trend_score * 0.1          # Reduced from 0.2
        )

        # Adjustments
        if matchup_rating == 'SMASH':
            base_confidence += 0.10
        elif matchup_rating == 'GOOD':
            base_confidence += 0.05
        elif matchup_rating == 'TOUGH':
            base_confidence -= 0.15  # Increased from 0.05 (TOUGH = 45.8% win rate)

        if is_b2b:
            base_confidence -= 0.05

        if blowout_risk == 'HIGH':
            base_confidence -= 0.05

        if teammate_boost and teammate_boost > 1.0:
            base_confidence += 0.05

        # Vol trend adjustment (from 12/19 calibration)
        # Vol down = 69.8% win rate, Vol up = 33% win rate
        if vol_trend == 'DOWN':
            base_confidence += 0.08  # Boost confidence for volume regression
        elif vol_trend == 'UP':
            base_confidence -= 0.10  # Penalize unsustainable high volume

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
        prop_type_lower = prop_type.lower()
        if prop_type_lower not in CONFIG.VALID_PROP_TYPES:
            raise InvalidPropTypeError(prop_type)
        prop_type = prop_type_lower  # Normalize to lowercase

        # Map prop type aliases to actual column names
        # 'threes' is a common alias for 'fg3m' (3-pointers made)
        PROP_TO_COLUMN = {
            'threes': 'fg3m',
            'three': 'fg3m',
            '3pm': 'fg3m',
            'steals_blocks': 'stl_blk',  # If this combo exists
        }
        column_name = PROP_TO_COLUMN.get(prop_type, prop_type)

        # Fetch player game logs
        logs = self.fetcher.get_player_game_logs(player_name, last_n_games=last_n_games)

        if logs.empty or column_name not in logs.columns:
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

        # Use the mapped column name to get the stat history
        history = logs[column_name]

        # Auto-detect context
        context = self._detect_context_from_logs(logs)
        player_team = context['player_team']

        # Validate/correct team using roster data (prevents misidentification)
        roster_team = get_player_team(player_name)
        if roster_team:
            if player_team and player_team.upper() != roster_team.upper():
                logger.warning(
                    f"Team mismatch for {player_name}: matchup={player_team}, "
                    f"roster={roster_team}. Using roster team."
                )
            player_team = roster_team

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

        # Early exit for OUT players - don't waste time analyzing
        if player_status in ['OUT', 'INJURED', 'SUSPENDED']:
            logger.info(f"Player {player_name} is {player_status} - returning PASS")
            return PropAnalysis(
                player=player_name,
                prop_type=prop_type,
                line=line,
                projection=0,
                base_projection=0,
                edge=0,
                confidence=0,
                pick='PASS',
                recent_avg=history.head(5).mean() if len(history) >= 5 else history.mean(),
                season_avg=history.mean(),
                over_rate=0,
                under_rate=0,
                std_dev=history.std(),
                games_analyzed=len(history),
                trend='NEUTRAL',
                flags=[f'PLAYER {player_status}'],
                player_status=player_status,
                context_quality=0,
                warnings=[f'Player is {player_status} - no pick possible'],
            )

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
            player_name=player_name,
            is_home=is_home,
            is_b2b=is_b2b,
            game_total=game_total,
            blowout_risk=blowout_risk,
            minutes_factor=minutes_factor,
            teammate_boost=teammate_boost,
            defense_data=defense_data,
            pace_data=pace_data,
            rest_days=context.get('rest_days'),
            game_logs=logs,
        )

        # Apply adjustments
        projection = base_projection * total_adj

        # Apply global projection bias (calibration correction)
        projection *= CONFIG.PROJECTION_BIAS

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
