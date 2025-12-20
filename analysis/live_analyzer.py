"""
LivePropAnalyzer - Combines NBA stats and betting odds for live prop analysis.

Uses UnifiedPropModel for consistent, context-aware predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import logging

from core.constants import normalize_team_abbrev
from core.news_intelligence import NewsIntelligence
from models import UnifiedPropModel
from data import NBADataFetcher, OddsAPIClient, InjuryTracker

# Set up logging
logger = logging.getLogger(__name__)


class LivePropAnalyzer:
    """
    Combines NBA stats and betting odds for live prop analysis.
    Uses UnifiedPropModel for consistent, context-aware predictions.
    """

    def __init__(self, nba_fetcher: NBADataFetcher = None,
                 odds_client: OddsAPIClient = None,
                 injury_tracker: InjuryTracker = None,
                 news_search_fn: callable = None):
        self.nba = nba_fetcher or NBADataFetcher()
        self.odds = odds_client
        self.injuries = injury_tracker or InjuryTracker()

        # Initialize unified model with shared resources
        self.model = UnifiedPropModel(
            data_fetcher=self.nba,
            injury_tracker=self.injuries,
            odds_client=self.odds
        )

        # Initialize news intelligence for real-time context
        self.news_intel = NewsIntelligence(search_fn=news_search_fn)
        self._news_cache = {}  # {game_key: {player: NewsContext}}

        # Track players confirmed OUT (refreshed before analysis)
        self._players_out = set()
        self._injury_report = None

    def refresh_injuries(self, force: bool = True) -> dict:
        """
        Refresh injury data from all sources before running analysis.

        Args:
            force: Force refresh even if cache is valid (default True)

        Returns:
            dict with keys:
                - players_out: set of player names confirmed OUT
                - players_gtd: set of player names with GTD status
                - players_questionable: set of player names who are QUESTIONABLE
                - injury_df: Full injury DataFrame
                - summary: Human-readable summary string
        """
        logger.info("=" * 60)
        logger.info("INJURY REFRESH: Fetching fresh injury data...")

        # Force refresh from all sources
        injury_df = self.injuries.get_all_injuries(force_refresh=force)

        players_out = set()
        players_gtd = set()
        players_questionable = set()
        team_injuries = {}  # team -> list of injured players

        if not injury_df.empty:
            for _, row in injury_df.iterrows():
                player = row['player'] if 'player' in row.index else ''
                status = str(row['status']).upper() if 'status' in row.index else ''
                team = row['team'] if 'team' in row.index else 'UNK'

                if status in ['OUT', 'O', 'DNP']:
                    players_out.add(player)
                    if team not in team_injuries:
                        team_injuries[team] = {'out': [], 'gtd': [], 'questionable': []}
                    team_injuries[team]['out'].append(player)
                elif status in ['GTD', 'GAME TIME DECISION', 'GAME-TIME DECISION']:
                    players_gtd.add(player)
                    if team not in team_injuries:
                        team_injuries[team] = {'out': [], 'gtd': [], 'questionable': []}
                    team_injuries[team]['gtd'].append(player)
                elif status in ['QUESTIONABLE', 'Q', 'DOUBTFUL', 'D']:
                    players_questionable.add(player)
                    if team not in team_injuries:
                        team_injuries[team] = {'out': [], 'gtd': [], 'questionable': []}
                    team_injuries[team]['questionable'].append(player)

        # Log summary
        summary_lines = [
            f"Total injuries: {len(injury_df)}",
            f"OUT: {len(players_out)} players",
            f"GTD: {len(players_gtd)} players",
            f"Questionable: {len(players_questionable)} players",
        ]

        logger.info(f"  {summary_lines[0]}")
        logger.info(f"  {summary_lines[1]}")
        logger.info(f"  {summary_lines[2]}")
        logger.info(f"  {summary_lines[3]}")

        # Log key players OUT by team
        if players_out:
            logger.info("  Players OUT:")
            for team, injuries in sorted(team_injuries.items()):
                if injuries['out']:
                    logger.info(f"    {team}: {', '.join(injuries['out'])}")

        # Store for use in analysis
        self._players_out = players_out
        self._injury_report = {
            'players_out': players_out,
            'players_gtd': players_gtd,
            'players_questionable': players_questionable,
            'team_injuries': team_injuries,
            'injury_df': injury_df,
            'summary': '\n'.join(summary_lines),
        }

        logger.info("=" * 60)
        return self._injury_report

    def is_player_out(self, player_name: str) -> bool:
        """Check if a player is confirmed OUT."""
        # Check exact match first
        if player_name in self._players_out:
            return True
        # Check partial match (for name variations)
        player_lower = player_name.lower()
        for out_player in self._players_out:
            if player_lower in out_player.lower() or out_player.lower() in player_lower:
                return True
        return False

    def analyze_prop(self, player_name: str, prop_type: str,
                     line: float, odds: int = -110,
                     last_n_games: int = 15,
                     opponent: str = None,
                     is_home: bool = None,
                     game_total: float = None,
                     blowout_risk: str = None) -> dict:
        """
        Analyze a single prop bet using UnifiedPropModel with full context.

        Args:
            player_name: Player's full name
            prop_type: 'points', 'rebounds', 'assists', 'pra', 'threes'
            line: The betting line (e.g., 24.5)
            odds: American odds (default -110)
            last_n_games: Games to analyze
            opponent: Optional opponent team abbreviation
            is_home: Optional home/away indicator
            game_total: Optional Vegas over/under total
            blowout_risk: Optional 'HIGH', 'MEDIUM', 'LOW'

        Returns:
            Analysis dict with projection, edge, recommendation, and full context
        """
        # Use UnifiedPropModel for analysis
        analysis = self.model.analyze(
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            odds=odds,
            opponent=opponent,
            is_home=is_home,
            game_total=game_total,
            blowout_risk=blowout_risk,
            last_n_games=last_n_games
        )

        # Check for no data
        if analysis.games_analyzed == 0:
            return {'error': f'Could not fetch data for {player_name}'}

        # Convert PropAnalysis to dict format (backwards compatible)
        return {
            'player': analysis.player,
            'prop_type': analysis.prop_type,
            'line': analysis.line,
            'odds': odds,
            'sample_size': analysis.games_analyzed,
            'recent_avg': analysis.recent_avg,
            'recent_median': analysis.season_avg,  # Using season_avg as median proxy
            'recent_std': analysis.std_dev,
            'hit_rate_over': round(analysis.over_rate * 100, 1),
            'hit_rate_under': round(analysis.under_rate * 100, 1),
            'last_5_trend': '↑' if analysis.trend == 'HOT' else '↓' if analysis.trend == 'COLD' else '→',
            'last_5_avg': analysis.recent_avg,
            'projection': analysis.projection,
            'base_projection': analysis.base_projection,
            'recommended_side': analysis.pick.lower(),
            'avg_edge': round(analysis.edge * 100, 2),
            'confidence': round(analysis.confidence, 2),
            # New context fields
            'opponent': analysis.opponent,
            'is_home': analysis.is_home,
            'is_b2b': analysis.is_b2b,
            'matchup': analysis.matchup_rating,
            'opp_rank': analysis.opp_rank,
            'game_total': analysis.game_total,
            'blowout_risk': analysis.blowout_risk,
            'trend': analysis.trend,
            'flags': analysis.flags,
            'adjustments': analysis.adjustments,
            'total_adjustment': round(analysis.total_adjustment * 100, 1),
            'player_status': analysis.player_status,
            'teammate_boost': analysis.teammate_boost,
            'stars_out': analysis.stars_out,
        }

    def analyze_multiple_props(self, props: List[dict]) -> pd.DataFrame:
        """
        Analyze multiple props at once using UnifiedPropModel.

        Args:
            props: List of dicts with keys: player, prop_type, line, odds
        """
        results = []

        for prop in props:
            logger.info(f"  Analyzing: {prop['player']} {prop['prop_type']} {prop['line']}...")

            analysis = self.analyze_prop(
                player_name=prop['player'],
                prop_type=prop['prop_type'],
                line=prop['line'],
                odds=prop.get('odds', -110),
                opponent=prop.get('opponent'),
                is_home=prop.get('is_home'),
                game_total=prop.get('game_total'),
                blowout_risk=prop.get('blowout_risk'),
            )

            if 'error' not in analysis:
                results.append({
                    'Player': analysis['player'],
                    'Prop': analysis['prop_type'],
                    'Line': analysis['line'],
                    'Proj': analysis['projection'],
                    'Avg': analysis['recent_avg'],
                    'Over%': analysis['hit_rate_over'],
                    'Under%': analysis['hit_rate_under'],
                    'Edge': analysis['avg_edge'],
                    'Conf': int(analysis['confidence'] * 100),
                    'Pick': analysis['recommended_side'].upper(),
                    'Trend': analysis['trend'],
                    'Matchup': analysis['matchup'],
                    'Flags': ', '.join(analysis['flags']) if analysis['flags'] else '',
                })

            time.sleep(0.5)  # Rate limiting

        return pd.DataFrame(results)

    def find_value_props(self, min_edge: float = 0.05, max_events: int = 5,
                         min_confidence: float = 0.4, bookmakers: list = None,
                         event_ids: list = None) -> pd.DataFrame:
        """
        Scan current odds for value props with FULL CONTEXTUAL ANALYSIS.

        Incorporates:
        - Opponent defense ratings
        - Home/away adjustments
        - Back-to-back detection
        - Pace factors
        - Minutes trends
        - Vig-adjusted edge calculation
        - Minimum sample sizes by prop type
        - Correlation filtering

        Args:
            min_edge: Minimum vig-adjusted edge threshold (default 5%)
            max_events: Max number of games to scan (saves API calls)
            min_confidence: Minimum confidence threshold (default 40%)
            bookmakers: List of bookmaker keys to include (e.g., ['fanduel']). None = all books.
            event_ids: Optional list of specific event IDs to analyze (overrides max_events)
        """
        if not self.odds:
            logger.warning("OddsAPIClient required for live odds scanning")
            return pd.DataFrame()

        # =================================================================
        # PHASE 0: INJURY REFRESH (CRITICAL - must run before analysis)
        # =================================================================
        injury_report = self.refresh_injuries(force=True)
        players_out = injury_report.get('players_out', set())

        if players_out:
            logger.info(f"Will exclude {len(players_out)} players confirmed OUT from analysis")

        # =================================================================
        # PHASE 1: DATA COLLECTION
        # =================================================================
        logger.info("=" * 60)
        logger.info("PHASE 1: Fetching market data...")

        raw_props = self.odds.get_all_player_props(max_events=max_events, event_ids=event_ids)
        props_df = self.odds.parse_player_props(raw_props)

        if props_df.empty:
            logger.info("No props available")
            return pd.DataFrame()

        # Filter to specific bookmakers if requested
        if bookmakers:
            props_df = props_df[props_df['bookmaker'].isin(bookmakers)]
            logger.info(f"Filtered to bookmakers: {', '.join(bookmakers)}")
            if props_df.empty:
                logger.info("No props from specified bookmakers")
                return pd.DataFrame()

        # Get game context
        game_lines = self.odds.get_game_lines()

        # Get best odds per prop
        best_odds = self.odds.get_best_odds(props_df)

        # FIX #1: Properly pair OVER and UNDER odds instead of using .first()
        # Split into over and under, then merge to get both odds in same row
        overs = best_odds[best_odds['side'] == 'over'].copy()
        unders = best_odds[best_odds['side'] == 'under'].copy()

        # Rename odds columns to distinguish over vs under
        overs = overs.rename(columns={'odds': 'over_odds'})
        unders = unders.rename(columns={'odds': 'under_odds'})

        # Merge to get both odds in same row
        unique_props = overs.merge(
            unders[['player', 'prop_type', 'line', 'under_odds', 'bookmaker']].rename(
                columns={'bookmaker': 'under_bookmaker'}
            ),
            on=['player', 'prop_type', 'line'],
            how='outer'
        )

        # Fill missing odds with standard -110
        unique_props['over_odds'] = unique_props['over_odds'].fillna(-110).astype(int)
        unique_props['under_odds'] = unique_props['under_odds'].fillna(-110).astype(int)

        logger.info(f"Found {len(unique_props)} unique props across {max_events} games")

        # =================================================================
        # PHASE 1.5: FILTER OUT INJURED PLAYERS
        # =================================================================
        if players_out:
            # Create a mask for players who are NOT out
            def player_is_out(player_name):
                player_lower = player_name.lower()
                for out_player in players_out:
                    if player_lower in out_player.lower() or out_player.lower() in player_lower:
                        return True
                return False

            before_count = len(unique_props)
            excluded_players = set()

            # Build mask of players to keep
            keep_mask = []
            for player in unique_props['player']:
                if player_is_out(player):
                    excluded_players.add(player)
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)

            unique_props = unique_props[keep_mask]

            excluded_count = before_count - len(unique_props)
            if excluded_count > 0:
                logger.info(f"  EXCLUDED {excluded_count} props for {len(excluded_players)} players confirmed OUT:")
                for player in sorted(excluded_players):
                    logger.info(f"    - {player}")

        # =================================================================
        # PHASE 2: CONTEXTUAL DATA LOADING
        # =================================================================
        logger.info("PHASE 2: Loading contextual data...")

        # Load defense ratings
        defense_ratings = self.nba.get_team_defense_ratings()
        if defense_ratings is not None:
            logger.info(f"  Defense ratings: {len(defense_ratings)} teams")
        else:
            logger.warning("  Defense ratings unavailable")
            defense_ratings = pd.DataFrame()

        # Load defense vs position data (has pts_rank, reb_rank, etc.)
        defense_vs_position = self.nba.get_team_defense_vs_position()
        if defense_vs_position is None:
            defense_vs_position = pd.DataFrame()

        # Load pace data
        pace_data = self.nba.get_team_pace()
        if pace_data is not None:
            logger.info(f"  Pace data: {len(pace_data)} teams")
        else:
            logger.warning("  Pace data unavailable")
            pace_data = pd.DataFrame()

        # =================================================================
        # PHASE 2.5: NEWS INTELLIGENCE (real-time injury/lineup context)
        # =================================================================
        logger.info("PHASE 2.5: Fetching news intelligence...")

        # Get unique games from props
        game_news_cache = {}
        if 'home_team' in unique_props.columns and 'away_team' in unique_props.columns:
            games = unique_props[['home_team', 'away_team']].drop_duplicates()

            for _, game in games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']
                if home_team and away_team:
                    game_key = f"{away_team}@{home_team}"
                    try:
                        # Fetch game-level news (injuries, lineup changes)
                        game_news = self.news_intel.fetch_game_news(
                            home_team=home_team,
                            away_team=away_team,
                            game_date=datetime.now().strftime('%Y-%m-%d')
                        )
                        game_news_cache[game_key] = game_news

                        # Log any breaking news found
                        for team, context in game_news.items():
                            if context.flags:
                                logger.info(f"  News {team}: {', '.join(context.flags)}")
                    except Exception as e:
                        logger.debug(f"News fetch failed for {game_key}: {e}")

            logger.info(f"  News scanned for {len(games)} games")
        else:
            logger.warning("  Game info not available for news scanning")

        # =================================================================
        # PHASE 3: PLAYER DATA FETCHING (with extended history)
        # =================================================================
        logger.info("PHASE 3: Fetching player data (30 games for statistical validity)...")

        unique_players = unique_props['player'].unique()
        player_cache = {}
        player_context = {}  # Store contextual info per player

        for i, player in enumerate(unique_players):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  Fetching {i+1}/{len(unique_players)}: {player}")
            try:
                # Fetch MORE games for better sample size
                logs = self.nba.get_player_game_logs(player, last_n_games=30)
                if not logs.empty and len(logs) >= 5:
                    player_cache[player] = logs

                    # Get contextual data for this player
                    context = {}

                    # Home/Away splits
                    if 'home' in logs.columns:
                        home_games = logs[logs['home'] == True]
                        away_games = logs[logs['home'] == False]
                        context['home_games'] = len(home_games)
                        context['away_games'] = len(away_games)

                    # Back-to-back check
                    b2b_info = self.nba.check_back_to_back(logs)
                    context['is_b2b'] = b2b_info.get('is_b2b', False)
                    context['rest_days'] = b2b_info.get('rest_days', 2)

                    # Minutes trend
                    mins_info = self.nba.get_player_minutes_trend(logs)
                    context['minutes_trend'] = mins_info.get('trend', 'stable')
                    context['minutes_factor'] = mins_info.get('minutes_factor', 1.0)
                    context['recent_minutes'] = mins_info.get('last_5_avg', 30)

                    # Get player's team and opponent from most recent game
                    if 'matchup' in logs.columns and len(logs) > 0:
                        last_matchup = logs.iloc[0]['matchup'] if not logs.empty else ''
                        context['last_matchup'] = last_matchup

                    player_context[player] = context

            except Exception as e:
                logger.debug(f"Failed to process player context: {e}")

        logger.info(f"  Cached {len(player_cache)} players with context")

        # =================================================================
        # PHASE 4: CONTEXTUAL PROP ANALYSIS
        # =================================================================
        logger.info("PHASE 4: Analyzing props with full context...")

        value_props = []
        skipped = {'sample_size': 0, 'no_edge': 0, 'low_confidence': 0, 'correlation': 0, 'injury_out': 0}

        # Map prop types to column names
        prop_to_column = {
            'points': 'points',
            'rebounds': 'rebounds',
            'assists': 'assists',
            'pra': 'pra',
            'threes': 'fg3m',
            'blocks': 'blocks',
            'steals': 'steals',
        }

        # MINIMUM SAMPLE SIZES by prop type (for statistical validity)
        min_samples = {
            'points': 10,      # Lower variance
            'rebounds': 12,    # Medium variance
            'assists': 15,     # Higher variance
            'pra': 10,         # Aggregated, lower variance
            'threes': 20,      # VERY high variance - need more data
            'blocks': 20,      # High variance
            'steals': 20,      # High variance
        }

        # Defense factor mapping
        defense_stat_map = {
            'points': 'pts_factor',
            'rebounds': 'reb_factor',
            'assists': 'ast_factor',
            'threes': 'threes_factor',
            'pra': 'pts_factor',  # Use points as proxy
        }

        # Track correlated picks to filter later
        player_picks = {}  # player -> list of props picked

        for i, row in unique_props.iterrows():
            player = row['player']
            if player not in player_cache:
                continue

            logs = player_cache[player]
            context = player_context.get(player, {})
            prop_type = row['prop_type']
            stat_column = prop_to_column.get(prop_type, prop_type)

            if stat_column not in logs.columns:
                logger.warning(f"Column '{stat_column}' not found for {player} - skipping {prop_type} prop")
                continue

            history = logs[stat_column]

            # MINIMUM SAMPLE SIZE CHECK
            min_required = min_samples.get(prop_type, 15)
            if len(history) < min_required:
                skipped['sample_size'] += 1
                continue

            # =============================================================
            # NEWS CONTEXT LOOKUP (10th adjustment factor)
            # =============================================================
            news_context = None
            news_factor = 1.0
            news_flags = []
            news_notes = []

            # Get game key for news lookup
            home_team = row.get('home_team', '')
            away_team = row.get('away_team', '')
            game_key = f"{away_team}@{home_team}" if home_team and away_team else None

            # Check game-level news cache first
            if game_key and game_key in game_news_cache:
                # Check if this player has team-level news
                player_team = context.get('team_abbrev', '')
                if player_team in game_news_cache[game_key]:
                    team_news = game_news_cache[game_key][player_team]
                    if team_news.flags:
                        news_flags.extend(team_news.flags)

            # Fetch player-specific news if preliminary edge looks promising
            line = row['line']
            prelim_avg = history.head(10).mean()
            preliminary_edge = abs((prelim_avg - line) / line) if line > 0 else 0

            if preliminary_edge > 0.05:  # Only fetch detailed news for promising picks
                try:
                    player_news = self.news_intel.fetch_player_news(
                        player_name=player,
                        team=context.get('team_abbrev'),
                        game_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    if player_news:
                        news_context = player_news
                        news_factor = player_news.adjustment_factor
                        news_flags = player_news.flags
                        news_notes = player_news.notes

                        # Skip if player confirmed out or on load management
                        if player_news.should_skip():
                            if 'news_out' not in skipped:
                                skipped['news_out'] = 0
                            skipped['news_out'] += 1
                            continue
                except Exception as e:
                    logger.debug(f"Player news fetch failed for {player}: {e}")

            try:
                # =============================================================
                # STEP 1: BASE PROJECTION (weighted average with mean reversion)
                # =============================================================
                recent_5 = history.head(5)  # Most recent 5
                recent_10 = history.head(10)
                season = history

                recent_avg = recent_5.mean()
                mid_avg = recent_10.mean()
                season_avg = season.mean()

                # Apply MEAN REVERSION - hot streaks regress, cold streaks recover
                # Weight: 40% recent, 35% mid-term, 25% season (regression to mean)
                base_projection = (recent_avg * 0.40) + (mid_avg * 0.35) + (season_avg * 0.25)

                # =============================================================
                # STEP 2: CONTEXTUAL ADJUSTMENTS
                # =============================================================
                adjustment_factors = []
                adjustment_notes = []

                # --- HOME/AWAY ADJUSTMENT ---
                if 'home' in logs.columns:
                    home_games = logs[logs['home'] == True]
                    away_games = logs[logs['home'] == False]

                    if len(home_games) >= 3 and len(away_games) >= 3:
                        home_avg = home_games[stat_column].mean()
                        away_avg = away_games[stat_column].mean()
                        overall = season_avg

                        # Determine if tonight is home or away (from matchup string)
                        # '@' in matchup means away game
                        is_home_tonight = True  # Default assumption
                        if 'event_info' in row and '@' in str(row.get('event_info', '')):
                            is_home_tonight = False

                        if is_home_tonight and home_avg > 0:
                            ha_factor = home_avg / overall if overall > 0 else 1.0
                            adjustment_factors.append(ha_factor)
                            if abs(ha_factor - 1.0) > 0.05:
                                adjustment_notes.append(f"Home: {ha_factor:.2f}x")
                        elif not is_home_tonight and away_avg > 0:
                            ha_factor = away_avg / overall if overall > 0 else 1.0
                            adjustment_factors.append(ha_factor)
                            if abs(ha_factor - 1.0) > 0.05:
                                adjustment_notes.append(f"Away: {ha_factor:.2f}x")

                # --- BACK-TO-BACK ADJUSTMENT ---
                if context.get('is_b2b', False):
                    b2b_factor = 0.93  # 7% reduction on back-to-backs
                    adjustment_factors.append(b2b_factor)
                    adjustment_notes.append("B2B: 0.93x")
                elif context.get('rest_days', 2) >= 3:
                    rest_factor = 1.03  # 3% boost with extra rest
                    adjustment_factors.append(rest_factor)
                    adjustment_notes.append("Rested: 1.03x")

                # --- MINUTES TREND ADJUSTMENT ---
                mins_factor = context.get('minutes_factor', 1.0)
                if mins_factor < 0.9 or mins_factor > 1.1:
                    # Only adjust if significant change
                    capped_factor = max(0.85, min(1.15, mins_factor))
                    adjustment_factors.append(capped_factor)
                    adjustment_notes.append(f"Mins: {capped_factor:.2f}x")

                # --- OPPONENT DEFENSE ADJUSTMENT ---
                # (would need opponent info from game data - placeholder)
                # For now, use neutral 1.0

                # --- USAGE/SHOT VOLUME TREND ADJUSTMENT (NEW) ---
                # Only apply to scoring-related props
                # NOTE: Vol↑ neutralized (33% win rate = noise), only Vol↓ applied (69.8% win rate = signal)
                scoring_props = ['points', 'pra', 'threes', 'pts_reb', 'pts_ast', 'field_goals_made', 'three_pointers_made']
                if prop_type in scoring_props and len(logs) >= 10:
                    usage_result = self.nba.calculate_usage_trend(logs)
                    if usage_result:
                        trend = usage_result.get('trend', 'STABLE')
                        usage_factor = usage_result.get('usage_factor', 1.0)
                        if trend == 'DOWN' and usage_factor < 1.0:
                            # Vol↓ is predictive (69.8% win rate) - apply the factor
                            adjustment_factors.append(usage_factor)
                            adjustment_notes.append(f"Vol↓: {usage_factor:.2f}x")
                        elif trend == 'UP':
                            # Vol↑ is noise (33% win rate) - log but DON'T apply factor
                            adjustment_notes.append(f"Vol↑: 1.00x (neutralized)")

                # --- TRUE SHOOTING MOMENTUM (BACKTEST VALIDATED) ---
                # Hot shooters stay hot (62.5% OVER), cold shooters stay cold (55.3% UNDER)
                if prop_type in scoring_props and len(logs) >= 10:
                    ts_result = self.nba.calculate_ts_efficiency(logs)
                    if ts_result:
                        ts_factor = ts_result.get('ts_factor', 1.0)
                        momentum = ts_result.get('regression', 'NONE')
                        if ts_factor != 1.0 and momentum in ('HOT', 'COLD'):
                            adjustment_factors.append(ts_factor)
                            recent_ts = ts_result.get('recent_ts_pct', 0)
                            adjustment_notes.append(f"{momentum}: {ts_factor:.2f}x ({recent_ts:.0f}% TS)")

                # --- NEWS INTELLIGENCE ADJUSTMENT (10th factor) ---
                if news_factor != 1.0:
                    adjustment_factors.append(news_factor)
                    news_adj_pct = (news_factor - 1) * 100
                    adjustment_notes.append(f"News: {news_factor:.2f}x ({news_adj_pct:+.0f}%)")

                # Add news flags to adjustment notes
                if news_flags:
                    adjustment_notes.extend(news_flags)

                # Apply all adjustments
                final_projection = base_projection
                for factor in adjustment_factors:
                    final_projection *= factor

                # =============================================================
                # STEP 3: EDGE CALCULATION WITH VIG ADJUSTMENT
                # FIX #1: Use correct odds based on side (over_odds vs under_odds)
                # FIX #2: Conservative probability estimation
                # =============================================================
                line = row['line']
                over_odds = row.get('over_odds', -110)
                under_odds = row.get('under_odds', -110)

                # Raw edge (projection vs line)
                raw_edge = (final_projection - line) / line if line > 0 else 0

                # Historical hit rates
                historical_over = (history > line).mean()
                historical_under = (history < line).mean()

                # FIX #1: Calculate no-vig true probabilities from BOTH sides
                def american_to_implied(odds):
                    if odds < 0:
                        return abs(odds) / (abs(odds) + 100)
                    return 100 / (odds + 100)

                over_implied = american_to_implied(over_odds)
                under_implied = american_to_implied(under_odds)

                # Remove vig by normalizing
                total_implied = over_implied + under_implied
                true_over_prob = over_implied / total_implied
                true_under_prob = under_implied / total_implied

                # Determine pick direction based on raw edge
                pick_over = raw_edge > 0

                # FIX #2: Conservative probability estimation
                # Use market probability as anchor, adjust based on:
                # 1. Historical hit rate deviation from market
                # 2. Projection deviation from line
                # Cap total adjustment at ±15% from market
                proj_vs_line = (final_projection - line) / line if line > 0 else 0

                if pick_over:
                    # Use OVER odds for implied probability
                    implied_prob = over_implied
                    breakeven = over_implied
                    market_prob = true_over_prob

                    # Historical edge over market
                    hist_edge = historical_over - market_prob

                    # Projection edge (capped)
                    proj_adjustment = min(0.10, max(-0.10, proj_vs_line * 0.3))

                    # Combined adjustment (capped at ±15%)
                    total_adjustment = min(0.15, max(-0.15, hist_edge * 0.5 + proj_adjustment))
                    our_prob = min(0.85, max(0.15, market_prob + total_adjustment))
                else:
                    # Use UNDER odds for implied probability
                    implied_prob = under_implied
                    breakeven = under_implied
                    market_prob = true_under_prob

                    # Historical edge over market
                    hist_edge = historical_under - market_prob

                    # Projection edge (capped)
                    proj_adjustment = min(0.10, max(-0.10, abs(proj_vs_line) * 0.3))

                    # Combined adjustment (capped at ±15%)
                    total_adjustment = min(0.15, max(-0.15, hist_edge * 0.5 + proj_adjustment))
                    our_prob = min(0.85, max(0.15, market_prob + total_adjustment))

                # VIG-ADJUSTED EDGE = our probability - breakeven probability
                vig_adjusted_edge = (our_prob - breakeven) * 100

                # =============================================================
                # STEP 4: CONFIDENCE SCORE (multi-factor)
                # =============================================================
                std = history.std()
                cv = std / season_avg if season_avg > 0 else 1

                # Base confidence from consistency
                consistency_score = max(0, min(1, 1 - cv))

                # Sample size factor (more games = more confidence)
                sample_factor = min(1.0, len(history) / 25)

                # Agreement factor (do historical and projection agree?)
                if raw_edge > 0:
                    agreement = historical_over
                else:
                    agreement = historical_under

                # Combined confidence
                confidence = (consistency_score * 0.4) + (sample_factor * 0.3) + (agreement * 0.3)
                confidence = max(0.2, min(0.85, confidence))  # Cap at 85%

                # NEWS-BASED CONFIDENCE ADJUSTMENT
                if news_context:
                    if news_context.status in ('GTD_LEANING_OUT', 'LIKELY_OUT'):
                        confidence *= 0.85  # -15% confidence for uncertain status
                    elif news_context.status in ('GTD_UNCERTAIN',):
                        confidence *= 0.90  # -10% confidence for unknown GTD
                    elif news_context.status in ('GTD_LEANING_PLAY',):
                        confidence *= 0.95  # -5% confidence even if likely playing
                    elif news_context.status == 'RETURNING':
                        confidence *= 0.90  # -10% confidence for returning players

                # =============================================================
                # STEP 5: PICK DETERMINATION
                # =============================================================
                if abs(vig_adjusted_edge) < 3:  # Need at least 3% edge after vig
                    skipped['no_edge'] += 1
                    continue

                if confidence < min_confidence:
                    skipped['low_confidence'] += 1
                    continue

                pick = 'OVER' if raw_edge > 0 else 'UNDER'

                # =============================================================
                # STEP 6: RECORD VALUE PROP
                # =============================================================
                trend = 'HOT' if recent_avg > season_avg * 1.05 else 'COLD' if recent_avg < season_avg * 0.95 else 'NEUTRAL'

                # FIX #4: Filter out nan values from adjustment_notes
                clean_adjustments = [
                    note for note in adjustment_notes
                    if note and 'nan' not in str(note).lower()
                ]

                # Store the odds for the recommended side
                pick_odds = over_odds if pick == 'OVER' else under_odds

                # Build game matchup string
                home_team = row.get('home_team', '')
                away_team = row.get('away_team', '')
                game_matchup = f"{away_team} @ {home_team}" if home_team and away_team else 'Unknown'

                # =============================================================
                # STEP 6B: CONTEXTUAL DATA FOR OUTPUT
                # =============================================================
                # Determine player's team and opponent from matchup info
                player_team = None
                opponent = None
                opp_defense_rank = None
                matchup_label = 'NEUTRAL'
                team_pace_rank = None
                opp_pace_rank = None

                # Extract player team from their most recent matchup
                if not logs.empty and 'matchup' in logs.columns:
                    last_matchup = logs.iloc[0]['matchup']
                    # Format: "DAL @ TOR" or "DAL vs. TOR" - first part is player's team
                    if ' @ ' in last_matchup:
                        player_team = normalize_team_abbrev(last_matchup.split(' @ ')[0].strip())
                    elif ' vs. ' in last_matchup:
                        player_team = normalize_team_abbrev(last_matchup.split(' vs. ')[0].strip())

                # Determine opponent for tonight's game
                if player_team and home_team and away_team:
                    home_norm = normalize_team_abbrev(home_team)
                    away_norm = normalize_team_abbrev(away_team)
                    if player_team == home_norm:
                        opponent = away_norm
                    elif player_team == away_norm:
                        opponent = home_norm

                # Look up opponent defense rank (use defense_vs_position which has ranks)
                if opponent and defense_vs_position is not None and not defense_vs_position.empty:
                    opp_row = defense_vs_position[defense_vs_position['team_abbrev'] == opponent]
                    if not opp_row.empty and 'pts_rank' in opp_row.columns:
                        opp_defense_rank = int(opp_row['pts_rank'].values[0])
                        # Determine matchup label based on rank (1=worst defense, 30=best)
                        if opp_defense_rank <= 5:
                            matchup_label = 'SMASH'
                        elif opp_defense_rank <= 10:
                            matchup_label = 'GOOD'
                        elif opp_defense_rank >= 26:
                            matchup_label = 'TOUGH'
                        elif opp_defense_rank >= 21:
                            matchup_label = 'HARD'

                # Look up pace ranks for both teams
                if pace_data is not None and not pace_data.empty:
                    if player_team:
                        team_row = pace_data[pace_data['team_abbrev'] == player_team]
                        if not team_row.empty and 'pace_rank' in team_row.columns:
                            team_pace_rank = int(team_row['pace_rank'].values[0])
                    if opponent:
                        opp_pace_row = pace_data[pace_data['team_abbrev'] == opponent]
                        if not opp_pace_row.empty and 'pace_rank' in opp_pace_row.columns:
                            opp_pace_rank = int(opp_pace_row['pace_rank'].values[0])

                value_props.append({
                    'player': player,
                    'prop_type': prop_type,
                    'line': line,
                    'projection': round(final_projection, 1),
                    'raw_edge': round(raw_edge * 100, 1),
                    'avg_edge': round(vig_adjusted_edge, 1),  # VIG-ADJUSTED PROBABILITY EDGE
                    'confidence': round(confidence * 100, 0),
                    'recommended_side': pick,
                    'recent_avg': round(recent_avg, 1),
                    'season_avg': round(season_avg, 1),
                    'hit_rate_over': round(historical_over * 100, 0),
                    'hit_rate_under': round(historical_under * 100, 0),
                    'trend': trend,
                    'games_analyzed': len(history),
                    'bookmaker': row.get('bookmaker', 'unknown'),
                    'odds': pick_odds,  # Odds for the recommended side
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'implied_prob': round(implied_prob * 100, 1),  # Breakeven for recommended side
                    'market_prob': round(market_prob * 100, 1),  # No-vig probability
                    'our_prob': round(our_prob * 100, 1),
                    'adjustments': ' | '.join(clean_adjustments) if clean_adjustments else 'None',
                    'is_b2b': context.get('is_b2b', False),
                    'rest_days': context.get('rest_days'),
                    'minutes_factor': round(context.get('minutes_factor', 1.0), 2),
                    'minutes_trend': context.get('minutes_trend', 'STABLE'),
                    'game': game_matchup,
                    'home_team': home_team,
                    'away_team': away_team,
                    # MATCHUP CONTEXT COLUMNS
                    'opponent': opponent,
                    'opp_defense_rank': opp_defense_rank,
                    'matchup_label': matchup_label,
                    'team_pace_rank': team_pace_rank,
                    'opp_pace_rank': opp_pace_rank,
                    # NEWS INTELLIGENCE COLUMNS
                    'news_status': news_context.status if news_context else 'NO_NEWS',
                    'news_adjustment': f"{(news_factor - 1) * 100:+.1f}%" if news_factor != 1.0 else '',
                    'news_flags': ' | '.join(news_flags) if news_flags else '',
                    'news_notes': '; '.join(news_notes[:2]) if news_notes else '',  # Limit to 2 notes
                    'news_sources': ' | '.join(news_context.sources[:3]) if news_context and news_context.sources else '',  # Source URLs
                })

                # Track for correlation filtering
                if player not in player_picks:
                    player_picks[player] = []
                player_picks[player].append(prop_type)

            except Exception as e:
                logger.warning(f"Failed to analyze prop for {player} {prop_type}: {e}")
                skipped['analysis_error'] = skipped.get('analysis_error', 0) + 1
                continue

        logger.info(f"Skipped: {skipped}")

        # =================================================================
        # PHASE 5: CORRELATION FILTERING + ALT LINE DEDUPLICATION
        # =================================================================
        if value_props:
            df = pd.DataFrame(value_props)
            original_count = len(df)

            # Step 1: Remove highly correlated props (points vs pra, etc.)
            # If player has both 'points' and 'pra', keep only the higher edge one
            correlated_pairs = [('points', 'pra'), ('rebounds', 'pra'), ('assists', 'pra')]

            rows_to_drop = []
            for player in df['player'].unique():
                player_df = df[df['player'] == player]
                for prop1, prop2 in correlated_pairs:
                    p1 = player_df[player_df['prop_type'] == prop1]
                    p2 = player_df[player_df['prop_type'] == prop2]
                    if len(p1) > 0 and len(p2) > 0:
                        # Keep the one with higher vig-adjusted edge
                        if p1.iloc[0]['avg_edge'] > p2.iloc[0]['avg_edge']:
                            rows_to_drop.extend(p2.index.tolist())
                        else:
                            rows_to_drop.extend(p1.index.tolist())

            df = df.drop(index=list(set(rows_to_drop)))
            correlation_removed = original_count - len(df)

            # FIX #3: Alt Line Deduplication
            # Keep only ONE line per player per prop type (the one with highest edge)
            # This prevents picks like: Garland UNDER 23.5, 22.5, 21.5 all appearing
            before_alt_dedup = len(df)
            df = df.sort_values(['avg_edge', 'confidence'], ascending=[False, False])
            df = df.drop_duplicates(subset=['player', 'prop_type'], keep='first')
            alt_lines_removed = before_alt_dedup - len(df)

            skipped['correlation'] = correlation_removed
            skipped['alt_lines'] = alt_lines_removed

            # Sort by vig-adjusted edge
            df = df.sort_values('avg_edge', ascending=False)

            logger.info(f"PHASE 5: Filtering removed {correlation_removed} correlated + {alt_lines_removed} alt lines")
            logger.info("=" * 60)
            logger.info(f"FINAL: {len(df)} value props (min {min_edge*100:.0f}% edge after vig)")
            logger.info("=" * 60)

            return df

        return pd.DataFrame()
