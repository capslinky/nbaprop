"""
NBA Prop Analysis System - Quick Start Guide
=============================================

INSTALLATION
------------
pip install nba_api pandas numpy scipy openpyxl requests streamlit

QUICK START
-----------
# 1. Use UnifiedPropModel for context-aware prop analysis
from models import UnifiedPropModel

model = UnifiedPropModel()

# Analyze a specific prop (all context is auto-detected)
analysis = model.analyze(
    player_name="Luka Doncic",
    prop_type="points",
    line=32.5,
    odds=-110
)

print(f"Projection: {analysis.projection}")
print(f"Edge: {analysis.edge:.1%}")
print(f"Confidence: {analysis.confidence:.0%}")
print(f"Pick: {analysis.pick}")
print(f"Flags: {analysis.flags}")

# 2. Use LivePropAnalyzer for batch analysis with odds integration
from analysis import LivePropAnalyzer
from data import OddsAPIClient

odds = OddsAPIClient(api_key="YOUR_API_KEY")
analyzer = LivePropAnalyzer(odds_client=odds)

props_to_check = [
    {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
    {'player': 'Shai Gilgeous-Alexander', 'prop_type': 'points', 'line': 30.5},
    {'player': 'Jayson Tatum', 'prop_type': 'rebounds', 'line': 8.5},
]

results = analyzer.analyze_multiple_props(props_to_check)
print(results)

# 3. Find value props automatically from live odds
value_plays = analyzer.find_value_props(min_edge=0.05, min_confidence=0.4)
print(value_plays)

# 4. Run backtest on historical data
from models import Backtester, EnsembleModel
from models import generate_sample_dataset, generate_prop_lines

game_logs = generate_sample_dataset()
props = generate_prop_lines(game_logs)

backtester = Backtester(initial_bankroll=1000, unit_size=10)
results = backtester.run_backtest(props, game_logs, EnsembleModel(), min_edge=0.05)
backtester.print_report()

# 5. Launch Streamlit dashboard
# streamlit run app.py
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import centralized config from core module
from core.config import CONFIG as _CORE_CONFIG

# Backward-compatible CONFIG dict that reads from core.config
# This allows existing code to use CONFIG['MIN_EDGE_THRESHOLD'] syntax
CONFIG = {
    # The Odds API (https://the-odds-api.com/)
    # Note: API key now preferably set via ODDS_API_KEY environment variable
    'ODDS_API_KEY': _CORE_CONFIG.ODDS_API_KEY,  # Set via ODDS_API_KEY environment variable

    # Analysis settings (from core.config)
    'MIN_EDGE_THRESHOLD': _CORE_CONFIG.MIN_EDGE_THRESHOLD,
    'MIN_CONFIDENCE': _CORE_CONFIG.MIN_CONFIDENCE,
    'LOOKBACK_GAMES': _CORE_CONFIG.LOOKBACK_GAMES,

    # Bankroll settings (from core.config)
    'INITIAL_BANKROLL': _CORE_CONFIG.INITIAL_BANKROLL,
    'UNIT_SIZE': _CORE_CONFIG.UNIT_SIZE,
    'MAX_UNITS_PER_BET': _CORE_CONFIG.MAX_UNITS_PER_BET,

    # Preferred sportsbooks (from core.config)
    'PREFERRED_BOOKS': _CORE_CONFIG.PREFERRED_BOOKS,
}


# =============================================================================
# LOGGING SETUP
# =============================================================================

from core.logging_config import setup_logging, get_logger
setup_logging(level="INFO")
logger = get_logger(__name__)

# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

import os
import sys

def run_daily_analysis():
    """
    Complete daily prop analysis workflow.
    Run this each day before games start.

    Output format:
    - Breaks down each game individually
    - Shows top 5 picks per game with explanations
    - Uses FanDuel odds only
    - Saves full results to Excel/CSV
    """
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data import NBADataFetcher, OddsAPIClient, InjuryTracker, TEAM_ABBREVIATIONS
    from models import UnifiedPropModel
    from datetime import datetime
    import pandas as pd

    print("=" * 70)
    print("        NBA PROP ANALYSIS - DAILY BREAKDOWN BY GAME")
    print("        FanDuel Odds Only | Top 5 Picks Per Game")
    print("=" * 70)

    # Initialize components
    fetcher = NBADataFetcher()
    injury_tracker = InjuryTracker()

    api_key = CONFIG['ODDS_API_KEY'] or os.environ.get('ODDS_API_KEY')
    if not api_key:
        logger.error("No Odds API key configured")
        logger.error("Set ODDS_API_KEY in config or environment variable")
        logger.error("Get key at: https://the-odds-api.com/")
        return

    odds_client = OddsAPIClient(api_key=api_key)
    model = UnifiedPropModel(
        data_fetcher=fetcher,
        injury_tracker=injury_tracker,
        odds_client=odds_client
    )

    # Output filename with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    excel_file = f"nba_daily_picks_{today}.xlsx"

    print(f"\nDate: {today}")
    print(f"Min Edge: {CONFIG['MIN_EDGE_THRESHOLD']*100:.0f}%")
    print(f"Min Confidence: {CONFIG['MIN_CONFIDENCE']*100:.0f}%")

    # =================================================================
    # FETCH GAMES AND PROPS (FanDuel Only)
    # =================================================================
    print("\n" + "-" * 70)
    print("Fetching games and FanDuel odds...")

    events = odds_client.get_events()
    if not events:
        logger.warning("No games found today.")
        return

    logger.info(f"Found {len(events)} games today")

    # Props we want to analyze
    markets = [
        'player_points',
        'player_rebounds',
        'player_assists',
        'player_points_rebounds_assists',
        'player_threes'
    ]

    all_picks = []  # Store all picks for export
    game_count = 0

    for event in events:
        game_count += 1
        event_id = event.get('id')
        home_team = event.get('home_team', 'Unknown')
        away_team = event.get('away_team', 'Unknown')
        commence_time = event.get('commence_time', '')

        # Parse game time
        try:
            game_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            time_str = game_time.strftime('%I:%M %p ET')
        except (ValueError, TypeError):
            time_str = ''

        # Get team abbreviations
        home_abbrev = TEAM_ABBREVIATIONS.get(home_team, home_team[:3].upper())
        away_abbrev = TEAM_ABBREVIATIONS.get(away_team, away_team[:3].upper())

        print("=" * 70)
        print(f"GAME {game_count}: {away_team} @ {home_team}")
        print(f"         {away_abbrev} @ {home_abbrev} | {time_str}")
        print("=" * 70)

        # Get props for this game
        props_data = odds_client.get_player_props(event_id, markets)
        props_df = odds_client.parse_player_props(props_data)

        if props_df.empty:
            logger.info("  No props available for this game")
            continue

        # FILTER TO FANDUEL ONLY
        fanduel_props = props_df[props_df['bookmaker'] == 'fanduel'].copy()

        if fanduel_props.empty:
            logger.info("  No FanDuel odds available for this game")
            continue

        # Get unique props (over side only - we'll determine direction)
        overs = fanduel_props[fanduel_props['side'] == 'over']

        logger.info(f"  Analyzing {len(overs)} props from FanDuel...")

        # Analyze each prop
        game_results = []

        for _, row in overs.iterrows():
            player = row['player']
            prop_type = row['prop_type']
            line = row['line']
            over_odds = row['odds']

            # Get under odds
            under_row = fanduel_props[
                (fanduel_props['player'] == player) &
                (fanduel_props['prop_type'] == prop_type) &
                (fanduel_props['line'] == line) &
                (fanduel_props['side'] == 'under')
            ]
            under_odds = under_row['odds'].values[0] if len(under_row) > 0 else -110

            try:
                # Use UnifiedPropModel for full contextual analysis
                analysis = model.analyze(
                    player_name=player,
                    prop_type=prop_type,
                    line=line,
                    odds=over_odds if over_odds else -110,
                    opponent=home_abbrev if player in str(away_team) else away_abbrev,
                    is_home=player not in str(away_team)
                )

                if analysis and analysis.projection > 0:
                    # Use the model's pick direction
                    pick_side = analysis.pick
                    edge = analysis.edge

                    # Get the correct odds based on recommended side
                    display_odds = over_odds if pick_side == 'OVER' else under_odds

                    game_results.append({
                        'player': player,
                        'prop_type': prop_type,
                        'line': line,
                        'pick': pick_side,
                        'odds': display_odds,
                        'projection': analysis.projection,
                        'edge': edge,
                        'confidence': analysis.confidence,
                        'recent_avg': analysis.recent_avg,
                        'season_avg': analysis.season_avg,
                        'over_rate': analysis.over_rate,
                        'trend': analysis.trend,
                        'matchup_rating': analysis.matchup_rating,
                        'is_b2b': analysis.is_b2b,
                        'total_adjustment': analysis.total_adjustment,
                        'adjustments': analysis.adjustments,
                        'flags': analysis.flags,
                        'context_quality': analysis.context_quality,
                        'game': f"{away_abbrev} @ {home_abbrev}",
                        'home_team': home_team,
                        'away_team': away_team,
                    })
            except Exception as e:
                logger.debug(f"Skipped prop analysis: {e}")

        if not game_results:
            logger.info("  No analyzable props found")
            continue

        # Filter by minimum edge and confidence
        min_edge = CONFIG['MIN_EDGE_THRESHOLD']
        min_conf = CONFIG['MIN_CONFIDENCE']

        qualified = [r for r in game_results
                     if abs(r['edge']) >= min_edge
                     and r['confidence'] >= min_conf
                     and r['pick'] != 'PASS']

        if not qualified:
            logger.info(f"  No picks meet criteria (edge >= {min_edge*100:.0f}%, conf >= {min_conf*100:.0f}%)")
            continue

        # Sort by edge * confidence (best overall value)
        qualified.sort(key=lambda x: abs(x['edge']) * x['confidence'], reverse=True)

        # Take top 5
        top_5 = qualified[:5]

        print(f"\n  TOP {len(top_5)} PICKS:\n")
        print("  " + "-" * 66)

        for i, pick in enumerate(top_5, 1):
            # Format the pick display
            edge_pct = pick['edge'] * 100
            conf_pct = pick['confidence'] * 100
            odds_str = f"+{pick['odds']}" if pick['odds'] > 0 else str(pick['odds'])

            print(f"  {i}. {pick['player']}")
            print(f"     {pick['prop_type'].upper()} {pick['pick']} {pick['line']} ({odds_str})")
            print(f"     Projection: {pick['projection']:.1f} | Edge: {edge_pct:+.1f}% | Confidence: {conf_pct:.0f}%")

            # Explanation
            reasons = []

            # Recent performance
            if pick['recent_avg'] > pick['line'] and pick['pick'] == 'OVER':
                reasons.append(f"L5 avg {pick['recent_avg']:.1f} (above line)")
            elif pick['recent_avg'] < pick['line'] and pick['pick'] == 'UNDER':
                reasons.append(f"L5 avg {pick['recent_avg']:.1f} (below line)")

            # Hit rate
            if pick['pick'] == 'OVER' and pick['over_rate'] >= 0.6:
                reasons.append(f"Hit over {pick['over_rate']*100:.0f}% of games")
            elif pick['pick'] == 'UNDER' and (1 - pick['over_rate']) >= 0.6:
                reasons.append(f"Hit under {(1-pick['over_rate'])*100:.0f}% of games")

            # Trend
            if pick['trend'] == 'HOT' and pick['pick'] == 'OVER':
                reasons.append("HOT streak")
            elif pick['trend'] == 'COLD' and pick['pick'] == 'UNDER':
                reasons.append("COLD streak")

            # Matchup
            if pick['matchup_rating'] in ['SMASH', 'GOOD']:
                reasons.append(f"{pick['matchup_rating']} matchup")
            elif pick['matchup_rating'] in ['TOUGH', 'HARD']:
                reasons.append(f"{pick['matchup_rating']} matchup")

            # B2B
            if pick['is_b2b']:
                reasons.append("B2B game (-8%)")

            # Key adjustments
            adj = pick['adjustments']
            if adj.get('opp_defense', 0) > 2:
                reasons.append(f"Weak opp defense (+{adj['opp_defense']:.0f}%)")
            elif adj.get('opp_defense', 0) < -2:
                reasons.append(f"Strong opp defense ({adj['opp_defense']:.0f}%)")

            if adj.get('injury_boost', 0) > 0:
                reasons.append(f"Injury boost (+{adj['injury_boost']:.0f}%)")

            # Print explanation
            if reasons:
                print(f"     Why: {' | '.join(reasons[:4])}")

            # Flags
            if pick['flags']:
                print(f"     Flags: {', '.join(pick['flags'][:3])}")

            print()

        # Add to all picks
        all_picks.extend(top_5)

    # =================================================================
    # SUMMARY AND EXPORT
    # =================================================================
    print("\n" + "=" * 70)
    print("                         DAILY SUMMARY")
    print("=" * 70)

    if all_picks:
        print(f"\nTotal Games: {game_count}")
        print(f"Total Top Picks: {len(all_picks)}")

        # Best overall picks
        all_picks.sort(key=lambda x: abs(x['edge']) * x['confidence'], reverse=True)

        print("\nBEST OVERALL PICKS TODAY:")
        print("-" * 40)
        for i, pick in enumerate(all_picks[:10], 1):
            edge_pct = pick['edge'] * 100
            odds_str = f"+{pick['odds']}" if pick['odds'] > 0 else str(pick['odds'])
            print(f"  {i}. {pick['player']} {pick['prop_type'].upper()} {pick['pick']} {pick['line']} ({odds_str})")
            print(f"     Game: {pick['game']} | Edge: {edge_pct:+.1f}% | Proj: {pick['projection']:.1f}")

        # Save to Excel
        try:
            export_df = pd.DataFrame(all_picks)
            export_cols = ['game', 'player', 'prop_type', 'pick', 'line', 'odds',
                          'projection', 'edge', 'confidence', 'recent_avg', 'trend',
                          'matchup_rating', 'context_quality']
            available_cols = [c for c in export_cols if c in export_df.columns]
            export_df = export_df[available_cols]
            export_df['edge'] = (export_df['edge'] * 100).round(1)
            export_df['confidence'] = (export_df['confidence'] * 100).round(0)
            export_df.to_excel(excel_file, index=False, sheet_name='Top Picks')
            logger.info(f"Results saved to: {excel_file}")
        except Exception as e:
            csv_file = f"nba_daily_picks_{today}.csv"
            export_df.to_csv(csv_file, index=False)
            logger.info(f"Results saved to: {csv_file}")
    else:
        logger.warning("No qualified picks found today.")

    print("\n" + "=" * 70)


def analyze_custom_props(props_list: list):
    """
    Analyze a custom list of props.
    
    Args:
        props_list: List of dicts with keys: player, prop_type, line
                   Optional: odds (default -110)
    
    Example:
        props = [
            {'player': 'LeBron James', 'prop_type': 'points', 'line': 25.5},
            {'player': 'LeBron James', 'prop_type': 'assists', 'line': 7.5, 'odds': -115},
        ]
        analyze_custom_props(props)
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nba_integrations import NBADataFetcher, LivePropAnalyzer
    
    fetcher = NBADataFetcher()
    analyzer = LivePropAnalyzer(nba_fetcher=fetcher)
    
    print(f"\n🔍 Analyzing {len(props_list)} props...")
    results = analyzer.analyze_multiple_props(props_list)
    
    if not results.empty:
        # Sort by edge
        results = results.sort_values('Edge', ascending=False, key=abs)
        
        print("\n" + "=" * 70)
        print("                      PROP ANALYSIS RESULTS")
        print("=" * 70)
        print(results.to_string(index=False))
        
        # Highlight best plays
        best_plays = results[abs(results['Edge']) >= 5]
        if not best_plays.empty:
            print("\n" + "=" * 70)
            print("                      🎯 TOP PLAYS (5%+ Edge)")
            print("=" * 70)
            for _, row in best_plays.iterrows():
                emoji = "🟢" if row['Edge'] > 0 else "🔴"
                direction = "OVER" if row['Edge'] > 0 else "UNDER"
                print(f"  {emoji} {row['Player']:<25} {row['Prop'].upper():<10} "
                      f"{direction} {row['Line']:<6} (Edge: {row['Edge']:+.1f}%)")
    
    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Prop Analysis System')
    parser.add_argument('--daily', action='store_true', help='Run daily analysis')
    parser.add_argument('--player', type=str, help='Analyze specific player')
    parser.add_argument('--prop', type=str, choices=['points', 'rebounds', 'assists', 'pra'],
                       help='Prop type to analyze')
    parser.add_argument('--line', type=float, help='Betting line')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    
    args = parser.parse_args()

    # Log current date context for debugging
    from datetime import date
    from core.constants import get_current_nba_season
    current_season = get_current_nba_season()
    print(f"Running analysis for {date.today()} (Season: {current_season})")

    if args.daily:
        run_daily_analysis()
    elif args.player and args.prop and args.line:
        analyze_custom_props([{
            'player': args.player,
            'prop_type': args.prop,
            'line': args.line
        }])
    elif args.backtest:
        print("Running backtest...")
        from nba_prop_model import main as run_backtest
        run_backtest()
    else:
        # Default: run daily analysis
        run_daily_analysis()
