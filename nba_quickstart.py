"""
NBA Prop Analysis System - Quick Start Guide
=============================================

INSTALLATION
------------
pip install nba_api pandas numpy scipy openpyxl requests

QUICK START
-----------
# 1. Fetch player data and analyze a prop
from nba_integrations import NBADataFetcher, LivePropAnalyzer

fetcher = NBADataFetcher()
analyzer = LivePropAnalyzer(nba_fetcher=fetcher)

# Analyze a specific prop
result = analyzer.analyze_prop(
    player_name="Luka Doncic",
    prop_type="points",
    line=32.5,
    odds=-110
)

print(f"Projection: {result['recent_avg']}")
print(f"Edge: {result['avg_edge']}%")
print(f"Recommendation: {result['recommended_side']}")

# 2. Analyze multiple props at once
props_to_check = [
    {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
    {'player': 'Shai Gilgeous-Alexander', 'prop_type': 'points', 'line': 30.5},
    {'player': 'Jayson Tatum', 'prop_type': 'rebounds', 'line': 8.5},
    {'player': 'Anthony Edwards', 'prop_type': 'assists', 'line': 4.5},
]

results = analyzer.analyze_multiple_props(props_to_check)
print(results)

# 3. Get live odds (requires free API key from the-odds-api.com)
from nba_integrations import OddsAPIClient

odds = OddsAPIClient(api_key="YOUR_API_KEY")  # Get free key at the-odds-api.com
props = odds.get_player_props()
parsed = odds.parse_player_props(props)
print(parsed.head())

# 4. Find value props automatically
value_plays = analyzer.find_value_props(min_edge=0.05)
print(value_plays)

# 5. Run backtest on historical data
from nba_prop_model import Backtester, EnsembleModel, generate_sample_dataset, generate_prop_lines

game_logs = generate_sample_dataset()  # Or use real data
props = generate_prop_lines(game_logs)

backtester = Backtester(initial_bankroll=1000, unit_size=10)
results = backtester.run_backtest(props, game_logs, EnsembleModel(), min_edge=0.05)
backtester.print_report()
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # The Odds API (https://the-odds-api.com/)
    # Free tier: 500 requests/month
    'ODDS_API_KEY': 'ed075e0b818c7b977da240e06c5f06a5',  # Set your key here or use environment variable
    
    # Analysis settings
    'MIN_EDGE_THRESHOLD': 0.05,  # 5% minimum edge to bet
    'MIN_CONFIDENCE': 0.40,      # 40% minimum confidence
    'LOOKBACK_GAMES': 15,        # Games to analyze
    
    # Bankroll settings
    'INITIAL_BANKROLL': 1000,
    'UNIT_SIZE': 10,
    'MAX_UNITS_PER_BET': 3,
    
    # Preferred sportsbooks (in order)
    'PREFERRED_BOOKS': ['draftkings', 'fanduel', 'betmgm', 'caesars'],
}


# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

import os
import sys

def run_daily_analysis():
    """
    Complete daily prop analysis workflow.
    Run this each day before games start.
    """
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from nba_integrations import NBADataFetcher, OddsAPIClient, LivePropAnalyzer
    
    print("🏀 NBA PROP ANALYSIS - DAILY RUN")
    print("=" * 50)
    
    # Initialize components
    fetcher = NBADataFetcher()
    
    api_key = CONFIG['ODDS_API_KEY'] or os.environ.get('ODDS_API_KEY')
    odds_client = OddsAPIClient(api_key=api_key) if api_key else None
    
    analyzer = LivePropAnalyzer(nba_fetcher=fetcher, odds_client=odds_client)
    
    if odds_client:
        print("\n📊 Fetching live odds...")
        value_props = analyzer.find_value_props(min_edge=CONFIG['MIN_EDGE_THRESHOLD'])
        
        if not value_props.empty:
            print(f"\n🎯 Found {len(value_props)} value plays:")
            print(value_props.to_string(index=False))
        else:
            print("No value plays found meeting criteria")
    else:
        print("\n⚠️ No Odds API key configured")
        print("Set ODDS_API_KEY in config or environment variable")
        print("Get free key at: https://the-odds-api.com/")
        
        # Run manual analysis instead
        print("\n📊 Running manual analysis on sample props...")
        
        sample_props = [
            {'player': 'Luka Doncic', 'prop_type': 'points', 'line': 32.5},
            {'player': 'Shai Gilgeous-Alexander', 'prop_type': 'points', 'line': 30.5},
            {'player': 'Giannis Antetokounmpo', 'prop_type': 'rebounds', 'line': 11.5},
            {'player': 'Tyrese Haliburton', 'prop_type': 'assists', 'line': 10.5},
            {'player': 'Nikola Jokic', 'prop_type': 'pra', 'line': 47.5},
        ]
        
        results = analyzer.analyze_multiple_props(sample_props)
        
        if not results.empty:
            print("\n" + "=" * 60)
            print("                 ANALYSIS RESULTS")
            print("=" * 60)
            print(results.to_string(index=False))


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
