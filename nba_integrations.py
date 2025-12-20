"""
DEPRECATED: Import from 'data' and 'analysis' modules instead.

This file remains for backward compatibility only. All implementations
have been moved to the data/ and analysis/ packages.

Example:
    # Old (deprecated):
    from nba_integrations import NBADataFetcher, LivePropAnalyzer

    # New (preferred):
    from data import NBADataFetcher, OddsAPIClient, InjuryTracker
    from analysis import LivePropAnalyzer
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'nba_integrations' is deprecated. "
    "Import from 'data' and 'analysis' modules instead: "
    "from data import NBADataFetcher; from analysis import LivePropAnalyzer",
    DeprecationWarning,
    stacklevel=2
)

# Re-export data fetchers from data package for backward compatibility
from data import (
    NBADataFetcher,
    ResilientFetcher,
    get_resilient_fetcher,
    OddsAPIClient,
    InjuryTracker,
)

# Re-export LivePropAnalyzer from analysis package
from analysis import LivePropAnalyzer

# Re-export core utilities that were previously in this file
from core.constants import (
    TEAM_ABBREVIATIONS,
    STAR_PLAYERS,
    STAR_OUT_BOOST,
    normalize_team_abbrev,
    get_current_nba_season,
)
from core.config import CONFIG
from core.exceptions import (
    NetworkError,
    AuthenticationError,
    RateLimitError,
    OddsAPIError,
)

# Backward compatibility aliases
TEAM_ABBREV_MAP = TEAM_ABBREVIATIONS


def demo_nba_data():
    """Demonstrate NBA data fetching capabilities."""
    print("\n" + "="*60)
    print("        NBA DATA FETCHER DEMO")
    print("="*60)

    fetcher = NBADataFetcher()

    # Example players to analyze
    players = ["Luka Doncic", "Shai Gilgeous-Alexander", "Jayson Tatum"]

    for player in players:
        print(f"\n{player}")
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
            print("\nVALUE PLAYS (5%+ Edge):")
            print("-" * 40)
            for _, row in value_plays.iterrows():
                indicator = "[+]" if row['Pick'] != 'PASS' else "[ ]"
                print(f"  {indicator} {row['Player']} {row['Prop'].upper()} {row['Pick']} {row['Line']} "
                      f"(Edge: {row['Edge']:+.1f}%)")

    return results


def main():
    """Run full demonstration."""
    print("NBA PROP ANALYSIS - LIVE DATA INTEGRATION")
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

    from data import OddsAPIClient
    from analysis import LivePropAnalyzer

    odds = OddsAPIClient(api_key="YOUR_API_KEY")
    analyzer = LivePropAnalyzer(odds_client=odds)

    # Find value props
    value_props = analyzer.find_value_props(min_edge=0.05)

Free tier includes 500 requests/month - plenty for daily analysis!
    """)

    return fetcher, results


if __name__ == "__main__":
    fetcher, results = main()
