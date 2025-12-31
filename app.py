"""
NBA Prop Analysis Dashboard
Run with: streamlit run app.py

Uses UnifiedPropModel for consistent, context-aware analysis.
"""

import streamlit as st
import pandas as pd
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use new modular imports
from data import NBADataFetcher, OddsAPIClient, InjuryTracker, TEAM_ABBREVIATIONS
from models import UnifiedPropModel
from core.config import CONFIG
from core.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="NBA Props", page_icon="🏀", layout="wide")

# Cache resources
@st.cache_resource
def get_odds_client():
    api_key = CONFIG.ODDS_API_KEY or os.environ.get('ODDS_API_KEY')
    return OddsAPIClient(api_key=api_key) if api_key else None

@st.cache_resource
def get_fetcher():
    return NBADataFetcher()

@st.cache_resource
def get_model():
    """Get unified model with all dependencies."""
    return UnifiedPropModel(
        data_fetcher=get_fetcher(),
        injury_tracker=InjuryTracker(),
        odds_client=get_odds_client()
    )

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_player_stats(player_name, num_games):
    """Cached player stats for Player Lookup tab."""
    return get_fetcher().get_player_game_logs(player_name, last_n_games=num_games)

odds_client = get_odds_client()
model = get_model()
fetcher = get_fetcher()

st.title("🏀 NBA Prop Analyzer")

# Sidebar
st.sidebar.header("Settings")
min_edge = st.sidebar.slider("Min Edge %", 1, 25, 5)
num_games = st.sidebar.slider("Lookback Games", 5, 30, 15)
max_games_to_scan = st.sidebar.slider("Max Games to Scan", 1, 10, 5)

if odds_client and odds_client.remaining_requests:
    st.sidebar.metric("API Calls Left", odds_client.remaining_requests)

# =============================================================================
# MAIN: FIND BEST PLAYS ACROSS ALL GAMES
# =============================================================================

if not odds_client:
    st.error("Add your Odds API key to nba_quickstart.py")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎯 Find Best Plays", "🔍 Single Game", "📊 Player Lookup"])

with tab1:
    st.markdown("### Scan all games and find the best value plays")

    col1, col2 = st.columns([1, 3])
    with col1:
        scan_btn = st.button("🔍 SCAN ALL GAMES", type="primary", use_container_width=True)
    with col2:
        prop_types = st.multiselect(
            "Props to scan",
            ["points", "rebounds", "assists", "threes", "pra"],
            default=["points", "rebounds", "assists"],
            label_visibility="collapsed"
        )

    if scan_btn:
        # Get all events and game lines
        with st.spinner("Finding games and loading context..."):
            events = odds_client.get_events()
            game_lines = odds_client.get_game_lines()

        if not events:
            st.warning("No games found")
            st.stop()

        events = events[:max_games_to_scan]
        st.info(f"Scanning {len(events)} games with full situational analysis...")

        # Market mapping
        market_map = {
            "points": "player_points",
            "rebounds": "player_rebounds",
            "assists": "player_assists",
            "threes": "player_threes",
            "pra": "player_points_rebounds_assists"
        }
        markets = [market_map[p] for p in prop_types]

        all_results = []
        progress = st.progress(0, text="Loading props...")

        # Team name to abbreviation mapping (imported from core.constants)
        team_abbrev_map = TEAM_ABBREVIATIONS

        for i, event in enumerate(events):
            game_name = f"{event['away_team']} @ {event['home_team']}"
            progress.progress((i + 0.5) / len(events), text=f"Loading: {game_name}")

            # Get props for this game
            props_data = odds_client.get_player_props(event['id'], markets)
            props_df = odds_client.parse_player_props(props_data)

            if props_df.empty:
                continue

            # Get best odds per prop
            best = odds_client.get_best_odds(props_df)
            overs = best[best['side'] == 'over']

            # Get team abbreviations
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            event_id = event.get('id', '')
            home_abbrev = team_abbrev_map.get(home_team, home_team[:3].upper() if home_team else '')
            away_abbrev = team_abbrev_map.get(away_team, away_team[:3].upper() if away_team else '')

            # Get game context (total, spread, blowout risk)
            game_total = None
            blowout_risk = None
            if not game_lines.empty:
                game_line = game_lines[game_lines['game_id'] == event_id]
                if not game_line.empty:
                    game_total = game_line['total'].values[0]
                    blowout_risk = game_line['blowout_risk'].values[0]

            # Analyze each prop using UnifiedPropModel
            players_done = set()
            for _, row in overs.iterrows():
                player = row['player']
                prop = row['prop_type']
                line = row['line']

                # Skip if we already did this player (for this prop type)
                key = f"{player}_{prop}"
                if key in players_done:
                    continue
                players_done.add(key)

                try:
                    # Get under odds for display
                    under_row = best[(best['player'] == player) &
                                    (best['prop_type'] == prop) &
                                    (best['side'] == 'under')]
                    under_odds = under_row['odds'].values[0] if len(under_row) > 0 else -110
                    over_odds = int(row['odds']) if row['odds'] else -110

                    # Use UnifiedPropModel - it handles ALL context internally
                    analysis = model.analyze(
                        player_name=player,
                        prop_type=prop,
                        line=line,
                        odds=over_odds,
                        game_total=game_total,
                        blowout_risk=blowout_risk,
                        last_n_games=num_games
                    )

                    # Skip if no data
                    if analysis.games_analyzed == 0:
                        continue

                    edge = analysis.edge * 100  # Convert to percentage

                    # Only keep if meets edge threshold
                    if abs(edge) >= min_edge:
                        all_results.append({
                            'Game': game_name,
                            'Player': player,
                            'Prop': prop.upper(),
                            'Line': line,
                            'Proj': analysis.projection,
                            'L5': analysis.recent_avg,
                            'Avg': analysis.season_avg,
                            'O%': int(analysis.over_rate * 100),
                            'U%': int(analysis.under_rate * 100),
                            'Trend': analysis.trend,
                            'Matchup': analysis.matchup_rating,
                            'OppRk': analysis.opp_rank,
                            'Edge': edge,
                            'Conf': int(analysis.confidence * 100),
                            'Pick': analysis.pick,
                            'Over': over_odds,
                            'Under': int(under_odds) if under_odds else None,
                            'Book': row['bookmaker'],
                            'Home': '🏠' if analysis.is_home else '✈️' if analysis.is_home is False else '',
                            'Flags': analysis.flags,
                            'Total': analysis.game_total,
                            'B2B': '⚠️' if analysis.is_b2b else '',
                            'TotalAdj': round(analysis.total_adjustment * 100, 1),
                            'Adjustments': analysis.adjustments
                        })
                except Exception as e:
                    logger.debug(f"Skipped prop in dashboard: {e}")
                    continue

            progress.progress((i + 1) / len(events), text=f"Done: {game_name}")
            time.sleep(0.1)

        progress.empty()

        if all_results:
            df = pd.DataFrame(all_results)
            df = df.sort_values('Edge', key=abs, ascending=False)

            # Store in session
            st.session_state.results = df
            st.session_state.scan_time = datetime.now().strftime("%I:%M %p")
        else:
            st.warning(f"No plays found with {min_edge}%+ edge")

    # Display results if we have them
    if 'results' in st.session_state and len(st.session_state.results) > 0:
        df = st.session_state.results

        st.success(f"Found {len(df)} value plays (scanned at {st.session_state.get('scan_time', 'N/A')})")

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Plays", len(df))
        col2.metric("Overs", len(df[df['Pick'] == 'OVER']))
        col3.metric("Unders", len(df[df['Pick'] == 'UNDER']))
        col4.metric("Best Edge", f"{df['Edge'].abs().max():.1f}%")

        st.divider()

        # Top Picks - Clean action view
        st.subheader("🎯 TOP PICKS")

        for _, pick in df.head(10).iterrows():
            if pick['Pick'] == 'PASS':
                continue

            emoji = "🟢" if pick['Pick'] == 'OVER' else "🔴"
            trend_emoji = "🔥" if pick['Trend'] == 'HOT' else "❄️" if pick['Trend'] == 'COLD' else ""
            odds_val = pick['Over'] if pick['Pick'] == 'OVER' else pick['Under']
            hit_rate = pick['O%'] if pick['Pick'] == 'OVER' else pick['U%']

            # Get flags from analysis
            flags = pick.get('Flags', [])
            b2b_icon = pick.get('B2B', '')

            opp_rank = pick.get('OppRk')
            home_icon = pick.get('Home', '')
            total = pick.get('Total')

            col1, col2 = st.columns([3, 1])
            with col1:
                # Header with pick
                st.markdown(
                    f"### {emoji} {pick['Player']} - {pick['Prop']} **{pick['Pick']}** {pick['Line']} {trend_emoji} {b2b_icon}"
                )

                # Main stats line
                proj_str = f"Proj: {pick.get('Proj', pick['Avg'])}"
                st.caption(
                    f"{proj_str} | L5: {pick['L5']} | Hit: {hit_rate}% | "
                    f"Edge: **{pick['Edge']:+.1f}%** | Conf: {pick['Conf']}%"
                )

                # Show flags if any
                if flags:
                    st.caption(" ".join(flags))

            with col2:
                st.caption(f"{home_icon} {pick['Game']}")
                game_info = f"@ {pick['Book']} ({odds_val:+d})"
                if total:
                    game_info += f" | O/U: {total}"
                st.caption(game_info)

            st.divider()

        # Full table
        with st.expander("📋 View All Plays"):
            st.dataframe(df, hide_index=True, use_container_width=True)

        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Picks (CSV)",
            csv,
            f"nba_picks_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# =============================================================================
# TAB 2: SINGLE GAME ANALYSIS
# =============================================================================
with tab2:
    st.subheader("Analyze a Single Game")

    # Load games
    if 'all_events' not in st.session_state:
        events = odds_client.get_events()
        st.session_state.all_events = events if events else []

    events = st.session_state.all_events

    if not events:
        st.warning("No games found. Click refresh.")
        if st.button("Refresh"):
            st.session_state.all_events = odds_client.get_events() or []
            st.rerun()
    else:
        game_names = [f"{e['away_team']} @ {e['home_team']}" for e in events]
        selected = st.selectbox("Select Game", range(len(events)), format_func=lambda i: game_names[i])

        props = st.multiselect("Props", ["points", "rebounds", "assists", "threes"], default=["points", "rebounds", "assists"])

        if st.button("Load Game Props"):
            event = events[selected]
            markets = [{"points": "player_points", "rebounds": "player_rebounds",
                       "assists": "player_assists", "threes": "player_threes"}[p] for p in props]

            with st.spinner("Loading..."):
                data = odds_client.get_player_props(event['id'], markets)
                props_df = odds_client.parse_player_props(data)

            if props_df.empty:
                st.warning("No props available")
            else:
                best = odds_client.get_best_odds(props_df)

                # Show raw odds
                display = best[['player', 'prop_type', 'line', 'side', 'odds', 'bookmaker']]
                display.columns = ['Player', 'Prop', 'Line', 'Side', 'Odds', 'Book']
                display['Prop'] = display['Prop'].str.upper()

                st.dataframe(display.sort_values(['Player', 'Prop', 'Side']),
                           hide_index=True, use_container_width=True, height=500)

# =============================================================================
# TAB 3: PLAYER LOOKUP
# =============================================================================
with tab3:
    st.subheader("Player Stats Lookup")

    player = st.text_input("Player Name", "Luka Doncic")

    if st.button("Look Up"):
        with st.spinner("Fetching..."):
            logs = get_player_stats(player, num_games)

        if logs.empty:
            st.error(f"Could not find {player}")
        else:
            st.success(f"Last {len(logs)} games")

            # Averages
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Points", f"{logs['points'].mean():.1f}")
            col2.metric("Rebounds", f"{logs['rebounds'].mean():.1f}")
            col3.metric("Assists", f"{logs['assists'].mean():.1f}")
            col4.metric("PRA", f"{logs['pra'].mean():.1f}")

            # Quick check
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                check_prop = st.selectbox("Prop", ["points", "rebounds", "assists", "pra"])
            with col2:
                check_line = st.number_input("Line", value=round(logs[check_prop].mean(), 1), step=0.5)

            over_rate = (logs[check_prop] > check_line).mean() * 100
            under_rate = (logs[check_prop] < check_line).mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg", f"{logs[check_prop].mean():.1f}")
            col2.metric("Over Rate", f"{over_rate:.0f}%", delta=f"{over_rate-50:.0f}%" if over_rate != 50 else None)
            col3.metric("Under Rate", f"{under_rate:.0f}%", delta=f"{under_rate-50:.0f}%" if under_rate != 50 else None)

            # Game log
            st.divider()
            cols = ['date', 'matchup', 'points', 'rebounds', 'assists', 'pra', 'result']
            available = [c for c in cols if c in logs.columns]
            st.dataframe(logs[available], hide_index=True, use_container_width=True)

# Footer
st.divider()
st.caption(f"API: {odds_client.remaining_requests} calls left" if odds_client.remaining_requests else "")
