"""
NBA Schedule Utilities
======================
Fetch and display NBA game schedules with caching.

Usage:
    from data.schedule_utils import print_schedule, get_games_for_date

    # Print next 7 days of games
    print_schedule()

    # Get games for specific date
    games = get_games_for_date("2025-12-25")
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from zoneinfo import ZoneInfo

from core.config import CONFIG


# Cache settings
CACHE_FILE = Path(__file__).parent / "nba_schedule_cache.json"
CACHE_TTL_HOURS = 4


def _get_odds_client():
    """Get OddsAPIClient with API key from config."""
    from data.fetchers import OddsAPIClient
    api_key = CONFIG.ODDS_API_KEY
    if not api_key:
        api_key = os.environ.get('ODDS_API_KEY', '')
    return OddsAPIClient(api_key=api_key)


def _load_cache() -> Optional[Dict]:
    """Load cached schedule data if valid."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

        # Check if cache is still valid
        cached_at = datetime.fromisoformat(cache.get('cached_at', ''))
        if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
            return None  # Cache expired

        return cache
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _save_cache(events: List[Dict]):
    """Save schedule data to cache."""
    cache = {
        'cached_at': datetime.now().isoformat(),
        'events': events
    }
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save schedule cache: {e}")


def get_schedule(refresh: bool = False) -> List[Dict]:
    """
    Get upcoming NBA games from cache or API.

    Args:
        refresh: Force refresh from API even if cache is valid

    Returns:
        List of game dictionaries with keys:
        - id: Event ID
        - home_team: Home team name
        - away_team: Away team name
        - commence_time: ISO timestamp of game start
    """
    # Try cache first
    if not refresh:
        cache = _load_cache()
        if cache and cache.get('events'):
            return cache['events']

    # Fetch from API
    try:
        client = _get_odds_client()
        events = client.get_events()

        if events:
            _save_cache(events)
            return events
        else:
            # Return empty list if API returns nothing
            return []

    except Exception as e:
        print(f"Error fetching schedule: {e}")
        # Try to return stale cache if available
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                return cache.get('events', [])
            except Exception:
                pass
        return []


def get_games_for_date(date_str: str) -> List[Dict]:
    """
    Get games for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of games scheduled for that date
    """
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        print(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
        return []

    events = get_schedule()
    games_on_date = []

    for event in events:
        commence_time = event.get('commence_time', '')
        if not commence_time:
            continue

        try:
            # Parse ISO timestamp and convert to ET
            game_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            et_tz = ZoneInfo('America/New_York')
            game_dt_et = game_dt.astimezone(et_tz)

            if game_dt_et.date() == target_date:
                games_on_date.append({
                    **event,
                    'game_time_et': game_dt_et.strftime('%I:%M %p').lstrip('0')
                })
        except (ValueError, TypeError):
            continue

    # Sort by game time
    games_on_date.sort(key=lambda x: x.get('commence_time', ''))
    return games_on_date


def print_schedule(days: int = 7, date_str: str = None):
    """
    Print formatted NBA schedule to console.

    Args:
        days: Number of days to show (default 7)
        date_str: Specific date to show (YYYY-MM-DD), overrides days param
    """
    if date_str:
        # Show specific date
        games = get_games_for_date(date_str)
        try:
            target = datetime.strptime(date_str, '%Y-%m-%d')
            day_name = target.strftime('%A, %b %d').upper()
        except ValueError:
            day_name = date_str

        print()
        print("=" * 60)
        print(f"  NBA SCHEDULE - {day_name}")
        print("=" * 60)
        print()

        if not games:
            print("  No games scheduled")
        else:
            for game in games:
                time_str = game.get('game_time_et', 'TBD')
                away = game.get('away_team', 'Unknown')
                home = game.get('home_team', 'Unknown')
                print(f"  {time_str:>8} ET  {away} @ {home}")

        print()
        return

    # Show multiple days
    events = get_schedule()
    if not events:
        print("\nNo schedule data available. Check your ODDS_API_KEY.\n")
        return

    # Parse and organize by date
    et_tz = ZoneInfo('America/New_York')
    today = datetime.now(et_tz).date()
    end_date = today + timedelta(days=days)

    # Group games by date
    games_by_date: Dict[str, List[Dict]] = {}

    for event in events:
        commence_time = event.get('commence_time', '')
        if not commence_time:
            continue

        try:
            game_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            game_dt_et = game_dt.astimezone(et_tz)
            game_date = game_dt_et.date()

            if today <= game_date <= end_date:
                date_key = game_date.isoformat()
                if date_key not in games_by_date:
                    games_by_date[date_key] = []
                games_by_date[date_key].append({
                    **event,
                    'game_time_et': game_dt_et.strftime('%I:%M %p').lstrip('0')
                })
        except (ValueError, TypeError):
            continue

    # Sort games within each date by time
    for date_key in games_by_date:
        games_by_date[date_key].sort(key=lambda x: x.get('commence_time', ''))

    # Print header
    if games_by_date:
        dates = sorted(games_by_date.keys())
        start = datetime.fromisoformat(dates[0]).strftime('%b %d')
        end = datetime.fromisoformat(dates[-1]).strftime('%b %d, %Y')
        header = f"NBA SCHEDULE - {start} - {end}"
    else:
        header = f"NBA SCHEDULE - Next {days} Days"

    print()
    print("=" * 60)
    print(f"  {header}")
    print("=" * 60)

    if not games_by_date:
        print("\n  No games scheduled in this period\n")
        return

    # Print each day
    for date_str in sorted(games_by_date.keys()):
        games = games_by_date[date_str]
        date_obj = datetime.fromisoformat(date_str)
        day_name = date_obj.strftime('%A, %b %d').upper()

        print()
        print(f"{day_name}")
        print("-" * len(day_name))

        for game in games:
            time_str = game.get('game_time_et', 'TBD')
            away = game.get('away_team', 'Unknown')
            home = game.get('home_team', 'Unknown')
            print(f"  {time_str:>8} ET  {away} @ {home}")

    print()


def get_todays_games() -> List[Dict]:
    """Get today's NBA games."""
    today = datetime.now().strftime('%Y-%m-%d')
    return get_games_for_date(today)


def print_todays_games():
    """Print today's games to console."""
    today = datetime.now().strftime('%Y-%m-%d')
    print_schedule(date_str=today)


# Key season dates for 2025-26
SEASON_2025_26 = {
    'regular_season_start': '2025-10-21',
    'regular_season_end': '2026-04-12',
    'nba_cup_start': '2025-10-31',
    'nba_cup_end': '2025-12-16',
    'all_star_game': '2026-02-15',
    'play_in_start': '2026-04-14',
    'play_in_end': '2026-04-17',
    'playoffs_start': '2026-04-18',
    'finals_start': '2026-06-04',
    'finals_game7': '2026-06-21',
}


def print_key_dates():
    """Print key 2025-26 season dates."""
    print()
    print("=" * 60)
    print("  NBA 2025-26 KEY DATES")
    print("=" * 60)
    print()
    print(f"  Regular Season:    Oct 21, 2025 - Apr 12, 2026")
    print(f"  NBA Cup:           Oct 31 - Dec 16, 2025")
    print(f"  All-Star Game:     Feb 15, 2026 (Inglewood, CA)")
    print(f"  Play-In:           Apr 14-17, 2026")
    print(f"  Playoffs:          Apr 18, 2026")
    print(f"  NBA Finals:        Jun 4-21, 2026")
    print()


if __name__ == '__main__':
    # When run directly, print upcoming schedule
    print_schedule(days=7)
