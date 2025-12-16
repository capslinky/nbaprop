"""Injury tracking for NBA prop analysis.

This module provides the InjuryTracker class for fetching and tracking
player injuries from multiple sources, and calculating usage boosts
when star teammates are out.
"""

import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple
import time
import logging
import re
import requests

from core.constants import (
    STAR_PLAYERS,
    STAR_OUT_BOOST,
    get_current_nba_season,
)

logger = logging.getLogger(__name__)


class InjuryTracker:
    """
    Tracks NBA injuries and lineup changes from multiple sources:
    1. NBA API (official injury reports)
    2. CBS Sports / Rotowire (web scraping)
    3. Manual overrides for late-breaking news

    Usage:
        tracker = InjuryTracker()
        injuries = tracker.get_all_injuries()
        status = tracker.get_player_status("LeBron James")
        boost = tracker.get_teammate_boost("Anthony Davis", "LAL")
    """

    # Use imported constants from core.constants (single source of truth)
    # These are class attributes for backward compatibility
    STAR_PLAYERS = STAR_PLAYERS
    STAR_OUT_BOOST = STAR_OUT_BOOST

    def __init__(self, perplexity_fn=None):
        """Initialize InjuryTracker.

        Args:
            perplexity_fn: Optional callable for Perplexity MCP queries.
                          Signature: perplexity_fn(messages: List[Dict]) -> str
                          If None, Perplexity source is skipped.
        """
        self._injury_cache = {}
        self._cache_time = None
        self._cache_ttl = 1800  # 30 minutes cache
        self._manual_injuries = {}  # Manual overrides
        self._perplexity_fn = perplexity_fn

    def get_injuries_from_nba_api(self) -> pd.DataFrame:
        """Fetch injury data from NBA API."""
        try:
            from nba_api.stats.endpoints import playerindex

            # Get all players with their current status
            time.sleep(0.6)  # Rate limit
            players = playerindex.PlayerIndex(season=get_current_nba_season())
            df = players.get_data_frames()[0]

            # Filter to players with injury info if available
            if 'INJURY_STATUS' in df.columns:
                injured = df[df['INJURY_STATUS'].notna() & (df['INJURY_STATUS'] != '')]
                return injured[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'INJURY_STATUS', 'INJURY_DESCRIPTION']].copy()

            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"NBA API injury fetch error: {e}")
            return pd.DataFrame()

    def get_injuries_from_rotowire(self) -> pd.DataFrame:
        """
        Scrape injury data from Rotowire/CBS Sports.
        Falls back to CBS if Rotowire fails.
        """
        injuries = []

        # Try CBS Sports NBA injuries page
        try:
            url = "https://www.cbssports.com/nba/injuries/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                # Simple parsing - look for injury patterns
                content = response.text

                # Parse the page for injury info
                # CBS format typically has player name, team, status, injury
                # This is a simplified parser

                # Look for injury table patterns
                # Format: Player Name | Team | Status | Injury Type
                team_sections = re.findall(r'class="TeamName[^"]*"[^>]*>([^<]+)</a>', content)
                player_patterns = re.findall(
                    r'class="CellPlayerName[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>.*?'
                    r'class="[^"]*injury[^"]*"[^>]*>([^<]+)<',
                    content, re.DOTALL | re.IGNORECASE
                )

                for player, status in player_patterns:
                    injuries.append({
                        'player': player.strip(),
                        'status': status.strip().upper(),
                        'source': 'CBS Sports'
                    })

        except Exception as e:
            logger.warning(f"CBS Sports scrape error: {e}")

        # Try Rotowire as backup
        if not injuries:
            try:
                url = "https://www.rotowire.com/basketball/injury-report.php"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    content = response.text

                    # Look for injury entries
                    # Rotowire format: Player - Team - Status - Injury
                    matches = re.findall(
                        r'<a[^>]*player[^>]*>([^<]+)</a>.*?'
                        r'<span[^>]*team[^>]*>([^<]+)</span>.*?'
                        r'<span[^>]*status[^>]*>([^<]+)</span>',
                        content, re.DOTALL | re.IGNORECASE
                    )

                    for player, team, status in matches:
                        injuries.append({
                            'player': player.strip(),
                            'team': team.strip(),
                            'status': status.strip().upper(),
                            'source': 'Rotowire'
                        })

            except Exception as e:
                logger.warning(f"Rotowire scrape error: {e}")

        return pd.DataFrame(injuries) if injuries else pd.DataFrame()

    def get_injuries_from_perplexity(self) -> pd.DataFrame:
        """
        Fetch breaking injury news via Perplexity AI.
        Higher priority than web scraping for real-time updates.

        Returns:
            DataFrame with columns: player, team, status, injury, source
        """
        if self._perplexity_fn is None:
            logger.debug("Perplexity not configured, skipping")
            return pd.DataFrame()

        try:
            from datetime import date

            query = (
                f"NBA injury report for games on {date.today().strftime('%B %d, %Y')}. "
                "List all players who are OUT, DOUBTFUL, QUESTIONABLE, or GTD (game-time decision). "
                "Format each player as: PLAYER_NAME | TEAM_ABBREV | STATUS | INJURY_TYPE"
            )

            messages = [{"role": "user", "content": query}]
            response = self._perplexity_fn(messages)

            if not response:
                return pd.DataFrame()

            # Parse the response text
            return self._parse_perplexity_injuries(response)

        except Exception as e:
            logger.warning(f"Perplexity injury fetch error: {e}")
            return pd.DataFrame()

    def _parse_perplexity_injuries(self, response: str) -> pd.DataFrame:
        """
        Parse Perplexity response into injury DataFrame.

        Args:
            response: Text response from Perplexity

        Returns:
            DataFrame with injury records
        """
        injuries = []

        # Handle dict response (MCP format)
        if isinstance(response, dict):
            response = response.get('content', str(response))

        # Normalize to string
        text = str(response)

        # Parse lines looking for injury format: PLAYER | TEAM | STATUS | INJURY
        lines = text.split('\n')
        for line in lines:
            # Skip empty lines and headers
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue

            # Try pipe-delimited format first
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    player = parts[0].strip('* ')
                    team = parts[1].strip().upper()
                    status = parts[2].strip().upper()
                    injury = parts[3] if len(parts) > 3 else ''

                    # Validate team abbreviation (3 letters)
                    if len(team) == 3 and team.isalpha():
                        injuries.append({
                            'player': player,
                            'team': team,
                            'status': status,
                            'injury': injury,
                            'source': 'Perplexity'
                        })
                continue

            # Try to extract from natural language patterns

            # Pattern: "Player Name (TEAM) is OUT/QUESTIONABLE..."
            match = re.match(
                r'^([A-Za-z\.\'\-\s]+?)\s*\(([A-Z]{3})\)\s*(?:is|-)?\s*(OUT|QUESTIONABLE|DOUBTFUL|GTD|PROBABLE)',
                line, re.IGNORECASE
            )
            if match:
                injuries.append({
                    'player': match.group(1).strip(),
                    'team': match.group(2).upper(),
                    'status': match.group(3).upper(),
                    'injury': '',
                    'source': 'Perplexity'
                })
                continue

            # Pattern: "TEAM: Player Name - STATUS (injury)"
            match = re.match(
                r'^([A-Z]{3}):\s*([A-Za-z\.\'\-\s]+?)\s*-\s*(OUT|QUESTIONABLE|DOUBTFUL|GTD|PROBABLE)',
                line, re.IGNORECASE
            )
            if match:
                injuries.append({
                    'player': match.group(2).strip(),
                    'team': match.group(1).upper(),
                    'status': match.group(3).upper(),
                    'injury': '',
                    'source': 'Perplexity'
                })

        if injuries:
            logger.info(f"Perplexity returned {len(injuries)} injury records")

        return pd.DataFrame(injuries) if injuries else pd.DataFrame()

    def get_all_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get combined injury data from all sources.
        Caches results for 30 minutes.

        Priority order (lower = higher priority):
            0. Manual overrides
            1. Perplexity AI (real-time breaking news)
            2. NBA API (official injury reports)
            3. CBS Sports (web scraping)
            4. Rotowire (web scraping fallback)
        """
        now = datetime.now()

        # Check cache
        if not force_refresh and self._cache_time:
            if (now - self._cache_time).total_seconds() < self._cache_ttl:
                return self._injury_cache.get('all', pd.DataFrame())

        # Fetch from all sources
        all_injuries = []

        # 1. Perplexity AI (real-time, highest priority after manual)
        perplexity_injuries = self.get_injuries_from_perplexity()
        if not perplexity_injuries.empty:
            all_injuries.append(perplexity_injuries)

        # 2. NBA API
        nba_injuries = self.get_injuries_from_nba_api()
        if not nba_injuries.empty:
            nba_injuries['source'] = 'NBA API'
            all_injuries.append(nba_injuries)

        # 3. Web scraping (CBS Sports + Rotowire)
        web_injuries = self.get_injuries_from_rotowire()
        if not web_injuries.empty:
            all_injuries.append(web_injuries)

        # 4. Manual overrides (always included, highest priority)
        if self._manual_injuries:
            manual_df = pd.DataFrame(list(self._manual_injuries.values()))
            manual_df['source'] = 'Manual'
            all_injuries.append(manual_df)

        # Combine and deduplicate
        if all_injuries:
            combined = pd.concat(all_injuries, ignore_index=True)
            # Priority: Manual > Perplexity > NBA API > CBS Sports > Rotowire
            combined['priority'] = combined['source'].map({
                'Manual': 0,
                'Perplexity': 1,
                'NBA API': 2,
                'CBS Sports': 3,
                'Rotowire': 4
            })
            combined = combined.sort_values('priority').drop_duplicates(subset=['player'], keep='first')
            combined = combined.drop(columns=['priority'])

            self._injury_cache['all'] = combined
            self._cache_time = now
            return combined

        return pd.DataFrame()

    def get_player_status(self, player_name: str) -> dict:
        """
        Get injury status for a specific player.

        Returns:
            dict with keys: status, is_out, is_gtd, is_questionable, source
            status: 'HEALTHY', 'OUT', 'GTD', 'QUESTIONABLE', 'PROBABLE', 'DOUBTFUL'
        """
        injuries = self.get_all_injuries()

        if injuries.empty:
            return {'status': 'HEALTHY', 'is_out': False, 'is_gtd': False,
                    'is_questionable': False, 'source': None}

        # Find player (case-insensitive partial match)
        player_lower = player_name.lower()
        mask = injuries['player'].str.lower().str.contains(player_lower, regex=False)

        if mask.any():
            row = injuries[mask].iloc[0]
            status = row.get('status', '').upper()

            return {
                'status': status,
                'is_out': status in ['OUT', 'O', 'DNP'],
                'is_gtd': status in ['GTD', 'GAME TIME DECISION', 'GAME-TIME DECISION'],
                'is_questionable': status in ['QUESTIONABLE', 'Q', 'DOUBTFUL', 'D'],
                'is_probable': status in ['PROBABLE', 'P'],
                'injury': row.get('injury', row.get('INJURY_DESCRIPTION', '')),
                'source': row.get('source', 'Unknown')
            }

        return {'status': 'HEALTHY', 'is_out': False, 'is_gtd': False,
                'is_questionable': False, 'source': None}

    def set_manual_injury(self, player_name: str, team: str, status: str,
                          injury: str = '') -> None:
        """
        Manually set a player's injury status (for late-breaking news).

        Args:
            player_name: Full player name
            team: Team abbreviation (e.g., 'LAL')
            status: 'OUT', 'GTD', 'QUESTIONABLE', 'PROBABLE', 'HEALTHY'
            injury: Optional injury description
        """
        self._manual_injuries[player_name] = {
            'player': player_name,
            'team': team,
            'status': status.upper(),
            'injury': injury
        }
        # Invalidate cache
        self._cache_time = None

    def clear_manual_injuries(self) -> None:
        """Clear all manual injury overrides."""
        self._manual_injuries = {}
        self._cache_time = None

    def get_team_injuries(self, team_abbrev: str) -> pd.DataFrame:
        """Get all injured players for a specific team."""
        injuries = self.get_all_injuries()

        if injuries.empty or 'team' not in injuries.columns:
            return pd.DataFrame()

        team_upper = team_abbrev.upper()
        return injuries[injuries['team'].str.upper() == team_upper]

    def get_stars_out(self, team_abbrev: str) -> List[str]:
        """Get list of star players who are OUT for a team."""
        team_upper = team_abbrev.upper()
        stars = self.STAR_PLAYERS.get(team_upper, [])

        stars_out = []
        for star in stars:
            status = self.get_player_status(star)
            if status['is_out']:
                stars_out.append(star)

        return stars_out

    def get_teammate_boost(self, player_name: str, team_abbrev: str,
                           prop_type: str = 'points') -> dict:
        """
        Calculate usage/production boost when star teammates are out.

        Returns:
            dict with keys: boost_factor, stars_out, reason
        """
        team_upper = team_abbrev.upper()
        stars_out = self.get_stars_out(team_upper)

        # Remove the player themselves from stars_out
        stars_out = [s for s in stars_out if s.lower() != player_name.lower()]

        if not stars_out:
            return {
                'boost_factor': 1.0,
                'stars_out': [],
                'reason': 'No star teammates out'
            }

        # Calculate boost based on number of stars out
        base_boost = self.STAR_OUT_BOOST.get(prop_type, 1.05)

        # Compound boost for multiple stars out (diminishing returns)
        total_boost = 1.0
        for i, star in enumerate(stars_out):
            # First star gives full boost, subsequent stars give 50% of boost
            multiplier = 1.0 if i == 0 else 0.5
            star_boost = (base_boost - 1.0) * multiplier
            total_boost += star_boost

        return {
            'boost_factor': round(total_boost, 3),
            'stars_out': stars_out,
            'reason': f"{len(stars_out)} star(s) OUT: {', '.join(stars_out)}"
        }

    def should_exclude_player(self, player_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if player should be excluded from analysis (OUT status).

        Returns:
            Tuple of (should_exclude: bool, reason: Optional[str])
        """
        status = self.get_player_status(player_name)

        if status['is_out']:
            return True, f"Player is OUT ({status.get('injury', 'injury')})"

        return False, None

    def get_injury_adjustment(self, player_name: str, team_abbrev: str,
                               prop_type: str = 'points') -> dict:
        """
        Get complete injury-related adjustments for a player.

        Returns:
            dict with keys:
                - player_status: Player's own injury status
                - exclude: Whether to exclude from analysis
                - teammate_boost: Boost from stars being out
                - flags: List of warning/info flags
        """
        player_status = self.get_player_status(player_name)
        exclude, exclude_reason = self.should_exclude_player(player_name)
        teammate_boost = self.get_teammate_boost(player_name, team_abbrev, prop_type)

        flags = []

        if player_status['is_gtd']:
            flags.append('GTD')
        elif player_status['is_questionable']:
            flags.append('QUESTIONABLE')

        if teammate_boost['stars_out']:
            flags.append(f"USAGE BOOST (+{(teammate_boost['boost_factor']-1)*100:.0f}%)")

        return {
            'player_status': player_status,
            'exclude': exclude,
            'exclude_reason': exclude_reason,
            'teammate_boost': teammate_boost,
            'total_injury_factor': teammate_boost['boost_factor'],
            'flags': flags
        }
