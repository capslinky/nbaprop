"""Injury tracking for NBA prop analysis.

This module provides the InjuryTracker class for fetching and tracking
player injuries from multiple sources, and calculating usage boosts
when star teammates are out.
"""

import io
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
    TEAM_ABBREVIATIONS,
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

        # Team name to abbreviation mapping
        self._team_name_map = {
            'atlanta hawks': 'ATL', 'hawks': 'ATL',
            'boston celtics': 'BOS', 'celtics': 'BOS',
            'brooklyn nets': 'BKN', 'nets': 'BKN',
            'charlotte hornets': 'CHA', 'hornets': 'CHA',
            'chicago bulls': 'CHI', 'bulls': 'CHI',
            'cleveland cavaliers': 'CLE', 'cavaliers': 'CLE', 'cavs': 'CLE',
            'dallas mavericks': 'DAL', 'mavericks': 'DAL', 'mavs': 'DAL',
            'denver nuggets': 'DEN', 'nuggets': 'DEN',
            'detroit pistons': 'DET', 'pistons': 'DET',
            'golden state warriors': 'GSW', 'warriors': 'GSW',
            'houston rockets': 'HOU', 'rockets': 'HOU',
            'indiana pacers': 'IND', 'pacers': 'IND',
            'la clippers': 'LAC', 'los angeles clippers': 'LAC', 'clippers': 'LAC',
            'los angeles lakers': 'LAL', 'la lakers': 'LAL', 'lakers': 'LAL',
            'memphis grizzlies': 'MEM', 'grizzlies': 'MEM',
            'miami heat': 'MIA', 'heat': 'MIA',
            'milwaukee bucks': 'MIL', 'bucks': 'MIL',
            'minnesota timberwolves': 'MIN', 'timberwolves': 'MIN', 'wolves': 'MIN',
            'new orleans pelicans': 'NOP', 'pelicans': 'NOP',
            'new york knicks': 'NYK', 'knicks': 'NYK',
            'oklahoma city thunder': 'OKC', 'thunder': 'OKC',
            'orlando magic': 'ORL', 'magic': 'ORL',
            'philadelphia 76ers': 'PHI', '76ers': 'PHI', 'sixers': 'PHI',
            'phoenix suns': 'PHX', 'suns': 'PHX',
            'portland trail blazers': 'POR', 'trail blazers': 'POR', 'blazers': 'POR',
            'sacramento kings': 'SAC', 'kings': 'SAC',
            'san antonio spurs': 'SAS', 'spurs': 'SAS',
            'toronto raptors': 'TOR', 'raptors': 'TOR',
            'utah jazz': 'UTA', 'jazz': 'UTA',
            'washington wizards': 'WAS', 'wizards': 'WAS',
        }

    def _normalize_team_name(self, team_str: str) -> str:
        """Convert full team name to 3-letter abbreviation."""
        if not team_str:
            return 'UNK'

        team_lower = team_str.lower().strip()

        # Already an abbreviation
        if len(team_lower) == 3 and team_lower.isalpha():
            return team_lower.upper()

        # Look up in mapping
        if team_lower in self._team_name_map:
            return self._team_name_map[team_lower]

        # Try partial match
        for name, abbrev in self._team_name_map.items():
            if name in team_lower or team_lower in name:
                return abbrev

        return team_str.upper()[:3] if team_str else 'UNK'

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

            # 1) Try to locate the latest official NBA.com injury report PDF
            url_query = (
                f"Return ONLY the latest official NBA injury report PDF URL for "
                f"{date.today().strftime('%B %d, %Y')}. "
                "The URL must be from ak-static.cms.nba.com/referee/injury/ "
                "and look like Injury-Report_YYYY-MM-DD_HHPM.pdf. "
                "If no URL is found, respond with 'NONE'."
            )
            url_response = self._perplexity_fn([{"role": "user", "content": url_query}])
            if isinstance(url_response, dict):
                url_text = url_response.get('content', str(url_response))
            else:
                url_text = str(url_response)

            report_urls = self._extract_official_report_urls(url_text)
            if report_urls:
                report_url = self._select_latest_report_url(report_urls)
                if report_url:
                    official_df = self._parse_official_report_pdf(report_url)
                    if not official_df.empty:
                        return official_df

            # 2) Fall back to Perplexity's summarized injury list
            query = (
                f"From nba.com only, list ALL NBA players on the official NBA injury report for "
                f"{date.today().strftime('%B %d, %Y')}. "
                "Use the official NBA injury report page on NBA.com as the source. "
                "Include every player who is OUT, DOUBTFUL, QUESTIONABLE, or GTD. "
                "Format EXACTLY as: PLAYER_NAME | TEAM_NAME | STATUS | INJURY (one player per line). "
                "If the official report is not published yet, respond with "
                "'No official NBA.com injury report found.' "
                "Do not ask clarifying questions - provide the complete list immediately."
            )

            messages = [{"role": "user", "content": query}]
            response = self._perplexity_fn(messages)

            if not response:
                return pd.DataFrame()

            # Normalize response text and parse directly
            if isinstance(response, dict):
                response_text = response.get('content', str(response))
            else:
                response_text = str(response)

            return self._parse_perplexity_injuries(response_text)

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
                # Remove citation markers like [1][2][3] from the end
                line = re.sub(r'\[\d+\]', '', line).strip()

                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    # Remove bold markers (**) and clean player name
                    player = parts[0].strip('* ').replace('**', '').strip()
                    team_raw = parts[1].strip()
                    status = parts[2].strip().upper()
                    injury = parts[3] if len(parts) > 3 else ''

                    # Convert full team name to abbreviation if needed
                    team = self._normalize_team_name(team_raw)

                    # Accept if we have a valid player and status
                    if player and status in ['OUT', 'O', 'DNP', 'GTD', 'GAME TIME DECISION',
                                              'QUESTIONABLE', 'Q', 'DOUBTFUL', 'D', 'PROBABLE', 'P']:
                        injuries.append({
                            'player': player,
                            'team': team,
                            'status': status,
                            'injury': injury.replace(';', ' -'),
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

    def _extract_official_report_urls(self, text: str) -> List[str]:
        """Extract official NBA injury report PDF URLs from text."""
        if not text:
            return []
        cleaned = re.sub(r"\[\d+\]", "", text)
        urls = re.findall(
            r"https?://ak-static\.cms\.nba\.com/referee/injury/"
            r"Injury-Report_\d{4}-\d{2}-\d{2}_\d{2}(?:AM|PM)\.pdf",
            cleaned
        )
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    def _select_latest_report_url(self, urls: List[str]) -> Optional[str]:
        """Pick the latest report URL based on its timestamp."""
        if not urls:
            return None

        def parse_timestamp(url: str) -> Optional[datetime]:
            match = re.search(
                r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(AM|PM)\.pdf",
                url
            )
            if not match:
                return None
            date_str, hour_str, meridiem = match.groups()
            try:
                return datetime.strptime(f"{date_str} {hour_str}{meridiem}", "%Y-%m-%d %I%p")
            except ValueError:
                return None

        latest_url = None
        latest_ts = None
        for url in urls:
            ts = parse_timestamp(url)
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts
                latest_url = url
        return latest_url or urls[0]

    def _parse_official_report_pdf(self, url: str) -> pd.DataFrame:
        """Parse the official NBA injury report PDF into a DataFrame."""
        try:
            from PyPDF2 import PdfReader
        except Exception as e:
            logger.warning(f"PyPDF2 not available for injury PDF parsing: {e}")
            return pd.DataFrame()

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Official injury report download failed: {e}")
            return pd.DataFrame()

        try:
            reader = PdfReader(io.BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Failed to read injury report PDF: {e}")
            return pd.DataFrame()

        status_tokens = {"Out", "Questionable", "Doubtful", "Probable", "GTD", "Available"}
        suffix_tokens = {"Jr.,", "Sr.,", "II,", "III,", "IV,", "V,", "VI,", "Jr.", "Sr.", "II", "III", "IV"}
        team_tokens = sorted(
            [(name, name.lower().split()) for name in TEAM_ABBREVIATIONS.keys() if " " in name],
            key=lambda x: len(x[1]),
            reverse=True
        )

        def combine_hyphens(tokens: List[str]) -> List[str]:
            combined = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token.endswith("-") and len(token) > 1 and i + 1 < len(tokens):
                    combined.append(token[:-1] + "-" + tokens[i + 1])
                    i += 2
                    continue
                combined.append(token)
                i += 1
            return combined

        def merge_suffixes(tokens: List[str]) -> List[str]:
            merged = []
            for token in tokens:
                if token in suffix_tokens and merged:
                    merged[-1] = f"{merged[-1]} {token}"
                else:
                    merged.append(token)
            return merged

        def match_team(tokens: List[str], idx: int) -> Tuple[Optional[str], int]:
            for name, parts in team_tokens:
                if idx + len(parts) <= len(tokens):
                    window = [t.lower() for t in tokens[idx:idx + len(parts)]]
                    if window == parts:
                        return name, len(parts)
            return None, 0

        def is_date_token(token: str) -> bool:
            return bool(re.match(r"\d{1,2}/\d{1,2}/\d{4}", token))

        def is_time_token(token: str) -> bool:
            return bool(re.match(r"\d{1,2}:\d{2}", token))

        def is_matchup_token(token: str) -> bool:
            return bool(re.match(r"^[A-Z]{2,3}@[A-Z]{2,3}$", token))

        def normalize_reason(tokens: List[str]) -> str:
            if not tokens:
                return ""
            text = " ".join(tokens).strip()
            text = re.sub(r"\s+([;:,])", r"\1", text)
            text = re.sub(r"\s+-\s+", " - ", text)
            text = re.sub(r"\s{2,}", " ", text)
            return text

        def is_player_start(tokens: List[str], idx: int) -> bool:
            token = tokens[idx]
            if "," not in token:
                return False
            for offset in range(1, 5):
                if idx + offset >= len(tokens):
                    break
                if tokens[idx + offset] in status_tokens:
                    return True
            return False

        def extract_page_tokens(text: str) -> List[str]:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for i, line in enumerate(lines):
                if line.lower() == "reason":
                    return lines[i + 1:]
            if len(lines) > 6 and lines[0].lower() == "injury" and lines[1].lower().startswith("report"):
                if "Page" in lines:
                    page_idx = lines.index("Page")
                    return lines[min(page_idx + 4, len(lines)):]
            return lines

        tokens = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            tokens.extend(extract_page_tokens(page_text))

        tokens = combine_hyphens(tokens)
        tokens = merge_suffixes(tokens)

        rows = []
        current_team = None
        player_tokens = []
        status = None
        reason_tokens = []

        def flush_row() -> None:
            nonlocal player_tokens, status, reason_tokens
            if not player_tokens or not status:
                player_tokens = []
                status = None
                reason_tokens = []
                return
            player_raw = " ".join(player_tokens)
            if "," in player_raw:
                last, rest = player_raw.split(",", 1)
                player_name = f"{rest.strip()} {last.strip()}".strip()
            else:
                player_name = player_raw.strip()
            rows.append({
                "player": player_name,
                "team": self._normalize_team_name(current_team or ""),
                "status": status.upper(),
                "injury": normalize_reason(reason_tokens),
                "source": "NBA Official Report",
            })
            player_tokens = []
            status = None
            reason_tokens = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in {"Page", "of"}:
                i += 1
                continue
            if token in {"AM", "PM", "(ET)"}:
                i += 1
                continue
            if is_date_token(token) or is_time_token(token) or is_matchup_token(token):
                i += 1
                continue

            team_name, team_len = match_team(tokens, i)
            if team_name:
                if status:
                    flush_row()
                current_team = team_name
                i += team_len
                continue

            if status is None:
                if token in status_tokens:
                    status = token
                else:
                    player_tokens.append(token)
                i += 1
                continue

            if is_player_start(tokens, i):
                flush_row()
                player_tokens.append(token)
                i += 1
                continue

            reason_tokens.append(token)
            i += 1

        flush_row()

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_all_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get combined injury data from all sources.
        Caches results for 30 minutes.

        Priority order (lower = higher priority):
            0. Manual overrides
            1. NBA Official Report (nba.com PDF)
            2. Perplexity AI (real-time breaking news)
            3. NBA API (official injury reports)
            4. CBS Sports (web scraping)
            5. Rotowire (web scraping fallback)
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
                'NBA Official Report': 1,
                'Perplexity': 2,
                'NBA API': 3,
                'CBS Sports': 4,
                'Rotowire': 5
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
