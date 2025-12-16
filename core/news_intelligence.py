"""Real-time news intelligence for NBA prop analysis.

This module fetches and parses news from beat writers, official reports,
and news sites to inform prop betting analysis with late-breaking context.

Perplexity Integration:
    The module can use Perplexity MCP for real-time news intelligence.
    Pass a Perplexity wrapper function to NewsIntelligence constructor:

        from core.news_intelligence import NewsIntelligence, create_perplexity_search

        # If you have access to the MCP function
        search_fn = create_perplexity_search(mcp_perplexity_ask_fn)
        news_intel = NewsIntelligence(search_fn=search_fn)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


def create_perplexity_search(perplexity_fn: Callable) -> Callable[[str], str]:
    """
    Create a search function wrapper for Perplexity MCP.

    Args:
        perplexity_fn: The MCP function (mcp__perplexity-ask__perplexity_ask)
                      Signature: perplexity_fn(messages: List[Dict]) -> response

    Returns:
        Callable that takes a query string and returns search results as text

    Usage:
        # With MCP function available:
        search_fn = create_perplexity_search(mcp__perplexity_ask__perplexity_ask)
        news_intel = NewsIntelligence(search_fn=search_fn)
    """
    def search(query: str) -> str:
        """Execute search via Perplexity."""
        try:
            messages = [{"role": "user", "content": query}]
            response = perplexity_fn(messages)

            # Handle different response formats
            if isinstance(response, dict):
                return response.get('content', str(response))
            return str(response) if response else ""

        except Exception as e:
            logger.warning(f"Perplexity search failed for '{query}': {e}")
            return ""

    return search


# Source trust hierarchy - beat writers > official reports (more accurate close to game time)
SOURCE_CONFIDENCE = {
    # Beat writers (trust most)
    'shams': 0.95,
    'charania': 0.95,
    'woj': 0.95,
    'wojnarowski': 0.95,
    'theathletic': 0.90,
    'the athletic': 0.90,
    'espn': 0.85,
    'bleacherreport': 0.80,
    'bleacher report': 0.80,

    # Official sources (trust less close to game time - often delayed)
    'nba.com': 0.75,
    'official': 0.70,

    # General sports news
    'cbssports': 0.70,
    'cbs sports': 0.70,
    'rotowire': 0.70,
    'yahoo': 0.65,
    'fantasylabs': 0.65,

    # Social/unverified
    'twitter': 0.50,
    'x.com': 0.50,
    'reddit': 0.30,
}


@dataclass
class NewsContext:
    """Parsed news context for a player or team."""
    status: str  # 'CONFIRMED_OUT', 'LIKELY_OUT', 'GTD_LEANING_OUT', 'GTD_LEANING_PLAY', 'GTD_UNCERTAIN', 'HEALTHY', 'RETURNING', 'LOAD_MANAGEMENT'
    adjustment_factor: float  # Multiplier (0.85 = -15%, 1.10 = +10%)
    confidence: float  # 0-1 how reliable the source
    flags: List[str] = field(default_factory=list)  # Display flags
    notes: List[str] = field(default_factory=list)  # Detailed context
    sources: List[str] = field(default_factory=list)  # URLs/source names
    minutes_impact: Optional[str] = None  # 'REDUCED', 'NORMAL', 'INCREASED'
    timestamp: datetime = field(default_factory=datetime.now)

    def should_skip(self) -> bool:
        """Return True if this player should be skipped entirely."""
        return self.status in ('CONFIRMED_OUT', 'LOAD_MANAGEMENT') or self.adjustment_factor == 0.0


class NewsIntelligence:
    """Fetches and parses real-time news for NBA prop analysis.

    Uses web search to gather injury reports, beat writer updates,
    and contextual news that affects player projections.
    """

    def __init__(self, search_fn: Callable = None):
        """Initialize with optional search function.

        Args:
            search_fn: Callable that takes a query string and returns search results.
                      If None, searches will return empty results (for testing).
        """
        self._search_fn = search_fn
        self._cache: Dict[str, NewsContext] = {}
        self._cache_ttl = 1800  # 30 minutes
        self._cache_timestamps: Dict[str, datetime] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached result is still valid."""
        if key not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return age < self._cache_ttl

    def _search(self, query: str) -> str:
        """Execute search query and return results as text."""
        if self._search_fn is None:
            logger.debug(f"No search function configured, skipping: {query}")
            return ""

        try:
            result = self._search_fn(query)
            return str(result) if result else ""
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return ""

    def fetch_game_news(self, home_team: str, away_team: str,
                        game_date: str = None) -> Dict[str, NewsContext]:
        """Fetch news for both teams in a game.

        Args:
            home_team: Home team abbreviation (e.g., 'NYK')
            away_team: Away team abbreviation (e.g., 'SAS')
            game_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dict mapping team abbreviation to NewsContext
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')

        results = {}

        for team in [home_team, away_team]:
            cache_key = f"team_{team}_{game_date}"

            if self._is_cache_valid(cache_key):
                results[team] = self._cache[cache_key]
                continue

            # Search for team-level news
            query = f"{team} NBA injury report lineup {game_date}"
            search_results = self._search(query)

            context = self._parse_team_news(team, search_results)

            # Cache result
            self._cache[cache_key] = context
            self._cache_timestamps[cache_key] = datetime.now()
            results[team] = context

        return results

    def fetch_player_news(self, player_name: str, team: str = None,
                          game_date: str = None) -> NewsContext:
        """Fetch player-specific news.

        Args:
            player_name: Player's full name
            team: Team abbreviation (optional, for context)
            game_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            NewsContext with parsed player status and adjustments
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')

        cache_key = f"player_{player_name}_{game_date}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Build search queries prioritizing injury/status info
        queries = [
            f"{player_name} NBA injury status today",
            f"{player_name} questionable GTD game time decision",
        ]

        all_results = []
        for query in queries:
            result = self._search(query)
            if result:
                all_results.append(result)

        combined_text = '\n'.join(all_results)
        context = self._parse_player_news(player_name, combined_text)

        # Cache result
        self._cache[cache_key] = context
        self._cache_timestamps[cache_key] = datetime.now()

        return context

    def _parse_team_news(self, team: str, text: str) -> NewsContext:
        """Parse team-level news for injuries and lineup changes."""
        if not text:
            return NewsContext(
                status='NO_NEWS',
                adjustment_factor=1.0,
                confidence=0.5,
                flags=[],
                notes=['No team news found']
            )

        text_lower = text.lower()
        flags = []
        notes = []

        # Check for multiple players out
        out_count = len(re.findall(r'\b(out|ruled out|will not play|sidelined)\b', text_lower))
        if out_count >= 3:
            flags.append(f'🏥 {out_count}+ players out')
            notes.append(f'Multiple injuries reported for {team}')

        # Check for star player news
        star_patterns = [
            r'(star|all-star|superstar).*(out|injured|questionable)',
            r'(out|injured|questionable).*(star|all-star)',
        ]
        for pattern in star_patterns:
            if re.search(pattern, text_lower):
                flags.append('⭐ Star player news')
                break

        # Extract source confidence
        confidence = self._extract_source_confidence(text_lower)

        return NewsContext(
            status='TEAM_NEWS',
            adjustment_factor=1.0,  # Team news doesn't directly adjust
            confidence=confidence,
            flags=flags,
            notes=notes,
            sources=self._extract_sources(text_lower)
        )

    def _parse_player_news(self, player_name: str, text: str) -> NewsContext:
        """Parse player-specific news into structured context."""
        if not text:
            return NewsContext(
                status='NO_NEWS',
                adjustment_factor=1.0,
                confidence=0.5,
                flags=[],
                notes=['No player news found']
            )

        text_lower = text.lower()
        player_lower = player_name.lower()

        # Initialize defaults
        status = 'HEALTHY'
        adjustment = 1.0
        flags = []
        notes = []
        minutes_impact = 'NORMAL'

        # Check for CONFIRMED OUT indicators
        out_patterns = [
            r'ruled out',
            r'will not play',
            r'sidelined',
            r'out tonight',
            r'out today',
            r'expected to miss',
            r'will miss',
            r'not in lineup',
            r'has been ruled out',
            r'is out',
            r'won\'t play',
            r'not playing',
        ]

        for pattern in out_patterns:
            if re.search(pattern, text_lower):
                status = 'CONFIRMED_OUT'
                adjustment = 0.0  # Skip entirely
                flags.append('❌ OUT')
                notes.append('Player ruled out')
                break

        # Check for GTD/questionable (only if not already OUT)
        if status == 'HEALTHY':
            gtd_patterns = [
                r'game.?time decision',
                r'\bgtd\b',
                r'questionable',
                r'doubtful',
                r'uncertain',
                r'day.?to.?day',
            ]

            is_gtd = any(re.search(p, text_lower) for p in gtd_patterns)

            if is_gtd:
                # Determine lean based on beat writer language
                play_indicators = [
                    r'expected to play',
                    r'likely to play',
                    r'leaning towards playing',
                    r'should play',
                    r'plans to play',
                    r'intends to play',
                    r'good chance',
                    r'optimistic',
                ]

                out_indicators = [
                    r'unlikely to play',
                    r'leaning towards sitting',
                    r'expected to sit',
                    r'probably won\'t',
                    r'probably out',
                    r'not expected to play',
                    r'doubtful',
                ]

                leaning_play = any(re.search(p, text_lower) for p in play_indicators)
                leaning_out = any(re.search(p, text_lower) for p in out_indicators)

                if leaning_play and not leaning_out:
                    status = 'GTD_LEANING_PLAY'
                    adjustment = 0.95  # -5%
                    flags.append('⚠️ GTD (likely playing)')
                    notes.append('Game-time decision, expected to play')
                elif leaning_out:
                    status = 'GTD_LEANING_OUT'
                    adjustment = 0.88  # -12%
                    flags.append('⚠️ GTD (likely OUT)')
                    notes.append('Game-time decision, leaning towards sitting')
                else:
                    status = 'GTD_UNCERTAIN'
                    adjustment = 0.92  # -8%
                    flags.append('⚠️ GTD')
                    notes.append('Game-time decision, status uncertain')

        # Check for minutes restriction (compounds with other adjustments)
        minutes_patterns = [
            r'minutes restriction',
            r'minutes limit',
            r'on a minutes? limit',
            r'reduced minutes',
            r'limited minutes',
            r'minute.?restriction',
            r'load management',
        ]

        if any(re.search(p, text_lower) for p in minutes_patterns):
            if 'load management' in text_lower and status == 'HEALTHY':
                status = 'LOAD_MANAGEMENT'
                adjustment = 0.0  # Skip
                flags.append('💤 REST/LOAD MGMT')
                notes.append('Load management - likely resting')
            else:
                adjustment *= 0.88  # Additional -12%
                minutes_impact = 'REDUCED'
                flags.append('⏱️ MINS LIMIT')
                notes.append('Minutes restriction reported')

        # Check for return from injury
        return_patterns = [
            r'return from injury',
            r'back from.{1,20}injury',
            r'first game back',
            r'making his return',
            r'cleared to play',
            r'returning from',
            r'back in the lineup',
            r'return to action',
        ]

        if any(re.search(p, text_lower) for p in return_patterns):
            if status == 'HEALTHY':
                status = 'RETURNING'
            adjustment *= 0.92  # -8% for rust
            flags.append('🔄 RETURNING')
            notes.append('Returning from injury - rust factor applied')

        # Check for positive indicators (star teammate out = usage boost)
        if status == 'HEALTHY':
            boost_patterns = [
                r'with.{1,30}out.{1,30}(star|all-star)',
                r'without.{1,30}(star|all-star)',
                r'in absence of',
                r'while.{1,30}sidelined',
                r'increased usage',
                r'usage boost',
                r'more touches',
                r'bigger role',
            ]

            if any(re.search(p, text_lower) for p in boost_patterns):
                adjustment *= 1.08  # +8% usage boost
                flags.append('📈 USAGE BOOST')
                notes.append('Teammate out - potential usage increase')

        # Extract source confidence
        confidence = self._extract_source_confidence(text_lower)

        # Reduce confidence for uncertain statuses
        if status in ('GTD_UNCERTAIN', 'GTD_LEANING_OUT'):
            confidence *= 0.85
        elif status == 'GTD_LEANING_PLAY':
            confidence *= 0.90

        return NewsContext(
            status=status,
            adjustment_factor=adjustment,
            confidence=confidence,
            flags=flags,
            notes=notes,
            sources=self._extract_sources(text_lower),
            minutes_impact=minutes_impact
        )

    def _extract_source_confidence(self, text: str) -> float:
        """Extract confidence level based on sources mentioned."""
        confidence = 0.6  # Default baseline

        for source, conf in SOURCE_CONFIDENCE.items():
            if source in text:
                confidence = max(confidence, conf)

        return confidence

    def _extract_sources(self, text: str) -> List[str]:
        """Extract mentioned sources from text."""
        found = []
        for source in SOURCE_CONFIDENCE.keys():
            if source in text:
                found.append(source)
        return found

    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("News intelligence cache cleared")

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        valid_count = sum(1 for k in self._cache if self._is_cache_valid(k))
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_count,
            'expired_entries': len(self._cache) - valid_count,
        }
