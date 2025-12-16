"""
Unit tests for NewsIntelligence module.

Tests the news parsing, adjustment calculation, and caching logic.
"""

import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.news_intelligence import NewsIntelligence, NewsContext, SOURCE_CONFIDENCE


class TestNewsContextDataclass:
    """Tests for the NewsContext dataclass."""

    def test_should_skip_confirmed_out(self):
        """CONFIRMED_OUT status should trigger skip."""
        context = NewsContext(
            status='CONFIRMED_OUT',
            adjustment_factor=0.0,
            confidence=0.0,
            flags=['❌ OUT'],
            notes=['Player ruled out']
        )
        assert context.should_skip() is True

    def test_should_skip_load_management(self):
        """LOAD_MANAGEMENT status should trigger skip."""
        context = NewsContext(
            status='LOAD_MANAGEMENT',
            adjustment_factor=0.0,
            confidence=0.0,
            flags=['💤 REST LIKELY'],
            notes=['Load management']
        )
        assert context.should_skip() is True

    def test_should_not_skip_healthy(self):
        """HEALTHY status should not trigger skip."""
        context = NewsContext(
            status='HEALTHY',
            adjustment_factor=1.0,
            confidence=0.7,
            flags=[],
            notes=[]
        )
        assert context.should_skip() is False

    def test_should_not_skip_gtd(self):
        """GTD status should not trigger skip (reduced but not skipped)."""
        context = NewsContext(
            status='GTD_LEANING_PLAY',
            adjustment_factor=0.95,
            confidence=0.7,
            flags=['⚠️ GTD (likely playing)'],
            notes=['Game-time decision']
        )
        assert context.should_skip() is False


class TestNewsIntelligenceInit:
    """Tests for NewsIntelligence initialization."""

    def test_init_without_search_fn(self):
        """Initialize without search function."""
        intel = NewsIntelligence()
        assert intel._search_fn is None
        assert intel._cache == {}

    def test_init_with_search_fn(self):
        """Initialize with custom search function."""
        mock_search = lambda q: "mock results"
        intel = NewsIntelligence(search_fn=mock_search)
        assert intel._search_fn is mock_search


class TestNewsParsingRuledOut:
    """Tests for parsing RULED OUT scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    @pytest.mark.parametrize("text,expected_status", [
        ("LeBron James has been ruled out for tonight's game", "CONFIRMED_OUT"),
        ("Player will not play due to injury", "CONFIRMED_OUT"),
        ("Sidelined with knee soreness", "CONFIRMED_OUT"),
        ("He's out tonight according to reports", "CONFIRMED_OUT"),
        ("Expected to miss the game", "CONFIRMED_OUT"),
        ("Will miss tonight's matchup", "CONFIRMED_OUT"),
        ("Not in lineup for tonight", "CONFIRMED_OUT"),
    ])
    def test_parse_out_indicators(self, intel, text, expected_status):
        """Various OUT phrases should return CONFIRMED_OUT."""
        context = intel._parse_player_news("Test Player", text)
        assert context.status == expected_status
        assert context.adjustment_factor == 0.0
        assert '❌ OUT' in context.flags


class TestNewsParsingGTD:
    """Tests for parsing Game-Time Decision scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    def test_parse_gtd_leaning_play(self, intel):
        """GTD with 'expected to play' should lean towards playing."""
        text = "Player is questionable but expected to play tonight"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'GTD_LEANING_PLAY'
        assert context.adjustment_factor == 0.95
        assert '⚠️ GTD (likely playing)' in context.flags

    def test_parse_gtd_leaning_out(self, intel):
        """GTD with 'unlikely to play' should lean towards sitting."""
        text = "Player is game-time decision but unlikely to play"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'GTD_LEANING_OUT'
        assert context.adjustment_factor == 0.88
        assert '⚠️ GTD (likely OUT)' in context.flags

    def test_parse_gtd_uncertain(self, intel):
        """GTD with no lean should be uncertain."""
        text = "Player is questionable for tonight's game"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'GTD_UNCERTAIN'
        assert context.adjustment_factor == 0.92
        assert '⚠️ GTD' in context.flags

    def test_parse_doubtful(self, intel):
        """Doubtful should be GTD leaning out."""
        text = "Player is doubtful for tonight"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'GTD_LEANING_OUT'
        assert context.adjustment_factor == 0.88


class TestNewsParsingMinutesRestriction:
    """Tests for parsing minutes restriction scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    @pytest.mark.parametrize("text", [
        "Player will be on a minutes restriction tonight",
        "Limited minutes expected",
        "On a minutes limit per coach",
        "Will play on reduced minutes",
    ])
    def test_parse_minutes_restriction(self, intel, text):
        """Minutes restriction should reduce adjustment by 12%."""
        context = intel._parse_player_news("Test Player", text)
        # If healthy + minutes restriction: 1.0 * 0.88 = 0.88
        assert context.adjustment_factor == pytest.approx(0.88, rel=0.01)
        assert '⏱️ MINS LIMIT' in context.flags
        assert context.minutes_impact == 'REDUCED'

    def test_parse_gtd_plus_minutes_restriction(self, intel):
        """GTD + minutes restriction should compound adjustments."""
        text = "Player is questionable and if he plays will be on minutes restriction"
        context = intel._parse_player_news("Test Player", text)
        # GTD uncertain (0.92) * minutes restriction (0.88) = ~0.81
        assert context.adjustment_factor == pytest.approx(0.92 * 0.88, rel=0.01)
        assert '⚠️ GTD' in context.flags
        assert '⏱️ MINS LIMIT' in context.flags


class TestNewsParsingReturnFromInjury:
    """Tests for parsing return from injury scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    @pytest.mark.parametrize("text", [
        "Player making his return from injury tonight",
        "First game back from ankle sprain",
        "Cleared to play after missing 3 weeks",
        "Back from knee injury",
    ])
    def test_parse_return_from_injury(self, intel, text):
        """Return from injury should apply rust factor."""
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'RETURNING'
        assert context.adjustment_factor == pytest.approx(0.92, rel=0.01)
        assert '🔄 RETURNING' in context.flags


class TestNewsParsingLoadManagement:
    """Tests for parsing load management scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    def test_parse_load_management(self, intel):
        """Load management should trigger skip."""
        text = "Player expected to sit for load management"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'LOAD_MANAGEMENT'
        assert context.adjustment_factor == 0.0
        assert context.should_skip() is True


class TestNewsParsingUsageBoost:
    """Tests for parsing teammate out / usage boost scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    def test_parse_star_teammate_out(self, intel):
        """Star teammate out should boost usage."""
        text = "With LeBron out, player expected to see increased usage"
        context = intel._parse_player_news("Test Player", text)
        assert context.adjustment_factor == pytest.approx(1.08, rel=0.01)
        assert '📈 USAGE BOOST' in context.flags


class TestSourceConfidenceExtraction:
    """Tests for source confidence extraction."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    def test_shams_high_confidence(self, intel):
        """Shams Charania should have high confidence."""
        text = "According to Shams Charania, player is out tonight"
        context = intel._parse_player_news("Test Player", text)
        assert context.confidence >= 0.95

    def test_espn_medium_high_confidence(self, intel):
        """ESPN should have medium-high confidence."""
        text = "ESPN reports player is cleared and ready to play"
        context = intel._parse_player_news("Test Player", text)
        assert context.confidence >= 0.85

    def test_twitter_lower_confidence(self, intel):
        """Twitter/X should have lower confidence."""
        text = "Per twitter rumors, player might be out"
        context = intel._parse_player_news("Test Player", text)
        assert context.confidence <= 0.70

    def test_default_confidence(self, intel):
        """No source mentioned should have default confidence."""
        text = "Player is healthy and ready to play"
        context = intel._parse_player_news("Test Player", text)
        assert context.confidence == 0.6  # Default baseline


class TestNewsCaching:
    """Tests for news caching functionality."""

    def test_cache_validity(self):
        """Cache should expire after TTL."""
        intel = NewsIntelligence()

        # Manually set cache with old timestamp
        intel._cache['test_key'] = NewsContext(
            status='HEALTHY', adjustment_factor=1.0, confidence=0.7
        )
        intel._cache_timestamps['test_key'] = datetime.now() - timedelta(seconds=3600)

        # Should be invalid (TTL is 1800 seconds)
        assert intel._is_cache_valid('test_key') is False

    def test_cache_stats(self):
        """get_cache_stats should return correct counts."""
        intel = NewsIntelligence()

        # Add valid and expired entries
        intel._cache['valid'] = NewsContext(status='HEALTHY', adjustment_factor=1.0, confidence=0.7)
        intel._cache_timestamps['valid'] = datetime.now()

        intel._cache['expired'] = NewsContext(status='HEALTHY', adjustment_factor=1.0, confidence=0.7)
        intel._cache_timestamps['expired'] = datetime.now() - timedelta(seconds=3600)

        stats = intel.get_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 1
        assert stats['expired_entries'] == 1

    def test_clear_cache(self):
        """clear_cache should empty all caches."""
        intel = NewsIntelligence()
        intel._cache['test'] = NewsContext(status='HEALTHY', adjustment_factor=1.0, confidence=0.7)
        intel._cache_timestamps['test'] = datetime.now()

        intel.clear_cache()

        assert len(intel._cache) == 0
        assert len(intel._cache_timestamps) == 0


class TestNoSearchFunction:
    """Tests for behavior when no search function is provided."""

    def test_fetch_player_news_without_search(self):
        """Should return NO_NEWS when no search function."""
        intel = NewsIntelligence()  # No search_fn
        context = intel.fetch_player_news("LeBron James", "LAL")

        # Should return default context with NO_NEWS
        assert context.status == 'NO_NEWS'
        assert context.adjustment_factor == 1.0

    def test_fetch_game_news_without_search(self):
        """Should return empty results when no search function."""
        intel = NewsIntelligence()
        results = intel.fetch_game_news("LAL", "BOS")

        # Should return contexts for both teams
        assert 'LAL' in results or 'BOS' in results


class TestHealthyPlayerParsing:
    """Tests for parsing healthy player scenarios."""

    @pytest.fixture
    def intel(self):
        return NewsIntelligence()

    def test_healthy_no_news(self, intel):
        """No negative indicators should return healthy."""
        text = "Player had a great practice today and is ready for the game"
        context = intel._parse_player_news("Test Player", text)
        assert context.status == 'HEALTHY'
        assert context.adjustment_factor == 1.0
        assert len(context.flags) == 0

    def test_empty_text(self, intel):
        """Empty text should return NO_NEWS."""
        context = intel._parse_player_news("Test Player", "")
        assert context.status == 'NO_NEWS'
        assert context.adjustment_factor == 1.0
