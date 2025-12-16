"""Unit tests for Closing Line Value (CLV) tracking functionality."""

import pytest
import tempfile
import os
from datetime import datetime

from pick_tracker import PickTracker, PickRecord


class TestCLVSchema:
    """Tests for CLV database schema."""

    def test_clv_columns_exist(self):
        """Test that CLV columns are created in new database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            tracker = PickTracker(db_path=db_path)

            # Check columns exist by querying
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("PRAGMA table_info(picks)")
                columns = [row[1] for row in cursor.fetchall()]

            assert 'closing_line' in columns
            assert 'closing_odds' in columns
            assert 'clv' in columns
        finally:
            os.unlink(db_path)

    def test_clv_migration_adds_columns(self):
        """Test that CLV columns are added to existing databases."""
        import sqlite3

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Create a database without CLV columns (old schema)
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE picks (
                        id INTEGER PRIMARY KEY,
                        date TEXT NOT NULL,
                        player TEXT NOT NULL,
                        prop_type TEXT NOT NULL,
                        line REAL NOT NULL,
                        pick TEXT NOT NULL,
                        edge REAL,
                        confidence REAL,
                        projection REAL,
                        context_quality INTEGER,
                        warnings TEXT,
                        adjustments TEXT,
                        flags TEXT,
                        opponent TEXT,
                        is_home INTEGER,
                        is_b2b INTEGER DEFAULT 0,
                        game_total REAL,
                        matchup_rating TEXT,
                        bookmaker TEXT,
                        odds INTEGER DEFAULT -110,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, player, prop_type, line)
                    )
                """)

            # Now initialize PickTracker which should migrate
            tracker = PickTracker(db_path=db_path)

            # Check columns were added
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("PRAGMA table_info(picks)")
                columns = [row[1] for row in cursor.fetchall()]

            assert 'closing_line' in columns
            assert 'closing_odds' in columns
            assert 'clv' in columns
        finally:
            os.unlink(db_path)


class TestCLVUpdate:
    """Tests for CLV update functionality."""

    @pytest.fixture
    def tracker_with_pick(self):
        """Create tracker with a test pick."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        tracker = PickTracker(db_path=db_path)

        # Record a test pick
        pick = PickRecord(
            date='2025-12-15',
            player='Test Player',
            prop_type='points',
            line=25.0,
            pick='OVER',
            edge=0.12,
            confidence=0.65,
            projection=28.0,
            context_quality=50,
            warnings='[]',
            adjustments='{}',
            flags='["ACTIVE"]',
            opponent='LAL',
            is_home=True,
            is_b2b=False,
            game_total=225.0,
            matchup_rating='NEUTRAL',
            bookmaker='fanduel',
            odds=-110
        )
        tracker.record_pick(pick)

        yield tracker, db_path

        os.unlink(db_path)

    def test_update_clv_over_positive(self, tracker_with_pick):
        """Test CLV calculation for OVER pick with positive CLV."""
        tracker, _ = tracker_with_pick

        # Closing line moved up (favorable for OVER)
        success = tracker.update_clv(
            date_str='2025-12-15',
            player='Test Player',
            prop_type='points',
            line=25.0,
            closing_line=26.0,  # Line moved up
            closing_odds=-115
        )

        assert success is True

        # Check CLV was calculated correctly
        # For OVER: CLV = (closing - our_line) / our_line = (26 - 25) / 25 = 0.04 = 4%
        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            row = conn.execute("""
                SELECT closing_line, closing_odds, clv FROM picks
                WHERE player = 'Test Player' AND prop_type = 'points'
            """).fetchone()

        assert row[0] == 26.0
        assert row[1] == -115
        assert abs(row[2] - 0.04) < 0.001  # CLV should be +4%

    def test_update_clv_over_negative(self, tracker_with_pick):
        """Test CLV calculation for OVER pick with negative CLV."""
        tracker, _ = tracker_with_pick

        # Closing line moved down (unfavorable for OVER)
        success = tracker.update_clv(
            date_str='2025-12-15',
            player='Test Player',
            prop_type='points',
            line=25.0,
            closing_line=24.0,  # Line moved down
            closing_odds=-105
        )

        assert success is True

        # For OVER: CLV = (closing - our_line) / our_line = (24 - 25) / 25 = -0.04 = -4%
        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            row = conn.execute("""
                SELECT clv FROM picks WHERE player = 'Test Player'
            """).fetchone()

        assert abs(row[0] - (-0.04)) < 0.001  # CLV should be -4%

    def test_update_clv_under_positive(self):
        """Test CLV calculation for UNDER pick with positive CLV."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            tracker = PickTracker(db_path=db_path)

            # Record UNDER pick
            pick = PickRecord(
                date='2025-12-15',
                player='Under Player',
                prop_type='points',
                line=30.0,
                pick='UNDER',
                edge=0.10,
                confidence=0.60,
                projection=27.0,
                context_quality=50,
                warnings='[]',
                adjustments='{}',
                flags='["ACTIVE"]',
                odds=-110
            )
            tracker.record_pick(pick)

            # Closing line moved down (favorable for UNDER)
            success = tracker.update_clv(
                date_str='2025-12-15',
                player='Under Player',
                prop_type='points',
                line=30.0,
                closing_line=28.0  # Line moved down
            )

            assert success is True

            # For UNDER: CLV = (our_line - closing) / our_line = (30 - 28) / 30 = 0.0667 = 6.67%
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("""
                    SELECT clv FROM picks WHERE player = 'Under Player'
                """).fetchone()

            assert row[0] > 0.06  # Positive CLV for favorable line move
        finally:
            os.unlink(db_path)

    def test_update_clv_not_found(self, tracker_with_pick):
        """Test CLV update for non-existent pick."""
        tracker, _ = tracker_with_pick

        success = tracker.update_clv(
            date_str='2025-12-15',
            player='Non Existent',
            prop_type='points',
            line=25.0,
            closing_line=26.0
        )

        assert success is False

    def test_update_clv_batch(self, tracker_with_pick):
        """Test batch CLV update."""
        tracker, db_path = tracker_with_pick

        # Add another pick
        pick2 = PickRecord(
            date='2025-12-15',
            player='Second Player',
            prop_type='rebounds',
            line=10.0,
            pick='OVER',
            edge=0.08,
            confidence=0.55,
            projection=11.0,
            context_quality=45,
            warnings='[]',
            adjustments='{}',
            flags='["ACTIVE"]',
            odds=-110
        )
        tracker.record_pick(pick2)

        clv_data = [
            {'date': '2025-12-15', 'player': 'Test Player', 'prop_type': 'points',
             'line': 25.0, 'closing_line': 26.0, 'closing_odds': -115},
            {'date': '2025-12-15', 'player': 'Second Player', 'prop_type': 'rebounds',
             'line': 10.0, 'closing_line': 10.5, 'closing_odds': -110},
        ]

        count = tracker.update_clv_batch(clv_data)

        assert count == 2


class TestCLVQueries:
    """Tests for CLV query functionality."""

    def test_get_picks_missing_clv(self):
        """Test query for picks without CLV data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            tracker = PickTracker(db_path=db_path)

            # Add pick without CLV
            pick = PickRecord(
                date='2025-12-15',
                player='Missing CLV',
                prop_type='points',
                line=25.0,
                pick='OVER',
                edge=0.10,
                confidence=0.60,
                projection=28.0,
                context_quality=50,
                warnings='[]',
                adjustments='{}',
                flags='["ACTIVE"]',
                odds=-110
            )
            tracker.record_pick(pick)

            missing = tracker.get_picks_missing_clv()
            assert len(missing) == 1
            assert missing.iloc[0]['player'] == 'Missing CLV'

            # Update CLV
            tracker.update_clv('2025-12-15', 'Missing CLV', 'points', 25.0, 26.0)

            # Should now be empty
            missing = tracker.get_picks_missing_clv()
            assert len(missing) == 0
        finally:
            os.unlink(db_path)

    def test_clv_stats_in_accuracy_report(self):
        """Test CLV statistics appear in accuracy report."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            tracker = PickTracker(db_path=db_path)

            # Add pick with CLV
            pick = PickRecord(
                date='2025-12-15',
                player='CLV Player',
                prop_type='points',
                line=25.0,
                pick='OVER',
                edge=0.10,
                confidence=0.60,
                projection=28.0,
                context_quality=50,
                warnings='[]',
                adjustments='{}',
                flags='["ACTIVE"]',
                odds=-110,
                closing_line=26.0,
                closing_odds=-115,
                clv=0.04
            )
            tracker.record_pick(pick)

            report = tracker.get_accuracy_report(days=30)

            # CLV stats should be in report
            assert 'clv_stats' in report
            assert 'clv_win_correlation' in report
        finally:
            os.unlink(db_path)
