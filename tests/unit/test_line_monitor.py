"""Unit tests for line movement monitoring functionality."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from core.line_monitor import LineMonitor, LineMovement, OddsSnapshot


class TestLineMovement:
    """Tests for LineMovement dataclass."""

    def test_line_movement_creation(self):
        """Test basic LineMovement creation."""
        movement = LineMovement(
            player="Luka Doncic",
            prop_type="points",
            old_line=32.5,
            new_line=33.5,
            movement_pct=0.031
        )

        assert movement.player == "Luka Doncic"
        assert movement.prop_type == "points"
        assert movement.old_line == 32.5
        assert movement.new_line == 33.5
        assert movement.movement_pct == 0.031

    def test_direction_up(self):
        """Test direction property for upward movement."""
        movement = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=27.0,
            movement_pct=0.08
        )
        assert movement.direction == "UP"

    def test_direction_down(self):
        """Test direction property for downward movement."""
        movement = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=23.0,
            movement_pct=-0.08
        )
        assert movement.direction == "DOWN"

    def test_direction_unchanged(self):
        """Test direction for no change."""
        movement = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=25.0,
            movement_pct=0.0
        )
        assert movement.direction == "UNCHANGED"

    def test_is_significant(self):
        """Test significance threshold."""
        significant = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=26.5,
            movement_pct=0.06
        )
        assert significant.is_significant is True

        not_significant = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=25.5,
            movement_pct=0.02
        )
        assert not_significant.is_significant is False

    def test_to_dict(self):
        """Test serialization to dict."""
        movement = LineMovement(
            player="Test Player",
            prop_type="rebounds",
            old_line=10.0,
            new_line=11.0,
            movement_pct=0.10,
            bookmaker="draftkings"
        )

        data = movement.to_dict()
        assert data['player'] == "Test Player"
        assert data['prop_type'] == "rebounds"
        assert data['direction'] == "UP"
        assert data['bookmaker'] == "draftkings"
        assert 'timestamp' in data

    def test_str_representation(self):
        """Test string representation."""
        movement = LineMovement(
            player="Test",
            prop_type="points",
            old_line=25.0,
            new_line=27.0,
            movement_pct=0.08,
            bookmaker="fanduel"
        )

        string = str(movement)
        assert "Test" in string
        assert "points" in string
        assert "25.0" in string
        assert "27.0" in string
        assert "fanduel" in string


class TestOddsSnapshot:
    """Tests for OddsSnapshot class."""

    def test_make_key(self):
        """Test key generation."""
        key = OddsSnapshot.make_key("Luka Doncic", "points")
        assert key == "luka doncic|points"

    def test_make_key_case_insensitive(self):
        """Test key is case insensitive."""
        key1 = OddsSnapshot.make_key("LUKA DONCIC", "POINTS")
        key2 = OddsSnapshot.make_key("luka doncic", "points")
        assert key1 == key2


class TestLineMonitor:
    """Tests for LineMonitor class."""

    def test_initialization(self):
        """Test LineMonitor initialization."""
        monitor = LineMonitor(alert_threshold=0.05)
        assert monitor.alert_threshold == 0.05
        assert monitor.snapshot_count == 0
        assert monitor.movement_count == 0

    def test_get_threshold_by_prop_type(self):
        """Test getting threshold for specific prop types."""
        monitor = LineMonitor()

        # Points has 5% threshold
        assert monitor.get_threshold('points') == 0.05

        # Threes is more volatile, 10%
        assert monitor.get_threshold('threes') == 0.10

        # Unknown falls back to default
        assert monitor.get_threshold('unknown') == 0.05

    def test_update_snapshot(self):
        """Test updating odds snapshot."""
        monitor = LineMonitor()

        odds_df = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'prop_type': ['points', 'rebounds'],
            'line': [25.0, 10.0],
            'bookmaker': ['fanduel', 'draftkings']
        })

        monitor.update_snapshot(odds_df)
        assert monitor.snapshot_count == 1

    def test_update_snapshot_empty(self):
        """Test updating with empty DataFrame."""
        monitor = LineMonitor()
        monitor.update_snapshot(pd.DataFrame())
        assert monitor.snapshot_count == 0

    def test_check_for_alerts_no_previous(self):
        """Test checking alerts without previous snapshot."""
        monitor = LineMonitor()

        current_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })

        movements = monitor.check_for_alerts(current_odds)
        assert len(movements) == 0

    def test_check_for_alerts_significant_movement(self):
        """Test detecting significant line movement."""
        monitor = LineMonitor(alert_threshold=0.05)

        # First snapshot
        old_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(old_odds)

        # New odds with 10% movement
        new_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [27.5],  # +10%
            'bookmaker': ['fanduel']
        })

        movements = monitor.check_for_alerts(new_odds)
        assert len(movements) == 1
        assert movements[0].player == "Player A"
        assert movements[0].direction == "UP"
        assert movements[0].old_line == 25.0
        assert movements[0].new_line == 27.5

    def test_check_for_alerts_no_significant_movement(self):
        """Test no alert for small movement."""
        monitor = LineMonitor(alert_threshold=0.05)

        # First snapshot
        old_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(old_odds)

        # New odds with only 2% movement
        new_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.5],  # +2%
            'bookmaker': ['fanduel']
        })

        movements = monitor.check_for_alerts(new_odds)
        assert len(movements) == 0

    def test_get_movement_history(self):
        """Test retrieving movement history."""
        monitor = LineMonitor(alert_threshold=0.05)

        # Create some movements
        old_odds = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'prop_type': ['points', 'rebounds'],
            'line': [25.0, 10.0],
            'bookmaker': ['fanduel', 'draftkings']
        })
        monitor.update_snapshot(old_odds)

        new_odds = pd.DataFrame({
            'player': ['Player A', 'Player B'],
            'prop_type': ['points', 'rebounds'],
            'line': [27.5, 11.5],  # Both significant
            'bookmaker': ['fanduel', 'draftkings']
        })
        monitor.check_for_alerts(new_odds)

        # Get all history
        history = monitor.get_movement_history()
        assert len(history) == 2

        # Filter by player
        history_a = monitor.get_movement_history(player='Player A')
        assert len(history_a) == 1
        assert history_a[0].player == 'Player A'

        # Filter by prop type
        history_pts = monitor.get_movement_history(prop_type='points')
        assert len(history_pts) == 1

    def test_get_steam_moves(self):
        """Test detecting coordinated multi-book movements."""
        monitor = LineMonitor(alert_threshold=0.03)

        # Initial snapshot
        old_odds = pd.DataFrame({
            'player': ['Player A', 'Player A', 'Player A'],
            'prop_type': ['points', 'points', 'points'],
            'line': [25.0, 25.0, 25.0],
            'bookmaker': ['fanduel', 'draftkings', 'betmgm']
        })
        monitor.update_snapshot(old_odds)

        # All three books move in same direction
        new_odds = pd.DataFrame({
            'player': ['Player A', 'Player A', 'Player A'],
            'prop_type': ['points', 'points', 'points'],
            'line': [26.5, 26.5, 27.0],  # All up 6-8%
            'bookmaker': ['fanduel', 'draftkings', 'betmgm']
        })
        monitor.check_for_alerts(new_odds)

        steam_moves = monitor.get_steam_moves(min_books=3)
        assert len(steam_moves) == 1
        assert steam_moves[0]['direction'] == 'UP'
        assert len(steam_moves[0]['books']) == 3

    def test_clear_history(self):
        """Test clearing monitor history."""
        monitor = LineMonitor()

        odds_df = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(odds_df)

        assert monitor.snapshot_count == 1

        monitor.clear_history()
        assert monitor.snapshot_count == 0
        assert monitor.movement_count == 0

    def test_export_movements(self):
        """Test exporting movements to JSON file."""
        monitor = LineMonitor(alert_threshold=0.05)

        # Create a movement
        old_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(old_odds)

        new_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [27.5],
            'bookmaker': ['fanduel']
        })
        monitor.check_for_alerts(new_odds)

        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            count = monitor.export_movements(filepath)
            assert count == 1

            # Verify file content
            import json
            with open(filepath) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]['player'] == 'Player A'
        finally:
            os.unlink(filepath)

    def test_snapshot_ttl(self):
        """Test that old snapshots are pruned."""
        monitor = LineMonitor(
            snapshot_ttl=timedelta(seconds=1)
        )

        # Add snapshot
        odds_df = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(odds_df)
        assert monitor.snapshot_count == 1

        # Manually age the snapshot
        monitor._snapshots[0].timestamp = datetime.now() - timedelta(seconds=2)

        # Update again should prune old
        monitor.update_snapshot(odds_df)

        # Old snapshot should be gone, only new one remains
        assert monitor.snapshot_count == 1


class TestLineMonitorEdgeCases:
    """Edge case tests for LineMonitor."""

    def test_different_bookmaker_fallback(self):
        """Test falling back to average when bookmaker doesn't match."""
        monitor = LineMonitor(alert_threshold=0.05)

        # Snapshot from fanduel
        old_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(old_odds)

        # New data from draftkings
        new_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [27.5],
            'bookmaker': ['draftkings']
        })

        movements = monitor.check_for_alerts(new_odds)
        # Should still detect movement using average fallback
        assert len(movements) == 1

    def test_missing_player_column_names(self):
        """Test with alternative column names."""
        monitor = LineMonitor()

        odds_df = pd.DataFrame({
            'player_name': ['Player A'],  # Alternative name
            'market': ['points'],  # Alternative name
            'point': [25.0],  # Alternative name
            'bookmaker': ['fanduel']
        })

        monitor.update_snapshot(odds_df)
        assert monitor.snapshot_count == 1

    def test_zero_line_ignored(self):
        """Test that zero lines are ignored."""
        monitor = LineMonitor(alert_threshold=0.05)

        old_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [25.0],
            'bookmaker': ['fanduel']
        })
        monitor.update_snapshot(old_odds)

        new_odds = pd.DataFrame({
            'player': ['Player A'],
            'prop_type': ['points'],
            'line': [0],  # Zero line
            'bookmaker': ['fanduel']
        })

        movements = monitor.check_for_alerts(new_odds)
        assert len(movements) == 0
