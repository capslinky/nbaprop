"""Line movement monitoring for real-time odds tracking.

This module provides functionality to track odds changes over time
and detect significant line movements that may indicate sharp action
or injury news.

Usage:
    from core.line_monitor import LineMonitor, LineMovement

    monitor = LineMonitor(alert_threshold=0.05)
    monitor.update_snapshot(current_odds_df)

    movements = monitor.check_for_alerts(new_odds_df)
    for m in movements:
        print(f"{m.player} {m.prop_type}: {m.old_line} -> {m.new_line} ({m.movement_pct:+.1%})")
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class LineMovement:
    """Represents a significant line movement for a player prop."""
    player: str
    prop_type: str
    old_line: float
    new_line: float
    movement_pct: float
    old_odds: int = -110
    new_odds: int = -110
    bookmaker: str = "unknown"
    event_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def direction(self) -> str:
        """Return 'UP', 'DOWN', or 'UNCHANGED'."""
        if self.new_line > self.old_line:
            return "UP"
        elif self.new_line < self.old_line:
            return "DOWN"
        return "UNCHANGED"

    @property
    def is_significant(self) -> bool:
        """Check if movement is significant (> 3%)."""
        return abs(self.movement_pct) > 0.03

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'player': self.player,
            'prop_type': self.prop_type,
            'old_line': self.old_line,
            'new_line': self.new_line,
            'movement_pct': self.movement_pct,
            'direction': self.direction,
            'old_odds': self.old_odds,
            'new_odds': self.new_odds,
            'bookmaker': self.bookmaker,
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        arrow = "↑" if self.direction == "UP" else "↓" if self.direction == "DOWN" else "→"
        return (
            f"{self.player} {self.prop_type}: {self.old_line} {arrow} {self.new_line} "
            f"({self.movement_pct:+.1%}) [{self.bookmaker}]"
        )


@dataclass
class OddsSnapshot:
    """Snapshot of odds at a point in time."""
    timestamp: datetime
    data: Dict[str, Dict[str, float]]  # {player_prop_key: {bookmaker: line}}

    @staticmethod
    def make_key(player: str, prop_type: str) -> str:
        """Create a unique key for a player prop."""
        return f"{player.lower()}|{prop_type.lower()}"


class LineMonitor:
    """
    Monitor betting lines for significant movements.

    Tracks odds over time and alerts when lines move significantly,
    which may indicate sharp money or late-breaking injury news.

    Attributes:
        alert_threshold: Minimum percentage movement to trigger alert (default 5%)
        snapshot_ttl: How long to keep historical snapshots (default 4 hours)
    """

    # Default thresholds for different prop types
    DEFAULT_THRESHOLDS = {
        'points': 0.05,      # 5% movement
        'rebounds': 0.08,    # 8% (lower volume, more volatile)
        'assists': 0.08,
        'threes': 0.10,      # 10% (very volatile)
        'steals': 0.15,
        'blocks': 0.15,
        'pra': 0.05,         # Points + Rebounds + Assists
        'default': 0.05,
    }

    def __init__(
        self,
        alert_threshold: float = 0.05,
        snapshot_ttl: timedelta = timedelta(hours=4),
        prop_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize line monitor.

        Args:
            alert_threshold: Default percentage movement to trigger alert
            snapshot_ttl: How long to keep snapshot history
            prop_thresholds: Optional dict of prop_type -> threshold overrides
        """
        self.alert_threshold = alert_threshold
        self.snapshot_ttl = snapshot_ttl
        self.prop_thresholds = prop_thresholds or self.DEFAULT_THRESHOLDS.copy()

        # Snapshot history: list of (timestamp, snapshot_dict)
        self._snapshots: List[OddsSnapshot] = []

        # Movement history for analysis
        self._movements: List[LineMovement] = []

        # Maximum movements to keep
        self._max_movements = 1000

    def get_threshold(self, prop_type: str) -> float:
        """Get the alert threshold for a prop type."""
        return self.prop_thresholds.get(
            prop_type.lower(),
            self.prop_thresholds.get('default', self.alert_threshold)
        )

    def update_snapshot(self, odds_df: pd.DataFrame) -> None:
        """
        Update the current odds snapshot.

        Args:
            odds_df: DataFrame with columns: player, prop_type, line, bookmaker
        """
        if odds_df.empty:
            return

        # Build snapshot data
        snapshot_data: Dict[str, Dict[str, float]] = {}

        for _, row in odds_df.iterrows():
            player = row.get('player', row.get('player_name', ''))
            prop_type = row.get('prop_type', row.get('market', ''))
            line = row.get('line', row.get('point', 0))
            bookmaker = row.get('bookmaker', 'unknown')

            if not player or not prop_type:
                continue

            key = OddsSnapshot.make_key(player, prop_type)
            if key not in snapshot_data:
                snapshot_data[key] = {}

            snapshot_data[key][bookmaker] = float(line)

        # Add new snapshot
        self._snapshots.append(OddsSnapshot(
            timestamp=datetime.now(),
            data=snapshot_data
        ))

        # Prune old snapshots
        self._prune_old_snapshots()

    def check_for_alerts(
        self,
        current_odds_df: pd.DataFrame,
        update_snapshot: bool = True
    ) -> List[LineMovement]:
        """
        Check for significant line movements compared to last snapshot.

        Args:
            current_odds_df: Current odds DataFrame
            update_snapshot: Whether to update snapshot after checking

        Returns:
            List of LineMovement objects for significant moves
        """
        if current_odds_df.empty:
            return []

        movements: List[LineMovement] = []

        # Get previous snapshot
        prev_snapshot = self._get_latest_snapshot()

        # Check each current line against previous
        for _, row in current_odds_df.iterrows():
            player = row.get('player', row.get('player_name', ''))
            prop_type = row.get('prop_type', row.get('market', ''))
            new_line = row.get('line', row.get('point', 0))
            new_odds = row.get('odds', -110)
            bookmaker = row.get('bookmaker', 'unknown')
            event_id = row.get('event_id')

            if not player or not prop_type or new_line == 0:
                continue

            key = OddsSnapshot.make_key(player, prop_type)

            # Find previous line for this prop
            old_line = self._get_previous_line(key, bookmaker, prev_snapshot)

            if old_line is None:
                continue

            # Calculate movement
            if old_line == 0:
                continue

            movement_pct = (new_line - old_line) / old_line
            threshold = self.get_threshold(prop_type)

            # Check if movement exceeds threshold
            if abs(movement_pct) >= threshold:
                movement = LineMovement(
                    player=player,
                    prop_type=prop_type,
                    old_line=old_line,
                    new_line=new_line,
                    movement_pct=movement_pct,
                    new_odds=new_odds,
                    bookmaker=bookmaker,
                    event_id=event_id
                )
                movements.append(movement)
                self._movements.append(movement)

                logger.info(f"Line movement detected: {movement}")

        # Update snapshot with current data
        if update_snapshot:
            self.update_snapshot(current_odds_df)

        # Prune old movements
        self._prune_old_movements()

        return movements

    def get_movement_history(
        self,
        player: Optional[str] = None,
        prop_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LineMovement]:
        """
        Get historical line movements, optionally filtered.

        Args:
            player: Filter by player name
            prop_type: Filter by prop type
            since: Only movements after this time
            limit: Maximum movements to return

        Returns:
            List of LineMovement objects
        """
        filtered = self._movements

        if player:
            player_lower = player.lower()
            filtered = [m for m in filtered if m.player.lower() == player_lower]

        if prop_type:
            prop_lower = prop_type.lower()
            filtered = [m for m in filtered if m.prop_type.lower() == prop_lower]

        if since:
            filtered = [m for m in filtered if m.timestamp >= since]

        # Sort by timestamp descending
        filtered = sorted(filtered, key=lambda m: m.timestamp, reverse=True)

        return filtered[:limit]

    def get_steam_moves(
        self,
        min_books: int = 3,
        time_window: timedelta = timedelta(minutes=30)
    ) -> List[Dict[str, Any]]:
        """
        Detect "steam moves" - coordinated line movements across multiple books.

        Steam moves often indicate sharp action and are good predictors of
        closing line value.

        Args:
            min_books: Minimum number of books with same direction move
            time_window: Time window to look for coordinated moves

        Returns:
            List of steam move dicts with player, prop, direction, books
        """
        since = datetime.now() - time_window
        recent_moves = self.get_movement_history(since=since, limit=500)

        # Group by player+prop
        grouped: Dict[str, List[LineMovement]] = {}
        for move in recent_moves:
            key = OddsSnapshot.make_key(move.player, move.prop_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(move)

        steam_moves = []
        for key, moves in grouped.items():
            # Check if multiple books moved in same direction
            up_moves = [m for m in moves if m.direction == "UP"]
            down_moves = [m for m in moves if m.direction == "DOWN"]

            if len(up_moves) >= min_books:
                steam_moves.append({
                    'player': up_moves[0].player,
                    'prop_type': up_moves[0].prop_type,
                    'direction': 'UP',
                    'books': [m.bookmaker for m in up_moves],
                    'avg_movement': sum(m.movement_pct for m in up_moves) / len(up_moves),
                    'movements': up_moves,
                })

            if len(down_moves) >= min_books:
                steam_moves.append({
                    'player': down_moves[0].player,
                    'prop_type': down_moves[0].prop_type,
                    'direction': 'DOWN',
                    'books': [m.bookmaker for m in down_moves],
                    'avg_movement': sum(m.movement_pct for m in down_moves) / len(down_moves),
                    'movements': down_moves,
                })

        return steam_moves

    def _get_latest_snapshot(self) -> Optional[OddsSnapshot]:
        """Get the most recent snapshot."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def _get_previous_line(
        self,
        key: str,
        bookmaker: str,
        snapshot: Optional[OddsSnapshot]
    ) -> Optional[float]:
        """Get previous line for a player prop from snapshot."""
        if snapshot is None:
            return None

        prop_data = snapshot.data.get(key)
        if prop_data is None:
            return None

        # Try exact bookmaker first
        if bookmaker in prop_data:
            return prop_data[bookmaker]

        # Fall back to average across books
        if prop_data:
            return sum(prop_data.values()) / len(prop_data)

        return None

    def _prune_old_snapshots(self) -> None:
        """Remove snapshots older than TTL."""
        cutoff = datetime.now() - self.snapshot_ttl
        self._snapshots = [s for s in self._snapshots if s.timestamp >= cutoff]

    def _prune_old_movements(self) -> None:
        """Keep movement history under limit."""
        if len(self._movements) > self._max_movements:
            # Keep most recent
            self._movements = sorted(
                self._movements,
                key=lambda m: m.timestamp,
                reverse=True
            )[:self._max_movements]

    def clear_history(self) -> None:
        """Clear all snapshots and movement history."""
        self._snapshots.clear()
        self._movements.clear()

    def export_movements(self, filepath: str) -> int:
        """
        Export movement history to JSON file.

        Returns:
            Number of movements exported
        """
        data = [m.to_dict() for m in self._movements]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return len(data)

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots in history."""
        return len(self._snapshots)

    @property
    def movement_count(self) -> int:
        """Number of movements in history."""
        return len(self._movements)
