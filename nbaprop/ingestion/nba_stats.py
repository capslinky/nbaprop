"""NBA stats ingestion."""

from typing import List, Dict


def fetch_player_logs(players: List[str]) -> List[Dict]:
    """Fetch raw player logs for a list of players."""
    raise NotImplementedError


def fetch_team_stats() -> List[Dict]:
    """Fetch raw team stats."""
    raise NotImplementedError
