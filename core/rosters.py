"""
NBA Team Rosters and Depth Charts.

Provides structured roster data for all 30 NBA teams including:
- Full rotations (10-12 players per team)
- Position and role information
- Average minutes per game
- Star player designations

Usage:
    from core.rosters import TEAM_ROSTERS, get_player_info, get_team_starters

    # Get full roster for a team
    roster = TEAM_ROSTERS.get('DAL')

    # Get player info
    info = get_player_info('Luka Doncic')

    # Get team's starting lineup
    starters = get_team_starters('DAL')
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class PlayerRoster:
    """Individual player roster information."""
    name: str
    position: str  # PG, SG, SF, PF, C
    role: str      # STARTER, ROTATION, BENCH
    avg_minutes: float
    is_star: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TeamRoster:
    """Full team roster with depth chart."""
    team_abbrev: str
    team_name: str
    players: List[PlayerRoster] = field(default_factory=list)
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def get_starters(self) -> List[PlayerRoster]:
        """Get starting lineup."""
        return [p for p in self.players if p.role == "STARTER"]

    def get_rotation(self) -> List[PlayerRoster]:
        """Get rotation players (starters + key bench)."""
        return [p for p in self.players if p.role in ("STARTER", "ROTATION")]

    def get_stars(self) -> List[PlayerRoster]:
        """Get star players whose absence significantly impacts teammates."""
        return [p for p in self.players if p.is_star]

    def get_player(self, name: str) -> Optional[PlayerRoster]:
        """Get specific player by name."""
        name_lower = name.lower()
        for p in self.players:
            if p.name.lower() == name_lower:
                return p
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_abbrev": self.team_abbrev,
            "team_name": self.team_name,
            "players": [p.to_dict() for p in self.players],
            "last_updated": self.last_updated
        }


# =============================================================================
# TEAM ROSTER DATA
# =============================================================================

# Global roster storage - populated from JSON file or Perplexity
TEAM_ROSTERS: Dict[str, TeamRoster] = {}


def load_rosters_from_json(filepath: str = None) -> Dict[str, TeamRoster]:
    """
    Load roster data from JSON file.

    Args:
        filepath: Path to JSON file. Defaults to data/rosters_2025_26.json

    Returns:
        Dictionary mapping team abbreviations to TeamRoster objects
    """
    global TEAM_ROSTERS

    if filepath is None:
        # Default path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "rosters_2025_26.json")

    if not os.path.exists(filepath):
        return TEAM_ROSTERS

    with open(filepath, 'r') as f:
        data = json.load(f)

    for team_abbrev, team_data in data.items():
        players = [
            PlayerRoster(**p) for p in team_data.get("players", [])
        ]
        TEAM_ROSTERS[team_abbrev] = TeamRoster(
            team_abbrev=team_data.get("team_abbrev", team_abbrev),
            team_name=team_data.get("team_name", ""),
            players=players,
            last_updated=team_data.get("last_updated", "")
        )

    return TEAM_ROSTERS


def save_rosters_to_json(rosters: Dict[str, TeamRoster], filepath: str = None) -> str:
    """
    Save roster data to JSON file.

    Args:
        rosters: Dictionary of team rosters
        filepath: Output path. Defaults to data/rosters_2025_26.json

    Returns:
        Path to saved file
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "rosters_2025_26.json")

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {abbrev: roster.to_dict() for abbrev, roster in rosters.items()}

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_info(player_name: str) -> Optional[Dict]:
    """
    Get player information across all teams.

    Args:
        player_name: Full player name

    Returns:
        Dict with player info and team, or None if not found
    """
    name_lower = player_name.lower()
    for team_abbrev, roster in TEAM_ROSTERS.items():
        for player in roster.players:
            if player.name.lower() == name_lower:
                return {
                    "player": player.to_dict(),
                    "team": team_abbrev,
                    "team_name": roster.team_name
                }
    return None


def get_team_starters(team_abbrev: str) -> List[str]:
    """
    Get list of starting players for a team.

    Args:
        team_abbrev: 3-letter team abbreviation

    Returns:
        List of player names in starting lineup
    """
    roster = TEAM_ROSTERS.get(team_abbrev.upper())
    if roster:
        return [p.name for p in roster.get_starters()]
    return []


def get_team_stars(team_abbrev: str) -> List[str]:
    """
    Get list of star players for a team.

    Args:
        team_abbrev: 3-letter team abbreviation

    Returns:
        List of star player names
    """
    roster = TEAM_ROSTERS.get(team_abbrev.upper())
    if roster:
        return [p.name for p in roster.get_stars()]
    return []


def is_player_starter(player_name: str) -> bool:
    """Check if player is a starter on their team."""
    info = get_player_info(player_name)
    if info:
        return info["player"]["role"] == "STARTER"
    return False


def get_player_avg_minutes(player_name: str) -> float:
    """Get player's average minutes per game."""
    info = get_player_info(player_name)
    if info:
        return info["player"]["avg_minutes"]
    return 0.0


# =============================================================================
# INITIALIZATION
# =============================================================================

# Attempt to load rosters on module import
load_rosters_from_json()
