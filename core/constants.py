"""
Constants and team data for the NBA prop analysis system.

Consolidates team abbreviation mappings from:
- nba_prop_model.py TEAM_ABBREV_MAP
- nba_props_v2.py TEAM_ABBREVS
- nba_integrations.py inline maps
- app.py inline map

Single source of truth for all team-related lookups.
"""

from datetime import datetime
from typing import Dict, List


# =============================================================================
# TEAM ABBREVIATIONS
# =============================================================================

# Complete mapping of all team name variants to standard 3-letter abbreviations
TEAM_ABBREVIATIONS: Dict[str, str] = {
    # Full official names -> Standard abbreviation
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',

    # Common variants and nicknames
    'Spurs': 'SAS',
    'Clippers': 'LAC',
    'Lakers': 'LAL',
    'Warriors': 'GSW',
    'Pelicans': 'NOP',
    '76ers': 'PHI',
    'Sixers': 'PHI',
    'Blazers': 'POR',
    'Trail Blazers': 'POR',
    'Timberwolves': 'MIN',
    'T-Wolves': 'MIN',
    'Wolves': 'MIN',
    'Knicks': 'NYK',
    'Nets': 'BKN',
    'Heat': 'MIA',
    'Bulls': 'CHI',
    'Cavs': 'CLE',
    'Cavaliers': 'CLE',
    'Mavs': 'DAL',
    'Mavericks': 'DAL',
    'Nuggets': 'DEN',
    'Pistons': 'DET',
    'Rockets': 'HOU',
    'Pacers': 'IND',
    'Grizzlies': 'MEM',
    'Bucks': 'MIL',
    'Thunder': 'OKC',
    'Magic': 'ORL',
    'Suns': 'PHX',
    'Kings': 'SAC',
    'Raptors': 'TOR',
    'Jazz': 'UTA',
    'Wizards': 'WAS',
    'Hawks': 'ATL',
    'Celtics': 'BOS',
    'Hornets': 'CHA',

    # First 3-letter fallbacks for partial matches
    'SAN': 'SAS',  # San Antonio
    'NEW': 'NOP',  # Could be New Orleans or New York - default to NOP
    'GOL': 'GSW',  # Golden State
    'LOS': 'LAL',  # Los Angeles - default to Lakers
    'PHO': 'PHX',  # Phoenix (PHO vs PHX)
    'BRO': 'BKN',  # Brooklyn

    # Standard abbreviations map to themselves
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
    'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
    'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
    'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHX', 'POR': 'POR',
    'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR', 'UTA': 'UTA', 'WAS': 'WAS',
}


def normalize_team_abbrev(team_input: str) -> str:
    """
    Normalize any team reference to standard 3-letter NBA abbreviation.

    Handles:
    - Full team names: "San Antonio Spurs" -> "SAS"
    - Partial names: "Spurs" -> "SAS"
    - Non-standard abbrevs: "SAN" -> "SAS", "PHO" -> "PHX"
    - Already standard: "SAS" -> "SAS"

    Args:
        team_input: Team name, abbreviation, or variant

    Returns:
        Standard 3-letter NBA abbreviation, or original if not found
    """
    if not team_input:
        return team_input

    team_str = str(team_input).strip()

    # Direct lookup
    if team_str in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_str]

    # Case-insensitive lookup
    team_upper = team_str.upper()
    if team_upper in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_upper]

    # Try matching by first 3 characters
    if len(team_str) >= 3:
        first_three = team_str[:3].upper()
        if first_three in TEAM_ABBREVIATIONS:
            return TEAM_ABBREVIATIONS[first_three]

    # If 3-letter code, check if it's already standard
    if len(team_str) == 3:
        return team_str.upper()

    return team_str


# =============================================================================
# STAR PLAYERS
# =============================================================================

# Star players by team - their absence significantly impacts teammates
STAR_PLAYERS: Dict[str, List[str]] = {
    'ATL': ['Trae Young', 'Dejounte Murray'],
    'BOS': ['Jayson Tatum', 'Jaylen Brown'],
    'BKN': ['Mikal Bridges', 'Cameron Johnson'],
    'CHA': ['LaMelo Ball', 'Brandon Miller'],
    'CHI': ['Zach LaVine', 'DeMar DeRozan', 'Coby White'],
    'CLE': ['Donovan Mitchell', 'Darius Garland', 'Evan Mobley'],
    'DAL': ['Luka Doncic', 'Kyrie Irving'],
    'DEN': ['Nikola Jokic', 'Jamal Murray'],
    'DET': ['Cade Cunningham', 'Jaden Ivey'],
    'GSW': ['Stephen Curry', 'Klay Thompson', 'Draymond Green'],
    'HOU': ['Jalen Green', 'Alperen Sengun', 'Fred VanVleet'],
    'IND': ['Tyrese Haliburton', 'Pascal Siakam', 'Myles Turner'],
    'LAC': ['Kawhi Leonard', 'Paul George', 'James Harden'],
    'LAL': ['LeBron James', 'Anthony Davis'],
    'MEM': ['Ja Morant', 'Desmond Bane', 'Jaren Jackson Jr.'],
    'MIA': ['Jimmy Butler', 'Bam Adebayo', 'Tyler Herro'],
    'MIL': ['Giannis Antetokounmpo', 'Damian Lillard', 'Khris Middleton'],
    'MIN': ['Anthony Edwards', 'Karl-Anthony Towns', 'Rudy Gobert'],
    'NOP': ['Zion Williamson', 'Brandon Ingram', 'CJ McCollum'],
    'NYK': ['Jalen Brunson', 'Julius Randle', 'RJ Barrett'],
    'OKC': ['Shai Gilgeous-Alexander', 'Chet Holmgren', 'Jalen Williams'],
    'ORL': ['Paolo Banchero', 'Franz Wagner', 'Jalen Suggs'],
    'PHI': ['Joel Embiid', 'Tyrese Maxey'],
    'PHX': ['Kevin Durant', 'Devin Booker', 'Bradley Beal'],
    'POR': ['Anfernee Simons', 'Jerami Grant', 'Scoot Henderson'],
    'SAC': ["De'Aaron Fox", 'Domantas Sabonis', 'Keegan Murray'],
    'SAS': ['Victor Wembanyama', 'Devin Vassell', 'Keldon Johnson'],
    'TOR': ['Scottie Barnes', 'Pascal Siakam', 'OG Anunoby'],
    'UTA': ['Lauri Markkanen', 'Jordan Clarkson', 'Collin Sexton'],
    'WAS': ['Jordan Poole', 'Kyle Kuzma', 'Deni Avdija'],
}


# Usage boost when star is out (by stat type)
STAR_OUT_BOOST: Dict[str, float] = {
    'points': 1.08,    # 8% boost to scoring
    'assists': 1.06,   # 6% boost to assists
    'rebounds': 1.04,  # 4% boost to rebounds
    'pra': 1.07,       # 7% boost to PRA
    'threes': 1.05,    # 5% boost to threes
}


# =============================================================================
# SEASON UTILITIES
# =============================================================================

def get_current_nba_season() -> str:
    """
    Get current NBA season string (e.g., '2024-25').

    NBA season runs from October to June.
    - Oct-Dec 2024: "2024-25"
    - Jan-Jun 2025: "2024-25"
    """
    today = datetime.now()
    return get_season_from_date(today)


def get_season_from_date(date: datetime) -> str:
    """
    Get NBA season string for a specific date.

    NBA season runs from October to June.
    Oct 2024 - Jun 2025 = "2024-25" season
    Oct 2025 - Jun 2026 = "2025-26" season
    """
    year = date.year
    month = date.month
    if month >= 10:  # Oct-Dec: start of new season
        return f"{year}-{str(year + 1)[-2:]}"
    else:  # Jan-Sep: continuation of previous season
        return f"{year - 1}-{str(year)[-2:]}"
