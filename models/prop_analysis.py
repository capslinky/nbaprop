"""
PropAnalysis dataclass for comprehensive analysis results.

Contains the PropAnalysis dataclass with validation, warnings, and explain() for debugging.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PropAnalysis:
    """
    Comprehensive analysis result from UnifiedPropModel.
    Contains projection, recommendations, and full context breakdown.
    Includes validation, warnings, and explain() for debugging.
    """
    # Core predictions
    player: str
    prop_type: str
    line: float
    projection: float
    base_projection: float
    edge: float  # As decimal (0.05 = 5%)
    confidence: float  # 0-1 scale
    pick: str  # 'OVER', 'UNDER', 'PASS'

    # Historical stats
    recent_avg: float  # Last 5 games
    season_avg: float  # All games analyzed
    over_rate: float  # % of games over line
    under_rate: float  # % of games under line
    std_dev: float
    games_analyzed: int
    trend: str  # 'HOT', 'COLD', 'NEUTRAL'

    # Context used
    opponent: Optional[str] = None
    is_home: Optional[bool] = None
    is_b2b: bool = False
    game_total: Optional[float] = None
    blowout_risk: Optional[str] = None
    matchup_rating: str = 'NEUTRAL'
    opp_rank: Optional[int] = None

    # Adjustment breakdown
    adjustments: dict = field(default_factory=dict)
    total_adjustment: float = 0.0
    flags: List[str] = field(default_factory=list)

    # Injury context
    player_status: str = 'HEALTHY'
    teammate_boost: float = 1.0
    stars_out: List[str] = field(default_factory=list)

    # Validation & Quality
    context_quality: int = 0  # 0-100 score
    warnings: List[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)  # What data supported each adjustment

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'player': self.player,
            'prop_type': self.prop_type,
            'line': self.line,
            'projection': self.projection,
            'edge': round(self.edge * 100, 1),
            'confidence': round(self.confidence * 100, 0),
            'pick': self.pick,
            'recent_avg': self.recent_avg,
            'season_avg': self.season_avg,
            'over_rate': round(self.over_rate * 100, 0),
            'under_rate': round(self.under_rate * 100, 0),
            'trend': self.trend,
            'matchup': self.matchup_rating,
            'opp_rank': self.opp_rank,
            'is_home': self.is_home,
            'is_b2b': self.is_b2b,
            'flags': self.flags,
            'total_adjustment': round(self.total_adjustment * 100, 1),
            'context_quality': self.context_quality,
            'warnings': self.warnings,
        }

    def explain(self, return_string: bool = False) -> Optional[str]:
        """
        Print detailed breakdown of the analysis for debugging.
        Shows exactly why this pick was made and what context was/wasn't applied.
        """
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"PICK ANALYSIS: {self.player} {self.prop_type.upper()} {self.line}")
        lines.append("=" * 60)

        # Raw Stats
        lines.append("")
        lines.append(f"RAW STATS ({self.games_analyzed} games)")
        lines.append("-" * 30)
        lines.append(f"  L5 Average:   {self.recent_avg:.1f}")
        lines.append(f"  L15 Average:  {self.season_avg:.1f}")
        lines.append(f"  Std Dev:      {self.std_dev:.1f}")
        lines.append(f"  Over Rate:    {self.over_rate*100:.0f}% ({int(self.over_rate * self.games_analyzed)}/{self.games_analyzed})")
        lines.append(f"  Under Rate:   {self.under_rate*100:.0f}% ({int(self.under_rate * self.games_analyzed)}/{self.games_analyzed})")
        lines.append(f"  Trend:        {self.trend}")

        # Base Projection
        lines.append("")
        lines.append(f"BASE PROJECTION: {self.base_projection:.1f}")
        lines.append(f"  (60% x {self.recent_avg:.1f}) + (40% x {self.season_avg:.1f}) + trend adj")

        # Adjustments
        lines.append("")
        lines.append("ADJUSTMENTS APPLIED")
        lines.append("-" * 30)

        adj_names = {
            'opp_defense': 'Opponent Defense',
            'location': 'Location (H/A)',
            'b2b': 'Back-to-Back',
            'rest': 'Rest Days',
            'pace': 'Pace Factor',
            'total': 'Game Total',
            'blowout': 'Blowout Risk',
            'vs_team': 'vs Team History',
            'minutes': 'Minutes Trend',
            'injury_boost': 'Injury Boost',
            'usage_rate': 'Usage Rate',
            'shot_volume': 'Shot Volume',
            'ts_regression': 'TS% Regression',
        }

        active_adj = 0
        for key, name in adj_names.items():
            adj_val = self.adjustments.get(key, 0)
            evidence = self.evidence.get(key, '')

            if adj_val != 0:
                sign = '+' if adj_val > 0 else ''
                lines.append(f"  {name:20s} {sign}{adj_val:+.1f}%  {evidence}")
                active_adj += 1
            else:
                lines.append(f"  {name:20s}   0.0%  (not applied)")

        lines.append("")
        lines.append(f"TOTAL ADJUSTMENT: {self.total_adjustment*100:+.1f}%")
        lines.append(f"Adjustments Active: {active_adj}/{len(adj_names)}")

        # Final Projection
        lines.append("")
        lines.append(f"FINAL PROJECTION: {self.projection:.1f}")
        lines.append(f"  Base ({self.base_projection:.1f}) x Total Adj ({1 + self.total_adjustment:.3f})")

        # Edge Calculation
        lines.append("")
        lines.append("EDGE CALCULATION")
        lines.append("-" * 30)
        lines.append(f"  Projection: {self.projection:.1f}")
        lines.append(f"  Line:       {self.line}")
        lines.append(f"  Edge:       {self.edge*100:+.1f}%")

        # Confidence
        lines.append("")
        lines.append(f"CONFIDENCE: {self.confidence*100:.0f}%")

        # Context Quality
        lines.append("")
        lines.append(f"CONTEXT QUALITY: {self.context_quality}/100")
        quality_desc = (
            "EXCELLENT" if self.context_quality >= 80 else
            "GOOD" if self.context_quality >= 60 else
            "FAIR" if self.context_quality >= 40 else
            "POOR"
        )
        lines.append(f"  Rating: {quality_desc}")

        # Pick
        lines.append("")
        if self.pick == 'PASS':
            lines.append(f"PICK: PASS (edge {self.edge*100:+.1f}% below threshold)")
        else:
            lines.append(f"PICK: {self.pick} {self.line}")
            lines.append(f"  Edge: {self.edge*100:+.1f}%, Confidence: {self.confidence*100:.0f}%")

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            lines.append("-" * 30)
            for w in self.warnings:
                lines.append(f"  * {w}")
        else:
            lines.append("")
            lines.append("WARNINGS: None")

        # Flags
        if self.flags:
            lines.append("")
            lines.append(f"FLAGS: {', '.join(self.flags)}")

        lines.append("")
        lines.append("=" * 60)

        output = '\n'.join(lines)

        if return_string:
            return output
        else:
            print(output)
            return None

    def is_high_quality(self) -> bool:
        """Returns True if context quality is sufficient for a confident pick."""
        return self.context_quality >= 50 and len(self.warnings) <= 2
