#!/usr/bin/env python3
"""
NBA Prop Picks PDF Report Generator
====================================
Generates a clean, organized PDF report of daily picks.

Usage:
    python generate_report.py                    # Use today's picks
    python generate_report.py --date 2024-12-12  # Specific date
    python generate_report.py --csv picks.csv    # From CSV file
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PicksReportGenerator:
    """Generate professional PDF reports for NBA prop picks."""

    def __init__(self, df: pd.DataFrame, date_str: str = None):
        self.df = df
        self.date_str = date_str or datetime.now().strftime('%Y-%m-%d')
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
        ))
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a4a6a'),
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e'),
        ))
        self.styles.add(ParagraphStyle(
            name='Explanation',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=15,
            textColor=colors.HexColor('#555555'),
            leftIndent=10,
        ))

    def _create_header(self):
        """Create report header."""
        elements = []

        # Title
        elements.append(Paragraph("NBA PROP PICKS REPORT", self.styles['ReportTitle']))
        elements.append(Paragraph(
            f"Generated: {self.date_str} | Total Picks: {len(self.df)}",
            self.styles['ReportSubtitle']
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
        elements.append(Spacer(1, 20))

        return elements

    def _create_summary_section(self):
        """Create executive summary."""
        elements = []

        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))

        # Calculate summary stats
        total_picks = len(self.df)
        overs = len(self.df[self.df['recommended_side'] == 'OVER'])
        unders = len(self.df[self.df['recommended_side'] == 'UNDER'])
        avg_edge = self.df['avg_edge'].mean()
        avg_confidence = self.df['confidence'].mean()

        # By prop type
        by_prop = self.df.groupby('prop_type').size().to_dict()

        # Calculate percentages safely
        over_pct = (overs / total_picks * 100) if total_picks > 0 else 0
        under_pct = (unders / total_picks * 100) if total_picks > 0 else 0

        summary_text = f"""
        <b>Overview:</b> This report contains {total_picks} value plays identified from today's NBA slate.
        Each pick has a projected edge of 5% or greater based on recent player performance data.<br/><br/>

        <b>Direction Breakdown:</b><br/>
        • OVER picks: {overs} ({over_pct:.0f}%)<br/>
        • UNDER picks: {unders} ({under_pct:.0f}%)<br/><br/>

        <b>Average Metrics:</b><br/>
        • Average Edge: {avg_edge:.1f}%<br/>
        • Average Confidence: {avg_confidence:.0f}%<br/><br/>

        <b>By Prop Type:</b><br/>
        """
        for prop_type, count in sorted(by_prop.items(), key=lambda x: -x[1]):
            summary_text += f"• {prop_type.upper()}: {count} picks<br/>"

        elements.append(Paragraph(summary_text, self.styles['Explanation']))
        elements.append(Spacer(1, 20))

        return elements

    def _create_methodology_section(self):
        """Explain the methodology."""
        elements = []

        elements.append(Paragraph("METHODOLOGY", self.styles['SectionHeader']))

        methodology_text = """
        <b>How Picks Are Generated (v2.0 - Full Context Model):</b><br/><br/>

        <b>1. Data Collection</b><br/>
        • Player game logs from last 30 games (not 15) for statistical validity<br/>
        • Live betting lines from major sportsbooks via The Odds API<br/>
        • Team defense ratings, pace data, and schedule info<br/><br/>

        <b>2. Base Projection (Mean Reversion Model)</b><br/>
        • 40% × Last 5 games + 35% × Last 10 games + 25% × Season avg<br/>
        • Incorporates mean reversion (hot streaks regress, cold streaks recover)<br/><br/>

        <b>3. Contextual Adjustments</b><br/>
        • Home/Away splits (player-specific boost/reduction)<br/>
        • Back-to-back: -7% reduction when fatigued<br/>
        • Extra rest (3+ days): +3% boost<br/>
        • Minutes trend: adjusts if recent minutes differ from average<br/>
        • News Intelligence: real-time injury reports, GTD status, minutes restrictions<br/><br/>

        <b>4. Vig-Adjusted Edge (Conservative Model)</b><br/>
        • Calculates no-vig market probability from BOTH over and under odds<br/>
        • Our probability = market probability + adjustment (capped at ±15%)<br/>
        • Adjustment based on: historical hit rate edge + projection movement<br/>
        • Edge = Our estimated probability - Breakeven probability<br/>
        • Example: -110/-110 odds = 50% market, +5% adjustment = 55% our prob, +2.6% edge<br/><br/>

        <b>5. Confidence Score (Multi-Factor)</b><br/>
        • 40% consistency (low variance = higher confidence)<br/>
        • 30% sample size (more games = more confidence)<br/>
        • 30% agreement (historical hit rate in same direction)<br/>
        • Capped at 85% - no overconfidence<br/><br/>

        <b>6. Minimum Sample Sizes</b><br/>
        • Points: 10 games | Rebounds: 12 | Assists: 15<br/>
        • 3-Pointers: 20 games (high variance stat)<br/><br/>

        <b>7. Correlation + Alt Line Filtering</b><br/>
        • Removes redundant picks (e.g., if both Points and PRA picked, keeps best)<br/>
        • Removes duplicate alt lines (e.g., UNDER 23.5, 22.5, 21.5 → keeps best)<br/>
        • Prevents over-exposure to single player outcomes<br/>
        """

        elements.append(Paragraph(methodology_text, self.styles['Explanation']))
        elements.append(Spacer(1, 10))

        return elements

    def _generate_explanation(self, row) -> str:
        """Generate a brief explanation for a pick with contextual factors."""
        side = row['recommended_side']
        hit_rate = row['hit_rate_over'] if side == 'OVER' else row['hit_rate_under']
        trend = row.get('trend', 'NEUTRAL')
        recent_avg = row.get('recent_avg', row['projection'])
        season_avg = row.get('season_avg', recent_avg)

        # Build explanation
        parts = []

        # NEWS FLAGS (if present) - show first for visibility
        news_flags = row.get('news_flags', '')
        if news_flags and news_flags.strip():
            parts.append(news_flags)

        # Core reasoning: projection vs line
        parts.append(f"Proj {row['projection']:.1f} vs {row['line']} line")

        # Edge (vig-adjusted if available)
        edge = row.get('avg_edge', 0)
        if edge > 0:
            parts.append(f"+{edge:.0f}% edge (after vig)")
        else:
            parts.append(f"{edge:.0f}% edge")

        # Probability comparison
        our_prob = row.get('our_prob', 0)
        implied = row.get('implied_prob', 50)
        if our_prob and implied:
            parts.append(f"Est {our_prob:.0f}% vs {implied:.0f}% implied")

        # Hit rate
        parts.append(f"{side} hit {hit_rate:.0f}% (L{int(row.get('games_analyzed', 15))})")

        # Contextual adjustments
        adjustments = row.get('adjustments', 'None')
        if adjustments and adjustments != 'None':
            parts.append(f"Adj: {adjustments}")

        # B2B warning
        if row.get('is_b2b', False):
            parts.append("⚠️ B2B")

        # Trend if notable
        if trend in ['HOT', 'COLD']:
            parts.append(f"Trend: {trend}")

        # News notes (if present and different from flags)
        news_notes = row.get('news_notes', '')
        if news_notes and news_notes.strip() and news_notes not in str(news_flags):
            parts.append(f"📰 {news_notes}")

        return " | ".join(parts)

    def _create_top_picks_section(self, n: int = 25):
        """Create top picks table with explanations."""
        elements = []

        elements.append(Paragraph("TOP PICKS WITH EXPLANATIONS", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Each pick includes a brief explanation of why it was selected.",
            self.styles['Explanation']
        ))

        # Get top picks
        top_df = self.df.nlargest(n, 'avg_edge')

        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            hit_rate = row['hit_rate_over'] if row['recommended_side'] == 'OVER' else row['hit_rate_under']

            # Pick header
            pick_header = f"<b>#{i}. {row['player']}</b> - {row['prop_type'].upper()} {row['recommended_side']} {row['line']}"
            elements.append(Paragraph(pick_header, ParagraphStyle(
                'PickHeader',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=8,
                spaceAfter=2,
                textColor=colors.HexColor('#1a1a2e'),
            )))

            # Explanation
            explanation = self._generate_explanation(row)
            elements.append(Paragraph(explanation, ParagraphStyle(
                'PickExplanation',
                parent=self.styles['Normal'],
                fontSize=9,
                spaceAfter=6,
                textColor=colors.HexColor('#555555'),
                leftIndent=15,
            )))

            # Stats line
            stats = f"Edge: +{row['avg_edge']:.0f}% | Confidence: {row['confidence']:.0f}% | Hit Rate: {hit_rate:.0f}% | Book: {row['bookmaker']}"
            elements.append(Paragraph(stats, ParagraphStyle(
                'PickStats',
                parent=self.styles['Normal'],
                fontSize=8,
                spaceAfter=4,
                textColor=colors.HexColor('#777777'),
                leftIndent=15,
            )))

        elements.append(Spacer(1, 20))
        return elements

    def _create_by_game_section(self):
        """Create picks organized by game."""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("PICKS BY GAME", self.styles['SectionHeader']))

        # Check if game column exists
        if 'game' not in self.df.columns:
            elements.append(Paragraph("Game data not available in picks.", self.styles['Explanation']))
            return elements

        # Get unique games
        games = self.df['game'].unique()

        for game in games:
            game_df = self.df[self.df['game'] == game].nlargest(10, 'avg_edge')

            if game_df.empty:
                continue

            elements.append(Spacer(1, 15))
            elements.append(Paragraph(
                f"<b>{game}</b> ({len(self.df[self.df['game'] == game])} total picks, top 10 shown)",
                ParagraphStyle('GameHeader', parent=self.styles['Normal'], fontSize=12,
                              spaceAfter=8, textColor=colors.HexColor('#1a1a2e'))
            ))

            for _, row in game_df.iterrows():
                hit_rate = row['hit_rate_over'] if row['recommended_side'] == 'OVER' else row['hit_rate_under']

                pick_line = f"<b>{row['player']}</b> - {row['prop_type'].upper()} {row['recommended_side']} {row['line']} | Edge: +{row['avg_edge']:.0f}% | Conf: {row['confidence']:.0f}% | Proj: {row['projection']:.1f}"
                elements.append(Paragraph(pick_line, ParagraphStyle(
                    'GamePick',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    spaceBefore=2,
                    spaceAfter=2,
                    textColor=colors.HexColor('#333333'),
                    leftIndent=10,
                )))

            elements.append(Spacer(1, 10))

        return elements

    def _create_by_prop_type_section(self):
        """Create picks organized by prop type with explanations."""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("ALL PICKS BY PROP TYPE", self.styles['SectionHeader']))

        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        prop_labels = {'points': 'POINTS', 'rebounds': 'REBOUNDS', 'assists': 'ASSISTS', 'threes': '3-POINTERS', 'pra': 'PTS+REB+AST'}

        for prop_type in prop_types:
            prop_df = self.df[self.df['prop_type'] == prop_type].nlargest(20, 'avg_edge')

            if prop_df.empty:
                continue

            total_count = len(self.df[self.df['prop_type'] == prop_type])
            elements.append(Spacer(1, 15))
            elements.append(Paragraph(
                f"<b>{prop_labels.get(prop_type, prop_type.upper())}</b> ({total_count} total picks, showing top 20)",
                ParagraphStyle('PropTypeHeader', parent=self.styles['Normal'], fontSize=12,
                              spaceAfter=8, textColor=colors.HexColor('#1a1a2e'))
            ))

            for i, (_, row) in enumerate(prop_df.iterrows(), 1):
                hit_rate = row['hit_rate_over'] if row['recommended_side'] == 'OVER' else row['hit_rate_under']

                # Compact pick line with explanation
                pick_line = f"<b>{row['player']}</b> {row['recommended_side']} {row['line']} → {self._generate_explanation(row)}"
                elements.append(Paragraph(pick_line, ParagraphStyle(
                    'PropPick',
                    parent=self.styles['Normal'],
                    fontSize=8,
                    spaceBefore=2,
                    spaceAfter=2,
                    textColor=colors.HexColor('#333333'),
                    leftIndent=10,
                )))

            elements.append(Spacer(1, 10))

        return elements

    def _create_key_terms_section(self):
        """Explain key terms."""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("KEY TERMS & DEFINITIONS", self.styles['SectionHeader']))

        terms = """
        <b>Edge (Probability Edge):</b> The difference between our estimated win probability and the
        breakeven probability implied by the odds. A +5% edge means we estimate a 5% higher chance of
        winning than what the odds suggest. Example: If odds imply 52% breakeven and we estimate 57%,
        the edge is +5%.<br/><br/>

        <b>Implied Probability:</b> The breakeven win rate required by the odds. At -110 odds, you need
        to win 52.4% to break even after juice.<br/><br/>

        <b>Market Probability:</b> The no-vig (true) probability derived from both over and under odds.<br/><br/>

        <b>Confidence:</b> A measure of how consistent the player has been. Higher confidence means
        the player's stats have low variance and are more predictable. Capped at 85%.<br/><br/>

        <b>Projection:</b> Our predicted stat total, based on weighted recent performance with
        contextual adjustments (home/away, B2B, minutes trend).<br/><br/>

        <b>Hit Rate:</b> The percentage of recent games where the player exceeded/missed this line.<br/><br/>

        <b>PRA:</b> Points + Rebounds + Assists combined.<br/><br/>

        <b>Trend:</b> HOT = recent 5 games above season average, COLD = below average.<br/><br/>

        <b>OVER/UNDER:</b> The recommended bet direction. OVER means bet the player exceeds the line.
        """

        elements.append(Paragraph(terms, self.styles['Explanation']))

        return elements

    def _create_disclaimer_section(self):
        """Add disclaimer."""
        elements = []

        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        elements.append(Spacer(1, 10))

        disclaimer = """
        <b>DISCLAIMER:</b> This report is for informational purposes only. Past performance does not guarantee
        future results. Sports betting involves risk. Please gamble responsibly and only bet what you can afford to lose.
        The projections in this report are based on statistical analysis and do not account for all factors that may
        affect game outcomes (injuries, rest, motivation, etc.).
        """

        elements.append(Paragraph(disclaimer, ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#888888'),
            alignment=TA_CENTER,
        )))

        return elements

    def generate(self, output_path: str = None) -> str:
        """Generate the PDF report."""
        if output_path is None:
            output_path = f"nba_picks_report_{self.date_str}.pdf"

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )

        elements = []
        elements.extend(self._create_header())
        elements.extend(self._create_summary_section())
        elements.extend(self._create_methodology_section())
        elements.extend(self._create_top_picks_section(25))
        elements.extend(self._create_by_game_section())
        elements.extend(self._create_by_prop_type_section())
        elements.extend(self._create_key_terms_section())
        elements.extend(self._create_disclaimer_section())

        doc.build(elements)

        return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate NBA Picks PDF Report')
    parser.add_argument('--csv', type=str, help='Path to CSV file with picks')
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', '-o', type=str, help='Output PDF path')

    args = parser.parse_args()

    # Determine date
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    # Load picks
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = f"nba_daily_picks_{date_str}.csv"

    if not Path(csv_path).exists():
        print(f"Error: Could not find {csv_path}")
        print("Run 'python daily_runner.py --pre-game' first to generate picks.")
        sys.exit(1)

    print(f"Loading picks from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} picks")

    # Generate report
    generator = PicksReportGenerator(df, date_str)
    output_path = args.output or f"nba_picks_report_{date_str}.pdf"

    print(f"Generating PDF report...")
    result = generator.generate(output_path)

    print(f"Report saved to: {result}")

    return result


if __name__ == "__main__":
    main()
