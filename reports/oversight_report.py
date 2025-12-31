"""
Oversight Report Generator
==========================
Generates daily oversight reports with full audit trail for pick verification.

Usage:
    from reports.oversight_report import OversightReportGenerator
    from pick_tracker import PickTracker

    tracker = PickTracker()
    generator = OversightReportGenerator(tracker)

    # Generate both PDF and console output
    generator.generate_full_report(days=7)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class OversightReportGenerator:
    """Generate daily oversight reports with full audit trail."""

    def __init__(self, tracker):
        """
        Initialize the oversight report generator.

        Args:
            tracker: PickTracker instance
        """
        self.tracker = tracker
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
            name='SubSection',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#555555'),
        ))
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#cc0000'),
        ))
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#006600'),
        ))

    def generate_report_data(self, days: int = 7) -> Dict:
        """
        Generate all data needed for the oversight report.

        Args:
            days: Number of days to include in report

        Returns:
            Dictionary with all report data
        """
        data = {
            'generated_at': datetime.now().isoformat(),
            'days': days,
            'daily_summary': self.tracker.get_daily_summary(days),
            'audit_trail': self.tracker.get_full_audit_trail(days),
            'performance_breakdown': self.tracker.get_performance_breakdown(days),
            'data_integrity': self.tracker.get_data_integrity_issues(),
        }

        # Calculate overall stats
        df = data['audit_trail']
        if not df.empty:
            with_results = df[df['result'] != 'PENDING']
            wins = len(with_results[with_results['result'] == 'WIN'])
            losses = len(with_results[with_results['result'] == 'LOSS'])
            pushes = len(with_results[with_results['result'] == 'PUSH'])

            data['overall'] = {
                'total_picks': len(df),
                'with_results': len(with_results),
                'pending': len(df[df['result'] == 'PENDING']),
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'win_rate': wins / (wins + losses) if (wins + losses) > 0 else None,
            }
        else:
            data['overall'] = {
                'total_picks': 0,
                'with_results': 0,
                'pending': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': None,
            }

        return data

    def print_console_report(self, days: int = 7):
        """
        Print formatted oversight report to console.

        Args:
            days: Number of days to include
        """
        data = self.generate_report_data(days)

        print("\n" + "=" * 70)
        print(f"  DAILY OVERSIGHT REPORT - {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 70)

        # Summary
        overall = data['overall']
        print(f"\nSUMMARY (Last {days} Days)")
        print("-" * 50)
        print(f"  Total Picks:     {overall['total_picks']:,}")
        print(f"  With Results:    {overall['with_results']:,} ({overall['with_results']/overall['total_picks']*100:.0f}%)" if overall['total_picks'] > 0 else "  With Results:    0")
        print(f"  Pending:         {overall['pending']:,}")
        print()

        if overall['wins'] + overall['losses'] > 0:
            win_rate = overall['win_rate'] * 100 if overall['win_rate'] else 0
            print(f"  Record:          {overall['wins']}-{overall['losses']}-{overall['pushes']} ({win_rate:.1f}%)")
        else:
            print("  Record:          No results yet")

        # Daily Breakdown
        print(f"\nDAILY BREAKDOWN")
        print("-" * 50)
        daily_df = data['daily_summary']
        if not daily_df.empty:
            print(f"  {'Date':<12} {'Picks':>6} {'W':>4} {'L':>4} {'P':>4} {'Win%':>7} {'Avg Edge':>9}")
            print("  " + "-" * 55)
            for _, row in daily_df.iterrows():
                win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row['win_rate']) else "---"
                avg_edge_str = f"{row['avg_edge']*100:.1f}%" if pd.notna(row['avg_edge']) else "---"
                print(f"  {row['date']:<12} {int(row['total_picks']):>6} {int(row['wins']):>4} {int(row['losses']):>4} {int(row['pushes']):>4} {win_rate_str:>7} {avg_edge_str:>9}")
        else:
            print("  No data available")

        # Performance Breakdowns
        breakdown = data['performance_breakdown']

        print(f"\nBY PROP TYPE")
        print("-" * 50)
        for row in breakdown.get('by_prop_type', []):
            wins, losses = int(row['wins']), int(row['losses'])
            total = wins + losses
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) and total > 0 else "---"
            print(f"  {row['prop_type'].upper():<12} : {wins:>3}-{losses:<3} ({win_rate_str})")

        print(f"\nBY DIRECTION")
        print("-" * 50)
        for row in breakdown.get('by_direction', []):
            wins, losses = int(row['wins']), int(row['losses'])
            total = wins + losses
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) and total > 0 else "---"
            print(f"  {row['direction']:<12} : {wins:>3}-{losses:<3} ({win_rate_str})")

        print(f"\nBY EDGE TIER")
        print("-" * 50)
        for row in breakdown.get('by_edge_tier', []):
            wins, losses = int(row['wins']), int(row['losses'])
            total = wins + losses
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) and total > 0 else "---"
            print(f"  {row['edge_tier']:<18} : {wins:>3}-{losses:<3} ({win_rate_str})")

        # Data Integrity
        integrity = data['data_integrity']
        print(f"\nDATA INTEGRITY")
        print("-" * 50)

        if integrity['missing_results_count'] > 0:
            print(f"  WARNING: {integrity['missing_results_count']} picks missing results!")
            print()
            print("  Dates with missing results:")
            for item in integrity['missing_results_dates'][:5]:  # Show top 5
                print(f"    {item['date']}: {item['count']} picks missing")
            if len(integrity['missing_results_dates']) > 5:
                print(f"    ... and {len(integrity['missing_results_dates']) - 5} more dates")
        else:
            print("  All picks have results recorded.")

        if integrity.get('unusual_values'):
            print()
            print(f"  WARNING: {len(integrity['unusual_values'])} unusual values detected!")
            for item in integrity['unusual_values'][:3]:
                print(f"    {item['date']} {item['player']} {item['prop_type']}: {item['actual']} (line: {item['line']})")

        coverage = integrity.get('date_coverage', {})
        if coverage:
            print()
            print(f"  Date Range: {coverage.get('first_date')} to {coverage.get('last_date')}")
            print(f"  Days with Data: {coverage.get('days_with_picks')}")

        print("\n" + "=" * 70 + "\n")

    def generate_pdf(self, days: int = 7, output_path: str = None) -> str:
        """
        Generate PDF oversight report.

        Args:
            days: Number of days to include
            output_path: Output file path (default: oversight_YYYY-MM-DD.pdf)

        Returns:
            Path to generated PDF
        """
        if output_path is None:
            output_path = f"oversight_{datetime.now().strftime('%Y-%m-%d')}.pdf"

        data = self.generate_report_data(days)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )

        elements = []

        # Header
        elements.append(Paragraph("DAILY OVERSIGHT REPORT", self.styles['ReportTitle']))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Period: Last {days} Days",
            self.styles['ReportSubtitle']
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e')))
        elements.append(Spacer(1, 20))

        # Executive Summary
        elements.extend(self._create_summary_section(data))

        # Daily Breakdown Table
        elements.extend(self._create_daily_breakdown_section(data))

        # Performance Breakdowns
        elements.extend(self._create_performance_section(data))

        # Data Integrity
        elements.extend(self._create_integrity_section(data))

        # Full Audit Trail
        elements.extend(self._create_audit_trail_section(data))

        doc.build(elements)

        return output_path

    def _create_summary_section(self, data: Dict) -> list:
        """Create executive summary section."""
        elements = []
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))

        overall = data['overall']
        if overall['total_picks'] > 0:
            win_rate_str = f"{overall['win_rate']*100:.1f}%" if overall['win_rate'] else "N/A"
            summary_text = f"""
            <b>Total Picks:</b> {overall['total_picks']:,}<br/>
            <b>With Results:</b> {overall['with_results']:,} ({overall['with_results']/overall['total_picks']*100:.0f}%)<br/>
            <b>Pending:</b> {overall['pending']:,}<br/><br/>
            <b>Record:</b> {overall['wins']}-{overall['losses']}-{overall['pushes']}<br/>
            <b>Win Rate:</b> {win_rate_str}<br/>
            """
        else:
            summary_text = "No picks found in the specified date range."

        elements.append(Paragraph(summary_text, self.styles['ReportBody']))
        elements.append(Spacer(1, 20))

        return elements

    def _create_daily_breakdown_section(self, data: Dict) -> list:
        """Create daily breakdown table section."""
        elements = []
        elements.append(Paragraph("DAILY BREAKDOWN", self.styles['SectionHeader']))

        daily_df = data['daily_summary']
        if daily_df.empty:
            elements.append(Paragraph("No daily data available.", self.styles['ReportBody']))
            return elements

        # Create table data
        table_data = [['Date', 'Picks', 'W', 'L', 'P', 'Win%', 'Avg Edge']]
        for _, row in daily_df.iterrows():
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row['win_rate']) else "---"
            avg_edge_str = f"{row['avg_edge']*100:.1f}%" if pd.notna(row['avg_edge']) else "---"
            table_data.append([
                row['date'],
                str(int(row['total_picks'])),
                str(int(row['wins'])),
                str(int(row['losses'])),
                str(int(row['pushes'])),
                win_rate_str,
                avg_edge_str,
            ])

        table = Table(table_data, colWidths=[1.2*inch, 0.6*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.7*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 20))

        return elements

    def _create_performance_section(self, data: Dict) -> list:
        """Create performance breakdown section."""
        elements = []
        elements.append(Paragraph("PERFORMANCE BREAKDOWNS", self.styles['SectionHeader']))

        breakdown = data['performance_breakdown']

        # By Prop Type
        elements.append(Paragraph("By Prop Type", self.styles['SubSection']))
        prop_data = [['Prop Type', 'W', 'L', 'Win Rate']]
        for row in breakdown.get('by_prop_type', []):
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) else "---"
            prop_data.append([
                row['prop_type'].upper(),
                str(int(row['wins'])),
                str(int(row['losses'])),
                win_rate_str,
            ])

        if len(prop_data) > 1:
            prop_table = Table(prop_data, colWidths=[1.5*inch, 0.6*inch, 0.6*inch, 1*inch])
            prop_table.setStyle(self._get_breakdown_table_style())
            elements.append(prop_table)
        elements.append(Spacer(1, 15))

        # By Direction
        elements.append(Paragraph("By Direction", self.styles['SubSection']))
        dir_data = [['Direction', 'W', 'L', 'Win Rate']]
        for row in breakdown.get('by_direction', []):
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) else "---"
            dir_data.append([
                row['direction'],
                str(int(row['wins'])),
                str(int(row['losses'])),
                win_rate_str,
            ])

        if len(dir_data) > 1:
            dir_table = Table(dir_data, colWidths=[1.5*inch, 0.6*inch, 0.6*inch, 1*inch])
            dir_table.setStyle(self._get_breakdown_table_style())
            elements.append(dir_table)
        elements.append(Spacer(1, 15))

        # By Edge Tier
        elements.append(Paragraph("By Edge Tier", self.styles['SubSection']))
        edge_data = [['Edge Tier', 'W', 'L', 'Win Rate']]
        for row in breakdown.get('by_edge_tier', []):
            win_rate_str = f"{row['win_rate']*100:.1f}%" if pd.notna(row.get('win_rate')) else "---"
            edge_data.append([
                row['edge_tier'],
                str(int(row['wins'])),
                str(int(row['losses'])),
                win_rate_str,
            ])

        if len(edge_data) > 1:
            edge_table = Table(edge_data, colWidths=[1.5*inch, 0.6*inch, 0.6*inch, 1*inch])
            edge_table.setStyle(self._get_breakdown_table_style())
            elements.append(edge_table)

        elements.append(Spacer(1, 20))

        return elements

    def _get_breakdown_table_style(self):
        """Get standard style for breakdown tables."""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a4a6a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ])

    def _create_integrity_section(self, data: Dict) -> list:
        """Create data integrity section."""
        elements = []
        elements.append(Paragraph("DATA INTEGRITY CHECK", self.styles['SectionHeader']))

        integrity = data['data_integrity']

        if integrity['missing_results_count'] > 0:
            elements.append(Paragraph(
                f"WARNING: {integrity['missing_results_count']} picks are missing results!",
                self.styles['Warning']
            ))
            elements.append(Spacer(1, 10))

            # Show missing dates
            missing_text = "<b>Dates with missing results:</b><br/>"
            for item in integrity['missing_results_dates'][:10]:
                missing_text += f"&bull; {item['date']}: {item['count']} picks<br/>"
            if len(integrity['missing_results_dates']) > 10:
                missing_text += f"<i>...and {len(integrity['missing_results_dates']) - 10} more dates</i>"

            elements.append(Paragraph(missing_text, self.styles['ReportBody']))
        else:
            elements.append(Paragraph(
                "All picks have results recorded.",
                self.styles['Success']
            ))

        if integrity.get('unusual_values'):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(
                f"WARNING: {len(integrity['unusual_values'])} unusual values detected!",
                self.styles['Warning']
            ))

        coverage = integrity.get('date_coverage', {})
        if coverage:
            elements.append(Spacer(1, 10))
            coverage_text = f"""
            <b>Data Coverage:</b><br/>
            &bull; First Pick: {coverage.get('first_date', 'N/A')}<br/>
            &bull; Last Pick: {coverage.get('last_date', 'N/A')}<br/>
            &bull; Days with Data: {coverage.get('days_with_picks', 0)}
            """
            elements.append(Paragraph(coverage_text, self.styles['ReportBody']))

        elements.append(Spacer(1, 20))

        return elements

    def _create_audit_trail_section(self, data: Dict) -> list:
        """Create full audit trail section."""
        elements = []
        elements.append(PageBreak())
        elements.append(Paragraph("FULL AUDIT TRAIL", self.styles['SectionHeader']))
        elements.append(Paragraph(
            "Every pick with prediction vs actual result. WIN = green, LOSS = red, PENDING = gray.",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 10))

        audit_df = data['audit_trail']
        if audit_df.empty:
            elements.append(Paragraph("No picks found.", self.styles['ReportBody']))
            return elements

        # Group by date
        for date in audit_df['date'].unique():
            date_df = audit_df[audit_df['date'] == date]

            elements.append(Paragraph(f"<b>{date}</b> ({len(date_df)} picks)", self.styles['SubSection']))

            # Create table for this date (limit to 50 per page)
            table_data = [['Player', 'Prop', 'Line', 'Pick', 'Proj', 'Actual', 'Result']]

            for _, row in date_df.head(50).iterrows():
                actual_str = f"{row['actual']:.1f}" if pd.notna(row['actual']) else "---"
                table_data.append([
                    row['player'][:20],  # Truncate long names
                    row['prop_type'].upper()[:6],
                    f"{row['line']:.1f}",
                    row['pick'],
                    f"{row['projection']:.1f}",
                    actual_str,
                    row['result'],
                ])

            table = Table(table_data, colWidths=[1.8*inch, 0.6*inch, 0.5*inch, 0.6*inch, 0.5*inch, 0.6*inch, 0.7*inch])

            # Build style with conditional coloring for results
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]

            # Color code results
            for i, (_, row) in enumerate(date_df.head(50).iterrows(), 1):
                if row['result'] == 'WIN':
                    style.append(('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#c8e6c9')))
                elif row['result'] == 'LOSS':
                    style.append(('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#ffcdd2')))
                elif row['result'] == 'PENDING':
                    style.append(('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#e0e0e0')))

            table.setStyle(TableStyle(style))
            elements.append(table)

            if len(date_df) > 50:
                elements.append(Paragraph(f"<i>...and {len(date_df) - 50} more picks</i>", self.styles['ReportBody']))

            elements.append(Spacer(1, 15))

        return elements

    def generate_full_report(self, days: int = 7, output_path: str = None) -> str:
        """
        Generate both console output and PDF report.

        Args:
            days: Number of days to include
            output_path: Optional PDF output path

        Returns:
            Path to generated PDF
        """
        # Print to console
        self.print_console_report(days)

        # Generate PDF
        pdf_path = self.generate_pdf(days, output_path)
        print(f"PDF Report saved to: {pdf_path}")

        return pdf_path
