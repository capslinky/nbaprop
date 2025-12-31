"""PDF output for research-ranked picks."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _parse_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_pct(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def _format_odds(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _format_score(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _parse_game_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _infer_opponent(row: Dict) -> str:
    team = row.get("player_team") or ""
    home = row.get("home_team") or ""
    away = row.get("away_team") or ""
    if team and home and away:
        if team == home:
            return away
        if team == away:
            return home
    return ""


def _player_label(row: Dict) -> str:
    player = row.get("player_name") or ""
    team = row.get("player_team") or ""
    opponent = row.get("opponent_team") or _infer_opponent(row)
    if team and opponent:
        return f"{player} ({team} vs {opponent})"
    if team:
        return f"{player} ({team})"
    return player


def _news_summary(row: Dict) -> str:
    player_status = row.get("player_news_status") or "NO_NEWS"
    team_status = row.get("team_news_status") or "NO_NEWS"
    player_flags = row.get("player_news_flags") or ""
    team_flags = row.get("team_news_flags") or ""
    injury_status = row.get("injury_report_status") or ""
    injury_note = row.get("injury_context_note") or ""
    parts = [f"Player: {player_status}", f"Team: {team_status}"]
    if player_flags:
        parts.append(player_flags)
    if team_flags:
        parts.append(team_flags)
    if injury_status:
        parts.append(f"Injury: {injury_status}")
    if injury_note:
        parts.append(injury_note)
    return " | ".join(part for part in parts if part)


def _scale_widths(widths: List[float], total_width: float) -> List[float]:
    if not widths:
        return widths
    sum_widths = sum(widths)
    if sum_widths <= 0:
        return widths
    return [(total_width * width / sum_widths) for width in widths]


def _sort_rows(rows: List[Dict]) -> List[Dict]:
    def sort_key(row: Dict) -> Tuple[int, float]:
        rank = _parse_int(row.get("research_rank"))
        score = _parse_float(row.get("research_score")) or 0.0
        if rank is None:
            rank = 10 ** 6
        return (rank, -score)

    return sorted(rows, key=sort_key)


def _first_value(rows: List[Dict], key: str) -> Optional[str]:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _resolved_line(row: Dict) -> str:
    return str(row.get("latest_line") or row.get("line") or "").strip()


def _resolved_odds(row: Dict) -> str:
    return _format_odds(row.get("latest_odds") or row.get("odds"))


def write_research_pdf(
    rows: List[Dict],
    output_path: str,
    title: Optional[str] = None,
    top_n: Optional[int] = None,
) -> Optional[str]:
    """Write research-ranked picks to a PDF report."""
    if not rows:
        logger.warning("No research rows available for PDF output.")
        return None

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
    except Exception as exc:
        logger.warning("reportlab not available; skipping PDF output: %s", exc)
        return None

    rows_sorted = _sort_rows(rows)
    if top_n:
        rows_sorted = rows_sorted[: max(1, top_n)]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=17,
        alignment=1,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=1,
        textColor=colors.HexColor("#334155"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0f172a"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#111827"),
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1f2937"),
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=10,
        textColor=colors.HexColor("#111827"),
    )
    cell_style_center = ParagraphStyle(
        "CellCenter",
        parent=styles["BodyText"],
        fontSize=9,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#111827"),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#111827"),
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    available_width = doc.width

    report_title = title or "NBA Prop Picks — Research Re-Rank"
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p").lstrip("0")
    injury_as_of = _first_value(rows_sorted, "injury_report_as_of")
    odds_as_of = _first_value(rows_sorted, "odds_snapshot_fetched_at") or _first_value(
        rows_sorted, "latest_odds_fetched_at",
    )
    odds_books = _first_value(rows_sorted, "odds_snapshot_bookmakers")

    elements = [
        Paragraph(report_title, title_style),
        Paragraph(f"Generated: {generated_at} | Picks: {len(rows_sorted)}", subtitle_style),
        Spacer(1, 4),
    ]
    if injury_as_of or odds_as_of or odds_books:
        context_parts = []
        if injury_as_of:
            context_parts.append(f"Injury Report: {injury_as_of}")
        if odds_as_of:
            context_parts.append(f"Odds Snapshot: {odds_as_of}")
        if odds_books:
            context_parts.append(f"Bookmakers: {odds_books}")
        elements.extend([
            Paragraph(" | ".join(context_parts), subtitle_style),
            Spacer(1, 6),
        ])
    else:
        elements.append(Spacer(1, 10))

    avg_score = sum(_parse_float(r.get("research_score")) or 0.0 for r in rows_sorted) / len(rows_sorted)
    summary_table = Table([
        ["Researched Picks", "Avg Research Score", "News Fields"],
        [len(rows_sorted), f"{avg_score:.2f}", "Player + Team"],
    ])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.extend([summary_table, Spacer(1, 12)])

    elements.append(Paragraph("TOP PICKS (RESEARCH RANK)", section_style))
    table_rows = [[
        "Rank",
        "Player (Team vs Opp)",
        "Prop / Line",
        "Pick / Odds",
        "Edge / Conf",
        "Research Score",
        "News Summary",
    ]]

    for row in rows_sorted:
        rank = _parse_int(row.get("research_rank")) or ""
        prop_line = f"{row.get('prop_type') or ''} {_resolved_line(row)}".strip()
        pick_odds = f"{row.get('pick') or ''} {_resolved_odds(row)}".strip()
        edge_val = _format_pct(_parse_float(row.get("edge")), digits=1)
        conf_val = _format_pct(_parse_float(row.get("confidence")), digits=0)
        research_score = _format_score(_parse_float(row.get("research_score")))
        table_rows.append([
            Paragraph(str(rank), cell_style_center),
            Paragraph(_player_label(row), cell_style),
            Paragraph(prop_line, cell_style_center),
            Paragraph(pick_odds, cell_style_center),
            Paragraph(f"{edge_val} / {conf_val}", cell_style_center),
            Paragraph(research_score, cell_style_center),
            Paragraph(_news_summary(row), small_style),
        ])

    picks_table = Table(
        table_rows,
        repeatRows=1,
        colWidths=_scale_widths([0.5, 2.6, 1.0, 0.9, 1.0, 1.0, 3.0], available_width),
    )
    picks_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elements.extend([picks_table, Spacer(1, 14), PageBreak()])

    elements.append(Paragraph("RESEARCH DETAILS", section_style))
    for row in rows_sorted:
        rank = _parse_int(row.get("research_rank")) or ""
        player = _player_label(row)
        prop_line = f"{row.get('prop_type') or ''} {_resolved_line(row)}".strip()
        pick_odds = f"{row.get('pick') or ''} {_resolved_odds(row)}".strip()
        edge_val = _format_pct(_parse_float(row.get("edge")), digits=1)
        conf_val = _format_pct(_parse_float(row.get("confidence")), digits=0)
        score = _format_score(_parse_float(row.get("research_score")))
        base_rank = _parse_int(row.get("base_rank"))

        heading = f"#{rank} {player} — {prop_line} {pick_odds}"
        elements.append(Paragraph(heading, styles["Heading3"]))
        meta_line = f"Research Score: {score} | Edge: {edge_val} | Conf: {conf_val}"
        if base_rank is not None:
            meta_line += f" | Base Rank: {base_rank}"
        elements.append(Paragraph(meta_line, body_style))

        latest_odds = _resolved_odds(row)
        latest_line = _resolved_line(row)
        latest_fetched = row.get("latest_odds_fetched_at") or row.get("odds_snapshot_fetched_at")
        orig_odds = _format_odds(row.get("odds"))
        orig_line = str(row.get("line") or "").strip()
        if latest_odds or latest_line:
            odds_line = f"Latest Odds/Line: {latest_line} {latest_odds}".strip()
            if latest_fetched:
                odds_line += f" (as of {latest_fetched})"
            if orig_odds and orig_line and (orig_odds != latest_odds or orig_line != latest_line):
                odds_line += f" | Original: {orig_line} {orig_odds}"
            elements.append(Paragraph(odds_line, label_style))

        research_notes = row.get("research_notes") or ""
        usage = row.get("usage_expectation") or ""
        if research_notes:
            elements.append(Paragraph(f"Research Notes: {research_notes}", body_style))
        if usage:
            elements.append(Paragraph(f"Usage / Context: {usage}", body_style))

        injury_status = row.get("injury_report_status") or ""
        injury_detail = row.get("injury_report_detail") or ""
        injury_source = row.get("injury_report_source") or ""
        injury_teammates_out = row.get("injury_teammates_out") or ""
        injury_teammates_questionable = row.get("injury_teammates_questionable") or ""
        injury_note = row.get("injury_context_note") or ""
        if (
            injury_status
            or injury_detail
            or injury_source
            or injury_teammates_out
            or injury_teammates_questionable
            or injury_note
        ):
            injury_parts = []
            if injury_status:
                injury_parts.append(f"Player Status: {injury_status}")
            if injury_detail:
                injury_parts.append(f"Injury: {injury_detail}")
            if injury_source:
                injury_parts.append(f"Source: {injury_source}")
            if injury_teammates_out:
                injury_parts.append(f"Teammates Out: {injury_teammates_out}")
            if injury_teammates_questionable:
                injury_parts.append(f"Teammates Q: {injury_teammates_questionable}")
            if injury_note:
                injury_parts.append(f"Context: {injury_note}")
            elements.append(Paragraph("Injury Context: " + " | ".join(injury_parts), label_style))

        player_status = row.get("player_news_status") or "NO_NEWS"
        player_notes = row.get("player_news_notes") or ""
        player_sources = row.get("player_news_sources") or ""
        if player_status or player_notes or player_sources:
            elements.append(Paragraph(
                f"Player News: {player_status} | {player_notes} | Sources: {player_sources}",
                label_style,
            ))

        team_status = row.get("team_news_status") or "NO_NEWS"
        team_notes = row.get("team_news_notes") or ""
        team_sources = row.get("team_news_sources") or ""
        if team_status or team_notes or team_sources:
            elements.append(Paragraph(
                f"Team News: {team_status} | {team_notes} | Sources: {team_sources}",
                label_style,
            ))

        elements.append(Spacer(1, 10))

    doc.build(elements)
    return str(path)
