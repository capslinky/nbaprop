"""PDF output helpers."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:  # Prefer full team name normalization when available.
    from core.constants import normalize_team_abbrev as _normalize_team_abbrev
except Exception:  # pragma: no cover - fallback in minimal installs
    from nbaprop.normalization.ids import canonicalize_team_abbrev as _normalize_team_abbrev


def _format_game_time(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%Y-%m-%d %I:%M %p ET").lstrip("0")
    except Exception:
        return str(value)


def _parse_game_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_game_time(value: Optional[str]) -> Optional[str]:
    parsed = _parse_game_time(value)
    if parsed:
        return parsed.replace(tzinfo=None).isoformat()
    return value


def _coerce_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_pct(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def _truncate(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


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


def _normalize_detail(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, list):
                return "; ".join(str(item) for item in parsed if item)
        except Exception:
            return text
    return text


def _build_usage_context(row: Dict) -> str:
    parts = []
    usage = row.get("usage_expectation")
    if usage:
        parts.append(usage)

    player_name = row.get("player_name") or "this player"
    out_detail = _normalize_detail(row.get("team_out_detail"))
    q_detail = _normalize_detail(row.get("team_questionable_detail"))
    p_detail = _normalize_detail(row.get("team_probable_detail"))
    if out_detail:
        parts.append(f"Teammates OUT: {out_detail} — usage/minutes likely up for {player_name}.")
    if q_detail:
        parts.append(f"Teammates Q: {q_detail} — rotation risk; monitor availability.")
    if p_detail:
        parts.append(f"Teammates P: {p_detail} — less risk of surprise absences.")

    injury_status = row.get("injury_status") or ""
    injury_detail = row.get("injury_detail") or ""
    if injury_status:
        injury_text = f"Status: {injury_status}"
        if injury_detail:
            injury_text = f"{injury_text} ({injury_detail})"
        parts.append(injury_text)

    if not out_detail and not q_detail and not p_detail:
        out_count = _coerce_int(row.get("team_out_count"))
        q_count = _coerce_int(row.get("team_questionable_count"))
        p_count = _coerce_int(row.get("team_probable_count"))
        if out_count or q_count or p_count:
            parts.append(f"Team OUT {out_count} Q {q_count} P {p_count}")

    rest_days = row.get("rest_days")
    if rest_days not in (None, ""):
        parts.append(f"Rest {rest_days}d")
    if str(row.get("b2b")).lower() == "true":
        parts.append("B2B")

    return " | ".join(parts)


def _build_explanation(row: Dict, limit: Optional[int] = None) -> str:
    parts = []
    projection = row.get("projection")
    line = row.get("line")
    if projection not in (None, "") and line not in (None, ""):
        parts.append(f"Proj {projection} vs line {line}")

    trend = row.get("trend")
    if trend:
        parts.append(f"Trend: {trend}")

    hit_10 = row.get("recent_hit_rate_10")
    hit_15 = row.get("recent_hit_rate_15")
    if hit_10 not in (None, ""):
        try:
            parts.append(f"Hit10 {float(hit_10) * 100:.0f}%")
        except (TypeError, ValueError):
            pass
    if hit_15 not in (None, ""):
        try:
            parts.append(f"Hit15 {float(hit_15) * 100:.0f}%")
        except (TypeError, ValueError):
            pass

    usage = row.get("usage_expectation")
    if usage:
        parts.append(usage)

    notes = row.get("adjustment_notes")
    if notes:
        parts.append(notes)

    injury = row.get("injury_status")
    detail = row.get("injury_detail")
    if injury:
        if detail:
            parts.append(f"Injury: {injury} ({detail})")
        else:
            parts.append(f"Injury: {injury}")

    text = "; ".join(parts)
    if limit:
        return _truncate(text, limit)
    return text


def _build_recent_summary(row: Dict) -> str:
    hit_10 = _safe_float(row.get("recent_hit_rate_10"))
    hit_15 = _safe_float(row.get("recent_hit_rate_15"))
    streak = row.get("recent_streak")
    trend = row.get("trend") or ""
    parts = []
    if hit_10 is not None:
        parts.append(f"H10 {hit_10 * 100:.0f}%")
    if hit_15 is not None:
        parts.append(f"H15 {hit_15 * 100:.0f}%")
    if streak not in (None, ""):
        parts.append(f"Stk {streak}")
    if trend:
        parts.append(trend)
    return " ".join(parts)


def _format_odds(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _cell_paragraph(text: str, style) -> "Paragraph":
    from reportlab.platypus import Paragraph
    return Paragraph(text or "", style)


def _summarize_game_injuries(game_rows: List[Dict]) -> str:
    team_counts: Dict[str, Tuple[int, int, int]] = {}
    team_details: Dict[str, Dict[str, str]] = {}
    for row in game_rows:
        team = row.get("player_team") or ""
        if not team:
            continue
        details = team_details.setdefault(team, {"out": "", "questionable": "", "probable": ""})
        if not details["out"]:
            details["out"] = _normalize_detail(row.get("team_out_detail"))
        if not details["questionable"]:
            details["questionable"] = _normalize_detail(row.get("team_questionable_detail"))
        if not details["probable"]:
            details["probable"] = _normalize_detail(row.get("team_probable_detail"))
        out_count = _coerce_int(row.get("team_out_count"))
        q_count = _coerce_int(row.get("team_questionable_count"))
        p_count = _coerce_int(row.get("team_probable_count"))
        existing = team_counts.get(team)
        if existing is None:
            team_counts[team] = (out_count, q_count, p_count)
        else:
            team_counts[team] = (
                max(existing[0], out_count),
                max(existing[1], q_count),
                max(existing[2], p_count),
            )
    if not team_counts:
        return ""
    parts = []
    for team, counts in sorted(team_counts.items()):
        details = team_details.get(team, {})
        out_detail = details.get("out") or ""
        q_detail = details.get("questionable") or ""
        p_detail = details.get("probable") or ""
        if out_detail or q_detail or p_detail:
            segments = []
            if out_detail:
                segments.append(f"OUT: {out_detail}")
            if q_detail:
                segments.append(f"Q: {q_detail}")
            if p_detail:
                segments.append(f"P: {p_detail}")
            parts.append(f"{team} " + " | ".join(segments))
        else:
            parts.append(f"{team} OUT {counts[0]} Q {counts[1]} P {counts[2]}")
    return " | ".join(parts)


def _event_key(event: Dict) -> Tuple:
    return (
        _normalize_game_time(event.get("commence_time") or event.get("game_time")),
        _normalize_team_abbrev(event.get("away_team") or ""),
        _normalize_team_abbrev(event.get("home_team") or ""),
    )


def _group_by_game(
    rows: List[Dict],
    all_games: Optional[List[Dict]] = None,
) -> List[Tuple[Tuple, List[Dict]]]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = (
            _normalize_game_time(row.get("game_time")),
            _normalize_team_abbrev(row.get("away_team") or ""),
            _normalize_team_abbrev(row.get("home_team") or ""),
        )
        grouped.setdefault(key, []).append(row)

    if all_games:
        for event in all_games:
            key = _event_key(event)
            grouped.setdefault(key, [])

    def sort_key(item: Tuple[Tuple, List[Dict]]):
        game_time = _parse_game_time(item[0][0])
        return game_time or datetime.min

    return sorted(grouped.items(), key=sort_key)


def _scale_widths(widths: List[float], total_width: float) -> List[float]:
    if not widths:
        return widths
    sum_widths = sum(widths)
    if sum_widths <= 0:
        return widths
    return [(total_width * width / sum_widths) for width in widths]


def write_picks_pdf(
    rows: List[Dict],
    output_path: str,
    title: Optional[str] = None,
    all_games: Optional[List[Dict]] = None,
) -> Optional[str]:
    """Write picks to a PDF report."""
    if not rows:
        logger.warning("No picks available for PDF output.")
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
        )
    except Exception as exc:
        logger.warning("reportlab not available; skipping PDF output: %s", exc)
        return None

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
    explanation_style = ParagraphStyle(
        "Explanation",
        parent=styles["Normal"],
        fontSize=9,
        leading=10,
        textColor=colors.HexColor("#1f2937"),
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=10,
        textColor=colors.HexColor("#111827"),
    )
    cell_style_small = ParagraphStyle(
        "CellSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=9,
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

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        leftMargin=0.3 * inch,
        rightMargin=0.3 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )
    available_width = doc.width

    report_title = title or "NBA Prop Picks"
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p").lstrip("0")

    elements = [
        Paragraph(report_title, title_style),
        Paragraph(f"Generated: {generated_at} | Total Picks: {len(rows)}", subtitle_style),
        Spacer(1, 10),
    ]

    overs = sum(1 for row in rows if row.get("pick") == "OVER")
    unders = sum(1 for row in rows if row.get("pick") == "UNDER")
    edges = []
    confs = []
    for row in rows:
        try:
            edges.append(float(row.get("edge")))
        except (TypeError, ValueError):
            continue
    for row in rows:
        try:
            confs.append(float(row.get("confidence")))
        except (TypeError, ValueError):
            continue
    avg_edge = (sum(edges) / len(edges)) if edges else 0
    avg_conf = (sum(confs) / len(confs)) if confs else 0

    summary_table = Table([
        ["Total Picks", "Overs", "Unders", "Avg Edge", "Avg Conf"],
        [len(rows), overs, unders, f"{avg_edge * 100:.1f}%", f"{avg_conf * 100:.0f}%"],
    ])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.extend([
        summary_table,
        Paragraph("All scheduled games are listed; games without picks are noted.", subtitle_style),
        Spacer(1, 12),
    ])

    elements.append(Paragraph("TOP 10 PICKS (BY EDGE)", section_style))
    rows_sorted = sorted(rows, key=lambda r: float(r.get("edge") or 0.0), reverse=True)
    top_rows = [["Player (Team)", "Prop / Line", "Pick / Odds", "Edge / Conf", "Why It Ranks"]]
    for row in rows_sorted[:10]:
        edge_val = _format_pct(_safe_float(row.get("edge")), digits=1)
        conf_val = _format_pct(_safe_float(row.get("confidence")), digits=0)
        prop_line = f"{row.get('prop_type') or ''} {row.get('line') or ''}".strip()
        pick_odds = f"{row.get('pick') or ''} {_format_odds(row.get('odds'))}".strip()
        notes = " | ".join(
            part
            for part in [
                _build_explanation(row),
                _build_usage_context(row),
            ]
            if part
        )
        top_rows.append([
            _cell_paragraph(_player_label(row), cell_style),
            _cell_paragraph(prop_line, cell_style_center),
            _cell_paragraph(pick_odds, cell_style_center),
            _cell_paragraph(f"{edge_val} / {conf_val}", cell_style_center),
            _cell_paragraph(notes, cell_style_small),
        ])
    top_table = Table(
        top_rows,
        repeatRows=1,
        colWidths=_scale_widths([2.6, 1.2, 1.0, 1.1, 4.1], available_width),
    )
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.extend([top_table, Spacer(1, 12)])

    elements.append(Paragraph("RECENT HITS LOG", section_style))
    hit_rows = [["Player (Team)", "Prop", "Hit10", "Hit15", "Streak / Trend"]]
    for row in rows_sorted[:15]:
        hit_10 = _safe_float(row.get("recent_hit_rate_10"))
        hit_15 = _safe_float(row.get("recent_hit_rate_15"))
        hit_10_val = _format_pct(hit_10, digits=0)
        hit_15_val = _format_pct(hit_15, digits=0)
        streak = row.get("recent_streak")
        trend = row.get("trend") or ""
        streak_label = f"{streak}" if streak not in (None, "") else ""
        streak_trend = " ".join(part for part in [streak_label, trend] if part)
        hit_rows.append([
            _cell_paragraph(_player_label(row), cell_style),
            _cell_paragraph(row.get("prop_type") or "", cell_style_center),
            _cell_paragraph(hit_10_val, cell_style_center),
            _cell_paragraph(hit_15_val, cell_style_center),
            _cell_paragraph(streak_trend, cell_style_center),
        ])
    hit_table = Table(
        hit_rows,
        repeatRows=1,
        colWidths=_scale_widths([3.4, 0.9, 0.8, 0.8, 1.3], available_width),
    )
    hit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.extend([hit_table, Spacer(1, 14)])

    elements.append(Paragraph("BREAKDOWN BY GAME", section_style))
    for (game_time, away, home), game_rows in _group_by_game(rows, all_games=all_games):
        matchup = f"{away} @ {home}" if home and away else "Matchup"
        elements.append(Spacer(1, 10))
        header_rows = [
            [Paragraph(matchup, styles["Heading3"]), Paragraph(_format_game_time(game_time), styles["Heading3"])],
            [
                Paragraph(
                    _summarize_game_injuries(game_rows) or "Injury summary unavailable",
                    explanation_style,
                ),
                Paragraph(f"Picks: {len(game_rows)}", explanation_style),
            ],
        ]
        header_table = Table(
            header_rows,
            colWidths=_scale_widths([7.0, 3.0], available_width),
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(header_table)

        if not game_rows:
            elements.append(Paragraph("No picks met thresholds for this game.", explanation_style))
            continue

        table_rows = [[
            "Player (Team)",
            "Prop / Line",
            "Pick / Odds",
            "Edge / Conf",
            "Recent",
            "Usage / Injury / Notes",
        ]]

        for row in game_rows:
            edge_val = _format_pct(_safe_float(row.get("edge")), digits=1)
            conf_val = _format_pct(_safe_float(row.get("confidence")), digits=0)
            recent_summary = _build_recent_summary(row)
            usage = _build_usage_context(row)
            explanation = _build_explanation(row)
            notes = " | ".join(part for part in [usage, explanation] if part)
            prop_line = f"{row.get('prop_type') or ''} {row.get('line') or ''}".strip()
            pick_odds = f"{row.get('pick') or ''} {_format_odds(row.get('odds'))}".strip()

            table_rows.append([
                _cell_paragraph(_player_label(row), cell_style),
                _cell_paragraph(prop_line, cell_style_center),
                _cell_paragraph(pick_odds, cell_style_center),
                _cell_paragraph(f"{edge_val} / {conf_val}", cell_style_center),
                _cell_paragraph(recent_summary, cell_style_center),
                _cell_paragraph(notes, cell_style_small),
            ])

        table = Table(
            table_rows,
            repeatRows=1,
            colWidths=_scale_widths([2.8, 1.2, 1.1, 1.1, 0.9, 4.0], available_width),
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)

    doc.build(elements)
    return str(path)
