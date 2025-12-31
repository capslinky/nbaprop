#!/usr/bin/env python3
"""Generate a PDF from research-ranked picks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbaprop.reporting.research_pdf_output import write_research_pdf  # noqa: E402


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _resolve_latest_research(cache_dir: Path) -> Optional[Path]:
    candidates = list(cache_dir.glob("research_*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PDF from research-ranked picks.")
    parser.add_argument("--research-path", help="Path to research CSV (defaults to latest in .cache/research).")
    parser.add_argument("--output-path", help="Output PDF path (defaults next to research CSV).")
    parser.add_argument("--top-n", type=int, help="Limit PDF to top N research rows.")
    parser.add_argument("--title", help="Custom report title.")
    args = parser.parse_args()

    research_path = Path(args.research_path) if args.research_path else None
    if research_path is None:
        research_path = _resolve_latest_research(Path(".cache/research"))
    if not research_path or not research_path.exists():
        raise SystemExit("No research CSV found. Provide --research-path or run research first.")

    rows = _load_rows(research_path)
    if not rows:
        raise SystemExit(f"No rows found in {research_path}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = research_path.with_suffix(".pdf")

    write_research_pdf(rows, str(output_path), title=args.title, top_n=args.top_n)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
