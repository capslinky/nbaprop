"""CSV output helpers."""

from typing import List, Dict
from pathlib import Path
import csv


def write_picks_csv(rows: List[Dict], output_path: str) -> None:
    """Write picks to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    extra_keys = set()
    for row in rows[1:]:
        extra_keys.update(key for key in row.keys() if key not in fieldnames)
    if extra_keys:
        fieldnames.extend(sorted(extra_keys))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
