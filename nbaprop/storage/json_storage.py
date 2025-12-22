"""JSON-based storage for raw and normalized tables."""

from pathlib import Path
from typing import List, Dict
import json


class JsonStorage:
    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, name: str, rows: List[Dict]) -> str:
        path = self._base_dir / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def read_table(self, name: str) -> List[Dict]:
        path = self._base_dir / f"{name}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))
