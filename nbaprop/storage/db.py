"""Database interface."""

from typing import List, Dict


class Database:
    def init_schema(self) -> None:
        raise NotImplementedError

    def write_table(self, name: str, rows: List[Dict]) -> None:
        raise NotImplementedError

    def read_table(self, name: str) -> List[Dict]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
