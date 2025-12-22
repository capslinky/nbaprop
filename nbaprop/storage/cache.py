"""Cache interfaces."""

from typing import Optional


class CacheStore:
    def get(self, key: str) -> Optional[object]:
        raise NotImplementedError

    def set(self, key: str, value: object, ttl_seconds: int) -> None:
        raise NotImplementedError

    def invalidate(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
