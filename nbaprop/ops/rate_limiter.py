"""Shared rate limiter for external sources."""

from typing import Dict, Optional
import threading
import time


class RateLimiter:
    def __init__(self, default_interval: float = 1.0, overrides: Optional[Dict[str, float]] = None) -> None:
        self._default_interval = max(0.0, default_interval)
        self._overrides: Dict[str, float] = dict(overrides or {})
        self._last_called: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set_interval(self, source: str, interval: float) -> None:
        self._overrides[source] = max(0.0, interval)

    def wait(self, source: str, interval: Optional[float] = None) -> float:
        min_interval = self._default_interval if interval is None else max(0.0, interval)
        min_interval = self._overrides.get(source, min_interval)

        with self._lock:
            now = time.monotonic()
            last = self._last_called.get(source, 0.0)
            sleep_for = max(0.0, min_interval - (now - last))
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._last_called[source] = now

        return sleep_for


_DEFAULT_RATE_LIMITER = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _DEFAULT_RATE_LIMITER
