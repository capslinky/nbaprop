"""Metrics collection."""

from typing import Dict, List
import threading


class MetricsRecorder:
    def increment(self, key: str, value: int = 1) -> None:
        raise NotImplementedError

    def timing(self, key: str, value_ms: float) -> None:
        raise NotImplementedError

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError


class InMemoryMetricsRecorder(MetricsRecorder):
    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._timings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def increment(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + int(value)

    def timing(self, key: str, value_ms: float) -> None:
        with self._lock:
            self._timings.setdefault(key, []).append(float(value_ms))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            timing_summary = {}
            for key, values in self._timings.items():
                if not values:
                    continue
                timing_summary[key] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "max_ms": max(values),
                }
            return {
                "counters": dict(self._counters),
                "timings": timing_summary,
            }


_DEFAULT_RECORDER = InMemoryMetricsRecorder()


def get_metrics_recorder() -> MetricsRecorder:
    return _DEFAULT_RECORDER
