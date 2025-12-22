"""Metrics collection."""

class MetricsRecorder:
    def increment(self, key: str, value: int = 1) -> None:
        raise NotImplementedError

    def timing(self, key: str, value_ms: float) -> None:
        raise NotImplementedError
