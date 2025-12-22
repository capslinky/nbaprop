"""Cache interfaces."""

from pathlib import Path
from typing import Optional, Any, Dict, Tuple, Union
import hashlib
import json
import threading
import time


class CacheStore:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError

    def invalidate(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MemoryCache(CacheStore):
    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() >= expires_at:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._data[key] = (value, expires_at)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class FileCache(CacheStore):
    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        expires_at = payload.get("expires_at")
        if expires_at is not None and time.time() >= float(expires_at):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return payload.get("value")

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        payload = {
            "expires_at": time.time() + ttl_seconds,
            "value": value,
        }
        path = self._path_for_key(key)
        with self._lock:
            path.write_text(json.dumps(payload), encoding="utf-8")

    def invalidate(self, key: str) -> None:
        path = self._path_for_key(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def clear(self) -> None:
        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    continue

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.json"
