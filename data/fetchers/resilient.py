"""Resilient HTTP fetcher with retry logic and connection pooling.

This module provides the ResilientFetcher class for making API calls
with automatic retries, exponential backoff, and connection pooling.
"""

import random
import time
import logging
from typing import Callable, Tuple, Any

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class ResilientFetcher:
    """
    Utility class that wraps API calls with retry logic, exponential backoff,
    and connection pooling for improved reliability.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 15.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.consecutive_failures = 0

        # Create session with connection pooling
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=5,
            max_retries=0  # We handle retries ourselves
        )
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff: 1s, 2s, 4s, 8s...
        delay = self.base_delay * (2 ** attempt)
        # Add jitter (random 0-30% of delay)
        jitter = delay * random.uniform(0, 0.3)
        # Cap at max delay
        return min(delay + jitter, self.max_delay)

    def _should_retry(self, error: Exception) -> Tuple[bool, float]:
        """
        Determine if an error should trigger a retry and how long to wait.
        Returns (should_retry, delay_multiplier)
        """
        error_str = str(error).lower()

        # Rate limiting - back off aggressively
        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            return True, 3.0  # Triple the delay

        # 529 Site Overloaded - NBA API specific, back off very aggressively
        if '529' in error_str or 'site overloaded' in error_str or 'overloaded' in error_str:
            return True, 4.0  # Quadruple the delay for site overload

        # Connection errors - retry immediately
        if any(x in error_str for x in ['connection', 'timeout', 'timed out', 'reset', 'refused']):
            return True, 1.0

        # Server errors (5xx) - retry with normal backoff
        if any(x in error_str for x in ['500', '502', '503', '504', '529', 'server error', 'internal error']):
            return True, 1.5

        # Client errors (4xx except 429) - don't retry
        if any(x in error_str for x in ['400', '401', '403', '404', 'not found', 'forbidden']):
            return False, 0

        # JSON decode errors - might be transient, retry once
        if 'json' in error_str or 'decode' in error_str:
            return True, 1.0

        # Default: retry with normal backoff
        return True, 1.0

    def fetch_with_retry(
        self,
        fetch_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Execute a fetch function with automatic retry on failure.

        Args:
            fetch_func: The function to execute
            *args, **kwargs: Arguments to pass to fetch_func

        Returns:
            Tuple of (result, success). If all retries fail, result is None.
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = fetch_func(*args, **kwargs)
                self.consecutive_failures = 0  # Reset on success
                return result, True

            except Exception as e:
                last_error = e
                self.consecutive_failures += 1

                should_retry, delay_mult = self._should_retry(e)

                if not should_retry or attempt >= self.max_retries:
                    logger.warning(
                        f"Fetch failed after {attempt + 1} attempts: {e}"
                    )
                    break

                delay = self._calculate_delay(attempt) * delay_mult
                # Add extra delay if we've had many consecutive failures
                if self.consecutive_failures > 3:
                    delay *= 1.5

                logger.info(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        return None, False

    @property
    def session(self) -> requests.Session:
        """Get the shared session with connection pooling."""
        return self._session


# Global resilient fetcher instance
_resilient_fetcher = ResilientFetcher()


def get_resilient_fetcher() -> ResilientFetcher:
    """Get the global ResilientFetcher instance."""
    return _resilient_fetcher
