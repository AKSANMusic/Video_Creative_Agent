"""Shared global rate limiter for all Replicate API prediction calls.

Replicate's free tier enforces:
  - 6 requests/minute = 1 token every ~10 seconds
  - Burst of 1 (only one prediction may be submitted at a time)

This module provides a process-global token-bucket limiter that serialises
every ``client.run()`` call across FlorenceClient and LLaVAClient so we
never exceed the burst=1 constraint.  The interval is configurable and
defaults to 11 s (10 s window + 1 s safety buffer).
"""

from __future__ import annotations

import threading
import time
import logging

log = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 11.0  # seconds between each Replicate prediction call


class ReplicateRateLimiter:
    """Thread-safe token bucket: at most 1 Replicate call per ``interval_s``."""

    def __init__(self, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        self.interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0

    def acquire(self, context: str = "") -> None:
        """Block until enough time has elapsed since the last API call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait_s = self.interval_s - elapsed
            if wait_s > 0:
                log.debug(
                    "ReplicateRateLimiter: sleeping %.1fs before next call%s",
                    wait_s,
                    f" ({context})" if context else "",
                )
                time.sleep(wait_s)
            self._last_call_time = time.monotonic()

    def set_interval(self, interval_s: float) -> None:
        """Dynamically update the limiter interval (e.g. from UI slider)."""
        with self._lock:
            self.interval_s = float(interval_s)


# Singleton — import this from both FlorenceClient and LLaVAClient.
REPLICATE_LIMITER = ReplicateRateLimiter(interval_s=_DEFAULT_INTERVAL_S)

__all__ = ["REPLICATE_LIMITER", "ReplicateRateLimiter"]
