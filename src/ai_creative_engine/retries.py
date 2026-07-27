"""Shared tenacity retry policy for upstream API calls.

The policy retries only on transient/network-class errors with exponential
backoff + jitter, up to a bounded number of attempts. After exhaustion the
final exception is re-raised so the pipeline can record it as a per-image
failure.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .errors import APIError

log = logging.getLogger(__name__)

T = TypeVar("T")

# Default policy parameters (module-level for easy override in tests).
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_INITIAL_WAIT = 1.0  # seconds
DEFAULT_MAX_WAIT = 30.0  # seconds

# Retryable conditions: transient network/transport issues and our own APIError.
_RETRYABLE_EXC = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    APIError,
)


def api_retry(
    func: Callable[..., T],
) -> Callable[..., T]:
    """Decorator applying the standard API retry policy.

    Example
    -------
    >>> @api_retry
    ... def call_replicate(): ...
    """

    decorated = retry(
        stop=stop_after_attempt(DEFAULT_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=DEFAULT_INITIAL_WAIT, max=DEFAULT_MAX_WAIT),
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )(func)
    return decorated  # type: ignore[return-value]


__all__: list[Any] = ["api_retry"]
