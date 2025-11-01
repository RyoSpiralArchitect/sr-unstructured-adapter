"""Shared retry/backoff helpers for HTTP based LLM drivers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class BackoffPolicy:
    """Configuration describing exponential backoff with jitter."""

    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.5

    def compute(self, attempt: int) -> float:
        """Return the backoff delay for the given retry ``attempt``."""

        if attempt <= 0:
            return 0.0
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter > 0.0:
            delay += random.uniform(0.0, self.jitter)
        return max(delay, 0.0)

    def iter_delays(self, retries: int) -> Iterator[float]:
        """Yield the sequence of delays for ``retries`` attempts."""

        for attempt in range(1, retries + 1):
            yield self.compute(attempt)


class CircuitBreaker:
    """Simple circuit breaker tracking failure streaks."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_time: float = 30.0,
        window: float = 10.0,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_time <= 0:
            raise ValueError("recovery_time must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.failure_threshold = int(failure_threshold)
        self.recovery_time = float(recovery_time)
        self.window = float(window)
        self._failure_count = 0
        self._opened_at: float | None = None
        self._window_start: float | None = None

    def allow_request(self) -> bool:
        """Return ``True`` when new requests should be attempted."""

        if self._opened_at is None:
            return True
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.recovery_time:
            # Reset after the cooldown window expires.
            self.reset()
            return True
        return False

    def record_success(self) -> None:
        """Clear failure streak after a successful call."""

        self.reset()

    def record_failure(self) -> None:
        """Register a failed attempt and open the breaker if needed."""

        now = time.monotonic()
        if self._window_start is None or (now - self._window_start) > self.window:
            self._window_start = now
            self._failure_count = 0
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._opened_at = now

    def reset(self) -> None:
        """Return the breaker to the closed state."""

        self._failure_count = 0
        self._opened_at = None
        self._window_start = None

    @property
    def is_open(self) -> bool:
        """Expose whether the breaker is currently open."""

        return self._opened_at is not None and not self.allow_request()


__all__ = ["BackoffPolicy", "CircuitBreaker"]

