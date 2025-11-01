"""Lightweight in-memory metrics for LLM driver invocations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, MutableMapping


@dataclass
class LLMCallStats:
    """Aggregate counters for a single driver."""

    driver: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    last_error: str | None = None

    def record_success(
        self,
        *,
        latency_ms: float,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        self.calls += 1
        self.successes += 1
        self.total_latency_ms += max(latency_ms, 0.0)
        self.total_request_bytes += max(request_bytes, 0)
        self.total_response_bytes += max(response_bytes, 0)
        self.last_error = None

    def record_failure(self, *, latency_ms: float, error: str) -> None:
        self.calls += 1
        self.failures += 1
        self.total_latency_ms += max(latency_ms, 0.0)
        self.last_error = error

    @property
    def avg_latency_ms(self) -> float:
        if not self.calls:
            return 0.0
        return self.total_latency_ms / self.calls

    def to_dict(self) -> Dict[str, int | float | str | None]:
        return {
            "driver": self.driver,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "total_latency_ms": round(self.total_latency_ms, 4),
            "total_request_bytes": self.total_request_bytes,
            "total_response_bytes": self.total_response_bytes,
            "last_error": self.last_error,
        }


@dataclass
class LLMMetricsSnapshot:
    """Serializable view of the aggregated metrics."""

    collected_at: float
    stats: List[LLMCallStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "collected_at": self.collected_at,
            "drivers": [stat.to_dict() for stat in self.stats],
        }


class LLMMetricsRegistry:
    """Thread-safe registry accumulating driver telemetry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: MutableMapping[str, LLMCallStats] = {}

    def record_success(
        self,
        driver: str,
        *,
        latency_ms: float,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        with self._lock:
            stats = self._stats.setdefault(driver, LLMCallStats(driver))
            stats.record_success(
                latency_ms=latency_ms,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
            )

    def record_failure(self, driver: str, *, latency_ms: float, error: str) -> None:
        with self._lock:
            stats = self._stats.setdefault(driver, LLMCallStats(driver))
            stats.record_failure(latency_ms=latency_ms, error=error)

    def snapshot(self) -> LLMMetricsSnapshot:
        with self._lock:
            stats = [stat for stat in self._stats.values()]
        return LLMMetricsSnapshot(collected_at=time.time(), stats=list(stats))

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


_GLOBAL_REGISTRY = LLMMetricsRegistry()


def get_llm_registry() -> LLMMetricsRegistry:
    return _GLOBAL_REGISTRY


__all__ = [
    "LLMCallStats",
    "LLMMetricsRegistry",
    "LLMMetricsSnapshot",
    "get_llm_registry",
]

