"""Tests covering retry/backoff utilities for drivers."""

from __future__ import annotations

from dataclasses import dataclass

from sr_adapter.drivers import resilience


@dataclass
class _FakeClock:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value


def test_circuit_breaker_is_open_does_not_reset(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(resilience, "time", clock)

    breaker = resilience.CircuitBreaker(
        failure_threshold=1,
        recovery_time=10.0,
        window=1.0,
    )

    breaker.record_failure()
    assert breaker.is_open is True

    clock.value += 11.0
    assert breaker.is_open is False
    # ``is_open`` should not have reset internal state before an allow check.
    assert breaker._opened_at is not None  # type: ignore[attr-defined]

    assert breaker.allow_request() is True
