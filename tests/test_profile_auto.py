from __future__ import annotations

import json

from sr_adapter.profile_auto import AdaptiveProfileSelector
from sr_adapter.runtime import KernelStats, RuntimeSnapshot
from sr_adapter.settings import AutoProfileSettings


class _StubTelemetry:
    def __init__(self, *, kernel_ms: float = 0.0, calls: int = 0, failures: int = 0) -> None:
        self._kernel_ms = kernel_ms
        self._calls = calls
        self._failures = failures

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            text_enabled=True,
            layout_enabled=False,
            text_stats=KernelStats(name="text", calls=1, total_ms=self._kernel_ms, total_units=0),
            layout_stats=KernelStats(name="layout"),
        )

    def llm_snapshot(self) -> dict[str, object]:
        return {
            "drivers": [
                {
                    "driver": "stub",
                    "calls": self._calls,
                    "failures": self._failures,
                }
            ]
        }


def test_selector_prefers_archival_for_large_documents(tmp_path):
    settings = AutoProfileSettings(
        enabled=True,
        candidate_profiles=("balanced", "archival"),
        state_path=str(tmp_path / "state.json"),
        large_document_bytes=100,
    )
    telemetry = _StubTelemetry(kernel_ms=10.0)
    selector = AdaptiveProfileSelector(settings=settings, telemetry=telemetry)

    chosen = selector.select(context={"size_bytes": 10_000})
    assert chosen.name == "archival"


def test_selector_prefers_balanced_on_failures(tmp_path):
    settings = AutoProfileSettings(
        enabled=True,
        candidate_profiles=("balanced", "realtime"),
        state_path=str(tmp_path / "state.json"),
        max_llm_failure_rate=0.2,
    )
    telemetry = _StubTelemetry(calls=10, failures=5)
    selector = AdaptiveProfileSelector(settings=settings, telemetry=telemetry)

    chosen = selector.select()
    assert chosen.name == "balanced"


def test_selector_records_outcomes(tmp_path):
    settings = AutoProfileSettings(
        enabled=True,
        state_path=str(tmp_path / "state.json"),
        epsilon=0.0,
    )
    telemetry = _StubTelemetry()
    selector = AdaptiveProfileSelector(settings=settings, telemetry=telemetry)

    profile = selector.select()
    selector.record_outcome(
        profile,
        {
            "metrics_total_ms": 1200,
            "block_count": 8,
            "llm_escalations": 4,
            "truncated_blocks": 1,
        },
    )

    stats = selector._stats[profile.name]
    assert stats.trials == 1
    assert stats.reward_sum != 0

    payload = json.loads(settings.resolved_state_path.read_text(encoding="utf-8"))
    assert profile.name in payload["profiles"]
