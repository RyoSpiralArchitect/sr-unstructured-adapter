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


def test_selector_skips_invalid_saved_stats(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"profiles": {
        "balanced": {"trials": "broken"},
        "archival": {"trials": 2, "reward_sum": 1.5},
    }}), encoding="utf-8")
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    assert "balanced" not in selector._stats
    assert selector._stats["archival"].trials == 2


def test_selector_saves_complete_concurrent_outcomes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    profile = selector.select()
    def record(_):
        selector.record_outcome(profile, {"block_count": 1, "metrics_total_ms": 1})
        # Readers must always see one complete JSON object.
        json.loads(path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(20)))
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"][profile.name]["trials"] == 20
