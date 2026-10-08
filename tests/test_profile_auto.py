from __future__ import annotations

import json
import multiprocessing

import pytest

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


def _record_in_worker(path, ready, start, count):
    """Spawn target: every worker starts with the same old JSON snapshot."""
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=path),
        telemetry=_StubTelemetry(),
    )
    balanced = selector.store.load("balanced")
    archival = selector.store.load("archival")
    ready.set()
    if not start.wait(timeout=30):
        raise TimeoutError("Parent did not release profile workers")
    for _ in range(count):
        selector.record_outcome(balanced, {"block_count": 1})
        selector.record_outcome(archival, {"block_count": 1})
        # Unlocked readers also see a complete published JSON object.
        with open(path, encoding="utf-8") as handle:
            json.load(handle)


def test_spawned_processes_preserve_all_updates_and_legacy_stats(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"profiles": {
        "balanced": {"trials": 7, "reward_sum": 2.0, "last_updated": 1.0},
        "archival": {"trials": 3, "reward_sum": 1.0, "last_updated": 1.0},
        "custom-retired": {"trials": 9, "reward_sum": 4.5, "last_updated": 1.0},
    }}), encoding="utf-8")
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    ready = [ctx.Event() for _ in range(4)]
    workers = [ctx.Process(target=_record_in_worker, args=(str(path), event, start, 25)) for event in ready]
    try:
        for worker in workers:
            worker.start()
        assert all(event.wait(timeout=30) for event in ready)
        start.set()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        start.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=5)

    stats = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    assert stats["balanced"]["trials"] == 107
    assert stats["balanced"]["reward_sum"] == pytest.approx(62.0)
    assert stats["archival"]["trials"] == 103
    assert stats["archival"]["reward_sum"] == pytest.approx(61.0)
    assert stats["custom-retired"] == {"trials": 9, "reward_sum": 4.5, "last_updated": 1.0}


def test_existing_selector_refreshes_rewards_from_other_instances(tmp_path):
    settings = AutoProfileSettings(
        enabled=True, state_path=str(tmp_path / "state.json"), epsilon=0,
        candidate_profiles=("balanced", "archival"),
    )
    reader = AdaptiveProfileSelector(settings=settings, telemetry=_StubTelemetry())
    writer = AdaptiveProfileSelector(settings=settings, telemetry=_StubTelemetry())
    assert reader.select().name == "balanced"
    writer.record_outcome(writer.store.load("balanced"), {"metrics_total_ms": 10000})
    writer.record_outcome(writer.store.load("archival"), {"metrics_total_ms": 0})
    assert reader.select().name == "archival"


@pytest.mark.parametrize("contents", [b"{broken", b"\xff", b'null', b'{"profiles": []}'])
def test_corrupt_cache_recovers_on_next_outcome(tmp_path, contents):
    path = tmp_path / "state.json"
    path.write_bytes(contents)
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    assert selector._stats == {}
    selector.record_outcome(selector.store.load("balanced"), {})
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"]["balanced"]["trials"] == 1


def test_failed_replace_does_not_commit_in_memory_or_leave_temporary_files(tmp_path, monkeypatch):
    import sr_adapter.profile_auto as module

    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    profile = selector.store.load("balanced")
    selector.record_outcome(profile, {})
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated failed publication")

    with monkeypatch.context() as context:
        context.setattr(module.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated failed publication"):
            selector.record_outcome(profile, {})
    assert path.read_bytes() == before
    assert selector._stats["balanced"].trials == 1
    assert list(tmp_path.glob(".state.json.*")) == []
    selector.record_outcome(profile, {})
    assert selector._stats["balanced"].trials == 2


def test_lock_timeout_does_not_drop_stats_and_recovers(tmp_path, monkeypatch):
    import sr_adapter.profile_auto as module

    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    profile = selector.store.load("balanced")
    selector.record_outcome(profile, {})
    monkeypatch.setattr(module, "_STATE_LOCK_TIMEOUT", 0.03)
    with module._state_file_lock(path):
        with pytest.raises(TimeoutError, match="profile state lock"):
            selector.record_outcome(profile, {})
    assert selector._stats["balanced"].trials == 1
    selector.record_outcome(profile, {})
    assert selector._stats["balanced"].trials == 2


def test_windows_reader_sharing_conflict_retries_same_snapshot(tmp_path, monkeypatch):
    import sr_adapter.profile_auto as module

    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    replace_file = module.os.replace
    attempts = []

    def busy_once(source, destination):
        attempts.append((source, destination))
        if len(attempts) == 1:
            error = PermissionError("reader denies delete sharing")
            error.winerror = 32
            raise error
        replace_file(source, destination)

    monkeypatch.setattr(module.os, "replace", busy_once)
    selector.record_outcome(selector.store.load("balanced"), {})
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"]["balanced"]["trials"] == 1


def test_read_permission_error_does_not_reset_previous_state(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    profile = selector.store.load("balanced")
    selector.record_outcome(profile, {})
    before = path.read_bytes()

    def deny_read(*args, **kwargs):
        raise PermissionError("simulated unreadable state")

    with monkeypatch.context() as context:
        context.setattr(Path, "read_text", deny_read)
        selector._load_state()
        assert selector._stats["balanced"].trials == 1
        with pytest.raises(PermissionError, match="simulated unreadable state"):
            selector.record_outcome(profile, {})
    assert path.read_bytes() == before
    assert selector._stats["balanced"].trials == 1


def test_symlink_aliases_share_the_same_state_and_lock(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"profiles": {}}', encoding="utf-8")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(path)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    direct = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    indirect = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(alias)),
        telemetry=_StubTelemetry(),
    )
    assert direct._state_path == indirect._state_path
    direct.record_outcome(direct.store.load("balanced"), {})
    indirect.record_outcome(indirect.store.load("balanced"), {})
    assert alias.is_symlink()
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"]["balanced"]["trials"] == 2


def _hold_state_lock(path, ready):
    from pathlib import Path
    from threading import Event

    from sr_adapter.profile_auto import _state_file_lock

    with _state_file_lock(Path(path)):
        ready.set()
        Event().wait(timeout=60)


def test_terminated_worker_releases_state_lock(tmp_path):
    path = tmp_path / "state.json"
    selector = AdaptiveProfileSelector(
        settings=AutoProfileSettings(enabled=True, state_path=str(path)),
        telemetry=_StubTelemetry(),
    )
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    worker = ctx.Process(target=_hold_state_lock, args=(str(path), ready))
    worker.start()
    try:
        assert ready.wait(timeout=30)
    finally:
        worker.terminate()
        worker.join(timeout=5)
        assert not worker.is_alive()
    selector.record_outcome(selector.store.load("balanced"), {})
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"]["balanced"]["trials"] == 1
