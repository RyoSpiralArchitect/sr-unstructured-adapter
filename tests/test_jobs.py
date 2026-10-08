from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event, Thread

import pytest

from sr_adapter.jobs import JobManager, JobRecord, MemoryJobStore, SQLiteJobStore


def test_shutdown_waits_for_sqlite_jobs_before_closing(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager(max_workers=1, store=SQLiteJobStore(path))
    started, release, shutdown_done = Event(), Event(), Event()

    def work():
        started.set()
        assert release.wait(3)
        return {"result": "saved"}

    record = manager.submit("test", work)
    assert started.wait(1)

    def shutdown():
        manager.shutdown()
        shutdown_done.set()

    thread = Thread(target=shutdown)
    thread.start()
    try:
        assert not shutdown_done.wait(0.05)
    finally:
        release.set()
        thread.join(timeout=3)
    assert shutdown_done.is_set()
    store = SQLiteJobStore(path)
    try:
        persisted = store.get(record.id)
        assert persisted.status == "succeeded"
        assert persisted.result == {"result": "saved"}
    finally:
        store.close()
    manager.shutdown()  # Idempotent.


def test_cancel_queued_job_releases_resources(tmp_path):
    manager = JobManager(max_workers=1)
    started, release = Event(), Event()
    upload = tmp_path / "upload.txt"
    upload.write_text("payload")

    def block_worker():
        started.set()
        assert release.wait(3)

    manager.submit("busy", block_worker)
    assert started.wait(1)
    try:
        queued = manager.submit("upload", lambda: pytest.fail("Canceled job ran"), on_done=upload.unlink)
        assert manager.cancel(queued.id)
        assert not upload.exists()
        assert manager.get(queued.id).status == "canceled"
    finally:
        release.set()
        manager.shutdown()


def test_submit_after_shutdown_releases_resources():
    manager = JobManager()
    manager.shutdown()
    cleaned = []
    with pytest.raises(RuntimeError, match="shut down"):
        manager.submit("test", lambda: None, on_done=lambda: cleaned.append(True))
    assert cleaned == [True]


def test_failed_job_runs_completion_cleanup():
    manager = JobManager()
    cleaned = Event()

    def fail():
        raise ValueError("conversion failed")

    record = manager.submit("test", fail, on_done=cleaned.set)
    try:
        assert cleaned.wait(1)
        result = manager.get(record.id)
        assert result.status == "failed"
        assert result.error == "conversion failed"
        assert result._future is None
    finally:
        manager.shutdown()


def test_completed_job_cleanup_can_submit_followup(monkeypatch):
    from concurrent.futures import Future

    manager = JobManager()
    completed = Future()
    completed.set_result(None)
    monkeypatch.setattr(manager._executor, "submit", lambda func: completed)
    done = Event()

    def cleanup():
        manager.submit("followup", lambda: None)
        done.set()

    thread = Thread(target=lambda: manager.submit("test", lambda: None, on_done=cleanup), daemon=True)
    thread.start()
    assert done.wait(1), "Completion callback ran while lifecycle lock was held"
    thread.join(timeout=1)
    manager.shutdown()


@pytest.mark.parametrize("from_callback", [False, True])
@pytest.mark.parametrize("external_draining", [False, True])
def test_worker_shutdown_is_rejected_without_poisoning_lifecycle(monkeypatch, from_callback, external_draining):
    from sr_adapter.jobs import MemoryJobStore

    class Store(MemoryJobStore):
        close_calls = 0
        def close(self):
            self.close_calls += 1

    store = Store()
    manager = JobManager(max_workers=1, store=store)
    started, release, finished, draining = Event(), Event(), Event(), Event()
    errors = []
    drain_thread = None

    def attempt_shutdown():
        try:
            manager.shutdown()
        except RuntimeError as exc:
            errors.append(str(exc))

    def work():
        started.set()
        assert release.wait(3)
        if not from_callback:
            attempt_shutdown()

    def on_done():
        if from_callback:
            attempt_shutdown()
        finished.set()

    original_shutdown = manager._executor.shutdown

    def observe_shutdown(*args, **kwargs):
        draining.set()
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(manager._executor, "shutdown", observe_shutdown)
    manager.submit("worker-shutdown", work, on_done=on_done)
    try:
        assert started.wait(1)
        if external_draining:
            drain_thread = Thread(target=manager.shutdown, daemon=True)
            drain_thread.start()
            assert draining.wait(1)
        release.set()
        assert finished.wait(2), "Worker waited for a shutdown which was joining it"
        assert len(errors) == 1
        assert "cannot be called from its worker thread" in errors[0]
        if not external_draining:
            # The rejected call must leave the manager open for ordinary work.
            manager.submit("still-open", lambda: None)
    finally:
        release.set()
        if drain_thread is not None:
            # Bound a failed regression too, instead of stranding executor
            # threads in an exit-time join after the assertion above fails.
            if not finished.is_set():
                manager._shutdown_done.set()
            drain_thread.join(timeout=2)
        manager.shutdown()
    assert store.close_calls == 1
    assert drain_thread is None or not drain_thread.is_alive()


@pytest.mark.parametrize("sqlite", [False, True])
def test_recovery_never_interrupts_live_jobs(tmp_path, sqlite):
    store = SQLiteJobStore(tmp_path / "jobs.db") if sqlite else MemoryJobStore()
    manager = JobManager(max_workers=1, store=store)
    started, release = Event(), Event()
    sibling = SQLiteJobStore(store.path) if sqlite else store

    def work():
        started.set()
        assert release.wait(3)
        return "complete"

    try:
        running = manager.submit("running", work)
        assert started.wait(1)
        queued = manager.submit("queued", lambda: "also complete")
        assert manager.reset_incomplete() == 0
        assert sibling.reset_incomplete(error="sibling started") == 0
        assert sibling.get(running.id).status == "running"
        assert sibling.get(queued.id).status == "queued"
        release.set()
        manager.shutdown()
        if sqlite:
            assert sibling.get(running.id).result == "complete"
            assert sibling.get(queued.id).result == "also complete"
    finally:
        release.set()
        manager.shutdown()
        if sqlite:
            sibling.close()


def test_process_crash_is_interrupted_without_replay_and_live_process_is_safe(tmp_path):
    database = tmp_path / "jobs.db"
    side_effect = tmp_path / "paid-request-must-not-run"
    child = r'''
import json, os, subprocess, sys
from pathlib import Path
from threading import Event
from sr_adapter.jobs import JobManager, SQLiteJobStore
manager = JobManager(max_workers=1, store=SQLiteJobStore(sys.argv[1]))
started, release = Event(), Event()
def work():
    started.set()
    release.wait()
    Path(sys.argv[2]).write_text("unexpected running replay")
running = manager.submit("running", work, request={"tenant": "alpha"})
assert started.wait(3)
queued = manager.submit("queued", lambda: Path(sys.argv[2]).write_text("unexpected queued replay"))
probe = "from sr_adapter.jobs import SQLiteJobStore; import sys; s=SQLiteJobStore(sys.argv[1]); assert s.reset_incomplete(error='other process') == 0; s.close()"
subprocess.run([sys.executable, "-c", probe, sys.argv[1]], check=True, timeout=10)
print(json.dumps([running.id, queued.id]), flush=True)
os._exit(19)
'''
    result = subprocess.run(
        [sys.executable, "-c", child, str(database), str(side_effect)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 19, result.stderr
    running_id, queued_id = json.loads(result.stdout)
    store = SQLiteJobStore(database)
    try:
        assert store.reset_incomplete(error="worker stopped; not replayed") == 2
        assert store.reset_incomplete(error="repeat") == 0
        running, queued = store.get(running_id), store.get(queued_id)
        for record in (running, queued):
            assert record.status == "interrupted"
            assert record.finished_at is not None
            assert record.error == "worker stopped; not replayed"
            assert record.result is None
        assert running.started_at is not None
        assert running.request == {"tenant": "alpha"}
        assert queued.started_at is None
        assert not side_effect.exists()
    finally:
        store.close()


def test_legacy_database_migrates_tenants_and_interrupts_only_incomplete(tmp_path):
    database = tmp_path / "jobs.db"
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, request_json TEXT NOT NULL, result_json TEXT, error TEXT)")
        conn.executemany(
            "INSERT INTO jobs VALUES (?, 'legacy', ?, ?, NULL, NULL, ?, ?, NULL)",
            [
                ("queued", "queued", timestamp, '{"tenant":"alpha"}', None),
                ("running", "running", timestamp, '{}', None),
                ("done", "succeeded", timestamp, '{"tenant":"beta"}', '{"kept":true}'),
            ],
        )
    store = SQLiteJobStore(database)
    try:
        assert [r.id for r in store.list(limit=1, tenants=frozenset({"alpha"}))] == ["queued"]
        assert store.reset_incomplete(error="upgrade") == 2
        assert store.get("queued").status == "interrupted"
        assert store.get("running").status == "interrupted"
        assert store.get("done").status == "succeeded"
        assert store.get("done").result == {"kept": True}
    finally:
        store.close()


@pytest.mark.parametrize("sqlite", [False, True])
def test_tenant_filter_applies_before_limit_and_excludes_unknown(tmp_path, sqlite):
    store = SQLiteJobStore(tmp_path / "jobs.db") if sqlite else MemoryJobStore()
    manager = JobManager(store=store)
    try:
        start = datetime.now(timezone.utc)
        for index, tenant in enumerate(["alpha", "alpha", "beta", None, ""]):
            store.create(JobRecord(id=str(index), kind="fixture", request={"tenant": tenant},
                                   created_at=start + timedelta(seconds=index)))
        assert [r.id for r in manager.list(limit=2, tenants=frozenset({"alpha"}))] == ["1", "0"]
        assert manager.list(tenants=frozenset()) == []
        assert len(manager.list()) == 5
    finally:
        manager.shutdown()


@pytest.mark.parametrize("sqlite", [False, True])
def test_interrupted_jobs_expire(tmp_path, sqlite):
    store = SQLiteJobStore(tmp_path / "jobs.db") if sqlite else MemoryJobStore()
    try:
        store.create(JobRecord(id="old", kind="fixture", status="interrupted",
                               finished_at=datetime.now(timezone.utc) - timedelta(hours=1)))
        assert store.cleanup(ttl_seconds=60) == ["old"]
        assert store.get("old") is None
    finally:
        if sqlite:
            store.close()


def test_sibling_store_cannot_overwrite_another_owners_status(tmp_path):
    first, second = SQLiteJobStore(tmp_path / "jobs.db"), SQLiteJobStore(tmp_path / "jobs.db")
    try:
        first.create(JobRecord(id="active", kind="fixture", status="running"))
        record = second.get("active")
        record.status = "failed"
        with pytest.raises(RuntimeError, match="another SQLite store"):
            second.save(record)
        assert first.get("active").status == "running"
    finally:
        first.close()
        second.close()


def test_concurrent_recovery_marks_each_abandoned_job_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    original = SQLiteJobStore(tmp_path / "jobs.db")
    for index in range(10):
        original.create(JobRecord(id=str(index), kind="abandoned"))
    original.close()
    first, second = SQLiteJobStore(original.path), SQLiteJobStore(original.path)
    barrier = Barrier(2)

    def recover(store):
        barrier.wait(timeout=3)
        return store.reset_incomplete(error="concurrent recovery")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            counts = list(executor.map(recover, (first, second)))
        assert sum(counts) == 10
        assert all(record.status == "interrupted" for record in first.list(limit=10))
    finally:
        first.close()
        second.close()


def test_concurrent_first_open_initializes_database(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    barrier = Barrier(4)

    class SimultaneousStore(SQLiteJobStore):
        def _initialize(self):
            barrier.wait(timeout=3)
            super()._initialize()

    for iteration in range(5):
        stores = []

        def open_store(index):
            store = SimultaneousStore(tmp_path / f"new-{iteration}.db")
            stores.append(store)
            store.create(JobRecord(id=str(index), kind="concurrent open"))

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(open_store, range(4)))
            assert len(stores[0].list(limit=10)) == 4
        finally:
            for store in stores:
                store.close()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require privileges")
def test_database_symlink_uses_same_owner_locks(tmp_path):
    first = SQLiteJobStore(tmp_path / "real.db")
    alias = tmp_path / "alias.db"
    alias.symlink_to(first.path)
    second = SQLiteJobStore(alias)
    try:
        first.create(JobRecord(id="live", kind="fixture", status="running"))
        assert second.reset_incomplete(error="alias startup") == 0
        assert first.get("live").status == "running"
    finally:
        first.close()
        second.close()


def test_baseexception_from_worker_is_recorded_and_cleanup_runs_once():
    manager, done = JobManager(), Event()
    calls = []

    def work():
        raise SystemExit()

    def cleanup():
        calls.append(True)
        done.set()

    try:
        record = manager.submit("exit", work, on_done=cleanup)
        assert done.wait(1)
        assert manager.get(record.id).status == "failed"
        assert manager.get(record.id).error == "SystemExit"
        assert calls == [True]
    finally:
        manager.shutdown()


def test_immediate_cleanup_failure_is_not_invoked_twice(monkeypatch):
    from concurrent.futures import Future

    manager = JobManager()
    completed = Future()
    completed.set_result(None)
    monkeypatch.setattr(manager._executor, "submit", lambda func: completed)
    calls = []

    def cleanup():
        calls.append(True)
        raise KeyboardInterrupt("cleanup failed")

    try:
        with pytest.raises(KeyboardInterrupt, match="cleanup failed"):
            manager.submit("cleanup", lambda: None, on_done=cleanup)
        assert calls == [True]
    finally:
        manager.shutdown()
