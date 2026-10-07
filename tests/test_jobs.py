from __future__ import annotations

from threading import Event, Thread

import pytest

from sr_adapter.jobs import JobManager, SQLiteJobStore


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
