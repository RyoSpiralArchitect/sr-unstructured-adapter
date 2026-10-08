# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lightweight in-process job manager with optional SQLite persistence.

This module is intentionally dependency-free so it can be reused by:
- FastAPI wrapper (SaaS shell)
- internal batch services
- research notebooks

Notes:
- Execution is still in-process via a thread pool.
- The SQLite backend persists job status/results for inspection across restarts,
  but does not provide distributed worker claiming or replay callables.
- Recovery marks abandoned work as interrupted, without touching live owners.
  SQLite databases and their owner lock sidecars must remain on a local disk.
"""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, local
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Protocol


JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled", "interrupted"]


def _tenant(record: JobRecord) -> str | None:
    value = record.request.get("tenant")
    return value if isinstance(value, str) and value else None


class _OwnerLease:
    """A nonblocking OS lock, released even when its owning process crashes.

    A unique ID is never reused. Recovery may therefore remove an abandoned
    owner's sidecar after acquiring it; no new worker can claim that identity.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: Any = None

    def acquire(self) -> bool:
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":  # pragma: no cover - Windows-specific backend
                import msvcrt
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        except BaseException:
            handle.close()
            raise
        self._file = handle
        return True

    def close(self, *, remove: bool = False) -> None:
        if self._file is not None:
            # On Windows an open file cannot be removed. No new live owner
            # will acquire this UUID, so close-before-unlink is safe too.
            self._file.close()
            self._file = None
        if remove:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                # Cleanup cannot affect recovery correctness; an unlocked
                # stale file still identifies a dead owner. In particular,
                # Windows may deny unlink while another probe has it open.
                pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(obj: object) -> object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _dumps(value: object) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, separators=(",", ":"))


def _loads(payload: str | None) -> Any:
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _format_dt(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(slots=True)
class JobRecord:
    id: str
    kind: str
    status: JobStatus = "queued"
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    request: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    _future: Optional[Future] = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_result: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "request": dict(self.request),
            "error": self.error,
        }
        if include_result:
            payload["result"] = self.result
        return payload


class JobStore(Protocol):
    def create(self, record: JobRecord) -> None: ...
    def save(self, record: JobRecord) -> None: ...
    def get(self, job_id: str) -> Optional[JobRecord]: ...
    def list(self, *, limit: int, tenants: frozenset[str] | None = None) -> list[JobRecord]: ...
    def cleanup(self, *, ttl_seconds: int) -> list[str]: ...
    def reset_incomplete(self, *, error: str) -> int: ...


class MemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.id] = record

    def save(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.id] = record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, limit: int, tenants: frozenset[str] | None = None) -> list[JobRecord]:
        limit = max(1, int(limit))
        with self._lock:
            records = [record for record in self._jobs.values()
                       if tenants is None or _tenant(record) in tenants]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def cleanup(self, *, ttl_seconds: int) -> list[str]:
        if ttl_seconds <= 0:
            return []
        cutoff = _now().timestamp() - float(ttl_seconds)
        removed: list[str] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                finished_at = record.finished_at
                if record.status in {"succeeded", "failed", "canceled", "interrupted"} and finished_at is not None:
                    if finished_at.timestamp() < cutoff:
                        self._jobs.pop(job_id, None)
                        removed.append(job_id)
        return removed

    def reset_incomplete(self, *, error: str) -> int:
        # Memory records cannot survive a process restart. Their workers may
        # still be active; resetting them would misreport work still executing.
        return 0


class SQLiteJobStore:
    def __init__(self, path: str | Path) -> None:
        # Alias paths must use the same sidecar directory as the database.
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._closed = False
        self._owner_id = uuid.uuid4().hex
        self._owners_path = self.path.with_name(self.path.name + ".owners")
        self._owners_path.mkdir(parents=True, exist_ok=True)
        self._lease = _OwnerLease(self._owners_path / (self._owner_id + ".lock"))
        if not self._lease.acquire():  # UUID collision or invalid filesystem.
            raise RuntimeError("Could not acquire SQLite job owner lock")
        try:
            self._initialize()
        except BaseException:
            if hasattr(self, "_conn"):
                self._conn.close()
            self._lease.close(remove=True)
            raise

    def _initialize(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # SQLite can return BUSY here without invoking its busy handler
            # when several workers open a brand-new database simultaneously.
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                    break
                except sqlite3.OperationalError as exc:
                    if str(exc) != "database is locked" or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
            self._conn.execute("BEGIN IMMEDIATE;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    runtime_owner TEXT,
                    tenant TEXT
                );
                """
            )
            columns = {str(row["name"]) for row in self._conn.execute("PRAGMA table_info(jobs);")}
            if "runtime_owner" not in columns:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN runtime_owner TEXT;")
            if "tenant" not in columns:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN tenant TEXT;")
                for row in self._conn.execute("SELECT id, request_json FROM jobs;").fetchall():
                    request = _loads(row["request_json"])
                    tenant = request.get("tenant") if isinstance(request, dict) else None
                    if isinstance(tenant, str) and tenant:
                        self._conn.execute("UPDATE jobs SET tenant=? WHERE id=?;", (tenant, row["id"]))
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_tenant_created_idx ON jobs(tenant, created_at);"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._conn.close()
                finally:
                    self._lease.close(remove=True)

    def create(self, record: JobRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, created_at, started_at, finished_at,
                    request_json, result_json, error, runtime_owner, tenant
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.id,
                    record.kind,
                    record.status,
                    _format_dt(record.created_at),
                    _format_dt(record.started_at),
                    _format_dt(record.finished_at),
                    _dumps(record.request),
                    _dumps(record.result) if record.result is not None else None,
                    record.error,
                    self._owner_id,
                    _tenant(record),
                ),
            )
            self._conn.commit()

    def save(self, record: JobRecord) -> None:
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, created_at, started_at, finished_at,
                    request_json, result_json, error, runtime_owner, tenant
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    request_json=excluded.request_json,
                    result_json=excluded.result_json,
                    error=excluded.error,
                    tenant=excluded.tenant
                WHERE jobs.runtime_owner=excluded.runtime_owner;
                """,
                (
                    record.id,
                    record.kind,
                    record.status,
                    _format_dt(record.created_at),
                    _format_dt(record.started_at),
                    _format_dt(record.finished_at),
                    _dumps(record.request),
                    _dumps(record.result) if record.result is not None else None,
                    record.error,
                    self._owner_id,
                    _tenant(record),
                ),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RuntimeError("Cannot update a job owned by another SQLite store")
            self._conn.commit()

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?;",
                (str(job_id),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self, *, limit: int, tenants: frozenset[str] | None = None) -> list[JobRecord]:
        limit = max(1, int(limit))
        if tenants is not None and not tenants:
            return []
        where = ""
        parameters: list[Any] = []
        if tenants is not None:
            where = " WHERE tenant IN (" + ",".join("?" for _ in tenants) + ")"
            parameters.extend(sorted(tenants))
        parameters.append(limit)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs" + where + " ORDER BY created_at DESC LIMIT ?;",
                parameters,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def cleanup(self, *, ttl_seconds: int) -> list[str]:
        if ttl_seconds <= 0:
            return []
        cutoff_dt = datetime.fromtimestamp(_now().timestamp() - float(ttl_seconds), timezone.utc)
        cutoff = cutoff_dt.isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM jobs
                WHERE status IN ('succeeded','failed','canceled','interrupted')
                  AND finished_at IS NOT NULL
                  AND finished_at < ?;
                """,
                (cutoff,),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            if ids:
                self._conn.executemany("DELETE FROM jobs WHERE id = ?;", [(job_id,) for job_id in ids])
                self._conn.commit()
        return ids

    def reset_incomplete(self, *, error: str) -> int:
        """Mark abandoned jobs interrupted; never replay their callables.

        A held owner lock proves that a sibling store is still alive, even if
        a long-running job has not written a status update. Legacy rows without
        owners are recoverable after an upgrade with old workers stopped.
        """
        count = 0
        with self._lock:
            owners = self._conn.execute(
                "SELECT DISTINCT runtime_owner FROM jobs WHERE status IN ('queued','running');"
            ).fetchall()
            for row in owners:
                owner = row["runtime_owner"]
                if owner == self._owner_id:
                    continue
                lease = None
                if owner is not None:
                    # Treat corrupt ownership as unknown, without consulting
                    # untrusted database text as a filesystem path.
                    if not isinstance(owner, str) or len(owner) != 32 or any(c not in "0123456789abcdef" for c in owner):
                        continue
                    lease = _OwnerLease(self._owners_path / (owner + ".lock"))
                    if not lease.acquire():
                        continue
                try:
                    cur = self._conn.execute(
                        """
                        UPDATE jobs SET status='interrupted', finished_at=?, error=?
                        WHERE status IN ('queued','running') AND runtime_owner IS ?;
                        """,
                        (_now().isoformat(), str(error), owner),
                    )
                    self._conn.commit()
                    count += int(cur.rowcount or 0)
                except BaseException:
                    self._conn.rollback()
                    raise
                finally:
                    if lease is not None:
                        lease.close(remove=True)
        return count

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=str(row["id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            created_at=_parse_dt(row["created_at"]) or _now(),
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
            request=_loads(row["request_json"]) or {},
            result=_loads(row["result_json"]),
            error=row["error"],
        )


class JobManager:
    """Submit callables to a thread pool while tracking status/result."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        ttl_seconds: int = 3600,
        thread_name_prefix: str = "sr-adapter-job",
        store: JobStore | None = None,
    ) -> None:
        self._worker_context = local()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix=thread_name_prefix,
            initializer=lambda: setattr(self._worker_context, "active", True),
        )
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._store: JobStore = store or MemoryJobStore()
        self._futures: Dict[str, Future] = {}
        self._lock = Lock()
        self._shutdown_lock = Lock()
        self._closed = False
        self._shutdown_done = Event()

    def submit(
        self,
        kind: str,
        func: Callable[[], Any],
        *,
        request: Optional[Mapping[str, Any]] = None,
        on_done: Optional[Callable[[], None]] = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            kind=str(kind),
            request=dict(request or {}),
        )
        cleanup_lock = Lock()
        cleanup_done = False

        def _cleanup_once() -> None:
            nonlocal cleanup_done
            with cleanup_lock:
                if cleanup_done:
                    return
                cleanup_done = True
            if on_done is not None:
                on_done()

        def _run() -> None:
            self._mark_running(job_id)
            try:
                result = func()
            except BaseException as exc:  # Executor threads also capture SystemExit.
                self._mark_failed(job_id, exc)
                return
            self._mark_succeeded(job_id, result)

        def _done(future: Future) -> None:
            try:
                _cleanup_once()
            finally:
                with self._lock:
                    self._futures.pop(job_id, None)
                # Do not retain completed callable closures (e.g. large uploads).
                record._future = None

        try:
            with self._shutdown_lock:
                if self._closed:
                    raise RuntimeError("Job manager is shut down")
                self.cleanup()
                self._store.create(record)
                try:
                    future = self._executor.submit(_run)
                except Exception as exc:
                    self._mark_failed(job_id, exc)
                    raise
                record._future = future
                with self._lock:
                    self._futures[job_id] = future
            # An already-finished future invokes callbacks immediately. Avoid
            # holding the lifecycle lock while calling client cleanup code.
            future.add_done_callback(_done)
        except BaseException:
            _cleanup_once()
            raise
        return record

    def reset_incomplete(
        self, *, error: str = "Worker stopped before recording completion; outcome unknown; not replayed"
    ) -> int:
        return self._store.reset_incomplete(error=str(error))

    def get(self, job_id: str) -> Optional[JobRecord]:
        self.cleanup()
        record = self._store.get(job_id)
        if record is None:
            return None
        with self._lock:
            record._future = self._futures.get(job_id)
        return record

    def list(self, *, limit: int = 100, tenants: frozenset[str] | None = None) -> list[JobRecord]:
        self.cleanup()
        if tenants is None:
            records = self._store.list(limit=max(1, int(limit)))
        else:
            records = self._store.list(limit=max(1, int(limit)), tenants=tenants)
        with self._lock:
            for record in records:
                record._future = self._futures.get(record.id)
        return records

    def cancel(self, job_id: str) -> bool:
        record = self._store.get(job_id)
        if record is None:
            return False
        if record.status not in {"queued", "running"}:
            return False
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return False
        if future.cancel():
            record.status = "canceled"
            record.finished_at = _now()
            record.error = "canceled"
            self._store.save(record)
            return True
        return False

    def cleanup(self) -> int:
        removed = self._store.cleanup(ttl_seconds=self._ttl_seconds)
        if removed:
            with self._lock:
                for job_id in removed:
                    self._futures.pop(job_id, None)
        return len(removed)

    def shutdown(self) -> None:
        """Drain accepted jobs before closing their persistence connection."""
        if getattr(self._worker_context, "active", False):
            # A worker cannot join itself or wait for an external shutdown
            # which is already joining it. Reject before changing any state.
            raise RuntimeError("JobManager.shutdown() cannot be called from its worker thread")
        with self._shutdown_lock:
            already_closed = self._closed
            self._closed = True
        if already_closed:
            self._shutdown_done.wait()
            return
        try:
            # Workers may try to submit more work; they must be able to observe
            # _closed and fail immediately while the executor is draining.
            self._executor.shutdown(wait=True, cancel_futures=False)
            close = getattr(self._store, "close", None)
            if callable(close):
                close()
        finally:
            self._shutdown_done.set()

    # ----------------------------------------------------------------- helpers
    def _mark_running(self, job_id: str) -> None:
        record = self._store.get(job_id)
        if record is None:
            return
        record.status = "running"
        record.started_at = _now()
        self._store.save(record)

    def _mark_succeeded(self, job_id: str, result: Any) -> None:
        record = self._store.get(job_id)
        if record is None:
            return
        record.status = "succeeded"
        record.finished_at = _now()
        record.result = result
        record.error = None
        self._store.save(record)

    def _mark_failed(self, job_id: str, exc: BaseException) -> None:
        record = self._store.get(job_id)
        if record is None:
            return
        record.status = "failed"
        record.finished_at = _now()
        record.error = str(exc) or type(exc).__name__
        self._store.save(record)


__all__ = [
    "JobManager",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "MemoryJobStore",
    "SQLiteJobStore",
]
