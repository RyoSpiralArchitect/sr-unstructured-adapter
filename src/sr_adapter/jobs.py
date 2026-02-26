# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lightweight in-process job manager with optional SQLite persistence.

This module is intentionally dependency-free so it can be reused by:
- FastAPI wrapper (SaaS shell)
- internal batch services
- research notebooks

Notes:
- Execution is still in-process via a thread pool.
- The SQLite backend persists job status/results for inspection across restarts,
  but does not provide distributed worker claiming.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Protocol


JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


def _now() -> datetime:
    return datetime.now(UTC)


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
        return parsed.replace(tzinfo=UTC)
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
    def list(self, *, limit: int) -> list[JobRecord]: ...
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

    def list(self, *, limit: int) -> list[JobRecord]:
        limit = max(1, int(limit))
        with self._lock:
            records = list(self._jobs.values())
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
                if record.status in {"succeeded", "failed", "canceled"} and finished_at is not None:
                    if finished_at.timestamp() < cutoff:
                        self._jobs.pop(job_id, None)
                        removed.append(job_id)
        return removed

    def reset_incomplete(self, *, error: str) -> int:
        count = 0
        with self._lock:
            now = _now()
            for record in self._jobs.values():
                if record.status in {"queued", "running"}:
                    record.status = "failed"
                    record.finished_at = now
                    record.error = error
                    count += 1
        return count


class SQLiteJobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
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
                    error TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, record: JobRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, created_at, started_at, finished_at,
                    request_json, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                ),
            )
            self._conn.commit()

    def save(self, record: JobRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, created_at, started_at, finished_at,
                    request_json, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    request_json=excluded.request_json,
                    result_json=excluded.result_json,
                    error=excluded.error;
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
                ),
            )
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

    def list(self, *, limit: int) -> list[JobRecord]:
        limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def cleanup(self, *, ttl_seconds: int) -> list[str]:
        if ttl_seconds <= 0:
            return []
        cutoff_dt = datetime.fromtimestamp(_now().timestamp() - float(ttl_seconds), UTC)
        cutoff = cutoff_dt.isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id FROM jobs
                WHERE status IN ('succeeded','failed','canceled')
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
        now = _now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE jobs
                SET status='failed',
                    finished_at=?,
                    error=?
                WHERE status IN ('queued','running');
                """,
                (now, str(error)),
            )
            self._conn.commit()
            return int(cur.rowcount or 0)

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
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix=thread_name_prefix,
        )
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._store: JobStore = store or MemoryJobStore()
        self._futures: Dict[str, Future] = {}
        self._lock = Lock()

    def submit(
        self,
        kind: str,
        func: Callable[[], Any],
        *,
        request: Optional[Mapping[str, Any]] = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            kind=str(kind),
            request=dict(request or {}),
        )
        self._store.create(record)

        def _run() -> None:
            self._mark_running(job_id)
            try:
                result = func()
            except Exception as exc:  # pragma: no cover - runtime failures vary
                self._mark_failed(job_id, exc)
                return
            self._mark_succeeded(job_id, result)

        future = self._executor.submit(_run)
        record._future = future
        with self._lock:
            self._futures[job_id] = future
        return record

    def reset_incomplete(self, *, error: str = "abandoned") -> int:
        return self._store.reset_incomplete(error=str(error))

    def get(self, job_id: str) -> Optional[JobRecord]:
        self.cleanup()
        record = self._store.get(job_id)
        if record is None:
            return None
        with self._lock:
            record._future = self._futures.get(job_id)
        return record

    def list(self, *, limit: int = 100) -> list[JobRecord]:
        self.cleanup()
        records = self._store.list(limit=max(1, int(limit)))
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
        self._executor.shutdown(wait=False, cancel_futures=False)
        store = getattr(self._store, "close", None)
        if callable(store):
            try:
                store()
            except Exception:
                pass

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
        record.error = str(exc)
        self._store.save(record)


__all__ = [
    "JobManager",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "MemoryJobStore",
    "SQLiteJobStore",
]

