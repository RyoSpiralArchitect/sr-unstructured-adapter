# SPDX-License-Identifier: AGPL-3.0-or-later
"""FastAPI shell around the conversion pipeline (optional extra).

Install with:
  pip install "sr-unstructured-adapter[api]"
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from .pipeline import batch_convert, convert
from .jobs import JobManager, SQLiteJobStore
from .semantic import list_semantic_annotators
from .settings import get_settings
from .version import get_adapter_version

try:  # pragma: no cover - optional dependency
    from fastapi import UploadFile as UploadFile
except Exception:  # pragma: no cover - when the api extra is not installed
    UploadFile = object  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from starlette.requests import Request as Request
except Exception:  # pragma: no cover - when the api extra is not installed
    Request = object  # type: ignore[assignment]


def create_app():  # type: ignore[no-untyped-def]
    try:
        from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI dependencies are not installed. "
            "Install with: pip install \"sr-unstructured-adapter[api]\""
        ) from exc

    from .telemetry import TelemetryExporter
    from .drivers.manager import DriverManager
    from .profiles import get_profile_store

    exporter = TelemetryExporter()
    profile_store = get_profile_store()
    settings = get_settings()

    allow_paths = os.getenv("SR_ADAPTER_API_ALLOW_PATHS", "").strip().lower() in {"1", "true", "yes"}
    trust_proxy_headers = os.getenv("SR_ADAPTER_API_TRUST_PROXY_HEADERS", "").strip().lower() in {"1", "true", "yes"}
    max_upload_mb_env = os.getenv("SR_ADAPTER_API_MAX_UPLOAD_MB", "").strip()
    if not max_upload_mb_env:
        max_upload_mb_env = os.getenv("SR_ADAPTER_MAX_SIZE_MB", "").strip()
    max_upload_mb_default = 200.0
    max_upload_mb: float
    if max_upload_mb_env:
        try:
            max_upload_mb = float(max_upload_mb_env)
        except Exception:
            max_upload_mb = max_upload_mb_default
    else:
        max_upload_mb = max_upload_mb_default
    max_upload_bytes = int(max_upload_mb * 1024 * 1024) if max_upload_mb > 0 else None

    def _configured_api_keys() -> set[str]:
        single = os.getenv("SR_ADAPTER_API_KEY", "").strip()
        multi = os.getenv("SR_ADAPTER_API_KEYS", "").strip()
        keys: set[str] = set()
        if single:
            keys.add(single)
        for item in multi.replace("\n", ",").split(","):
            item = item.strip()
            if item:
                keys.add(item)
        return keys

    api_keys = _configured_api_keys()

    def _request_id_from(request: Request) -> str:
        candidate = request.headers.get("x-request-id")
        if candidate:
            candidate = candidate.strip()
            if 0 < len(candidate) <= 128:
                return candidate
        return uuid.uuid4().hex

    def _extract_key(request: Request) -> str | None:
        header_key = request.headers.get("x-api-key")
        if header_key:
            return header_key.strip()
        auth = request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        return None

    def _require_api_key(request: Request) -> None:
        if not api_keys:
            return
        candidate = _extract_key(request)
        if not candidate or candidate not in api_keys:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth_required = Depends(_require_api_key)

    rate_limit_rpm = 0
    env_rate_limit = os.getenv("SR_ADAPTER_API_RATE_LIMIT_RPM", "").strip()
    if env_rate_limit:
        try:
            rate_limit_rpm = max(0, int(env_rate_limit))
        except Exception:
            rate_limit_rpm = 0
    rate_limit_window_s = 60
    rate_limit_state: dict[str, tuple[float, int]] = {}
    rate_limit_lock = Lock()

    def _rate_limit_key(request: Request) -> str:
        api_key = _extract_key(request)
        if api_key:
            return f"key:{api_key}"
        if trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                first = forwarded.split(",", 1)[0].strip()
                if first:
                    return f"ip:{first}"
        client = getattr(request, "client", None)
        host = getattr(client, "host", None) if client else None
        if isinstance(host, str) and host:
            return f"ip:{host}"
        return "anon"

    async def _upload_to_tempfile(file: UploadFile) -> Path:  # type: ignore[no-untyped-def]
        suffix = Path(getattr(file, "filename", "") or "").suffix
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = Path(handle.name)
        written = 0
        try:
            await file.seek(0)
            chunk_size = 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if max_upload_bytes is not None and written > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large")
                handle.write(chunk)
            handle.flush()
            handle.close()
            return tmp_path
        except Exception:
            try:
                handle.close()
            except Exception:
                pass
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass
            raise

    job_workers = settings.distributed.max_workers or 4
    env_workers = os.getenv("SR_ADAPTER_API_JOBS_MAX_WORKERS", "").strip()
    if env_workers:
        try:
            job_workers = max(1, int(env_workers))
        except Exception:
            pass
    job_ttl = 3600
    env_ttl = os.getenv("SR_ADAPTER_API_JOBS_TTL_SECONDS", "").strip()
    if env_ttl:
        try:
            job_ttl = max(0, int(env_ttl))
        except Exception:
            pass
    jobs_backend = os.getenv("SR_ADAPTER_API_JOBS_BACKEND", "memory").strip().lower() or "memory"
    job_store = None
    if jobs_backend == "sqlite":
        db_path = os.getenv("SR_ADAPTER_API_JOBS_DB_PATH", "").strip() or "sr_adapter_jobs.sqlite3"
        job_store = SQLiteJobStore(db_path)
    job_manager = JobManager(max_workers=job_workers, ttl_seconds=job_ttl, store=job_store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        reset_incomplete = os.getenv("SR_ADAPTER_API_JOBS_RESET_INCOMPLETE", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if reset_incomplete and jobs_backend == "sqlite":
            try:
                job_manager.reset_incomplete(error="server restarted")
            except Exception:
                pass
        try:
            yield
        finally:
            job_manager.shutdown()

    app = FastAPI(
        title="SR Unstructured Adapter",
        version=get_adapter_version(),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _request_context_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = _request_id_from(request)
        request.state.request_id = request_id

        if rate_limit_rpm > 0 and request.url.path not in {"/healthz", "/metrics"}:
            now = time.time()
            key = _rate_limit_key(request)
            with rate_limit_lock:
                window_start, count = rate_limit_state.get(key, (now, 0))
                if now - window_start >= float(rate_limit_window_s):
                    window_start, count = now, 0
                count += 1
                rate_limit_state[key] = (window_start, count)
                if count > rate_limit_rpm:
                    retry_after = max(0, int(rate_limit_window_s - (now - window_start)))
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                        headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
                    )

        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        return response

    @app.get("/")
    def root() -> Dict[str, object]:
        return {
            "service": "sr-unstructured-adapter",
            "version": get_adapter_version(),
        }

    @app.get("/healthz")
    def healthz() -> Dict[str, object]:
        return {
            "ok": True,
            "version": get_adapter_version(),
        }

    @app.get("/telemetry")
    def telemetry(_: None = auth_required) -> Dict[str, object]:
        return exporter.snapshot_dict()

    @app.get("/metrics")
    def metrics(_: None = auth_required) -> PlainTextResponse:
        try:
            payload = exporter.render_prometheus()
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")

    @app.get("/inspect/{kind}")
    def inspect(
        kind: str,
        *,
        as_json: bool = Query(False, alias="json"),
        _: None = auth_required,
    ):  # type: ignore[no-untyped-def]
        kind = kind.strip().lower()
        if kind == "drivers":
            items = list(DriverManager.registered_driver_names())
        elif kind == "profiles":
            items = list(profile_store.list_available())
        elif kind == "parsers":
            from .pipeline import REGISTRY  # local import to avoid import-time side effects

            items = sorted(set(REGISTRY.alias_to_key.values()))
        elif kind in {"semantic", "semantic-annotators", "semantic_annotators"}:
            items = list(list_semantic_annotators())
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown inspect kind '{kind}'",
            )
        if as_json:
            return {"kind": kind, "items": items}
        return PlainTextResponse("\n".join(items) + ("\n" if items else ""))

    @app.post("/convert")
    async def convert_upload(
        *,
        file: UploadFile = File(...),
        recipe: str = Query("default"),
        profile: str = Query("balanced"),
        llm_ok: bool = Query(True),
        deadline_ms: Optional[int] = Query(None),
        max_blocks: Optional[int] = Query(None),
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        tmp_path = await _upload_to_tempfile(file)
        try:
            document = convert(
                tmp_path,
                recipe=recipe,
                llm_ok=llm_ok,
                mime=file.content_type,
                deadline_ms=deadline_ms,
                max_blocks=max_blocks,
                profile=profile,
                tenant=tenant_value,
            )
            return document.model_dump()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass

    @app.post("/convert-path")
    def convert_path(
        payload: Dict[str, Any] = Body(...),
        *,
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(status_code=422, detail="Field 'path' is required")
        recipe = str(payload.get("recipe") or "default")
        profile = str(payload.get("profile") or "balanced")
        llm_ok = bool(payload.get("llm_ok", True))
        mime = payload.get("mime")
        mime_value = str(mime) if isinstance(mime, str) and mime else None
        deadline_ms = payload.get("deadline_ms")
        deadline_value = int(deadline_ms) if isinstance(deadline_ms, int) else None
        max_blocks = payload.get("max_blocks")
        max_blocks_value = int(max_blocks) if isinstance(max_blocks, int) else None
        path = Path(raw_path).expanduser()
        if max_upload_bytes is not None:
            try:
                if path.stat().st_size > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File too large")
            except OSError as exc:
                raise HTTPException(status_code=422, detail=f"Cannot stat path: {exc}") from exc
        document = convert(
            path,
            recipe=recipe,
            llm_ok=llm_ok,
            mime=mime_value,
            deadline_ms=deadline_value,
            max_blocks=max_blocks_value,
            profile=profile,
            tenant=tenant_value,
        )
        return document.model_dump()

    @app.post("/batch-convert-paths")
    def batch_convert_paths(
        payload: Dict[str, Any] = Body(...),
        *,
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> list[Dict[str, Any]]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        paths = payload.get("paths")
        if not isinstance(paths, list) or not all(isinstance(p, str) and p.strip() for p in paths):
            raise HTTPException(status_code=422, detail="Field 'paths' must be a list of strings")
        recipe = str(payload.get("recipe") or "default")
        profile = str(payload.get("profile") or "balanced")
        llm_ok = bool(payload.get("llm_ok", True))
        deadline_ms = payload.get("deadline_ms")
        deadline_value = int(deadline_ms) if isinstance(deadline_ms, int) else None
        max_blocks = payload.get("max_blocks")
        max_blocks_value = int(max_blocks) if isinstance(max_blocks, int) else None
        path_list = [Path(p).expanduser() for p in paths]
        if max_upload_bytes is not None:
            for path in path_list:
                try:
                    if path.stat().st_size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File too large")
                except OSError as exc:
                    raise HTTPException(status_code=422, detail=f"Cannot stat path: {exc}") from exc
        documents = batch_convert(
            path_list,
            recipe=recipe,
            llm_ok=llm_ok,
            deadline_ms=deadline_value,
            max_blocks=max_blocks_value,
            profile=profile,
            tenant=tenant_value,
        )
        return [doc.model_dump() for doc in documents]

    # -------------------------------------------------------------------- jobs
    @app.get("/jobs")
    def list_jobs(
        *,
        limit: int = Query(50, ge=1, le=500),
        _: None = auth_required,
    ) -> list[Dict[str, Any]]:
        return [record.to_dict() for record in job_manager.list(limit=limit)]

    @app.get("/jobs/{job_id}")
    def job_status(
        job_id: str,
        *,
        include_result: bool = Query(False),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        record = job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return record.to_dict(include_result=bool(include_result and record.status == "succeeded"))

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str, _: None = auth_required):  # type: ignore[no-untyped-def]
        record = job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.status == "succeeded":
            return record.result
        if record.status == "failed":
            raise HTTPException(status_code=409, detail=f"Job failed: {record.error}")
        if record.status == "canceled":
            raise HTTPException(status_code=409, detail="Job canceled")
        raise HTTPException(status_code=409, detail=f"Job not ready (status={record.status})")

    @app.delete("/jobs/{job_id}")
    def job_cancel(job_id: str, _: None = auth_required) -> Dict[str, Any]:
        record = job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job_manager.cancel(job_id):
            updated = job_manager.get(job_id)
            return (updated or record).to_dict()
        raise HTTPException(status_code=409, detail=f"Job cannot be canceled (status={record.status})")

    @app.post("/jobs/convert")
    async def job_convert_upload(
        *,
        file: UploadFile = File(...),
        recipe: str = Query("default"),
        profile: str = Query("balanced"),
        llm_ok: bool = Query(True),
        deadline_ms: Optional[int] = Query(None),
        max_blocks: Optional[int] = Query(None),
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        filename = file.filename
        content_type = file.content_type
        tmp_path = await _upload_to_tempfile(file)

        def _task() -> Dict[str, Any]:
            try:
                document = convert(
                    tmp_path,
                    recipe=recipe,
                    llm_ok=llm_ok,
                    mime=content_type,
                    deadline_ms=deadline_ms,
                    max_blocks=max_blocks,
                    profile=profile,
                    tenant=tenant_value,
                )
                return document.model_dump()
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except Exception:
                    pass

        record = job_manager.submit(
            "convert",
            _task,
            request={
                "filename": filename,
                "content_type": content_type,
                "recipe": recipe,
                "profile": profile,
                "llm_ok": llm_ok,
                "deadline_ms": deadline_ms,
                "max_blocks": max_blocks,
                "tenant": tenant_value,
            },
        )
        return record.to_dict()

    @app.post("/jobs/convert-path")
    def job_convert_path(
        payload: Dict[str, Any] = Body(...),
        *,
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise HTTPException(status_code=422, detail="Field 'path' is required")
        recipe = str(payload.get("recipe") or "default")
        profile = str(payload.get("profile") or "balanced")
        llm_ok = bool(payload.get("llm_ok", True))
        mime = payload.get("mime")
        mime_value = str(mime) if isinstance(mime, str) and mime else None
        deadline_ms = payload.get("deadline_ms")
        deadline_value = int(deadline_ms) if isinstance(deadline_ms, int) else None
        max_blocks = payload.get("max_blocks")
        max_blocks_value = int(max_blocks) if isinstance(max_blocks, int) else None
        path = Path(raw_path).expanduser()
        if max_upload_bytes is not None:
            try:
                if path.stat().st_size > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File too large")
            except OSError as exc:
                raise HTTPException(status_code=422, detail=f"Cannot stat path: {exc}") from exc

        def _task() -> Dict[str, Any]:
            document = convert(
                path,
                recipe=recipe,
                llm_ok=llm_ok,
                mime=mime_value,
                deadline_ms=deadline_value,
                max_blocks=max_blocks_value,
                profile=profile,
                tenant=tenant_value,
            )
            return document.model_dump()

        record = job_manager.submit(
            "convert-path",
            _task,
            request={
                "path": str(path),
                "recipe": recipe,
                "profile": profile,
                "llm_ok": llm_ok,
                "mime": mime_value,
                "deadline_ms": deadline_value,
                "max_blocks": max_blocks_value,
                "tenant": tenant_value,
            },
        )
        return record.to_dict()

    @app.post("/jobs/batch-convert-paths")
    def job_batch_convert_paths(
        payload: Dict[str, Any] = Body(...),
        *,
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        paths = payload.get("paths")
        if not isinstance(paths, list) or not all(isinstance(p, str) and p.strip() for p in paths):
            raise HTTPException(status_code=422, detail="Field 'paths' must be a list of strings")
        recipe = str(payload.get("recipe") or "default")
        profile = str(payload.get("profile") or "balanced")
        llm_ok = bool(payload.get("llm_ok", True))
        deadline_ms = payload.get("deadline_ms")
        deadline_value = int(deadline_ms) if isinstance(deadline_ms, int) else None
        max_blocks = payload.get("max_blocks")
        max_blocks_value = int(max_blocks) if isinstance(max_blocks, int) else None
        backend = payload.get("backend")
        backend_value = str(backend) if isinstance(backend, str) and backend.strip() else None
        concurrency = payload.get("concurrency")
        concurrency_value = int(concurrency) if isinstance(concurrency, int) else 0
        dask_scheduler = payload.get("dask_scheduler")
        dask_value = str(dask_scheduler) if isinstance(dask_scheduler, str) and dask_scheduler.strip() else None
        ray_address = payload.get("ray_address")
        ray_value = str(ray_address) if isinstance(ray_address, str) and ray_address.strip() else None
        path_list = [Path(p).expanduser() for p in paths]
        if max_upload_bytes is not None:
            for path in path_list:
                try:
                    if path.stat().st_size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File too large")
                except OSError as exc:
                    raise HTTPException(status_code=422, detail=f"Cannot stat path: {exc}") from exc

        def _task() -> list[Dict[str, Any]]:
            documents = batch_convert(
                path_list,
                recipe=recipe,
                llm_ok=llm_ok,
                deadline_ms=deadline_value,
                max_blocks=max_blocks_value,
                profile=profile,
                tenant=tenant_value,
                backend=backend_value,
                concurrency=concurrency_value,
                dask_scheduler=dask_value,
                ray_address=ray_value,
            )
            return [doc.model_dump() for doc in documents]

        record = job_manager.submit(
            "batch-convert-paths",
            _task,
            request={
                "paths": [str(p) for p in path_list],
                "recipe": recipe,
                "profile": profile,
                "llm_ok": llm_ok,
                "deadline_ms": deadline_value,
                "max_blocks": max_blocks_value,
                "tenant": tenant_value,
                "backend": backend_value,
                "concurrency": concurrency_value,
                "dask_scheduler": dask_value,
                "ray_address": ray_value,
            },
        )
        return record.to_dict()

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is not installed. Install with: pip install \"sr-unstructured-adapter[api]\""
        ) from exc

    uvicorn.run(
        "sr_adapter.api:create_app",
        factory=True,
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level=str(args.log_level),
    )
    return 0


__all__ = ["create_app", "main"]
