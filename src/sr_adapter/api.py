# SPDX-License-Identifier: AGPL-3.0-or-later
"""FastAPI shell around the conversion pipeline (optional extra).

Install with:
  pip install "sr-unstructured-adapter[api]"
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from .pipeline import batch_convert, convert, stream_convert
from .jobs import JobManager, JobRecord, SQLiteJobStore
from .semantic import list_semantic_annotators
from .settings import get_settings, load_api_key_tenants
from .sniff import detect_type
from .version import get_adapter_version

try:  # pragma: no cover - optional dependency
    from fastapi import UploadFile as UploadFile
except Exception:  # pragma: no cover - when the api extra is not installed
    UploadFile = object  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from starlette.requests import Request as Request
except Exception:  # pragma: no cover - when the api extra is not installed
    Request = object  # type: ignore[assignment]


class ConversionOptions(BaseModel):
    """JSON options are strict so a misspelled flag cannot enable paid LLM calls."""

    model_config = ConfigDict(extra="forbid")

    recipe: str = "default"
    profile: str = "balanced"
    llm_ok: StrictBool = True
    deadline_ms: Optional[StrictInt] = Field(default=None, ge=0)
    max_blocks: Optional[StrictInt] = Field(default=None, ge=0)


class PathConversionRequest(ConversionOptions):
    path: str
    mime: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _nonempty_path(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("path must be a non-empty file path")
        return value


class BatchConversionRequest(ConversionOptions):
    paths: list[str] = Field(min_length=1)
    backend: Optional[Literal[
        "auto", "sync", "sequential", "none", "thread", "threads",
        "threadpool", "async", "asyncio", "dask", "ray",
    ]] = None
    concurrency: StrictInt = Field(default=0, ge=0)
    dask_scheduler: Optional[str] = None
    ray_address: Optional[str] = None

    @field_validator("paths")
    @classmethod
    def _nonempty_paths(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or "\0" in value for value in values):
            raise ValueError("paths must contain non-empty file paths")
        return values


def create_app():  # type: ignore[no-untyped-def]
    try:
        from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
        from starlette.concurrency import run_in_threadpool
        from starlette.formparsers import MultiPartException
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
    if not math.isfinite(max_upload_mb * 1024 * 1024):
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

    key_tenants = load_api_key_tenants()
    api_keys = _configured_api_keys() | key_tenants.keys()

    def _request_id_from(request: Request) -> str:
        candidate = request.headers.get("x-request-id")
        if candidate:
            candidate = candidate.strip()
            if 0 < len(candidate) <= 128 and all(32 <= ord(char) < 127 for char in candidate):
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

    def _require_api_key(request: Request) -> frozenset[str] | None:
        if not api_keys:
            return None
        candidate = _extract_key(request)
        if not candidate or candidate not in api_keys:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # An explicit scope takes precedence over legacy unrestricted keys.
        return key_tenants.get(candidate)

    auth_required = Depends(_require_api_key)

    def _resolve_tenant(request: Request, tenant: str | None) -> str | None:
        allowed_tenants = _require_api_key(request)
        tenant_value = tenant.strip() if isinstance(tenant, str) and tenant.strip() else None
        if tenant_value is None:
            if allowed_tenants is None:
                # Preserve recipe.llm.tenant precedence for legacy callers.
                return None
            tenant_value = os.getenv("SR_ADAPTER_TENANT", "default")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tenant_value):
            raise HTTPException(status_code=422, detail="Invalid tenant name")
        if allowed_tenants is not None and tenant_value not in allowed_tenants:
            raise HTTPException(status_code=403, detail="Tenant access denied")
        return tenant_value

    def _tenant_dependency(
        request: Request,
        tenant: str | None = Header(default=None, alias="X-SR-Tenant"),
    ) -> str | None:
        return _resolve_tenant(request, tenant)

    tenant_required = Depends(_tenant_dependency)

    def _job_tenant(tenant: str | None, recipe: str) -> str | None:
        from .recipe import load_recipe

        effective = str(tenant or load_recipe(recipe).llm.get("tenant") or os.getenv("SR_ADAPTER_TENANT", "default"))
        # Invalid trusted configuration does not establish tenant ownership.
        return effective if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", effective) else None

    def _visible_job(record: JobRecord | None, allowed_tenants: frozenset[str] | None) -> bool:
        if record is None:
            return False
        if allowed_tenants is None:
            return True
        # Historical jobs with no recorded tenant have no provable owner.
        tenant = record.request.get("tenant")
        return isinstance(tenant, str) and tenant in allowed_tenants

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
        if api_key and api_key in api_keys:
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
        except BaseException:
            try:
                handle.close()
            except Exception:
                pass
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass
            raise

    def _validate_selection(recipe: str, profile: str) -> None:
        from .recipe import load_recipe

        # These are registry names, never arbitrary filesystem paths.
        for name, value in (("recipe", recipe), ("profile", profile)):
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
                raise HTTPException(status_code=422, detail=f"Invalid {name} name")
        try:
            load_recipe(recipe)
            if profile.lower() != "auto":
                profile_store.load(profile)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Unknown or invalid recipe/profile") from exc

    def _path_to_file(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        try:
            info = path.stat()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Cannot access input file") from exc
        if not stat.S_ISREG(info.st_mode):
            raise HTTPException(status_code=422, detail="Path must refer to a regular file")
        if max_upload_bytes is not None and info.st_size > max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        return path

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
        try:
            if reset_incomplete and jobs_backend == "sqlite":
                try:
                    await run_in_threadpool(
                        job_manager.reset_incomplete,
                        error="Worker stopped before recording completion; outcome unknown; not replayed"
                    )
                except Exception:
                    raise RuntimeError("Job recovery failed; server startup aborted") from None
            yield
        finally:
            await run_in_threadpool(job_manager.shutdown)

    app = FastAPI(
        title="SR Unstructured Adapter",
        version=get_adapter_version(),
        lifespan=lifespan,
    )

    upload_routes = {"/convert", "/convert-stream", "/jobs/convert"}
    json_body_routes = {
        "/convert-path", "/batch-convert-paths",
        "/jobs/convert-path", "/jobs/batch-convert-paths",
    }

    def _route_path(scope: dict) -> str:
        """Use the app-local route when mounted beneath an ASGI root path."""
        path = scope.get("path", "")
        root = scope.get("root_path", "").rstrip("/")
        if root and (path == root or path.startswith(root + "/")):
            path = path[len(root):] or "/"
        return path

    def _protected_route(path: str) -> bool:
        path = path.rstrip("/") or "/"
        return (
            path in upload_routes | json_body_routes | {"/telemetry", "/metrics", "/jobs"}
            or path.startswith("/jobs/")
            or path.startswith("/inspect/")
        )

    class _UploadBodyLimit:
        """Bound conversion bodies before their parsers allocate unlimited data."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            path = _route_path(scope)
            if path not in upload_routes | json_body_routes or max_upload_bytes is None:
                await self.app(scope, receive, send)
                return
            # File size is checked separately; leave room for multipart headers.
            request_limit = max_upload_bytes + 1024 * 1024
            length = dict(scope.get("headers", [])).get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError
                except ValueError:
                    response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
                    await response(scope, receive, send)
                    return
                if declared > request_limit:
                    response = JSONResponse(status_code=413, content={"detail": "Upload request too large"})
                    await response(scope, receive, send)
                    return
            received = 0
            exceeded = False
            content_type = dict(scope.get("headers", [])).get(b"content-type", b"").lower()

            async def bounded_receive():
                nonlocal received, exceeded
                message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > request_limit:
                        exceeded = True
                        if content_type.startswith(b"multipart/form-data"):
                            # Use the parser's exception type so older supported
                            # Starlette releases also close spooled files. Its
                            # resulting 400 is mapped back to 413 below.
                            raise MultiPartException("Upload request too large")
                        raise HTTPException(status_code=413, detail="Upload request too large")
                return message

            async def bounded_send(message):
                if exceeded and message["type"] == "http.response.start":
                    message = {**message, "status": 413}
                await send(message)

            await self.app(scope, bounded_receive, bounded_send)

    app.add_middleware(_UploadBodyLimit)

    @app.middleware("http")
    async def _request_context_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = _request_id_from(request)
        request.state.request_id = request_id
        path = _route_path(request.scope)

        if rate_limit_rpm > 0 and path not in {"/healthz", "/metrics"}:
            now = time.monotonic()
            key = _rate_limit_key(request)
            with rate_limit_lock:
                # Expired clients must not accumulate for the server lifetime.
                for old_key, (started, _) in list(rate_limit_state.items()):
                    if now - started >= rate_limit_window_s:
                        rate_limit_state.pop(old_key, None)
                window_start, count = rate_limit_state.get(key, (now, 0))
                if now - window_start >= float(rate_limit_window_s):
                    window_start, count = now, 0
                count += 1
                rate_limit_state[key] = (window_start, count)
                if count > rate_limit_rpm:
                    retry_after = max(1, math.ceil(rate_limit_window_s - (now - window_start)))
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                        headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
                    )

        # Dependencies run after body parsing. Authenticate every protected
        # route before accepting multipart or JSON data, including mounted apps.
        if _protected_route(path):
            try:
                allowed_tenants = _require_api_key(request)
                route = path.rstrip("/") or "/"
                if allowed_tenants is not None and route in {"/telemetry", "/metrics"}:
                    raise HTTPException(status_code=403, detail="Aggregate telemetry requires an unrestricted API key")
                if route in upload_routes | json_body_routes:
                    _resolve_tenant(request, request.headers.get("x-sr-tenant"))
                    # A tenant header cannot scope a host filesystem path.
                    if allowed_tenants is not None and route in json_body_routes:
                        raise HTTPException(status_code=403, detail="Path conversion requires an unrestricted API key")
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers={**(exc.headers or {}), "X-Request-ID": request_id},
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

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(_: None = auth_required) -> Any:
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
        deadline_ms: Optional[int] = Query(None, ge=0),
        max_blocks: Optional[int] = Query(None, ge=0),
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> Dict[str, Any]:
        _validate_selection(recipe, profile)
        tmp_path = await _upload_to_tempfile(file)
        try:
            document = await run_in_threadpool(
                convert,
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

    @app.post("/convert-stream")
    async def convert_stream_upload(
        request: Request,
        *,
        file: UploadFile = File(...),
        recipe: str = Query("default"),
        profile: str = Query("balanced"),
        llm_ok: bool = Query(False),
        max_blocks: Optional[int] = Query(None, ge=0),
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ):  # type: ignore[no-untyped-def]
        if llm_ok:
            raise HTTPException(
                status_code=422,
                detail="Streaming conversion requires llm_ok=false (LLM escalation is not streamable).",
            )
        _validate_selection(recipe, profile)
        tmp_path = await _upload_to_tempfile(file)
        doc_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        mime_value = file.content_type or ""
        request_id = getattr(getattr(request, "state", None), "request_id", None)

        def _to_payload(value: object) -> Dict[str, Any]:
            if hasattr(value, "model_dump"):
                return value.model_dump()  # type: ignore[attr-defined]
            if hasattr(value, "dict"):
                return value.dict()  # type: ignore[call-arg]
            candidate = getattr(value, "__dict__", {})
            return candidate if isinstance(candidate, dict) else {}

        def _iter():  # type: ignore[no-untyped-def]
            try:
                meta: Dict[str, Any] = {
                    "kind": "document",
                    "id": doc_id,
                    "source": getattr(file, "filename", "") or "",
                    "type": detect_type(tmp_path),
                    "mime": mime_value,
                    "recipe": str(recipe),
                    "profile": str(profile),
                    "adapter_version": get_adapter_version(),
                    "created_at": created_at,
                }
                if request_id:
                    meta["request_id"] = request_id
                if tenant_value:
                    meta["tenant"] = tenant_value
                if isinstance(max_blocks, int):
                    meta["max_blocks"] = max_blocks
                yield (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")

                count = 0
                for block in stream_convert(
                    tmp_path,
                    recipe=recipe,
                    profile=profile,
                    max_blocks=max_blocks,
                    mime=mime_value or None,
                ):
                    record = {
                        "kind": "block",
                        "id": doc_id,
                        "source": getattr(file, "filename", "") or "",
                        "index": count,
                        "block": _to_payload(block),
                    }
                    yield (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    count += 1

                summary: Dict[str, Any] = {
                    "kind": "summary",
                    "id": doc_id,
                    "source": getattr(file, "filename", "") or "",
                    "block_count": count,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                }
                yield (json.dumps(summary, ensure_ascii=False) + "\n").encode("utf-8")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except Exception:
                    pass

        class _UploadStreamResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Covers disconnects before the generator's first iteration.
                    tmp_path.unlink(missing_ok=True)

        return _UploadStreamResponse(
            _iter(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post("/convert-path")
    def convert_path(
        payload: PathConversionRequest = Body(...),
        *,
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        _validate_selection(payload.recipe, payload.profile)
        recipe, profile, llm_ok = payload.recipe, payload.profile, payload.llm_ok
        mime_value = payload.mime
        deadline_value, max_blocks_value = payload.deadline_ms, payload.max_blocks
        path = _path_to_file(payload.path)

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
        payload: BatchConversionRequest = Body(...),
        *,
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> list[Dict[str, Any]]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        _validate_selection(payload.recipe, payload.profile)
        recipe, profile, llm_ok = payload.recipe, payload.profile, payload.llm_ok
        deadline_value, max_blocks_value = payload.deadline_ms, payload.max_blocks
        backend_value, concurrency_value = payload.backend, payload.concurrency
        dask_value, ray_value = payload.dask_scheduler, payload.ray_address
        path_list = [_path_to_file(path) for path in payload.paths]

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

    # -------------------------------------------------------------------- jobs
    @app.get("/jobs")
    def list_jobs(
        *,
        limit: int = Query(50, ge=1, le=500),
        allowed_tenants: frozenset[str] | None = auth_required,
    ) -> list[Dict[str, Any]]:
        return [record.to_dict() for record in job_manager.list(limit=limit, tenants=allowed_tenants)]

    @app.get("/jobs/{job_id}")
    def job_status(
        job_id: str,
        *,
        include_result: bool = Query(False),
        allowed_tenants: frozenset[str] | None = auth_required,
    ) -> Dict[str, Any]:
        record = job_manager.get(job_id)
        if not _visible_job(record, allowed_tenants):
            raise HTTPException(status_code=404, detail="Job not found")
        return record.to_dict(include_result=bool(include_result and record.status == "succeeded"))

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str, allowed_tenants: frozenset[str] | None = auth_required):  # type: ignore[no-untyped-def]
        record = job_manager.get(job_id)
        if not _visible_job(record, allowed_tenants):
            raise HTTPException(status_code=404, detail="Job not found")
        if record.status == "succeeded":
            return record.result
        if record.status in {"failed", "interrupted"}:
            raise HTTPException(status_code=409, detail=f"Job {record.status}: {record.error}")
        if record.status == "canceled":
            raise HTTPException(status_code=409, detail="Job canceled")
        raise HTTPException(status_code=409, detail=f"Job not ready (status={record.status})")

    @app.delete("/jobs/{job_id}")
    def job_cancel(job_id: str, allowed_tenants: frozenset[str] | None = auth_required) -> Dict[str, Any]:
        record = job_manager.get(job_id)
        if not _visible_job(record, allowed_tenants):
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
        deadline_ms: Optional[int] = Query(None, ge=0),
        max_blocks: Optional[int] = Query(None, ge=0),
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> Dict[str, Any]:
        filename = file.filename
        content_type = file.content_type
        _validate_selection(recipe, profile)
        tmp_path = await _upload_to_tempfile(file)

        def _task() -> Dict[str, Any]:
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

        def _cleanup() -> None:
            tmp_path.unlink(missing_ok=True)

        record = job_manager.submit(
            "convert",
            _task,
            on_done=_cleanup,
            request={
                "filename": filename,
                "content_type": content_type,
                "recipe": recipe,
                "profile": profile,
                "llm_ok": llm_ok,
                "deadline_ms": deadline_ms,
                "max_blocks": max_blocks,
                "tenant": _job_tenant(tenant_value, recipe),
            },
        )
        return record.to_dict()

    @app.post("/jobs/convert-path")
    def job_convert_path(
        payload: PathConversionRequest = Body(...),
        *,
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        _validate_selection(payload.recipe, payload.profile)
        recipe, profile, llm_ok = payload.recipe, payload.profile, payload.llm_ok
        mime_value = payload.mime
        deadline_value, max_blocks_value = payload.deadline_ms, payload.max_blocks
        path = _path_to_file(payload.path)

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
                "tenant": _job_tenant(tenant_value, recipe),
            },
        )
        return record.to_dict()

    @app.post("/jobs/batch-convert-paths")
    def job_batch_convert_paths(
        payload: BatchConversionRequest = Body(...),
        *,
        tenant_value: str | None = tenant_required,
        _: None = auth_required,
    ) -> Dict[str, Any]:
        if not allow_paths:
            raise HTTPException(
                status_code=403,
                detail="Path conversion is disabled. Set SR_ADAPTER_API_ALLOW_PATHS=1 to enable.",
            )
        _validate_selection(payload.recipe, payload.profile)
        recipe, profile, llm_ok = payload.recipe, payload.profile, payload.llm_ok
        deadline_value, max_blocks_value = payload.deadline_ms, payload.max_blocks
        backend_value, concurrency_value = payload.backend, payload.concurrency
        dask_value, ray_value = payload.dask_scheduler, payload.ray_address
        path_list = [_path_to_file(path) for path in payload.paths]

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
                "tenant": _job_tenant(tenant_value, recipe),
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
