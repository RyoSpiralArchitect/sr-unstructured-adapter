# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import pytest


fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from sr_adapter.api import create_app  # noqa: E402
from sr_adapter.drivers.base import LLMDriver  # noqa: E402


def test_api_healthz() -> None:
    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert isinstance(data["version"], str)
    assert resp.headers.get("x-request-id")


def test_api_request_id_roundtrip() -> None:
    client = TestClient(create_app())
    resp = client.get("/healthz", headers={"x-request-id": "req-123"})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == "req-123"


def test_api_metrics_disabled_by_default() -> None:
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "sr_adapter_kernel_calls_total" in resp.text


def test_api_convert_upload(tmp_path: Path) -> None:
    client = TestClient(create_app())
    payload = b"Hello\nWorld\n"
    resp = client.post(
        "/convert?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", payload, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "text"
    assert data["meta"]["block_count"] >= 1
    assert any("Hello" in block["text"] for block in data["blocks"])


def test_api_convert_stream_upload() -> None:
    client = TestClient(create_app())
    payload = b"Hello\nWorld\n"
    resp = client.post(
        "/convert-stream?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", payload, "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("application/x-ndjson")
    assert resp.headers.get("x-request-id")

    lines = [line for line in resp.text.splitlines() if line.strip()]
    assert len(lines) >= 3
    events = [json.loads(line) for line in lines]
    assert events[0]["kind"] == "document"
    assert events[-1]["kind"] == "summary"
    block_events = [event for event in events if event.get("kind") == "block"]
    assert block_events
    assert events[-1]["block_count"] == len(block_events)


def test_api_convert_stream_rejects_llm_ok() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/convert-stream?recipe=default&profile=balanced&llm_ok=true",
        files={"file": ("sample.txt", b"Hello\n", "text/plain")},
    )
    assert resp.status_code == 422


def test_api_upload_size_limit(monkeypatch) -> None:
    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", "0.0001")
    client = TestClient(create_app())
    payload = b"x" * 1024
    resp = client.post(
        "/convert?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", payload, "text/plain")},
    )
    assert resp.status_code == 413


def test_api_convert_path_requires_flag(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("Alpha", encoding="utf-8")
    client = TestClient(create_app())
    resp = client.post("/convert-path", json={"path": str(target)})
    assert resp.status_code == 403


def test_api_convert_path_when_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    target = tmp_path / "note.txt"
    target.write_text("Alpha", encoding="utf-8")
    client = TestClient(create_app())
    resp = client.post(
        "/convert-path",
        json={"path": str(target), "recipe": "default", "profile": "balanced", "llm_ok": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "text"
    assert any("Alpha" in block["text"] for block in data["blocks"])


def test_api_inspect_profiles() -> None:
    client = TestClient(create_app())
    resp = client.get("/inspect/profiles?json=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "profiles"
    assert "balanced" in set(data["items"])


def test_api_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("SR_ADAPTER_API_RATE_LIMIT_RPM", "1")
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    resp = client.get("/")
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") is not None
    assert resp.headers.get("x-request-id") is not None


def test_api_auth_is_optional_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SR_ADAPTER_API_KEYS", "k1,k2")
    client = TestClient(create_app())

    healthz = client.get("/healthz")
    assert healthz.status_code == 200

    resp = client.post(
        "/convert?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", b"Hello\n", "text/plain")},
    )
    assert resp.status_code == 401

    resp_ok = client.post(
        "/convert?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", b"Hello\n", "text/plain")},
        headers={"x-api-key": "k1"},
    )
    assert resp_ok.status_code == 200


def test_api_tenant_header_overrides_llm_tenant(monkeypatch) -> None:
    class _DummyDriver(LLMDriver):
        def __init__(self) -> None:
            super().__init__("dummy", {})

        def generate(self, prompt: str, *, metadata=None):
            return {
                "model": "dummy-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"Reviewed: {prompt}"},
                    }
                ],
                "usage": {"total_tokens": 32},
            }

    class _DummyTenantManager:
        def get_default_tenant(self) -> str:
            return "default"

    class _DummyDriverManager:
        def __init__(self) -> None:
            self.tenant_manager = _DummyTenantManager()

        def get_driver(self, tenant: str, llm_config):
            assert tenant == "alpha"
            return _DummyDriver()

    monkeypatch.setattr("sr_adapter.delegate._driver_manager", _DummyDriverManager())

    client = TestClient(create_app())
    resp = client.post(
        "/convert?recipe=call_log&profile=balanced",
        files={"file": ("sample.txt", b"Hello world\n", "text/plain")},
        headers={"x-sr-tenant": "alpha"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["llm_escalations"] >= 1


def test_api_jobs_convert_upload() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/jobs/convert?recipe=default&profile=balanced&llm_ok=false",
        files={"file": ("sample.txt", b"Hello\nWorld\n", "text/plain")},
    )
    assert resp.status_code == 200
    job = resp.json()
    assert "id" in job
    job_id = job["id"]

    # Job should complete quickly; poll result endpoint.
    for _ in range(100):
        result = client.get(f"/jobs/{job_id}/result")
        if result.status_code == 200:
            payload = result.json()
            assert payload["meta"]["type"] == "text"
            return
        assert result.status_code == 409
    raise AssertionError("Job did not complete in time")


def test_api_jobs_backend_sqlite_persists(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_BACKEND", "sqlite")
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_DB_PATH", str(db_path))

    with TestClient(create_app()) as client:
        resp = client.post(
            "/jobs/convert?recipe=default&profile=balanced&llm_ok=false",
            files={"file": ("sample.txt", b"Hello\nWorld\n", "text/plain")},
        )
        assert resp.status_code == 200
        job_id = resp.json()["id"]

        for _ in range(100):
            result = client.get(f"/jobs/{job_id}/result")
            if result.status_code == 200:
                break
            assert result.status_code == 409
        else:
            raise AssertionError("Job did not complete in time")

    with TestClient(create_app()) as client2:
        resp = client2.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        result = client2.get(f"/jobs/{job_id}/result")
        assert result.status_code == 200
        payload = result.json()
        assert payload["meta"]["type"] == "text"


@pytest.mark.parametrize("route,field", [
    ("/convert-path", "path"),
    ("/jobs/convert-path", "path"),
    ("/batch-convert-paths", "paths"),
    ("/jobs/batch-convert-paths", "paths"),
])
@pytest.mark.parametrize("options", [
    {"llm_ok": "false"},
    {"llm_ok": 0},
    {"deadline_ms": True},
    {"deadline_ms": -1},
    {"max_blocks": -1},
    {"max_blocks": "2"},
    {"unknown_option": False},
])
def test_api_path_options_are_validated(monkeypatch, tmp_path, route, field, options):
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    target = tmp_path / "input.txt"
    target.write_text("input")
    payload = {field: str(target) if field == "path" else [str(target)], **options}
    with TestClient(create_app()) as client:
        assert client.post(route, json=payload).status_code == 422


@pytest.mark.parametrize("route", ["/convert", "/convert-stream", "/jobs/convert"])
@pytest.mark.parametrize("option", ["max_blocks=-1", "recipe=missing-recipe", "profile=missing-profile", "recipe=../settings", "profile=../settings"])
def test_api_upload_invalid_options_fail_before_conversion(monkeypatch, route, option):
    def unexpected(*args, **kwargs):
        raise AssertionError("Conversion must not start")

    monkeypatch.setattr("sr_adapter.api.convert", unexpected)
    monkeypatch.setattr("sr_adapter.api.stream_convert", unexpected)
    with TestClient(create_app()) as client:
        response = client.post(
            f"{route}?llm_ok=false&{option}",
            files={"file": ("sample.txt", b"Hello", "text/plain")},
        )
    assert response.status_code == 422


@pytest.mark.parametrize("max_size", ["0", "1"])
def test_api_path_must_be_regular_file(monkeypatch, tmp_path, max_size):
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", max_size)
    with TestClient(create_app()) as client:
        response = client.post("/convert-path", json={"path": str(tmp_path), "llm_ok": False})
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_api_nonfinite_upload_limit_does_not_crash(monkeypatch, value):
    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", value)
    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 200


def test_api_fake_keys_do_not_bypass_rate_limit(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_API_RATE_LIMIT_RPM", "1")
    with TestClient(create_app()) as client:
        assert client.get("/", headers={"X-API-Key": "invented-1"}).status_code == 200
        assert client.get("/", headers={"X-API-Key": "invented-2"}).status_code == 429


def test_api_batch_paths_passes_backend_options(monkeypatch, tmp_path):
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    target = tmp_path / "input.txt"
    target.write_text("input")
    seen = {}

    def fake_batch(paths, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("sr_adapter.api.batch_convert", fake_batch)
    with TestClient(create_app()) as client:
        response = client.post("/batch-convert-paths", json={
            "paths": [str(target)], "llm_ok": False,
            "backend": "threadpool", "concurrency": 2,
        })
    assert response.status_code == 200
    assert seen["llm_ok"] is False
    assert seen["backend"] == "threadpool"
    assert seen["concurrency"] == 2


def test_api_upload_conversion_does_not_block_health(monkeypatch):
    import asyncio
    from threading import Event
    import httpx

    started, release = Event(), Event()

    class Result:
        def model_dump(self):
            return {"ok": True}

    def fake_convert(*args, **kwargs):
        started.set()
        release.wait(3)
        return Result()

    monkeypatch.setattr("sr_adapter.api.convert", fake_convert)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/convert?llm_ok=false", files={
                "file": ("sample.txt", b"Hello", "text/plain"),
            }))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                assert (await asyncio.wait_for(client.get("/healthz"), 1)).status_code == 200
                assert not task.done(), "Conversion ran on the event loop"
            finally:
                release.set()
                response = await task
            assert response.status_code == 200

    asyncio.run(scenario())


def test_api_stream_conversion_runs_outside_event_loop(monkeypatch):
    import asyncio
    from sr_adapter.schema import Block

    def fake_stream(*args, **kwargs):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        yield Block(id="block", type="paragraph", text="Hello")

    monkeypatch.setattr("sr_adapter.api.stream_convert", fake_stream)
    with TestClient(create_app()) as client:
        response = client.post("/convert-stream", files={"file": ("sample.txt", b"Hello", "text/plain")})
    assert response.status_code == 200
    assert json.loads(response.text.splitlines()[-1])["block_count"] == 1


def test_api_upload_auth_precedes_multipart_parsing(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_API_KEY", "test-only")
    with TestClient(create_app()) as client:
        response = client.post("/convert", content=b"not multipart")
    assert response.status_code == 401


def test_api_rejects_oversized_request_before_parsing(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", "0.0001")
    with TestClient(create_app()) as client:
        response = client.post("/convert", content=b"x" * (1024 * 1024 + 512))
    assert response.status_code == 413


def test_api_chunked_upload_request_is_bounded(monkeypatch):
    import asyncio
    import httpx

    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", "0.0001")

    async def body():
        yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
        for _ in range(18):
            yield b"x" * 65536
        yield b"\r\n--test--\r\n"

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            response = await client.post("/convert", content=body(), headers={"Content-Type": "multipart/form-data; boundary=test"})
        assert response.status_code == 413

    asyncio.run(scenario())


def test_api_stream_disconnect_before_first_chunk_cleans_upload(monkeypatch, tmp_path):
    import asyncio
    import io
    import tempfile
    from starlette.datastructures import UploadFile
    from starlette.requests import Request

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = create_app()
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/convert-stream")

    async def scenario():
        scope = {"type": "http", "method": "POST", "path": "/convert-stream", "headers": [], "asgi": {"spec_version": "2.4"}}
        upload = UploadFile(io.BytesIO(b"Hello"), filename="sample.txt")
        response = await endpoint(Request(scope), file=upload, recipe="default", profile="balanced", llm_ok=False, max_blocks=None, tenant_value="default", _=None)
        assert list(tmp_path.iterdir())

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise RuntimeError("connection closed")

        with pytest.raises(RuntimeError, match="connection closed"):
            await response(scope, receive, send)
        assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("route", ["/convert", "/convert-path", "/jobs/convert", "/jobs/batch-convert-paths"])
def test_api_protected_routes_authenticate_before_reading_body(monkeypatch, mounted, route):
    import asyncio
    import httpx

    monkeypatch.setenv("SR_ADAPTER_API_KEY", "test-only")
    app = create_app()
    prefix = ""
    if mounted:
        parent = fastapi.FastAPI()
        parent.mount("/adapter", app)
        app = parent
        prefix = "/adapter"
    reads = []

    async def body():
        reads.append(True)
        yield b'{"path":"never read"}'

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(prefix + route, content=body(), headers={"Content-Type": "application/json"})
        assert response.status_code == 401
        assert response.headers.get("x-request-id")

    asyncio.run(scenario())
    assert reads == []


def test_api_mounted_public_endpoints_and_authenticated_routes(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_API_KEY", "test-only")
    parent = fastapi.FastAPI()
    parent.mount("/adapter", create_app())
    with TestClient(parent) as client:
        for route in ("/", "/healthz", "/docs", "/openapi.json"):
            assert client.get("/adapter" + route).status_code == 200
        for route in ("/telemetry", "/metrics", "/inspect/drivers", "/jobs"):
            assert client.get("/adapter" + route).status_code == 401
            assert client.get("/adapter" + route, headers={"X-API-Key": "test-only"}).status_code == 200


@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("json_body", [False, True])
def test_api_body_limits_cover_mounted_multipart_and_json(monkeypatch, mounted, json_body):
    import asyncio
    import httpx

    monkeypatch.setenv("SR_ADAPTER_API_KEY", "test-only")
    monkeypatch.setenv("SR_ADAPTER_API_MAX_UPLOAD_MB", "0.0001")
    app = create_app()
    prefix = ""
    if mounted:
        parent = fastapi.FastAPI()
        parent.mount("/adapter", app)
        app = parent
        prefix = "/adapter"
    route = "/convert-path" if json_body else "/convert"
    content_type = "application/json" if json_body else "multipart/form-data; boundary=test"

    async def body():
        if json_body:
            yield b'{"path":"'
        else:
            yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
        for _ in range(18):
            yield b"x" * 65536
        yield b'"}' if json_body else b'\r\n--test--\r\n'

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(prefix + route, content=body(), headers={"Content-Type": content_type, "X-API-Key": "test-only"})
        assert response.status_code == 413

    asyncio.run(scenario())


def _configure_tenant_keys(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_API_KEY_TENANTS", json.dumps({
        "alpha-key": ["alpha"], "alpha-rotated": ["alpha"], "beta-key": ["beta"],
        "both-key": ["alpha", "beta"],
    }))
    monkeypatch.setenv("SR_ADAPTER_API_KEY", "admin-key")
    monkeypatch.delenv("SR_ADAPTER_API_KEYS", raising=False)
    monkeypatch.setenv("SR_ADAPTER_TENANT", "default")


def _tenant_headers(key="alpha-key", tenant="alpha"):
    headers = {"X-API-Key": key}
    if tenant is not None:
        headers["X-SR-Tenant"] = tenant
    return headers


@pytest.mark.parametrize("route", ["/convert", "/convert-stream", "/jobs/convert"])
@pytest.mark.parametrize("tenant", ["beta", "", None])
def test_scoped_uploads_reject_other_or_implicit_tenants_before_body(monkeypatch, route, tenant):
    import asyncio
    import httpx

    _configure_tenant_keys(monkeypatch)
    reads = []

    async def body():
        reads.append(True)
        yield b"must not be parsed"

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
            response = await client.post(route + "?llm_ok=false", content=body(), headers=_tenant_headers(tenant=tenant))
        assert response.status_code == 403
        assert response.headers.get("x-request-id")

    asyncio.run(scenario())
    assert reads == []


@pytest.mark.parametrize("route,field", [
    ("/convert-path", "path"), ("/jobs/convert-path", "path"),
    ("/batch-convert-paths", "paths"), ("/jobs/batch-convert-paths", "paths"),
])
@pytest.mark.parametrize("tenant", ["alpha", "beta"])
def test_scoped_keys_cannot_use_host_paths_even_when_paths_enabled(monkeypatch, route, field, tenant):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    with TestClient(create_app()) as client:
        response = client.post(route, headers=_tenant_headers(tenant=tenant), json={
            field: "/must/not/be/read" if field == "path" else ["/must/not/be/read"],
            "llm_ok": False,
        })
    assert response.status_code == 403


def test_scoped_uploads_keep_options_and_resolve_default_tenant(monkeypatch):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_TENANT", "alpha")
    seen = []

    class Result:
        def model_dump(self):
            return {"ok": True}

    def fake_convert(path, **options):
        seen.append(options)
        return Result()

    monkeypatch.setattr("sr_adapter.api.convert", fake_convert)
    with TestClient(create_app()) as client:
        response = client.post(
            "/convert?recipe=default&profile=balanced&llm_ok=false&deadline_ms=100&max_blocks=2",
            headers={"Authorization": "Bearer alpha-key"},
            files={"file": ("sample.txt", b"Hello", "text/plain")},
        )
        assert response.status_code == 200
        invalid = client.post("/convert?max_blocks=-1", headers=_tenant_headers(), files={
            "file": ("sample.txt", b"Hello", "text/plain"),
        })
        assert invalid.status_code == 422
    assert len(seen) == 1
    assert seen[0]["tenant"] == "alpha"
    assert seen[0]["llm_ok"] is False
    assert seen[0]["deadline_ms"] == 100
    assert seen[0]["max_blocks"] == 2


def test_scoped_stream_reports_authorized_tenant(monkeypatch):
    from sr_adapter.schema import Block

    _configure_tenant_keys(monkeypatch)
    monkeypatch.setattr("sr_adapter.api.stream_convert", lambda *a, **k: iter([
        Block(id="one", type="paragraph", text="Hello"),
    ]))
    with TestClient(create_app()) as client:
        response = client.post("/convert-stream", headers=_tenant_headers(), files={
            "file": ("sample.txt", b"Hello", "text/plain"),
        })
    assert response.status_code == 200
    assert json.loads(response.text.splitlines()[0])["tenant"] == "alpha"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_scoped_jobs_hide_other_tenants_and_allow_rotated_keys(monkeypatch, tmp_path, backend):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_BACKEND", backend)
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    class Result:
        def __init__(self, tenant):
            self.tenant = tenant

        def model_dump(self):
            return {"private_result": self.tenant}

    monkeypatch.setattr("sr_adapter.api.convert", lambda *a, **k: Result(k["tenant"]))
    with TestClient(create_app()) as client:
        jobs = {}
        for tenant in ("alpha", "beta"):
            response = client.post("/jobs/convert?llm_ok=false", headers=_tenant_headers(f"{tenant}-key", tenant), files={
                "file": (f"{tenant}.txt", b"private input", "text/plain"),
            })
            assert response.status_code == 200
            jobs[tenant] = response.json()["id"]
            assert response.json()["request"]["tenant"] == tenant
        alpha_id, beta_id = jobs["alpha"], jobs["beta"]
        for suffix in ("", "?include_result=true", "/result"):
            denied = client.get(f"/jobs/{beta_id}{suffix}", headers=_tenant_headers())
            assert denied.status_code == 404
            assert denied.json() == {"detail": "Job not found"}
        assert client.delete(f"/jobs/{beta_id}", headers=_tenant_headers()).status_code == 404
        visible = client.get("/jobs?limit=1", headers=_tenant_headers()).json()
        assert [job["id"] for job in visible] == [alpha_id]
        assert len(client.get("/jobs", headers=_tenant_headers("both-key")).json()) == 2
        assert len(client.get("/jobs", headers=_tenant_headers("admin-key")).json()) == 2
        # Header spoofing cannot enlarge the credential's scope.
        assert client.get(f"/jobs/{beta_id}", headers=_tenant_headers(tenant="beta")).status_code == 404
        for _ in range(100):
            result = client.get(f"/jobs/{alpha_id}/result", headers=_tenant_headers("alpha-rotated"))
            if result.status_code == 200:
                break
            assert result.status_code == 409
        assert result.status_code == 200
        assert result.json() == {"private_result": "alpha"}
        assert "alpha-key" not in json.dumps(visible)

    if backend == "sqlite":
        with TestClient(create_app()) as client:
            assert client.get(f"/jobs/{alpha_id}/result", headers=_tenant_headers("alpha-rotated")).status_code == 200
            assert client.get(f"/jobs/{alpha_id}/result", headers=_tenant_headers("beta-key", "beta")).status_code == 404


def test_scoped_cancel_cannot_cancel_another_tenants_queued_job(monkeypatch):
    from threading import Event

    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_MAX_WORKERS", "1")
    started, release = Event(), Event()

    class Result:
        def model_dump(self):
            return {"ok": True}

    def fake_convert(*args, **kwargs):
        started.set()
        release.wait(3)
        return Result()

    monkeypatch.setattr("sr_adapter.api.convert", fake_convert)
    with TestClient(create_app()) as client:
        try:
            for idx in range(2):
                response = client.post("/jobs/convert?llm_ok=false", headers=_tenant_headers(), files={
                    "file": (f"sample-{idx}.txt", b"Hello", "text/plain"),
                })
                assert response.status_code == 200
                assert started.wait(1)
            job_id = response.json()["id"]
            assert client.delete(f"/jobs/{job_id}", headers=_tenant_headers("beta-key", "beta")).status_code == 404
            assert client.get(f"/jobs/{job_id}", headers=_tenant_headers()).json()["status"] == "queued"
            canceled = client.delete(f"/jobs/{job_id}", headers=_tenant_headers("alpha-rotated"))
            assert canceled.status_code == 200
            assert canceled.json()["status"] == "canceled"
        finally:
            release.set()


def test_scoped_keys_cannot_read_legacy_jobs_with_unknown_owner(monkeypatch, tmp_path):
    from sr_adapter.jobs import JobRecord, SQLiteJobStore

    _configure_tenant_keys(monkeypatch)
    path = tmp_path / "legacy.db"
    store = SQLiteJobStore(path)
    store.create(JobRecord(id="old-job", kind="convert", status="succeeded", request={"tenant": None}, result={"private": "legacy"}))
    store.close()
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_BACKEND", "sqlite")
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_DB_PATH", str(path))
    with TestClient(create_app()) as client:
        assert client.get("/jobs", headers=_tenant_headers()).json() == []
        assert client.get("/jobs/old-job/result", headers=_tenant_headers()).status_code == 404
        assert client.get("/jobs/old-job/result", headers=_tenant_headers("admin-key")).json() == {"private": "legacy"}


def test_scoped_key_overlap_never_becomes_unrestricted(monkeypatch):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_API_KEYS", "alpha-key")
    with TestClient(create_app()) as client:
        for path in ("/telemetry", "/metrics"):
            assert client.get(path, headers=_tenant_headers()).status_code == 403
            assert client.get(path, headers=_tenant_headers("admin-key")).status_code == 200
        assert client.get("/inspect/drivers", headers=_tenant_headers()).status_code == 200
        assert client.get("/jobs", headers={"X-API-Key": "unknown"}).status_code == 401


def test_scoped_only_configuration_requires_authentication_and_fails_closed(monkeypatch):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.delenv("SR_ADAPTER_API_KEY")
    with TestClient(create_app()) as client:
        assert client.get("/jobs").status_code == 401
        assert client.get("/jobs", headers=_tenant_headers()).status_code == 200
    monkeypatch.setenv("SR_ADAPTER_API_KEY_TENANTS", "{}")
    with pytest.raises(ValueError, match="SR_ADAPTER_API_KEY_TENANTS"):
        create_app()


@pytest.mark.parametrize("route", ["/convert", "/convert/", "/convert-path", "/jobs/convert", "/jobs/convert-path"])
def test_scoped_authorization_covers_mounted_routes(monkeypatch, route):
    _configure_tenant_keys(monkeypatch)
    parent = fastapi.FastAPI()
    parent.mount("/adapter", create_app())
    with TestClient(parent) as client:
        response = client.post("/adapter" + route, headers=_tenant_headers(tenant="beta"), content=b"invalid body")
    assert response.status_code == 403


def test_legacy_admin_keeps_path_access_when_scoped_keys_are_configured(monkeypatch, tmp_path):
    _configure_tenant_keys(monkeypatch)
    monkeypatch.setenv("SR_ADAPTER_API_ALLOW_PATHS", "1")
    target = tmp_path / "admin.txt"
    target.write_text("admin path input")
    seen = []

    class Result:
        def model_dump(self):
            return {"ok": True}

    def fake_convert(*args, **kwargs):
        seen.append(kwargs["tenant"])
        return Result()

    monkeypatch.setattr("sr_adapter.api.convert", fake_convert)
    with TestClient(create_app()) as client:
        response = client.post("/convert-path", headers=_tenant_headers("admin-key", "beta"), json={
            "path": str(target), "llm_ok": False,
        })
    assert response.status_code == 200
    assert seen == ["beta"]


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_scoped_jobs_do_not_leak_other_tenants_errors(monkeypatch, tmp_path, status):
    from sr_adapter.jobs import JobRecord, SQLiteJobStore

    _configure_tenant_keys(monkeypatch)
    path = tmp_path / "errors.db"
    store = SQLiteJobStore(path)
    store.create(JobRecord(
        id="beta-job", kind="convert", status=status, request={"tenant": "beta"},
        error="private tenant input in provider error",
    ))
    store.close()
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_BACKEND", "sqlite")
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_DB_PATH", str(path))
    with TestClient(create_app()) as client:
        for suffix in ("", "?include_result=true", "/result"):
            denied = client.get(f"/jobs/beta-job{suffix}", headers=_tenant_headers())
            assert denied.status_code == 404
            assert "private" not in denied.text
        own = client.get("/jobs/beta-job/result", headers=_tenant_headers("beta-key", "beta"))
        assert own.status_code == 409
        assert f"Job {status}:" in own.json()["detail"]


@pytest.mark.parametrize("scoped,expected", [(False, "beta"), (True, "alpha")])
def test_recipe_tenant_precedence_is_preserved_for_legacy_and_scoped_explicit_override(monkeypatch, scoped, expected):
    from sr_adapter.recipe import RecipeConfig, load_recipe

    _configure_tenant_keys(monkeypatch)
    original = load_recipe("call_log")
    recipe = RecipeConfig(
        name=original.name, patterns=original.patterns, fallback=original.fallback,
        llm={**original.llm, "tenant": "beta"},
    )

    def configured_recipe(name):
        return recipe if name == "call_log" else load_recipe(name)

    monkeypatch.setattr("sr_adapter.recipe.load_recipe", configured_recipe)
    monkeypatch.setattr("sr_adapter.pipeline.load_recipe", configured_recipe)
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", configured_recipe)
    seen = []

    class Manager:
        tenant_manager = type("Tenants", (), {"get_default_tenant": lambda self: "default"})()

        def get_driver(self, tenant, llm_config):
            seen.append(tenant)
            raise RuntimeError("Stop after observing tenant selection; no provider call")

    monkeypatch.setattr("sr_adapter.delegate._driver_manager", Manager())
    headers = _tenant_headers("alpha-key", "alpha") if scoped else _tenant_headers("admin-key", None)
    with TestClient(create_app()) as client:
        for route in ("/convert", "/jobs/convert"):
            response = client.post(route + "?recipe=call_log", headers=headers, files={
                "file": ("sample.txt", b"Hello world\n", "text/plain"),
            })
            assert response.status_code == 200
            if route.startswith("/jobs"):
                assert response.json()["request"]["tenant"] == expected
    assert seen == [expected, expected]


def test_sqlite_startup_recovery_failure_is_visible_and_closes_store(monkeypatch, tmp_path):
    import asyncio
    from sr_adapter.jobs import JobManager

    monkeypatch.setenv("SR_ADAPTER_API_JOBS_BACKEND", "sqlite")
    monkeypatch.setenv("SR_ADAPTER_API_JOBS_DB_PATH", str(tmp_path / "recovery.db"))
    closed = []
    original_shutdown = JobManager.shutdown

    def failed_recovery(*args, **kwargs):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        raise OSError("private backend connection details")

    def shutdown(manager, *args, **kwargs):
        closed.append(True)
        return original_shutdown(manager, *args, **kwargs)

    monkeypatch.setattr(JobManager, "reset_incomplete", failed_recovery)
    monkeypatch.setattr(JobManager, "shutdown", shutdown)
    with pytest.raises(RuntimeError, match="Job recovery failed; server startup aborted") as error:
        with TestClient(create_app()):
            raise AssertionError("Application must not serve after failed recovery")
    assert "private" not in str(error.value)
    assert closed == [True]
