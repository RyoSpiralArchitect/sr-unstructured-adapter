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
        response = await endpoint(Request(scope), file=upload, recipe="default", profile="balanced", llm_ok=False, max_blocks=None, tenant=None, _=None)
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
