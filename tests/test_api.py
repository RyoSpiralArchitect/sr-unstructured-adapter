# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

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
