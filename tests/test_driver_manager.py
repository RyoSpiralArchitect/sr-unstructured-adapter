from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sr_adapter.drivers import (
    DriverError,
    DriverManager,
    DockerDriver,
    LLMDriver,
    TenantManager,
    available_drivers,
    register_driver,
    unregister_driver,
)


def test_available_drivers_expose_new_providers() -> None:
    names = available_drivers()
    assert {"openai", "anthropic", "vllm"}.issubset(set(names))


def test_driver_manager_returns_docker_driver(tmp_path: Path) -> None:
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tenant_dir / "demo.yaml").write_text(
        """
        driver: docker
        settings:
          url: http://localhost:8000/v1/chat/completions
        """,
        encoding="utf-8",
    )

    manager = DriverManager(TenantManager(tenant_dir))
    driver = manager.get_driver("demo", {"enable": True})
    assert isinstance(driver, DockerDriver)


def test_driver_manager_cache_key_is_stable(tmp_path: Path) -> None:
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tenant_dir / "demo.yaml").write_text(
        """
        driver: docker
        settings:
          url: http://localhost:8000/v1/chat/completions
        """,
        encoding="utf-8",
    )

    manager = DriverManager(TenantManager(tenant_dir))
    driver_a = manager.get_driver("demo", {"enable": True})
    driver_b = manager.get_driver("demo", {"enable": True})
    assert driver_a is driver_b


class _DummyDriver(LLMDriver):
    def generate(  # type: ignore[override]
        self, prompt: str, *, metadata: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        payload = {"echo": prompt}
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload


def test_custom_driver_registration(tmp_path: Path) -> None:
    register_driver("dummy", "anthropic", factory=_DummyDriver, overwrite=True)
    try:
        tenant_dir = tmp_path / "tenants"
        tenant_dir.mkdir()
        (tenant_dir / "demo.yaml").write_text("driver: anthropic\n", encoding="utf-8")

        manager = DriverManager(TenantManager(tenant_dir))
        driver = manager.get_driver("demo", {})
        assert isinstance(driver, _DummyDriver)
        assert "anthropic" in available_drivers()
    finally:
        unregister_driver("dummy")
        unregister_driver("anthropic")


def test_unknown_driver_raises(tmp_path: Path) -> None:
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tenant_dir / "demo.yaml").write_text("driver: madeup\n", encoding="utf-8")

    manager = DriverManager(TenantManager(tenant_dir))
    try:
        manager.get_driver("demo", {})
    except DriverError as exc:
        assert "Unknown driver" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected DriverError for unknown driver")


def test_tenant_manager_lists_tenants(tmp_path: Path) -> None:
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tenant_dir / "demo.yaml").write_text("driver: docker\n", encoding="utf-8")
    (tenant_dir / "sample.yml").write_text("driver: docker\n", encoding="utf-8")

    manager = TenantManager(tenant_dir)

    assert manager.list_tenants() == ["demo", "sample"]
