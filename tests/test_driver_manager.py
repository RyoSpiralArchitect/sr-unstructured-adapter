from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sr_adapter.drivers import (
    AnthropicDriver,
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
    assert {"openai", "anthropic", "vllm", "mistral", "googleai", "xai", "bedrock"}.issubset(
        set(names)
    )


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
        register_driver("anthropic", factory=AnthropicDriver, overwrite=True)


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


def test_tenant_environment_defaults_and_required_values(tmp_path, monkeypatch):
    import pytest
    from sr_adapter.drivers.manager import _resolve_env

    monkeypatch.delenv("SR_TEST_ENDPOINT", raising=False)
    assert _resolve_env("${SR_TEST_ENDPOINT:https://example.test/v1}") == "https://example.test/v1"
    assert _resolve_env("${SR_TEST_ENDPOINT:-fallback}") == "fallback"
    monkeypatch.setenv("SR_TEST_ENDPOINT", "https://configured.test")
    assert _resolve_env("${SR_TEST_ENDPOINT:https://example.test/v1}") == "https://configured.test"
    assert _resolve_env({"nested": ["$SR_TEST_ENDPOINT"]}) == {"nested": ["https://configured.test"]}
    monkeypatch.delenv("SR_TEST_REQUIRED", raising=False)
    with pytest.raises(DriverError, match="SR_TEST_REQUIRED"):
        _resolve_env("${SR_TEST_REQUIRED}")


def test_tenant_names_cannot_traverse_directory(tmp_path):
    import pytest
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tmp_path / "outside.yaml").write_text("driver: docker\nsettings: {}", encoding="utf-8")
    manager = TenantManager(tenant_dir)
    for name in ("../outside", "/absolute", "..", "bad/name", "bad\\name"):
        with pytest.raises(DriverError, match="Tenant name"):
            manager.get(name)
    (tenant_dir / "link.yaml").symlink_to(tmp_path / "outside.yaml")
    with pytest.raises(DriverError, match="inside"):
        manager.get("link")


def test_tenant_directory_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_TENANT_DIR", str(tmp_path))
    (tmp_path / "demo.yaml").write_text("driver: docker\nsettings: {}", encoding="utf-8")
    assert TenantManager().list_tenants() == ["demo"]


def test_tenant_cache_does_not_leak_caller_mutations(tmp_path):
    (tmp_path / "demo.yaml").write_text("driver: docker\nsettings:\n  nested: [one]", encoding="utf-8")
    manager = TenantManager(tmp_path)
    manager.get("demo").settings["nested"].append("changed")
    assert manager.get("demo").settings["nested"] == ["one"]


def test_driver_cache_isolates_tenant_circuit_breakers(tmp_path):
    for tenant in ("a", "b"):
        (tmp_path / f"{tenant}.yaml").write_text("driver: docker\nsettings:\n  url: http://localhost:8000", encoding="utf-8")
    manager = DriverManager(TenantManager(tmp_path))
    assert manager.get_driver("a", {}) is not manager.get_driver("b", {})


def test_recipe_settings_expand_environment_variables(tmp_path, monkeypatch):
    (tmp_path / "demo.yaml").write_text("driver: docker\nsettings:\n  url: http://localhost:8000", encoding="utf-8")
    monkeypatch.setenv("SR_TEST_MODEL", "configured-model")
    driver = DriverManager(TenantManager(tmp_path)).get_driver("demo", {"settings": {"model": "${SR_TEST_MODEL}"}})
    assert driver.config["model"] == "configured-model"


def test_concurrent_driver_resolution_uses_one_circuit_breaker(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    created = []

    def factory(name, settings):
        time.sleep(0.005)
        instance = _DummyDriver(name, settings)
        created.append(instance)
        return instance

    register_driver("concurrent-test", factory=factory)
    try:
        (tmp_path / "demo.yaml").write_text("driver: concurrent-test\nsettings: {}", encoding="utf-8")
        manager = DriverManager(TenantManager(tmp_path))
        with ThreadPoolExecutor(max_workers=8) as executor:
            drivers = list(executor.map(lambda _: manager.get_driver("demo", {}), range(16)))
        assert len(created) == 1
        assert all(driver is drivers[0] for driver in drivers)
    finally:
        unregister_driver("concurrent-test")
