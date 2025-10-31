from __future__ import annotations

from pathlib import Path

from sr_adapter.drivers import DriverManager, TenantManager, DockerDriver


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


def test_tenant_manager_lists_tenants(tmp_path: Path) -> None:
    tenant_dir = tmp_path / "tenants"
    tenant_dir.mkdir()
    (tenant_dir / "demo.yaml").write_text("driver: docker\n", encoding="utf-8")
    (tenant_dir / "sample.yml").write_text("driver: docker\n", encoding="utf-8")

    manager = TenantManager(tenant_dir)

    assert manager.list_tenants() == ["demo", "sample"]
