"""Driver and tenant management utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from .azure_driver import AzureDriver
from .base import DriverError, LLMDriver
from .docker_driver import DockerDriver


@dataclass
class TenantConfig:
    """Resolved configuration for a tenant."""

    name: str
    driver: str
    settings: Dict[str, Any]


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _resolve_env(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


class TenantManager:
    """Loads tenant definitions from ``configs/tenants``."""

    def __init__(self, base_path: Path | None = None):
        if base_path is None:
            base_path = Path(__file__).resolve().parents[4] / "configs" / "tenants"
        self.base_path = Path(base_path)
        self._cache: Dict[str, TenantConfig] = {}

    def get_default_tenant(self) -> str:
        return os.getenv("SR_ADAPTER_TENANT", "default")

    def list_tenants(self) -> list[str]:
        """Return the names of configured tenants."""

        tenants: Iterable[Path]
        all_tenants: set[str] = set()
        if not self.base_path.exists():
            return []
        for pattern in ("*.yaml", "*.yml"):
            tenants = self.base_path.glob(pattern)
            all_tenants.update(path.stem for path in tenants)
        return sorted(all_tenants)

    def get(self, tenant: str) -> TenantConfig:
        if tenant in self._cache:
            return self._cache[tenant]
        for suffix in (".yaml", ".yml"):
            candidate = self.base_path / f"{tenant}{suffix}"
            if candidate.exists():
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                driver = data.get("driver")
                if not driver:
                    raise DriverError(f"Tenant '{tenant}' is missing a driver name")
                settings = _resolve_env(data.get("settings", {}))
                config = TenantConfig(name=tenant, driver=str(driver), settings=dict(settings))
                self._cache[tenant] = config
                return config
        raise DriverError(f"Tenant '{tenant}' not found under {self.base_path}")


class DriverManager:
    """Instantiate drivers for tenants on demand."""

    def __init__(self, tenant_manager: Optional[TenantManager] = None):
        self.tenant_manager = tenant_manager or TenantManager()
        self._driver_cache: MutableMapping[str, LLMDriver] = {}

    def get_driver(self, tenant: str, llm_config: Mapping[str, Any]) -> LLMDriver:
        tenant_config = self.tenant_manager.get(tenant)
        driver_name = str(llm_config.get("driver") or tenant_config.driver).lower()
        settings: Dict[str, Any] = dict(tenant_config.settings)
        recipe_settings = llm_config.get("settings")
        if isinstance(recipe_settings, Mapping):
            settings.update(recipe_settings)  # recipe level overrides
        cache_key = self._cache_key(driver_name, settings)
        if cache_key not in self._driver_cache:
            self._driver_cache[cache_key] = self._create_driver(driver_name, settings)
        return self._driver_cache[cache_key]

    @staticmethod
    def _cache_key(driver_name: str, settings: Mapping[str, Any]) -> str:
        return json.dumps({"driver": driver_name, "settings": settings}, sort_keys=True, default=str)

    @staticmethod
    def _create_driver(driver_name: str, settings: Mapping[str, Any]) -> LLMDriver:
        if driver_name in {"azure", "azure_openai"}:
            return AzureDriver(driver_name, settings)
        if driver_name in {"docker", "http"}:
            return DockerDriver(driver_name, settings)
        raise DriverError(f"Unknown driver '{driver_name}'")
