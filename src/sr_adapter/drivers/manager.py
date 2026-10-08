"""Driver and tenant management utilities."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from .anthropic_driver import AnthropicDriver  # noqa: F401  # ensure registration side effects
from .azure_driver import AzureDriver  # noqa: F401  # ensure registration side effects
from .base import (
    DriverError,
    LLMDriver,
    available_drivers,
    create_registered_driver,
)
from .docker_driver import DockerDriver  # noqa: F401  # ensure registration side effects
from .openai_driver import OpenAIDriver  # noqa: F401  # ensure registration side effects
from .vllm_driver import VLLMDriver  # noqa: F401  # ensure registration side effects
from ..settings import AdapterSettings, get_settings


@dataclass
class TenantConfig:
    """Resolved configuration for a tenant."""

    name: str
    driver: str
    settings: Dict[str, Any]


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def expand(match: re.Match[str]) -> str:
            name = match.group("braced") or match.group("plain")
            current = os.environ.get(name)
            separator = match.group("separator")
            if separator is not None and (current is None or (separator == ":-" and not current)):
                return match.group("default") or ""
            if current is None:
                raise DriverError(f"Required environment variable '{name}' is not set")
            return current

        return re.sub(
            r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?:(?P<separator>:-|-|:)(?P<default>[^}]*))?\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)",
            expand,
            value,
        )
    if isinstance(value, dict):
        return {key: _resolve_env(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


class TenantManager:
    """Loads tenant definitions from ``configs/tenants``."""

    def __init__(self, base_path: Path | None = None):
        if base_path is None:
            configured = os.getenv("SR_ADAPTER_TENANT_DIR")
            if configured:
                base_path = Path(configured).expanduser()
            else:
                candidates = (
                    Path.cwd() / "configs" / "tenants",
                    Path(__file__).resolve().parents[3] / "configs" / "tenants",
                    Path(__file__).resolve().parents[1] / "configs" / "tenants",
                )
                base_path = next((path for path in candidates if path.is_dir()), candidates[-1])
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
        if not isinstance(tenant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tenant):
            raise DriverError("Tenant name must be a simple configuration name")
        if tenant in self._cache:
            return deepcopy(self._cache[tenant])
        for suffix in (".yaml", ".yml"):
            candidate = self.base_path / f"{tenant}{suffix}"
            if not candidate.resolve().is_relative_to(self.base_path.resolve()):
                raise DriverError("Tenant configuration must remain inside the tenant directory")
            if candidate.exists():
                try:
                    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                except (OSError, yaml.YAMLError) as exc:
                    raise DriverError(f"Could not load configuration for tenant '{tenant}'") from exc
                if not isinstance(data, Mapping):
                    raise DriverError(f"Tenant '{tenant}' configuration must be a mapping")
                driver = data.get("driver")
                if not driver:
                    raise DriverError(f"Tenant '{tenant}' is missing a driver name")
                settings = _resolve_env(data.get("settings", {}))
                if not isinstance(settings, Mapping):
                    raise DriverError(f"Tenant '{tenant}' settings must be a mapping")
                config = TenantConfig(name=tenant, driver=str(driver), settings=dict(settings))
                self._cache[tenant] = config
                return deepcopy(config)
        raise DriverError(f"Tenant '{tenant}' not found under {self.base_path}")


class DriverManager:
    """Instantiate drivers for tenants on demand."""

    def __init__(
        self,
        tenant_manager: Optional[TenantManager] = None,
        *,
        settings: Optional[AdapterSettings] = None,
    ):
        self.tenant_manager = tenant_manager or TenantManager()
        self.settings = settings or get_settings()
        self._driver_cache: MutableMapping[str, LLMDriver] = {}
        self._cache_lock = Lock()

    def get_driver(self, tenant: str, llm_config: Mapping[str, Any]) -> LLMDriver:
        tenant_config = self.tenant_manager.get(tenant)
        driver_name = str(llm_config.get("driver") or tenant_config.driver).lower()
        settings: Dict[str, Any] = dict(tenant_config.settings)
        recipe_settings = llm_config.get("settings")
        if isinstance(recipe_settings, Mapping):
            settings.update(_resolve_env(dict(recipe_settings)))  # recipe level overrides
        driver_defaults = self.settings.drivers
        settings.setdefault("timeout", driver_defaults.default_timeout)
        if driver_defaults.user_agent and "user_agent" not in settings:
            settings["user_agent"] = driver_defaults.user_agent
        settings.setdefault("max_retries", driver_defaults.max_retries)
        settings.setdefault("retry_backoff_base", driver_defaults.retry_backoff_base)
        settings.setdefault("retry_backoff_max", driver_defaults.retry_backoff_max)
        settings.setdefault("retry_jitter", driver_defaults.retry_jitter)
        settings.setdefault("circuit_breaker_failures", driver_defaults.circuit_breaker_failures)
        settings.setdefault("circuit_breaker_recovery", driver_defaults.circuit_breaker_recovery)
        settings.setdefault("circuit_breaker_window", driver_defaults.circuit_breaker_window)
        cache_key = f"{tenant}:{self._cache_key(driver_name, settings)}"
        with self._cache_lock:
            if cache_key not in self._driver_cache:
                self._driver_cache[cache_key] = create_registered_driver(driver_name, settings)
            return self._driver_cache[cache_key]

    @staticmethod
    def _cache_key(driver_name: str, settings: Mapping[str, Any]) -> str:
        serialized = json.dumps({"driver": driver_name, "settings": settings}, sort_keys=True, default=str)
        return sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def registered_driver_names() -> tuple[str, ...]:
        """Expose registered drivers for CLI/introspection helpers."""

        return available_drivers()
