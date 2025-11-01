"""Central configuration loader with YAML + environment support."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:  # pragma: no cover - optional dependency guard
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - tests fallback when dependency missing
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        path = kwargs.get("dotenv_path")
        if path is None and args:
            path = args[0]
        if not path:
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                override = bool(kwargs.get("override", False))
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        continue
                    if override or key not in os.environ:
                        os.environ[key] = value
            return True
        except OSError:
            return False
from pydantic import BaseModel, Field, validator


class TelemetrySettings(BaseModel):
    """Telemetry configuration covering Sentry + Prometheus exports."""

    sentry_dsn: Optional[str] = None
    environment: str = "development"
    release: Optional[str] = None
    enable_prometheus: bool = False
    labels: Dict[str, str] = Field(default_factory=dict)

    @validator("environment")
    def _strip_environment(cls, value: str) -> str:  # noqa: D401
        return value.strip() or "development"


class DriverSettings(BaseModel):
    """Global defaults applied to driver configurations."""

    default_timeout: float = 30.0
    user_agent: Optional[str] = None
    max_retries: int = 1

    @validator("default_timeout")
    def _positive_timeout(cls, value: float) -> float:  # noqa: D401
        if value <= 0:
            raise ValueError("default_timeout must be > 0")
        return float(value)

    @validator("max_retries")
    def _non_negative_retry(cls, value: int) -> int:  # noqa: D401
        if value < 0:
            raise ValueError("max_retries must be >= 0")
        return int(value)


class DistributedSettings(BaseModel):
    """Configuration for distributed/concurrent execution backends."""

    default_backend: str = "auto"
    max_workers: Optional[int] = None
    dask_scheduler: Optional[str] = None
    ray_address: Optional[str] = None

    @validator("default_backend")
    def _normalize_backend(cls, value: str) -> str:  # noqa: D401
        return value.strip().lower() or "auto"


class AdapterSettings(BaseModel):
    """Composite settings object loaded from YAML + environment variables."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    drivers: DriverSettings = Field(default_factory=DriverSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)


def _default_settings_path() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "configs" / "settings.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings YAML at {path} must contain a dictionary")
    return data


def _resolve_settings_path(explicit: Optional[str | Path]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.getenv("SR_ADAPTER_SETTINGS_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return _default_settings_path()


def _resolve_env_path() -> Optional[Path]:
    candidate = os.getenv("SR_ADAPTER_DOTENV")
    if candidate:
        return Path(candidate).expanduser()
    base_dir = Path(__file__).resolve().parents[2]
    default = base_dir / ".env"
    return default if default.exists() else None


def _load_base_settings(path: Path | None = None) -> Dict[str, Any]:
    target = _resolve_settings_path(path)
    return _read_yaml(target)


def _collect_env_overrides() -> Dict[str, Any]:
    prefix = "SR_ADAPTER_"
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("__")
        cursor = overrides
        for idx, part in enumerate(parts):
            normalized = part.lower()
            if idx == len(parts) - 1:
                cursor[normalized] = value
            else:
                cursor = cursor.setdefault(normalized, {})  # type: ignore[assignment]
    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_settings(path: Path | None = None) -> AdapterSettings:
    """Load the global adapter settings, caching the resulting object."""

    env_path = _resolve_env_path()
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=False)
    data = _load_base_settings(path)
    merged = _deep_merge(data, _collect_env_overrides())
    return AdapterSettings.model_validate(merged)


def reset_settings_cache() -> None:
    """Clear the cached settings instance (useful for tests)."""

    get_settings.cache_clear()  # type: ignore[attr-defined]


__all__ = [
    "AdapterSettings",
    "DriverSettings",
    "DistributedSettings",
    "TelemetrySettings",
    "get_settings",
    "reset_settings_cache",
]

