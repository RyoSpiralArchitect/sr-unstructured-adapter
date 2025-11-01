"""Central configuration loader with YAML + environment support."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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
from pydantic import BaseModel, Field, field_validator


def _default_cache_dir() -> Path:
    base = Path(os.getenv("SR_ADAPTER_CACHE_DIR", "")).expanduser()
    if base and base.name:
        return base
    return Path.home() / ".cache" / "sr_adapter"


class TelemetrySettings(BaseModel):
    """Telemetry configuration covering Sentry + Prometheus exports."""

    sentry_dsn: Optional[str] = None
    environment: str = "development"
    release: Optional[str] = None
    enable_prometheus: bool = False
    labels: Dict[str, str] = Field(default_factory=dict)

    @field_validator("environment")
    @classmethod
    def _strip_environment(cls, value: str) -> str:  # noqa: D401
        return value.strip() or "development"


class DriverSettings(BaseModel):
    """Global defaults applied to driver configurations."""

    default_timeout: float = 30.0
    user_agent: Optional[str] = None
    max_retries: int = 1
    retry_backoff_base: float = 0.5
    retry_backoff_max: float = 8.0
    retry_jitter: float = 0.5
    circuit_breaker_failures: int = 3
    circuit_breaker_recovery: float = 30.0
    circuit_breaker_window: float = 10.0

    @field_validator("default_timeout")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:  # noqa: D401
        if value <= 0:
            raise ValueError("default_timeout must be > 0")
        return float(value)

    @field_validator("max_retries")
    @classmethod
    def _non_negative_retry(cls, value: int) -> int:  # noqa: D401
        if value < 0:
            raise ValueError("max_retries must be >= 0")
        return int(value)

    @field_validator("retry_backoff_base", "retry_backoff_max", "retry_jitter")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:  # noqa: D401
        if value < 0:
            raise ValueError("Retry backoff settings must be >= 0")
        return float(value)

    @field_validator("circuit_breaker_failures")
    @classmethod
    def _positive_failures(cls, value: int) -> int:  # noqa: D401
        if value <= 0:
            raise ValueError("circuit_breaker_failures must be > 0")
        return int(value)

    @field_validator("circuit_breaker_recovery", "circuit_breaker_window")
    @classmethod
    def _positive_window(cls, value: float) -> float:  # noqa: D401
        if value <= 0:
            raise ValueError("Circuit breaker timing must be > 0")
        return float(value)


class DistributedSettings(BaseModel):
    """Configuration for distributed/concurrent execution backends."""

    default_backend: str = "auto"
    max_workers: Optional[int] = None
    dask_scheduler: Optional[str] = None
    ray_address: Optional[str] = None

    @field_validator("default_backend")
    @classmethod
    def _normalize_backend(cls, value: str) -> str:  # noqa: D401
        return value.strip().lower() or "auto"

class EscalationSettings(BaseModel):
    """Controls for the escalation meta-model and logging pipeline."""

    logging_enabled: bool = True
    log_path: Optional[str] = None
    model_path: Optional[str] = None
    model_format: str = "json"
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    default_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "bias": 2.0,
            "confidence": -4.0,
            "layout_confidence": -1.5,
            "text_length": 0.0008,
            "has_spans": 0.25,
        }
    )
    feature_version: str = "v1"

    @field_validator("log_path", "model_path", mode="before")
    @classmethod
    def _expand_path(cls, value: Optional[str]) -> Optional[str]:  # noqa: D401
        if not value:
            return None
        return str(Path(value).expanduser())

    @property
    def resolved_log_path(self) -> Path:
        if self.log_path:
            return Path(self.log_path)
        return _default_cache_dir() / "escalation" / "events.jsonl"

    @property
    def resolved_model_path(self) -> Optional[Path]:
        if not self.model_path:
            return None
        path = Path(self.model_path)
        return path if path.exists() else None


class AutoProfileSettings(BaseModel):
    """Adaptive profile controller parameters."""

    enabled: bool = True
    epsilon: float = Field(default=0.15, ge=0.0, le=1.0)
    state_path: Optional[str] = None
    candidate_profiles: Tuple[str, ...] = ("balanced", "realtime", "archival")
    large_document_bytes: int = Field(default=5_000_000, ge=0)
    tight_deadline_ms: int = Field(default=1200, ge=0)
    high_kernel_latency_ms: float = Field(default=180.0, ge=0.0)
    max_llm_failure_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    latency_target_ms: float = Field(default=3600.0, ge=1.0)
    min_reward: float = -1.0
    max_reward: float = 1.5

    @field_validator("state_path", mode="before")
    @classmethod
    def _expand_state_path(cls, value: Optional[str]) -> Optional[str]:  # noqa: D401
        if not value:
            return None
        return str(Path(value).expanduser())

    @field_validator("candidate_profiles", mode="before")
    @classmethod
    def _normalise_candidates(
        cls, value: Optional[Sequence[str]] | str
    ) -> Tuple[str, ...]:  # noqa: D401
        if value is None:
            return ("balanced", "realtime", "archival")
        if isinstance(value, str):
            return (value.strip() or "balanced",)
        return tuple(str(item).strip() for item in value if str(item).strip())

    @property
    def resolved_state_path(self) -> Path:
        if self.state_path:
            return Path(self.state_path)
        return _default_cache_dir() / "profiles" / "bandit_state.json"


class AdapterSettings(BaseModel):
    """Composite settings object loaded from YAML + environment variables."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    drivers: DriverSettings = Field(default_factory=DriverSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    escalation: EscalationSettings = Field(default_factory=EscalationSettings)
    profile_automation: AutoProfileSettings = Field(default_factory=AutoProfileSettings)


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

