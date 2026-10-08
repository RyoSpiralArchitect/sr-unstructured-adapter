from pathlib import Path

import pytest

from sr_adapter.settings import get_settings, reset_settings_cache


def test_settings_loads_yaml_and_env(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        """
telemetry:
  sentry_dsn: https://example.ingest.sentry.io/1
  enable_prometheus: true
  labels:
    service: adapter
drivers:
  default_timeout: 12.5
  max_retries: 3
distributed:
  default_backend: threadpool
  max_workers: 6
        """,
        encoding="utf-8",
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text("SR_ADAPTER_DRIVERS__DEFAULT_TIMEOUT=24\n", encoding="utf-8")

    monkeypatch.setenv("SR_ADAPTER_DOTENV", str(dotenv))
    reset_settings_cache()
    settings = get_settings(path=config)

    assert settings.telemetry.sentry_dsn.endswith("/1")
    assert settings.telemetry.enable_prometheus is True
    assert settings.telemetry.labels["service"] == "adapter"
    assert settings.drivers.default_timeout == 24.0
    assert settings.drivers.max_retries == 3
    assert settings.drivers.retry_backoff_base == 0.5
    assert settings.drivers.circuit_breaker_failures == 3
    assert settings.distributed.default_backend == "threadpool"
    assert settings.distributed.max_workers == 6



def test_settings_reject_nonfinite_timings():
    import pytest
    from pydantic import ValidationError
    from sr_adapter.settings import DriverSettings

    for field in ("default_timeout", "retry_backoff_base", "retry_backoff_max", "retry_jitter", "circuit_breaker_recovery", "circuit_breaker_window"):
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError):
                DriverSettings(**{field: value})


def test_settings_validate_execution_configuration():
    import pytest
    from pydantic import ValidationError
    from sr_adapter.settings import DistributedSettings, KernelAutoTuneSettings

    with pytest.raises(ValidationError):
        DistributedSettings(max_workers=-1)
    with pytest.raises(ValidationError):
        DistributedSettings(default_backend="missing")
    with pytest.raises(ValidationError):
        KernelAutoTuneSettings(layout_batch_sizes=(0, 16))


def test_settings_env_conflict_is_deterministic(monkeypatch):
    import pytest
    from sr_adapter.settings import _collect_env_overrides

    monkeypatch.setenv("SR_ADAPTER_DRIVERS", "invalid scalar")
    monkeypatch.setenv("SR_ADAPTER_DRIVERS__DEFAULT_TIMEOUT", "2")
    with pytest.raises(ValueError, match="Conflicting settings environment variable"):
        _collect_env_overrides()


def test_legacy_autotune_path_does_not_shadow_nested_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("SR_ADAPTER_KERNEL_AUTOTUNE__ENABLED", raising=False)
    config = tmp_path / "settings.yaml"
    config.write_text("{}")
    monkeypatch.setenv("SR_ADAPTER_KERNEL_AUTOTUNE", str(tmp_path / "legacy-cache.json"))
    monkeypatch.setenv("SR_ADAPTER_KERNEL_AUTOTUNE__WARMUP_TRIALS", "0")
    reset_settings_cache()
    try:
        settings = get_settings(path=config)
        assert settings.kernel_autotune.warmup_trials == 0
        assert settings.kernel_autotune.enabled is True
    finally:
        reset_settings_cache()


def test_api_key_tenants_absent_and_explicit_scopes(monkeypatch):
    from sr_adapter.settings import load_api_key_tenants

    monkeypatch.delenv("SR_ADAPTER_API_KEY_TENANTS", raising=False)
    assert load_api_key_tenants() == {}
    monkeypatch.setenv("SR_ADAPTER_API_KEY_TENANTS", '{"test-key": ["alpha", "beta-2"]}')
    assert load_api_key_tenants() == {"test-key": frozenset({"alpha", "beta-2"})}


@pytest.mark.parametrize("raw", [
    "", "{}", "null", "[]", "not-json", '{"sensitive-test-key": ["alpha"]',
    '{"sensitive-test-key": []}', '{"sensitive-test-key": "alpha"}',
    '{"sensitive-test-key": ["alpha", "alpha"]}',
    '{"sensitive-test-key": ["*"]}', '{"sensitive-test-key": ["../alpha"]}',
    '{"sensitive-test-key": [" alpha"]}', '{"sensitive-test-key": [null]}',
    '{"sensitive-test-key": [1]}', '{"": ["alpha"]}', '{" spaced ": ["alpha"]}',
    '{"sensitive-test-key": ["alpha"], "sensitive-test-key": ["beta"]}',
])
def test_api_key_tenants_fails_closed_without_echoing_credentials(monkeypatch, raw):
    from sr_adapter.settings import load_api_key_tenants

    monkeypatch.setenv("SR_ADAPTER_API_KEY_TENANTS", raw)
    with pytest.raises(ValueError, match="SR_ADAPTER_API_KEY_TENANTS") as error:
        load_api_key_tenants()
    assert "sensitive-test-key" not in str(error.value)
