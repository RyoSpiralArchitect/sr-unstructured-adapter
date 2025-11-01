from pathlib import Path

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

