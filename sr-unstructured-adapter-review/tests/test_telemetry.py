from types import SimpleNamespace

from sr_adapter.llm_metrics import LLMMetricsRegistry
from sr_adapter.runtime import RuntimeSnapshot
from sr_adapter.telemetry import TelemetryExporter
from sr_adapter.settings import TelemetrySettings


class _RuntimeStub:
    def __init__(self, snapshot: RuntimeSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> RuntimeSnapshot:
        return self._snapshot


def test_prometheus_export_includes_labels(monkeypatch) -> None:
    snapshot = RuntimeSnapshot(text_enabled=True, layout_enabled=True)
    snapshot.text_stats.record(0.01, 12)
    snapshot.layout_stats.record(0.02, 4)
    runtime = _RuntimeStub(snapshot)
    settings = TelemetrySettings(enable_prometheus=True, labels={"service": "test"})
    registry = LLMMetricsRegistry()
    registry.record_success("openai", latency_ms=5.0, request_bytes=10, response_bytes=20)
    exporter = TelemetryExporter(settings=settings, runtime=runtime, llm_registry=registry)

    metrics = exporter.render_prometheus()

    assert 'sr_adapter_kernel_calls_total{kernel="text",service="test"}' in metrics
    assert 'sr_adapter_kernel_duration_ms_total{kernel="layout",service="test"}' in metrics
    assert 'sr_adapter_llm_calls_total{driver="openai",service="test",status="success"} 1' in metrics


def test_sentry_export_pushes_event(monkeypatch) -> None:
    snapshot = RuntimeSnapshot(text_enabled=True, layout_enabled=False)
    snapshot.text_stats.record(0.05, 20)
    runtime = _RuntimeStub(snapshot)
    captured: dict[str, object] = {}

    def _init(**kwargs) -> None:
        captured["init"] = kwargs

    def _capture(event: dict) -> None:
        captured["event"] = event

    hub = SimpleNamespace(current=SimpleNamespace(client=None))
    dummy = SimpleNamespace(Hub=hub, init=_init, capture_event=_capture)
    monkeypatch.setattr("sr_adapter.telemetry.sentry_sdk", dummy, raising=False)

    settings = TelemetrySettings(
        sentry_dsn="https://example.ingest.sentry.io/1",
        labels={"service": "test"},
    )
    registry = LLMMetricsRegistry()
    exporter = TelemetryExporter(settings=settings, runtime=runtime, llm_registry=registry)
    exporter.push_sentry()

    assert "init" in captured
    assert "event" in captured
    assert captured["event"]["extra"]["telemetry"]["text"]["enabled"] is True


def test_snapshot_includes_llm_metrics() -> None:
    snapshot = RuntimeSnapshot(text_enabled=True, layout_enabled=False)
    runtime = _RuntimeStub(snapshot)
    registry = LLMMetricsRegistry()
    registry.record_success("openai", latency_ms=123.0, request_bytes=42, response_bytes=2048)
    settings = TelemetrySettings(enable_prometheus=True, labels={"service": "test"})
    exporter = TelemetryExporter(settings=settings, runtime=runtime, llm_registry=registry)

    payload = exporter.snapshot_dict()

    assert "llm" in payload
    drivers = payload["llm"]["drivers"]
    assert drivers and drivers[0]["driver"] == "openai"
