from types import SimpleNamespace

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
    exporter = TelemetryExporter(settings=settings, runtime=runtime)

    metrics = exporter.render_prometheus()

    assert 'sr_adapter_kernel_calls_total{kernel="text",service="test"}' in metrics
    assert 'sr_adapter_kernel_duration_ms_total{kernel="layout",service="test"}' in metrics


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
    exporter = TelemetryExporter(settings=settings, runtime=runtime)
    exporter.push_sentry()

    assert "init" in captured
    assert "event" in captured
    assert captured["event"]["extra"]["telemetry"]["text"]["enabled"] is True
