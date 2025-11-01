import json

from sr_adapter.cli import main


def test_cli_kernels_status_json(capsys) -> None:
    exit_code = main(["kernels", "status", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"text", "layout"}


def test_cli_kernels_warm_summary(capsys) -> None:
    exit_code = main(["kernels", "warm"])
    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert any(line.startswith("Text kernel:") for line in lines)


def test_cli_kernels_export_prometheus(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    class _Exporter:
        def render_prometheus(self, *, extra_labels=None):
            calls["labels"] = extra_labels
            return "metric 1"

        def snapshot_json(self):
            return "{}"

        def push_sentry(self):  # pragma: no cover - not triggered here
            calls["sentry"] = True

    monkeypatch.setattr("sr_adapter.cli.TelemetryExporter", lambda: _Exporter())
    exit_code = main(["kernels", "export", "--format", "prometheus", "--label", "env=test"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "metric 1" in captured.out
    assert calls["labels"]["env"] == "test"


def test_cli_kernels_export_sentry(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    class _Exporter:
        def render_prometheus(self, *, extra_labels=None):
            return "metric"

        def snapshot_json(self):
            return "{}"

        def push_sentry(self):
            calls["sentry"] = True

    monkeypatch.setattr("sr_adapter.cli.TelemetryExporter", lambda: _Exporter())
    exit_code = main(["kernels", "export", "--format", "json", "--sentry"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "{}" in captured.out
    assert calls["sentry"] is True
