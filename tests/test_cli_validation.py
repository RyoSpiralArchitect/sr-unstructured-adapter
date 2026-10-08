from __future__ import annotations

import json

import pytest

from sr_adapter.cli import _resolve_metadata, _resolve_prompt, main
from sr_adapter.recipe import RecipeConfig


@pytest.mark.parametrize("stream", [False, True])
def test_convert_rejects_output_alias_of_input(tmp_path, stream):
    source = tmp_path / "source.txt"
    source.write_text("original text")
    output = tmp_path / "output.txt"
    output.hardlink_to(source)
    args = ["convert", str(source), "--out", str(output), "--no-llm"]
    if stream:
        args.append("--stream")
    assert main(args) == 2
    assert source.read_text() == "original text"


@pytest.mark.parametrize("flag", ["--max-blocks", "--concurrency"])
def test_convert_rejects_negative_limits(flag):
    with pytest.raises(SystemExit) as exc:
        main(["convert", "input.txt", "--out", "output.jsonl", flag, "-1"])
    assert exc.value.code == 2


def test_explicit_empty_prompt_does_not_read_stdin(monkeypatch):
    class NoRead:
        def read(self):
            raise AssertionError("Explicit empty prompt must not read stdin")
    monkeypatch.setattr("sys.stdin", NoRead())
    with pytest.raises(ValueError, match="empty"):
        _resolve_prompt("", None)


def test_unreadable_prompt_file_has_friendly_error(tmp_path):
    with pytest.raises(ValueError, match="Failed to read prompt file"):
        _resolve_prompt(None, tmp_path / "missing.txt")


@pytest.mark.parametrize("payload", ["[]", '"scalar"', "123", "true"])
def test_cli_metadata_must_be_object(payload, tmp_path):
    with pytest.raises(ValueError, match="JSON object"):
        _resolve_metadata(payload, None)
    file = tmp_path / "metadata.json"
    file.write_text(payload)
    with pytest.raises(ValueError, match="JSON object"):
        _resolve_metadata(None, file)


def test_replay_rejects_output_overwriting_dataset(monkeypatch, tmp_path):
    class TenantManager:
        def get_default_tenant(self):
            return "default"

    class Manager:
        tenant_manager = TenantManager()
        def get_driver(self, *args, **kwargs):
            pytest.fail("Driver must not be acquired for a destructive replay")

    monkeypatch.setattr("sr_adapter.cli.DriverManager", Manager)
    monkeypatch.setattr("sr_adapter.cli.load_recipe", lambda _: RecipeConfig(name="demo", patterns=[], llm={"enable": True}))
    dataset = tmp_path / "dataset.jsonl"
    contents = json.dumps({"prompt": "original prompt"})
    dataset.write_text(contents)
    assert main(["llm", "replay", "--input", str(dataset), "--output", str(dataset)]) == 2
    assert dataset.read_text() == contents


def test_replay_rejects_invalid_metadata_before_driver_call(monkeypatch, tmp_path):
    class TenantManager:
        def get_default_tenant(self):
            return "default"

    class Driver:
        name = "fake"
        def generate(self, *args, **kwargs):
            pytest.fail("Invalid metadata must not reach a live driver")

    class Manager:
        tenant_manager = TenantManager()
        def get_driver(self, *args, **kwargs):
            return Driver()

    monkeypatch.setattr("sr_adapter.cli.DriverManager", Manager)
    monkeypatch.setattr("sr_adapter.cli.load_recipe", lambda _: RecipeConfig(name="demo", patterns=[], llm={"enable": True}))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"prompt": "hello", "metadata": []}))
    assert main(["llm", "replay", "--input", str(dataset)]) == 8


def test_invalid_prometheus_label_returns_cli_failure(monkeypatch, capsys):
    from sr_adapter.settings import TelemetrySettings
    from sr_adapter.telemetry import TelemetryExporter

    monkeypatch.setenv("SR_ADAPTER_DISABLE_NATIVE_RUNTIME", "1")
    exporter = TelemetryExporter(settings=TelemetrySettings(enable_prometheus=True))
    monkeypatch.setattr("sr_adapter.cli.TelemetryExporter", lambda: exporter)
    assert main(["kernels", "export", "--format", "prometheus", "--label", "bad-key=x"]) == 12
    assert "Invalid Prometheus label name" in capsys.readouterr().err


def _stub_malformed_llm(monkeypatch):
    class TenantManager:
        def get_default_tenant(self):
            return "default"

    class Driver:
        name = "fake"
        def generate(self, prompt, **kwargs):
            if prompt == "bad":
                return {"choices": ["malformed"]}
            return {"choices": [{"message": {"content": "valid response"}}]}

    class Manager:
        tenant_manager = TenantManager()
        def get_driver(self, *args, **kwargs):
            return Driver()

    monkeypatch.setattr("sr_adapter.cli.DriverManager", Manager)
    monkeypatch.setattr("sr_adapter.cli.load_recipe", lambda _: RecipeConfig(name="demo", patterns=[], llm={"enable": True}))


def test_llm_run_malformed_response_returns_cli_failure(monkeypatch, capsys):
    _stub_malformed_llm(monkeypatch)
    assert main(["llm", "run", "--prompt", "bad"]) == 10
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "LLM choices must contain objects" in captured.err


def test_replay_malformed_response_preserves_existing_artifact(monkeypatch, tmp_path):
    _stub_malformed_llm(monkeypatch)
    dataset = tmp_path / "input.jsonl"
    contents = '\n'.join(json.dumps({"prompt": value}) for value in ("good", "bad"))
    dataset.write_text(contents)
    output = tmp_path / "output.jsonl"
    output.write_text("previous result\n")
    assert main(["llm", "replay", "--input", str(dataset), "--output", str(output)]) == 10
    assert output.read_text() == "previous result\n"
    assert dataset.read_text() == contents
    assert not list(tmp_path.glob(".output.jsonl.*.tmp"))


def test_replay_skip_errors_continues_after_malformed_response(monkeypatch, tmp_path):
    _stub_malformed_llm(monkeypatch)
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('\n'.join(json.dumps({"prompt": value}) for value in ("bad", "good")))
    output = tmp_path / "output.jsonl"
    assert main(["llm", "replay", "--input", str(dataset), "--output", str(output), "--skip-errors"]) == 10
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["record"]["prompt"] == "good"
    assert records[0]["response"]["choices"][0]["text"] == "valid response"
    assert not list(tmp_path.glob(".output.jsonl.*.tmp"))
