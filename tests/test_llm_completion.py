"""Completed HTTP requests must not masquerade as completed text generation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sr_adapter import cli, delegate, pipeline
from sr_adapter.escalation import SelectionResult
from sr_adapter.profiles import LLMPolicy, ProcessingProfile
from sr_adapter.recipe import RecipeConfig
from sr_adapter.schema import Block, Span


@pytest.fixture
def install_response(monkeypatch):
    def install(raw):
        calls = []

        def generate(prompt, **kwargs):
            calls.append(prompt)
            return raw

        driver = SimpleNamespace(name="dummy", generate=generate)
        manager = SimpleNamespace(
            tenant_manager=SimpleNamespace(get_default_tenant=lambda: "default"),
            get_driver=lambda *args: driver,
        )
        recipe = RecipeConfig(name="test", patterns=[], llm={"enable": True})
        monkeypatch.setattr(cli, "DriverManager", lambda: manager)
        monkeypatch.setattr(cli, "load_recipe", lambda _: recipe)
        monkeypatch.setattr(delegate, "_driver_manager", manager)
        monkeypatch.setattr(delegate, "load_recipe", lambda _: recipe)
        return calls
    return install


def response(reason="length"):
    return {"choices": [{"message": {"content": "partial answer"}, "finish_reason": reason}]}


@pytest.mark.parametrize("raw,status", [
    (response(), "truncated"), (response(None), "unknown"),
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked"),
    ({"choices": []}, "empty"),
])
def test_escalation_retains_incomplete_evidence_and_original_source(install_response, raw, status):
    calls = install_response(raw)
    original = Block(text="source", confidence=0.1, spans=[Span(start=0, end=6)], source="synthetic.txt")
    selected = SelectionResult(indices=[0], candidates=[], threshold=1.0, limit=1)
    output = delegate.escalate_low_conf([original], "test", selection=selected)[0]
    assert len(calls) == 1
    assert output.text == original.text
    assert output.spans == original.spans
    assert output.prov == original.prov
    assert output.confidence == original.confidence
    assert "llm_escalations" not in original.attrs
    assert output.attrs["llm_escalations"][0]["status"] == status
    assert output.attrs["llm_escalations"][0]["raw"] == raw


@pytest.mark.parametrize("reason,status,code", [("stop", "complete", 0), ("length", "truncated", 10), (None, "unknown", 10)])
def test_cli_run_exit_status_tracks_text_completion(install_response, capsys, reason, status, code):
    install_response(response(reason))
    assert cli.main(["llm", "run", "--prompt", "synthetic"]) == code
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == status


@pytest.mark.parametrize("skip_errors,expected_calls", [(False, 1), (True, 2)])
def test_replay_retains_incomplete_results_without_counting_them_as_success(
    install_response, tmp_path, capsys, skip_errors, expected_calls,
):
    calls = install_response(response())
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"prompt":"one"}\n{"prompt":"two"}\n', encoding="utf-8")
    output = tmp_path / "results.jsonl"
    output.write_text("previous result\n", encoding="utf-8")
    args = ["llm", "replay", "--input", str(dataset), "--output", str(output), "--limit", "1"]
    if skip_errors:
        args.append("--skip-errors")
    assert cli.main(args) == 10
    assert len(calls) == expected_calls
    if skip_errors:
        rows = [json.loads(row) for row in output.read_text().splitlines()]
        assert len(rows) == expected_calls
        assert all(row["response"]["status"] == "truncated" for row in rows)
    else:
        assert output.read_text() == "previous result\n"
        assert not list(tmp_path.glob("*.tmp"))
    assert "0 prompt(s) processed" in capsys.readouterr().err


@pytest.mark.parametrize("reason,complete", [("stop", 1), ("length", 0), (None, 0)])
def test_pipeline_reports_completion_separately_from_responses(
    install_response, monkeypatch, tmp_path, reason, complete,
):
    install_response(response(reason))
    selected = SelectionResult(indices=[0], candidates=[], threshold=1.0, limit=1)
    monkeypatch.setattr(pipeline, "get_escalation_policy", lambda: SimpleNamespace(evaluate=lambda *a, **k: selected))
    source = tmp_path / "input.txt"
    source.write_text("synthetic source", encoding="utf-8")
    profile = ProcessingProfile(name="test", llm_policy=LLMPolicy(enabled=True), warm_runtime=False)
    document = pipeline.convert(source, "default", profile=profile)
    assert document.meta.llm_escalations == 1
    assert document.meta.llm_completed == complete
    assert document.meta.llm_incomplete == 1 - complete
    assert bool(document.warnings) is (not complete)


def test_old_escalation_is_not_counted_again(monkeypatch, tmp_path):
    selected = SelectionResult(indices=[0], candidates=[], threshold=1.0, limit=1)
    monkeypatch.setattr(pipeline, "get_escalation_policy", lambda: SimpleNamespace(evaluate=lambda *a, **k: selected))
    monkeypatch.setattr(pipeline, "_parse", lambda *a, **k: [Block(text="source", attrs={"llm_escalations": [{"status": "complete"}]})])
    monkeypatch.setattr(pipeline, "escalate_low_conf", lambda blocks, *a, **k: list(blocks))
    source = tmp_path / "input.txt"
    source.write_text("source", encoding="utf-8")
    document = pipeline.convert(source, "default")
    assert document.meta.llm_escalations == document.meta.llm_completed == 0
