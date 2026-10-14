"""Offline tests of the opt-in live runner's checks and spending boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("smoke_llm", Path(__file__).parents[1] / "scripts" / "smoke_llm.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def response(reason="stop"):
    return {"model": "synthetic-model", "choices": [{"index": 0, "message": {"content": json.dumps(smoke.EXPECTED)}, "finish_reason": reason}]}


@pytest.mark.parametrize("reason,status", [("stop", "passed"), ("length", "failed"), (None, "failed")])
def test_matching_json_still_requires_completion(reason, status):
    receipt = smoke.check_result("openai", response(reason))
    assert receipt["content_matches"]
    assert receipt["status"] == status


def test_gemini_stream_ignores_thoughts_and_retains_trailing_usage():
    events = [
        {"modelVersion": "gemini-test", "candidates": [{"index": 0, "content": {"parts": [{"thought": True, "text": "private"}, {"text": '{"status":"ok",'}]}}]},
        {"candidates": [{"index": 0, "content": {"parts": [{"text": '"value":42}'}]}, "finishReason": "STOP"}]},
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18}},
    ]
    receipt = smoke.check_stream("gemini", events)
    assert receipt["status"] == "passed"
    assert receipt["response_model"] == "gemini-test"
    assert receipt["usage"]["total_tokens"] == 18
    assert receipt["events"] == 3


@pytest.mark.parametrize("reason,status", [("stop", "passed"), ("length", "failed"), (None, "failed")])
def test_openai_stream_checks_terminal_reason_and_separate_terminal_chunk(reason, status):
    events = [
        {"choices": [{"index": 0, "delta": {"content": json.dumps(smoke.EXPECTED)}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
        {"choices": [], "usage": {"total_tokens": 12}},
    ]
    receipt = smoke.check_stream("openai", events)
    assert receipt["status"] == status
    assert receipt["usage"]["total_tokens"] == 12


def test_multiple_candidates_are_not_concatenated_into_a_passing_response():
    receipt = smoke.check_stream("openai", [{"choices": [
        {"index": 0, "delta": {"content": '{"status":"ok",'}, "finish_reason": "stop"},
        {"index": 1, "delta": {"content": '"value":42}'}, "finish_reason": "stop"},
    ]}])
    assert receipt["status"] == "failed"
    assert not receipt["content_matches"]


def test_refusal_in_earlier_chunk_is_not_erased_by_terminal_chunk():
    events = [
        {"choices": [{"index": 0, "delta": {"refusal": "declined", "content": json.dumps(smoke.EXPECTED)}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    assert smoke.check_stream("openai", events)["completion_status"] == "blocked"


def test_no_more_modes_after_first_failure_and_all_planned_modes_are_recorded(monkeypatch):
    calls = []

    class BrokenDriver:
        def __init__(self, provider, config):
            assert config["max_retries"] == 0

        def generate(self, *args, **kwargs):
            calls.append(1)
            raise ValueError("secret must never enter receipts")

    monkeypatch.setenv("TEST_SMOKE_KEY", "synthetic-test-key")
    monkeypatch.setitem(smoke.PROVIDERS, "openai", ("TEST_SMOKE_KEY", BrokenDriver))
    receipts = smoke.run_provider("openai", "synthetic")
    assert len(calls) == 1
    assert [r["status"] for r in receipts] == ["failed", "skipped", "skipped", "skipped"]
    assert "secret must never" not in json.dumps(receipts)


def test_missing_credentials_keep_the_planned_denominator(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_SMOKE_KEY", raising=False)
    monkeypatch.setitem(smoke.PROVIDERS, "gemini", ("TEST_MISSING_SMOKE_KEY", object))
    receipts = smoke.run_provider("gemini", "synthetic")
    assert len(receipts) == 4
    assert all(r["status"] == "missing_credential" and not r["attempted"] for r in receipts)


def test_output_collision_stops_before_any_api_call(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(smoke, "run_provider", lambda *args: pytest.fail("unexpected API call"))
    with pytest.raises(SystemExit) as error:
        smoke.main(["--allow-live-api", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "preserve"


def test_pending_receipt_exists_before_first_call_and_survives_interrupt(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"

    class InterruptedDriver:
        def __init__(self, *args):
            pass

        def generate(self, *args, **kwargs):
            saved = json.loads(output.read_text())
            assert saved["max_requests"] == 4
            assert [r["status"] for r in saved["results"]] == ["running", "pending", "pending", "pending"]
            raise KeyboardInterrupt()

    monkeypatch.setenv("TEST_SMOKE_KEY", "synthetic-test-key")
    monkeypatch.setitem(smoke.PROVIDERS, "openai", ("TEST_SMOKE_KEY", InterruptedDriver))
    assert smoke.main(["--allow-live-api", "--provider", "openai", "--output", str(output)]) == 130
    saved = json.loads(output.read_text())
    assert saved["results"][0]["status"] == "interrupted"
    assert saved["results"][1]["status"] == "pending"
    assert not list(tmp_path.glob("*.tmp"))
