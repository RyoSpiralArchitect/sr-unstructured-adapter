"""Bounded Azure probe: frozen inputs, no gold leakage, raw failures, and replay."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.tables import canonical_json

REPO = Path(__file__).resolve().parents[1]
definition = importlib.util.spec_from_file_location("azure_relation_probe", REPO / "scripts/probe_azure_relation_context.py")
probe = importlib.util.module_from_spec(definition)
definition.loader.exec_module(probe)
ENDPOINT = "https://fixture.services.ai.azure.com/api/projects/test/openai/v1/responses"


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "frozen"
    probe.prepare(REPO / "docs/validation/foundry-context-2026-10-16/v2/frozen", root, ENDPOINT)
    return root


def credentials():
    def token(scope):
        assert scope == "https://ai.azure.com/.default"
        return SimpleNamespace(token="fixture-secret")
    return SimpleNamespace(get_token=token)


def successful_response(frozen, request):
    body = json.loads(request.content)
    assert set(body) == {"model", "reasoning", "store", "text", "max_output_tokens", "input"}
    assert body["model"] == "gpt-5.5" and body["max_output_tokens"] == 4096
    assert body["store"] is False and "temperature" not in body and "tools" not in body
    assert all(item["type"] == "message" and item["content"][0]["type"] == "input_text" for item in body["input"])
    questions = json.loads(body["input"][1]["content"][0]["text"].split("\nQuestions:\n", 1)[1])
    assert all(set(question) == {"id", "question"} for question in questions)
    case = "synthetic" if len(questions) == 8 else "roles"
    gold = json.loads((frozen / "inputs" / case / "gold.json").read_bytes())
    answers = [{key: value for key, value in question.items() if key != "question"} for question in gold]
    return {"status": "completed", "error": None, "model": "gpt-5.5-2026-04-24",
        "output": [{"type": "reasoning", "summary": []}, {"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": json.dumps({"answers": answers})}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 4}}}


def test_frozen_counterbalanced_eight_requests_and_replay(frozen, tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer fixture-secret"
        return httpx.Response(200, json=successful_response(frozen, request))

    out = tmp_path / "run"
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(handle))
    assert receipt["status"] == "complete" and len(calls) == 8
    assert [row["arm"] for row in receipt["requests"]] == [*probe.ARMS, *probe.ARMS, *probe.ARMS[::-1], *probe.ARMS[::-1]]
    replay = probe.replay(frozen, out)
    assert replay["winner"] is None and replay["comparison"] == "tie"
    assert all(arm["answer_match"] == arm["strict"] == arm["planned_answers"] == 24 for arm in replay["arms"].values())
    assert all(arm["completed_requests"] == 4 and arm["reasoning_tokens"] == 16 for arm in replay["arms"].values())
    assert "fixture-secret" not in (out / "receipt.json").read_text()
    with pytest.raises(FileExistsError):
        probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(handle))


def test_dry_run_does_not_construct_credentials_or_send(frozen, tmp_path):
    def forbidden(request):
        raise AssertionError("Network forbidden")
    receipt = probe.run(frozen, tmp_path / "dry", endpoint=ENDPOINT, transport=httpx.MockTransport(forbidden))
    assert receipt["status"] == "not_executed" and len(receipt["requests"]) == 8
    assert all(row["status"] == "not_executed" for row in receipt["requests"])


def test_http_400_is_saved_and_stops_without_retry(frozen, tmp_path):
    calls = []
    raw = b'{"error":{"code":"invalid_parameter","message":"unsupported parameter"}}'

    def handle(request):
        calls.append(request)
        return httpx.Response(400, content=raw)

    out = tmp_path / "failed"
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(handle))
    assert receipt["status"] == "incomplete" and receipt["completed_requests"] == 0 and len(calls) == 1
    first = receipt["requests"][0]
    assert first["http_status"] == 400 and first["wire_response_sha256"] == sha256(raw)
    assert (out / (first["id"] + "-response.json")).read_bytes() == raw
    assert all(row["status"] == "skipped_previous_failure" for row in receipt["requests"][1:])
    assert probe.replay(frozen, out)["comparison"] == "incomplete"


def test_incomplete_output_does_not_get_scored_or_retried(frozen, tmp_path):
    def handle(request):
        raw = successful_response(frozen, request)
        raw.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return httpx.Response(200, json=raw)
    receipt = probe.run(frozen, tmp_path / "incomplete", endpoint=ENDPOINT, execute=True,
                        credential=credentials(), transport=httpx.MockTransport(handle))
    assert receipt["status"] == "incomplete" and receipt["completed_requests"] == 0
    assert receipt["requests"][0]["completion_status"] == "incomplete"
    assert "scores" not in receipt["requests"][0]


def test_replay_rejects_rehashed_gold_leak_in_request(frozen, tmp_path):
    out = tmp_path / "run"
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(),
                        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=successful_response(frozen, request))))
    row = receipt["requests"][0]
    path = out / (row["id"] + "-request.json")
    request = json.loads(path.read_bytes())
    request["input"][1]["content"][0]["text"] += "\nGOLD: leaked answer"
    path.write_bytes(canonical_json(request))
    row["request_sha256"] = sha256(canonical_json(request))
    (out / "receipt.json").write_bytes(canonical_json(receipt))
    with pytest.raises(ValueError, match="differs from the frozen experiment"):
        probe.replay(frozen, out)


def test_changed_endpoint_and_frozen_inputs_stop_before_run(frozen, tmp_path):
    with pytest.raises(ValueError, match="Endpoint differs"):
        probe.run(frozen, tmp_path / "wrong", endpoint=ENDPOINT.replace("fixture.", "another."))
    assert not (tmp_path / "wrong").exists()
    (frozen / "inputs/system.txt").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.run(frozen, tmp_path / "changed", endpoint=ENDPOINT)
    assert not (tmp_path / "changed").exists()
