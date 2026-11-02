"""The paid probe admits one attempt and cites only inspected pinned references."""
from copy import deepcopy
import importlib
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_cross_format_ontology")


def test_transport_keeps_fixed_model_and_single_submission(probe, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: calls.append(request) or httpx.Response(200))
    transport = probe.OnceTransport(tmp_path / "submitted")
    bad = httpx.Request("POST", "https://fixture.invalid", json={"model": "other"})
    with pytest.raises(ValueError, match="frozen"):
        transport.handle_request(bad)
    request = httpx.Request("POST", "https://fixture.invalid", json={"model": probe.MODEL})
    assert transport.handle_request(request).status_code == 200
    with pytest.raises(FileExistsError):
        transport.handle_request(request)
    assert len(calls) == 1


def test_prepared_or_paid_request_is_never_overwritten(probe, tmp_path):
    path = tmp_path / "reuse-request.json"
    probe.freeze_request(path, b'{"model":"gpt-6-luna"}')
    probe.freeze_request(path, b'{"model":"gpt-6-luna"}')
    with pytest.raises(ValueError, match="keep the original"):
        probe.freeze_request(path, b'{"model":"changed"}')
    assert path.read_bytes() == b'{"model":"gpt-6-luna"}'


@pytest.mark.parametrize("change", ["snapshot", "document", "missing_question", "uncited"])
def test_answers_require_inspected_scope_and_complete_questions(probe, change):
    ref = {"collection": "cross-format", "name": "a.pdf", "snapshot_id": "a" * 64, "kind": "claim", "id": "c1"}
    payload = {"findings": [{"question_id": q, "answer": "fixture", "status": "supported", "citations": [deepcopy(ref)]}
                             for q in probe.QUESTIONS]}
    assert probe.validate_answers(payload, [{"reference": ref}]) == payload["findings"]
    if change == "snapshot":
        payload["findings"][0]["citations"][0]["snapshot_id"] = "b" * 64
    elif change == "document":
        payload["findings"][0]["citations"][0]["name"] = "same-label-other.pdf"
    elif change == "missing_question":
        payload["findings"].pop()
    else:
        payload["findings"][0]["citations"] = []
    with pytest.raises(ValueError):
        probe.validate_answers(payload, [{"reference": ref}])
