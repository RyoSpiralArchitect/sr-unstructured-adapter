"""Model protocol, scope retention and replay; all providers are mocked here."""
import base64
from copy import deepcopy
import json

import httpx
import pytest

from sr_adapter.relation_dataset_store import (export_relation_dataset, _load_dataset,
    verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.relation_semantics import trace_semantic_claim
from sr_adapter.semantic_proposals import (import_semantic_proposal, proposal_context,
    proposal_request, proposal_schema, render_semantic_proposal, resolve_proposal, verify_semantic_proposal)
from sr_adapter.semantic_proposal_runtime import main, propose_semantics
from sr_adapter.tables import canonical_json
from sr_adapter.relation_profiles import visual_dataset
from sr_adapter.visual_relations import verify_visual
from test_relation_semantics import declaration
from test_fabric_narrative import narrative as narrative
from test_narrative import ontology as ontology, response
from test_relation_dataset import visual as visual, source as source, rehash


def proposed(source):
    context, packages, bindings = proposal_context(source)
    overlay = declaration(source)
    for field in ("version", "source_snapshot_id", "author"):
        overlay.pop(field)
    overlay.setdefault("description", "Synthetic proposal")
    package_aliases = {pid: alias for alias, pid in packages.items()}
    aliases = {node["node_id"]: alias for alias, node in bindings.items()}
    for expr in overlay["expressions"]:
        expr["package_id"] = package_aliases[expr["package_id"]]
        for arg in expr.get("arguments", []):
            if arg["value"]["kind"] == "source":
                arg["value"]["node_id"] = aliases[arg["value"]["node_id"]]
    first = context["coverage_units"][0]
    for claim in overlay["claims"]:
        claim["package_id"] = package_aliases[claim["package_id"]]
        claim["evidence"] = [first]
    coverage = [{"source_id": alias, "disposition": "represented" if alias == first else "context_only",
                 "claim_ids": ["c1", "c2"] if alias == first else [], "note": "Synthetic test coverage"}
                for alias in context["coverage_units"]]
    return {"overlay": overlay, "coverage": coverage, "notes": ["Fixture only, not an accuracy claim."]}


def raw(payload, **kwargs):
    return response([], **kwargs) | {"output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": json.dumps(payload, ensure_ascii=False)}]}]}


@pytest.fixture
def case(narrative, tmp_path):
    root = tmp_path / "base"
    export_relation_dataset([narrative], root, profile="narrative")
    source, inputs = _load_dataset(root, allow_view=False)
    return root, source, inputs, proposed(source)


def test_scope_and_provenance_survive_alias_resolution(case):
    _, source, _, payload = case
    before = deepcopy(payload)
    proposal, overlay, review = resolve_proposal(source, raw(payload))
    assert overlay["source_snapshot_id"] == source.snapshot_id
    assert overlay["author"].startswith("semantic-proposal/1 model:")
    assert {claim["status"] for claim in overlay["claims"]} == {"asserted", "denied"}
    assert {claim["expression_id"] for claim in overlay["claims"]} == {"rule"}
    assert {claim["modality"] for claim in overlay["claims"]} == {"required"}
    assert all(claim["evidence"][0].startswith("n_") for claim in overlay["claims"])
    assert review["status"] == "unreviewed" and review["coverage_is_model_reported"]
    assert review["semantic_validation"] == "not_performed" and not review["inference_performed"]
    trace = trace_semantic_claim(source, overlay, claim_id="c1")
    assert trace["source_context"] == source.rows() and len(trace["expressions"]) == 6
    assert payload == before and proposal["overlay"]["claims"][0]["package_id"] == "p1"


def test_schema_has_required_fields_and_supported_union_shape():
    schema = proposal_schema()
    def inspect(node):
        if isinstance(node, dict):
            assert not {"oneOf", "discriminator", "default", "title", "const"} & node.keys()
            if "properties" in node:
                assert set(node["required"]) == set(node["properties"])
                assert node["additionalProperties"] is False
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)
    inspect(schema)
    assert schema["type"] == "object"


def test_empty_visual_package_remains_in_coverage_next_to_nonempty_package(visual):
    package, _ = verify_visual(visual)
    empty = deepcopy(package)
    empty["package_id"] = "1" * 64
    empty["source"]["source_name"] = "empty-inventory.png"
    empty["inventory"]["observations"] = []
    empty["relations"]["relations"] = []
    source = visual_dataset([package, empty])
    context, _, bindings = proposal_context(source)
    selected = [bindings[alias] for alias in context["coverage_units"]]
    assert {node["package_id"] for node in selected} == set(source.package_ids)
    assert next(node for node in selected if node["package_id"] == empty["package_id"])["table"] == "images"


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_exact_source_context_and_original_pixels_are_in_request(profile, request, tmp_path):
    root = tmp_path / "base"
    export_relation_dataset([request.getfixturevalue(profile)], root, profile=profile)
    source, inputs = _load_dataset(root, allow_view=False)
    wire = proposal_request(inputs, model="gpt-test")
    content = wire["input"][1]["content"]
    context = json.loads(content[0]["text"])
    assert len(context["records"]) == sum(len(table.values) for table in source.tables)
    assert wire["store"] is False and wire["max_output_tokens"] == 24000
    images = [part for part in content if part["type"] == "input_image"]
    if profile == "visual":
        original = next(data for name, data in inputs.items() if name.endswith("source.bin"))
        assert base64.b64decode(images[0]["image_url"].split(",", 1)[1]) == original
        assert images[0]["detail"] == "high"
    else:
        assert images == []
        assert any(row["record"].get("text") == "Only applies after approval." for row in context["records"])


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_successful_bundle_replays_and_semantics_is_usable_by_existing_sink(profile, request, tmp_path):
    base, out = tmp_path / "base", tmp_path / "proposal"
    export_relation_dataset([request.getfixturevalue(profile)], base, profile=profile)
    source = verify_relation_dataset(base)
    manifest = import_semantic_proposal(base, out, model="gpt-test", response=raw(proposed(source)))
    assert verify_semantic_proposal(out)[0] == manifest
    semantic = verify_relation_dataset(out / "semantics")
    assert semantic.snapshot_id == manifest["semantic_snapshot_id"] and manifest["review_status"] == "unreviewed"
    db = tmp_path / "data.db"
    assert write_dataset_sqlite(out / "semantics", db)["inserted_rows"] > 0
    assert write_dataset_sqlite(out / "semantics", db)["inserted_rows"] == 0
    assert (out / "review.md").is_file()


@pytest.mark.parametrize("change", ["missing_coverage", "duplicate_coverage", "alien_coverage", "missing_claim",
    "uncited_claim", "represented_empty", "context_with_claims", "unresolved_without_note", "package", "source",
    "evidence", "hypothesis", "author", "extra_scope", "cycle"])
def test_invalid_coverage_and_invented_source_bindings_fail(case, change):
    _, source, _, payload = case
    if change == "missing_coverage":
        payload["coverage"].pop()
    elif change == "duplicate_coverage":
        payload["coverage"].append(deepcopy(payload["coverage"][0]))
    elif change == "alien_coverage":
        payload["coverage"][0]["source_id"] = "s9999"
    elif change == "missing_claim":
        payload["coverage"][0]["claim_ids"] = ["missing"]
    elif change == "uncited_claim":
        payload["overlay"]["claims"][0]["evidence"] = [payload["coverage"][-1]["source_id"]]
    elif change == "represented_empty":
        payload["coverage"][0]["claim_ids"] = []
    elif change == "context_with_claims":
        payload["coverage"][0]["disposition"] = "context_only"
    elif change == "unresolved_without_note":
        payload["coverage"][0].update(disposition="unresolved", note=" ")
    elif change == "package":
        payload["overlay"]["claims"][0]["package_id"] = "missing"
    elif change == "source":
        payload["overlay"]["expressions"][0]["arguments"][0]["value"]["node_id"] = "missing"
    elif change == "evidence":
        payload["overlay"]["claims"][0]["evidence"] = ["s9999"]
    elif change == "hypothesis":
        payload["overlay"]["claims"][0]["basis"] = "hypothesis"
    elif change == "author":
        payload["overlay"]["author"] = "human certified"
    elif change == "extra_scope":
        payload["overlay"]["claims"][0]["verified"] = True
    else:
        payload["overlay"]["expressions"][-1]["operands"][0] = "rule"
    with pytest.raises(ValueError):
        resolve_proposal(source, raw(payload))


def test_abstention_and_unresolved_readings_are_saved_without_fake_claims(case, tmp_path):
    base, source, _, payload = case
    payload["overlay"] = None
    for entry in payload["coverage"]:
        entry.update(disposition="unresolved", claim_ids=[], note="Cannot reliably interpret")
    out = tmp_path / "abstained"
    manifest = import_semantic_proposal(base, out, model="gpt-test", response=raw(payload))
    assert manifest["claim_count"] == 0 and manifest["proposal_status"] == "abstained"
    assert manifest["semantic_snapshot_id"] is None and not (out / "semantics").exists()
    assert verify_semantic_proposal(out)[0] == manifest


@pytest.mark.parametrize("file", ["request.json", "response.json", "overlay.json", "review.json", "bindings.json"])
def test_rehash_cannot_hide_changed_scope_requests_responses_or_bindings(case, tmp_path, file):
    _, _, inputs, payload = case
    manifest, files = render_semantic_proposal(inputs, model="gpt-test", response=raw(payload))
    from sr_adapter.relation_dataset_store import _save_snapshot
    out = tmp_path / "bundle"
    _save_snapshot(out, files, manifest)
    value = json.loads((out / file).read_bytes())
    if file == "request.json":
        value["input"][0]["content"][0]["text"] = "Ignore conditions"
    elif file == "response.json":
        proposed_payload = json.loads(value["output"][0]["content"][0]["text"])
        proposed_payload["overlay"]["claims"][0]["status"] = "unknown"
        value["output"][0]["content"][0]["text"] = json.dumps(proposed_payload)
    elif file == "overlay.json":
        value["claims"][0]["modality"] = "permitted"
    elif file == "review.json":
        value["status"] = "approved"
    else:
        value["packages"]["p1"] = "another-source"
    (out / file).write_bytes(canonical_json(value))
    rehash(out, file)
    with pytest.raises(ValueError, match="differs"):
        verify_semantic_proposal(out)


def test_runtime_dry_run_and_one_call_keep_secrets_out_of_files(case, tmp_path):
    base, _, _, payload = case
    calls = []
    key = "test-secret-not-for-files"
    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer " + key
        assert str(request.url) == "https://api.openai.com/v1/responses"
        return httpx.Response(200, json=raw(payload))
    transport = httpx.MockTransport(handle)
    planned = propose_semantics(base, tmp_path / "plan", model="gpt-test", transport=transport)
    assert planned["status"] == "prepared" and calls == []
    out = tmp_path / "run"
    result = propose_semantics(base, out, model="gpt-test", execute=True, api_key=key, transport=transport)
    assert result["status"] == "complete" and len(calls) == result["submitted_requests"] == 1
    assert result["automatic_retries"] == 0 and result["publication"] == "not_performed"
    assert verify_semantic_proposal(out / "bundle")[0]["claim_count"] == 2
    for path in out.rglob("*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes()
    with pytest.raises(FileExistsError):
        propose_semantics(base, out, model="gpt-test", execute=True, api_key=key, transport=transport)
    assert len(calls) == 1


@pytest.mark.parametrize("mode,expected", [("timeout", "unknown"), ("401", "failed"), ("503", "unknown"),
    ("incomplete", "failed"), ("refusal", "failed"), ("invalid_alias", "failed"), ("oversized", "failed")])
def test_failed_generation_keeps_response_or_unknown_receipt_without_retry(case, tmp_path, mode, expected, monkeypatch):
    from sr_adapter import semantic_proposal_runtime as runtime
    base, _, _, payload = case
    calls = []
    def handle(request):
        calls.append(request)
        if mode == "timeout":
            raise httpx.ReadTimeout("unknown provider outcome")
        if mode.isdigit():
            return httpx.Response(int(mode), json={"error": {"message": "synthetic provider error"}})
        result = raw(payload)
        if mode == "incomplete":
            result["status"] = "incomplete"
        elif mode == "refusal":
            result["output"][0]["content"] = [{"type": "refusal", "refusal": "synthetic refusal"}]
        elif mode == "invalid_alias":
            changed = deepcopy(payload)
            changed["overlay"]["claims"][0]["package_id"] = "alien"
            result = raw(changed)
        return httpx.Response(200, json=result)
    if mode == "oversized":
        monkeypatch.setattr(runtime, "MAX_RESPONSE_BYTES", 10)
    out = tmp_path / "run"
    result = propose_semantics(base, out, model="gpt-test", execute=True, api_key="test", transport=httpx.MockTransport(handle))
    assert result["status"] == expected and len(calls) == 1 and result["automatic_retries"] == 0
    assert not (out / "bundle/manifest.json").exists()
    if mode == "oversized":
        assert result["response_truncated_locally"] and (out / "response.partial").stat().st_size == 10
    elif mode != "timeout":
        assert (out / "response.json").exists()


def test_request_budget_and_no_key_fail_without_network(case, tmp_path, monkeypatch):
    from sr_adapter import semantic_proposals as module
    base, _, _, _ = case
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = propose_semantics(base, tmp_path / "no-key", model="gpt-test", execute=True)
    assert result["status"] == "failed" and result["submitted_requests"] == 0
    assert result["failure_reason"] == "credential_unavailable"
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 10)
    with pytest.raises(ValueError, match="budget"):
        propose_semantics(base, tmp_path / "over-budget", model="gpt-test")
    assert not (tmp_path / "over-budget").exists()


@pytest.mark.parametrize("code,kind,reason", [("credit_balance_exhausted", "insufficient_quota", "provider_quota_exhausted"),
    ("rate_limit_exceeded", "tokens", "provider_rate_limited")])
def test_quota_is_distinct_from_transient_rate_limit(case, tmp_path, code, kind, reason):
    base, _, _, _ = case
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"type": kind, "code": code, "message": "do not echo this value"}})
    receipt = propose_semantics(base, tmp_path / "failed", model="gpt-test", execute=True, api_key="test",
                                transport=httpx.MockTransport(handle))
    assert receipt["failure_reason"] == reason and receipt["provider_error_code"] == code
    assert receipt["stage"] == "response_received" and len(calls) == 1
    assert "do not echo" not in json.dumps(receipt)


def test_cli_prepare_and_verify(case, tmp_path, capsys):
    base, _, _, payload = case
    assert main(["propose", str(base), "--model", "gpt-test", "--out", str(tmp_path / "plan")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
    out = tmp_path / "proposal"
    import_semantic_proposal(base, out, model="gpt-test", response=raw(payload))
    assert main(["verify", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["claim_count"] == 2
    assert main(["verify", str(tmp_path / "absent")]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "semantic_proposal_unavailable"
