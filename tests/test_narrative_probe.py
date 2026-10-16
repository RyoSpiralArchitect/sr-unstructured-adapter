"""The narrative pilot separates provenance, semantics and bounded live execution."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import httpx
import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_sources import read_source
from sr_adapter.tables import canonical_json

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data/knowledge/narrative-mock"
GOLD = json.loads((FIXTURE / "gold.json").read_bytes())
ONTOLOGY = NarrativeOntology.model_validate_json((FIXTURE / "ontology.json").read_bytes())
ENDPOINT = "https://fixture.services.ai.azure.com/api/projects/narrative/openai/v1/responses"
definition = importlib.util.spec_from_file_location("narrative_probe", REPO / "scripts/probe_narrative_relations.py")
probe = importlib.util.module_from_spec(definition)
definition.loader.exec_module(probe)


def response(payload, **extra):
    return {"status": "completed", "model": "gpt-5.5-2026-04-24", "output": [
        {"type": "reasoning", "summary": []}, {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": json.dumps(payload, ensure_ascii=False)}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 3}}, **extra}


def make_bundle(output, doc_id="memo-a", extension="md", transform=None):
    document = next(d for d in GOLD["documents"] if d["document_id"] == doc_id)
    path = FIXTURE / (f"{doc_id}.md" if extension == "md" else f"variants/{doc_id}.{extension}")
    data = path.read_bytes()
    transcription = None
    if extension == "png":
        transcription = response({"lines": [line for line in (FIXTURE / f"{doc_id}.md").read_text(encoding="utf-8").splitlines() if line.strip()]})
    source = read_source(data, path.name, transcription=probe._json_response(transcription) if transcription else None)
    candidates = []
    for assertion in document["assertions"]:
        unit = next(unit for unit in source["units"] if assertion["evidence"]["quote"] in unit["text"])
        candidates.append({**{key: assertion[key] for key in ("subject", "predicate", "object", "status")},
                           "evidence": [{"unit_id": unit["id"], "quote": unit["text"]}]})
    if transform:
        transform(candidates, source)
    manifest, files = render_narrative(data=data, source_name=path.name, ontology=ONTOLOGY, model="gpt-5.5",
        response=response({"assertions": candidates}), transcription_response=transcription)
    save_narrative(output, manifest, files)
    return output


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("narrative-bundles")
    return {name: make_bundle(root / name, doc_id, extension) for name, doc_id, extension in (
        ("memo-a", "memo-a", "md"), ("memo-b", "memo-b", "md"), ("csv", "memo-a", "csv"),
        ("pptx", "memo-a", "pptx"), ("png", "memo-a", "png"))}


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, bundles):
    root = tmp_path_factory.mktemp("frozen-parent") / "frozen"
    probe.prepare_comparison(FIXTURE / "gold.json", {key: bundles[key] for key in ("memo-a", "memo-b")}, root,
                             endpoint=ENDPOINT, png_bundle=bundles["png"])
    return root


def expected_payload(question_ids):
    assertions = {a["assertion_id"]: {"document_id": d["document_id"], **a}
                  for d in GOLD["documents"] for a in d["assertions"]}
    return {"answers": [{"question_id": q["question_id"], "answer_kind": q["answer_kind"],
        "claims": [{key: assertions[aid][key] for key in probe.CLAIM_FIELDS} for aid in q["assertion_ids"]],
        "explanation": q["expected_answer"]} for q in GOLD["questions"] if q["question_id"] in question_ids]}


def request_payload(request):
    body = json.loads(request.content)
    return json.loads(body["input"][1]["content"][0]["text"])


def success(request):
    payload = request_payload(request)
    return httpx.Response(200, json=response(expected_payload([q["question_id"] for q in payload["questions"]])))


def credentials(check=None):
    def token(scope):
        assert scope == "https://ai.azure.com/.default"
        if check:
            check()
        return SimpleNamespace(token="fixture-secret-never-persist")
    return SimpleNamespace(get_token=token)


def test_audit_five_inputs_counts_variants_without_claiming_ocr_quality(bundles, tmp_path):
    mapping = [("memo-b" if key == "memo-b" else "memo-a", value) for key, value in bundles.items()]
    root = tmp_path / "audit"
    report = probe.audit(FIXTURE / "gold.json", mapping, root)
    assert report["all_inputs_exact"] is True
    assert report["planned_inputs"] == report["source_verified_inputs"] == 5
    assert report["independent_gold_documents"] == 2
    assert report["totals_including_format_repeats"] == {
        "tp": 40, "fp": 0, "fn": 0, "expected": 40, "predicted": 40, "duplicate_count": 0}
    a = report["inputs"][0]
    assert a["semantic_gold"]["by_status"]["denied"]["tp"] == 1
    assert a["semantic_gold"]["by_status"]["unknown"]["tp"] == 1
    assert a["quote_source_validation"]["semantic_validation"] == "not_performed"
    image = report["inputs"][-1]
    assert image["quote_source_validation"]["scope"] == "saved model transcript units"
    assert image["image_transfer"] == {"matched_gold_facts": 8, "planned_gold_facts": 8,
        "all_ocr_text_validated": False, "pixel_or_layout_validation": "not_performed"}
    assert probe.replay_audit(root) == report


def test_matching_quote_is_not_semantic_gold_and_status_error_is_counted(tmp_path):
    def wrong(candidates, _source):
        candidates[0]["status"] = "denied"
    bundle = make_bundle(tmp_path / "bundle", transform=wrong)
    report = probe.audit(FIXTURE / "gold.json", [("memo-a", bundle)], tmp_path / "audit")
    row = report["inputs"][0]
    assert row["quote_source_validation"]["status"] == "verified"
    assert (row["semantic_gold"]["tp"], row["semantic_gold"]["fp"], row["semantic_gold"]["fn"]) == (7, 1, 1)
    assert row["semantic_gold"]["by_status"]["denied"]["fp"] == 1
    assert row["semantic_gold"]["by_status"]["asserted"]["fn"] == 1
    assert not report["all_inputs_exact"]


def test_semantic_duplicates_with_different_evidence_are_not_extra_true_positives(tmp_path):
    def duplicate(candidates, source):
        row = copy.deepcopy(candidates[0])
        row["evidence"].append({"unit_id": source["units"][0]["id"], "quote": source["units"][0]["text"]})
        candidates.append(row)
    bundle = make_bundle(tmp_path / "bundle", transform=duplicate)
    report = probe.audit(FIXTURE / "gold.json", [("memo-a", bundle)], tmp_path / "audit")
    metrics = report["inputs"][0]["semantic_gold"]
    assert metrics["tp"] == 8 and metrics["duplicate_count"] == 1 and metrics["predicted"] == 9
    assert not metrics["exact"] and not report["all_inputs_exact"]


def test_bad_quote_provenance_is_retained_without_semantic_scoring(bundles, tmp_path):
    bundle = tmp_path / "bad"
    shutil.copytree(bundles["memo-a"], bundle)
    (bundle / "assertions.json").write_bytes(b"[]")
    out = tmp_path / "audit"
    report = probe.audit(FIXTURE / "gold.json", [("memo-a", bundle)], out)
    assert report["source_verified_inputs"] == 0 and report["planned_inputs"] == 1
    assert report["planned_gold_tuples_including_format_repeats"] == 8
    assert report["inputs"][0]["semantic_gold"] == {"status": "not_scored"}
    assert report["inputs"][0]["quote_source_validation"]["status"] == "invalid"
    assert probe.replay_audit(out) == report


def test_audit_replay_detects_mutated_gold_or_report(bundles, tmp_path):
    out = tmp_path / "audit"
    probe.audit(FIXTURE / "gold.json", [("memo-a", bundles["memo-a"])], out)
    (out / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="differs from replay"):
        probe.replay_audit(out)
    (out / "gold/gold.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.replay_audit(out)


def test_frozen_requests_share_only_questions_ontology_and_complete_source(frozen):
    spec = probe._comparison_spec(frozen)
    assert spec["planned_requests"] == 8 and spec["planned_answers_per_arm"] == {"md": 24, "png": 12}
    assert [(r["case"], r["arm"]) for r in spec["order"]] == [
        ("md", "original"), ("md", "relations_with_source"), ("png", "original"), ("png", "relations_with_source"),
        ("png", "relations_with_source"), ("png", "original"), ("md", "relations_with_source"), ("md", "original")]
    questions_by_case = {}
    for row in spec["order"]:
        body = json.loads((frozen / "requests" / (row["id"] + ".json")).read_bytes())
        assert body["model"] == "gpt-5.5" and body["store"] is False and "tools" not in body
        assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
        parts = body["input"][1]["content"]
        payload = json.loads(parts[0]["text"])
        assert set(payload) == {"documents", "ontology", "questions"}
        assert all(set(q) == {"question_id", "question"} for q in payload["questions"])
        assert "expected_answer" not in parts[0]["text"] and "assertion_ids" not in parts[0]["text"]
        assert "subject_scope" not in parts[0]["text"] and "scope_evidence" not in parts[0]["text"]
        questions_by_case.setdefault(row["case"], payload["questions"])
        assert payload["questions"] == questions_by_case[row["case"]]
        if row["arm"] == "relations_with_source":
            assert len(parts) == 1
            for document in payload["documents"]:
                binding = next(i for c in spec["cases"] if c["id"] == row["case"] for i in c["inputs"]
                               if i["document_id"] == document["document_id"])
                source = json.loads((frozen / binding["bundle"] / "source.json").read_bytes())
                assert document["source_units"] == source["units"]
                assert len({u["id"] for u in document["source_units"]}) == len(source["units"])
                assert "quote" not in canonical_json(document["relations"]).decode()
        elif row["case"] == "md":
            for document in payload["documents"]:
                assert document["text"] == (FIXTURE / (document["document_id"] + ".md")).read_text(encoding="utf-8")
        else:
            import base64
            assert len(parts) == 2 and parts[1]["type"] == "input_image"
            assert base64.b64decode(parts[1]["image_url"].split(",", 1)[1]) == (FIXTURE / "variants/memo-a.png").read_bytes()
    assert [q["question_id"] for q in questions_by_case["png"]] == probe.PNG_QUESTIONS


def test_md_only_mode_has_explicit_smaller_denominator(bundles, tmp_path):
    spec = probe.prepare_comparison(FIXTURE / "gold.json", {key: bundles[key] for key in ("memo-a", "memo-b")},
                                    tmp_path / "frozen", endpoint=ENDPOINT)
    assert spec["planned_requests"] == 4 and spec["planned_answers_per_arm"] == {"md": 24}


def test_different_md_source_is_rejected_even_with_valid_bundle(bundles, tmp_path):
    with pytest.raises(ValueError, match="exact gold original bytes"):
        probe.prepare_comparison(FIXTURE / "gold.json", {"memo-a": bundles["memo-b"], "memo-b": bundles["memo-a"]},
                                 tmp_path / "wrong", endpoint=ENDPOINT)


@pytest.mark.parametrize("change,correct", [("missing", 11), ("extra", 0), ("duplicate_answer", 11),
                                          ("duplicate_claim", 11), ("extra_claim", 11),
                                          ("wrong_direction", 11), ("wrong_kind", 11)])
def test_strict_score_rejects_missing_extra_duplicates_and_semantic_mismatch(change, correct):
    ids = [q["question_id"] for q in GOLD["questions"]]
    payload = expected_payload(ids)
    if change == "missing":
        payload["answers"].pop(0)
    elif change == "extra":
        payload["answers"].append({**copy.deepcopy(payload["answers"][0]), "question_id": "Q99"})
    elif change == "duplicate_answer":
        payload["answers"].append(copy.deepcopy(payload["answers"][0]))
    elif change == "duplicate_claim":
        payload["answers"][0]["claims"].append(copy.deepcopy(payload["answers"][0]["claims"][0]))
    elif change == "extra_claim":
        payload["answers"][0]["claims"].append(copy.deepcopy(payload["answers"][1]["claims"][0]))
    elif change == "wrong_direction":
        claim = payload["answers"][0]["claims"][0]
        claim["subject"], claim["object"] = claim["object"], claim["subject"]
    else:
        payload["answers"][0]["answer_kind"] = "denied"
    scores = probe.score_answers(payload, GOLD, ids)
    assert scores["strict_correct"] == correct and scores["planned_answers"] == 12


def test_scorer_keeps_absence_unknown_denial_distinct_and_does_not_judge_explanation():
    ids = [q["question_id"] for q in GOLD["questions"]]
    payload = expected_payload(ids)
    assert probe.score_answers(payload, GOLD, ids)["strict_correct"] == 12
    for row in payload["answers"]:
        row["explanation"] = "This prose is deliberately not scored, including Q10 identity detail."
    assert probe.score_answers(payload, GOLD, ids)["strict_correct"] == 12
    payload["answers"][3]["answer_kind"] = "unknown"  # Q04 is absent, Q05 explicitly unknown.
    assert probe.score_answers(payload, GOLD, ids)["strict_correct"] == 11


def test_all_eight_calls_reserved_before_auth_and_replayed(frozen, tmp_path):
    out, calls = tmp_path / "run", []
    def before_auth():
        receipt = json.loads((out / "receipt.json").read_bytes())
        assert len(receipt["requests"]) == 8
        reserved = [r for r in receipt["requests"] if r["status"] == "reserved"]
        assert len(reserved) == 1 and (out / (reserved[0]["id"] + "-request.json")).exists()
    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer fixture-secret-never-persist"
        return success(request)
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True,
        credential=credentials(before_auth), transport=httpx.MockTransport(handle))
    assert receipt["status"] == "complete" and receipt["completed_requests"] == len(calls) == 8
    result = probe.replay(frozen, out)
    assert result["comparison"] == "complete_descriptive_only" and result["winner"] is None
    assert result["strict_delta_relations_minus_original"] == {"md": 0, "png": 0}
    for case, planned in (("md", 24), ("png", 12)):
        assert all(r["strict_correct"] == r["planned_answers"] == r["completed_answers"] == planned
                   for r in result["cases"][case].values())
    assert "fixture-secret" not in (out / "receipt.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(handle))


def test_without_execute_neither_auth_nor_transport_is_used(frozen, tmp_path):
    def forbidden(*_args):
        pytest.fail("No inference or token acquisition in a dry run")
    result = probe.run(frozen, tmp_path / "dry", endpoint=ENDPOINT,
                       credential=SimpleNamespace(get_token=forbidden), transport=httpx.MockTransport(forbidden))
    assert result["status"] == "not_executed" and result["completed_requests"] == 0
    assert probe.replay(frozen, tmp_path / "dry")["comparison"] == "incomplete"


@pytest.mark.parametrize("code,status", [(400, "failed"), (429, "failed"), (503, "unknown")])
def test_http_failures_save_raw_body_stop_and_never_retry(frozen, tmp_path, code, status):
    calls = []
    raw = b'{"error":{"message":"fixture failure"}}'
    def handle(request):
        calls.append(request)
        return httpx.Response(code, content=raw)
    out = tmp_path / "run"
    result = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(handle))
    assert len(calls) == 1 and result["status"] == "incomplete"
    row = result["requests"][0]
    assert row["status"] == status and row["wire_response_sha256"] == sha256(raw)
    assert (out / (row["id"] + "-response.json")).read_bytes() == raw
    assert all(r["status"] == "skipped_previous_failure" for r in result["requests"][1:])
    assert probe.replay(frozen, out)["completed_requests"] == 0


def test_auth_failure_is_recorded_without_leaking_exception_text(frozen, tmp_path):
    def fail(_scope):
        raise RuntimeError("credential-secret")
    result = probe.run(frozen, tmp_path / "run", endpoint=ENDPOINT, execute=True,
                       credential=SimpleNamespace(get_token=fail), transport=httpx.MockTransport(lambda _r: pytest.fail("No request")))
    assert result["requests"][0]["status"] == "failed" and result["completed_requests"] == 0
    assert "credential-secret" not in json.dumps(result)


def test_transport_timeout_is_unknown_and_reserves_run_against_resending(frozen, tmp_path):
    calls = []
    def fail(request):
        calls.append(request)
        raise httpx.ReadTimeout("fixture timeout", request=request)
    out = tmp_path / "run"
    result = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(fail))
    assert result["requests"][0]["status"] == "unknown" and len(calls) == 1
    with pytest.raises(FileExistsError):
        probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(fail))


@pytest.mark.parametrize("change", ["incomplete", "wrong_model", "missing_usage"])
def test_noncomparable_response_is_retained_but_not_scored(frozen, tmp_path, change):
    def handle(request):
        raw = json.loads(success(request).content)
        if change == "incomplete":
            raw["status"] = "incomplete"
        elif change == "wrong_model":
            raw["model"] = "gpt-4.1"
        else:
            raw.pop("usage")
        return httpx.Response(200, json=raw)
    result = probe.run(frozen, tmp_path / "run", endpoint=ENDPOINT, execute=True,
                       credential=credentials(), transport=httpx.MockTransport(handle))
    assert result["status"] == "incomplete" and result["completed_requests"] == 0
    assert "scores" not in result["requests"][0]
    assert probe.replay(frozen, tmp_path / "run")["comparison"] == "incomplete"


def test_frozen_file_schema_endpoint_and_inventory_are_checked_before_auth(frozen, tmp_path):
    with pytest.raises(ValueError, match="Endpoint differs"):
        probe.run(frozen, tmp_path / "endpoint", endpoint=ENDPOINT.replace("fixture.", "different."))
    changed = tmp_path / "frozen"
    shutil.copytree(frozen, changed)
    spec = json.loads((changed / "spec.json").read_bytes())
    spec["repeats"] = 3
    probe._write(changed / "spec.json", spec)
    with pytest.raises(ValueError, match="design differs"):
        probe.run(changed, tmp_path / "schema", endpoint=ENDPOINT)
    spec["repeats"] = 2
    spec["files"].pop("gold/gold.json")
    probe._write(changed / "spec.json", spec)
    with pytest.raises(ValueError, match="inventory differs"):
        probe.run(changed, tmp_path / "inventory", endpoint=ENDPOINT)
    assert not any((tmp_path / name).exists() for name in ("endpoint", "schema", "inventory"))


def test_rehashed_frozen_request_cannot_leak_gold(frozen, tmp_path):
    changed = tmp_path / "frozen"
    shutil.copytree(frozen, changed)
    spec = json.loads((changed / "spec.json").read_bytes())
    name = "requests/" + spec["order"][0]["id"] + ".json"
    body = json.loads((changed / name).read_bytes())
    body["input"][1]["content"][0]["text"] += "\nGOLD: " + json.dumps(GOLD)
    probe._write(changed / name, body)
    spec["files"][name] = sha256((changed / name).read_bytes())
    probe._write(changed / "spec.json", spec)
    with pytest.raises(ValueError, match="source-only reconstruction"):
        probe.run(changed, tmp_path / "run", endpoint=ENDPOINT)


def test_removed_raw_response_hash_cannot_reuse_previous_completed_response(frozen, tmp_path):
    out = tmp_path / "run"
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(success))
    receipt["requests"][1].pop("wire_response_sha256")
    probe._write(out / "receipt.json", receipt)
    with pytest.raises(ValueError, match="lacks saved inference evidence"):
        probe.replay(frozen, out)


def test_replay_rejects_rehashed_run_request_and_raw_score_drift(frozen, tmp_path):
    out = tmp_path / "run"
    receipt = probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(success))
    first = receipt["requests"][0]
    path = out / (first["id"] + "-request.json")
    old = path.read_bytes()
    path.write_bytes(old + b" ")
    first["request_sha256"] = sha256(path.read_bytes())
    probe._write(out / "receipt.json", receipt)
    with pytest.raises(ValueError, match="differs from frozen"):
        probe.replay(frozen, out)
    path.write_bytes(old)
    first["request_sha256"] = sha256(old)
    first["scores"]["strict_correct"] -= 1
    probe._write(out / "receipt.json", receipt)
    with pytest.raises(ValueError, match="differs from raw response"):
        probe.replay(frozen, out)


def test_recorded_gpt55_audit_and_comparison_replay_without_network():
    artifacts = Path(__file__).resolve().parents[1] / "docs/validation/narrative-foundry-2026-10-17"
    with pytest.raises(ValueError, match="frozen provenance"):
        probe.replay_audit(artifacts / "audit")
    audit = probe.replay_audit(artifacts / "audit", allow_code_drift=True)
    provenance = audit.pop("replay_provenance")
    assert {c["file"] for c in provenance["changes"]} == {"runner.py", "sr_adapter/narrative.py"}
    assert provenance["model_requests"] == 0 and provenance["frozen_files_unchanged"] is True
    assert audit == json.loads((artifacts / "audit/report.json").read_bytes())
    comparison = artifacts / "comparison"
    with pytest.raises(ValueError, match="frozen provenance"):
        probe.replay(comparison / "frozen", comparison / "run")
    replay = probe.replay(comparison / "frozen", comparison / "run", allow_code_drift=True)
    assert replay.pop("replay_provenance") == provenance
    assert replay == json.loads((comparison / "summary.json").read_bytes())
    assert replay["completed_requests"] == replay["planned_requests"] == 8
    assert replay["response_models"] == ["gpt-5.5"]
    assert replay["winner"] is None


def test_explicit_code_drift_replay_cannot_bypass_frozen_bytes_or_enable_inference(frozen, tmp_path, monkeypatch):
    out = tmp_path / "run"
    probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(success))
    paths = probe._provenance_files()
    changed_runner = tmp_path / "changed-runner.py"
    changed_runner.write_bytes(paths["runner.py"].read_bytes() + b"\n# changed implementation\n")
    paths["runner.py"] = changed_runner
    monkeypatch.setattr(probe, "_provenance_files", lambda: paths)
    with pytest.raises(ValueError, match="frozen provenance"):
        probe.run(frozen, tmp_path / "new-run", endpoint=ENDPOINT, execute=True,
                  credential=credentials(), transport=httpx.MockTransport(lambda _r: pytest.fail("No inference")))
    assert not (tmp_path / "new-run").exists()
    result = probe.replay(frozen, out, allow_code_drift=True)
    assert [c["file"] for c in result["replay_provenance"]["changes"]] == ["runner.py"]
    assert result["completed_requests"] == 8
    copied = tmp_path / "tampered"
    shutil.copytree(frozen, copied)
    p = copied / "gold/memo-a.md"
    p.write_bytes(p.read_bytes() + b"\ntampered source\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.replay(copied, out, allow_code_drift=True)


def test_explicit_drift_permission_still_rejects_rehashed_request_changes(frozen, tmp_path):
    out = tmp_path / "run"
    probe.run(frozen, out, endpoint=ENDPOINT, execute=True, credential=credentials(), transport=httpx.MockTransport(success))
    copied = tmp_path / "frozen"
    shutil.copytree(frozen, copied)
    spec = json.loads((copied / "spec.json").read_bytes())
    name = "requests/" + spec["order"][0]["id"] + ".json"
    body = json.loads((copied / name).read_bytes())
    body["input"][1]["content"][0]["text"] += " extra instructions"
    probe._write(copied / name, body)
    spec["files"][name] = sha256((copied / name).read_bytes())
    probe._write(copied / "spec.json", spec)
    with pytest.raises(ValueError, match="source-only reconstruction"):
        probe.replay(copied, out, allow_code_drift=True)
