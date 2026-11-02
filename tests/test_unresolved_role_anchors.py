"""V5 retains failed local text bindings as explicit unknown candidate roles."""
from copy import deepcopy
import json
import sqlite3

import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.local_ontology import read_ontology_source
from sr_adapter.ontology_ingest import _source_bundle, ingest_ontology_source, ontology_ingestion_status
from sr_adapter.ontology_recovery import recover_ontology_ingestion
from sr_adapter.ontology_tools import OntologyTools
from sr_adapter.relation_dataset_store import _load_dataset
from sr_adapter.relation_semantics import project_relation_semantics, trace_semantic_claim
from sr_adapter.semantic_proposals import proposal_request, render_semantic_proposal, resolve_proposal
from test_ocr_ontology import image
from test_ontology_ingest import proposed
from test_visual_relations import INVENTORY, response


TEXT = "資料Aは融資運用部の現行手順である。\n資料Bとの不一致は未解決であり、優先資料は未決定である。"
UNKNOWN = {"kind": "unknown", "reason": "unresolved_text_selector"}


def selected_answer(wire, *, exact="資料A", occurrence=0):
    context = json.loads(wire["input"][1]["content"][0]["text"])
    bound = next(record for record in context["records"]
                 if "未解決" in record["record"].get("text", ""))
    answer = proposed(wire)
    content = answer["output"][0]["content"][0]
    payload = json.loads(content["text"])
    overlay = payload["overlay"]
    overlay["terms"].append({"iri": "urn:test:qualification", "kind": "role", "label": "qualification"})
    overlay["expressions"][0]["arguments"] = [
        {"role": "urn:test:source", "value": {"kind": "source", "node_id": bound["id"],
            "selector": {"kind": "text", "exact": exact, "occurrence": occurrence}}},
        {"role": "urn:test:qualification", "value": {"kind": "source", "node_id": bound["id"],
            "selector": {"kind": "text", "exact": "未解決", "occurrence": 0}}},
    ]
    overlay["claims"][0]["evidence"] = [bound["id"]]
    for entry in payload["coverage"]:
        entry.update(disposition="represented" if entry["source_id"] == bound["id"] else "context_only",
                     claim_ids=["c1"] if entry["source_id"] == bound["id"] else [])
    content["text"] = json.dumps(payload)
    return answer


def change_payload(answer, mutate):
    answer = deepcopy(answer)
    content = answer["output"][0]["content"][0]
    payload = json.loads(content["text"])
    mutate(payload)
    content["text"] = json.dumps(payload)
    return answer


@pytest.fixture
def native_case(tmp_path):
    inputs = _source_bundle(TEXT.encode(), "source.txt", tmp_path)
    dataset = _load_dataset(tmp_path / "dataset", allow_view=False)[0]
    wire = proposal_request(inputs, model="fixture", version=5)
    return dataset, inputs, wire


@pytest.mark.parametrize("exact,occurrence,reason", [
    ("資料A", 0, "exact_not_found_in_bound_unit"),
    ("資料B", 1, "occurrence_not_found_in_bound_unit"),
])
def test_invalid_local_binding_becomes_unknown_without_retargeting(native_case, exact, occurrence, reason):
    dataset, _, wire = native_case
    answer = selected_answer(wire, exact=exact, occurrence=occurrence)
    original = deepcopy(answer)
    proposal, overlay, report = resolve_proposal(dataset, answer, version=5)
    assert answer == original
    original_value = proposal["overlay"]["expressions"][0]["arguments"][0]["value"]
    assert original_value["selector"] == {"kind": "text", "exact": exact, "occurrence": occurrence}
    assert overlay["expressions"][0]["arguments"][0]["value"] == UNKNOWN
    finding, = report["anchor_findings"]
    assert finding == {
        "expression_id": "e1", "position": 0, "role": "urn:test:source",
        "source_id": original_value["node_id"], "node_id": overlay["claims"][0]["evidence"][0],
        "selector": original_value["selector"], "failure_reason": reason, "requires_review": True,
    }
    assert report["anchor_counts"] == {"selected_text": 1, "whole_record": 0, "unresolved": 1}
    semantic = project_relation_semantics(dataset, overlay)
    argument = next(row for row in semantic.rows()["arguments"] if row["position"] == 0)
    assert argument["value_kind"] == "unknown" and argument["reason"] == "unresolved_text_selector"
    assert argument["node_id"] is None
    trace = trace_semantic_claim(dataset, overlay, claim_id="c1")
    assert [anchor["exact"] for anchor in trace["argument_anchors"]] == ["未解決"]
    assert all(record["record"]["text"].startswith("資料B") for record in trace["source_records"])
    # The exact phrase exists in another unit, but no binding or citation moves.
    assert any("資料A" in record["text"] for record in trace["source_context"]["evidence"])
    assert all("資料A" not in record["record"]["text"] for record in trace["source_records"])


def test_valid_selected_roles_and_replay_payloads_are_unchanged(native_case):
    dataset, inputs, wire = native_case
    answer = selected_answer(wire, exact="資料B")
    proposal, overlay, report = resolve_proposal(dataset, answer, version=5)
    assert "anchor_findings" not in report
    assert report["anchor_counts"] == {"selected_text": 2, "whole_record": 0}
    assert [anchor["exact"] for anchor in sorted(report["argument_anchors"], key=lambda item: item["position"])] == ["資料B", "未解決"]
    assert all(arg["value"]["kind"] == "source" for arg in overlay["expressions"][0]["arguments"])
    _, files = render_semantic_proposal(inputs, model="fixture", response=answer, version=5)
    assert json.loads(files["proposal.json"]) == proposal
    assert json.loads(files["response.json"]) == answer
    assert json.loads(files["request.json"]) == wire


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_protocols_keep_strict_anchor_failure(native_case, version):
    dataset, inputs, _ = native_case
    wire = proposal_request(inputs, model="fixture", version=version)
    with pytest.raises(ValueError):
        resolve_proposal(dataset, selected_answer(wire), version=version)


@pytest.mark.parametrize("change", ["unknown_alias", "wrong_table", "missing_selector_field", "negative_occurrence", "empty_exact", "source_only_signature", "missing_role"])
def test_degradation_does_not_relax_other_contracts(native_case, change):
    dataset, _, wire = native_case
    context = json.loads(wire["input"][1]["content"][0]["text"])

    def mutate(payload):
        overlay = payload["overlay"]
        argument = overlay["expressions"][0]["arguments"][0]
        if change == "unknown_alias":
            argument["value"]["node_id"] = "s999999"
        elif change == "wrong_table":
            argument["value"]["node_id"] = next(record["id"] for record in context["records"] if record["table"] == "documents")
        elif change == "missing_selector_field":
            argument["value"]["selector"].pop("occurrence")
        elif change == "negative_occurrence":
            argument["value"]["selector"]["occurrence"] = -1
        elif change == "empty_exact":
            argument["value"]["selector"]["exact"] = ""
        elif change == "source_only_signature":
            overlay["roles"] = [{"predicate": "urn:test:read", "role": role, "value_kinds": ["source"]}
                                for role in ("urn:test:source", "urn:test:qualification")]
        elif change == "missing_role":
            argument["role"] = "urn:test:undeclared"

    with pytest.raises(ValueError):
        resolve_proposal(dataset, change_payload(selected_answer(wire), mutate), version=5)


def test_visual_item_missing_occurrence_is_unknown_at_its_original_item(tmp_path):
    inputs = _source_bundle(image(), "source.png", tmp_path, visual_response=response(INVENTORY), model="fixture")
    dataset = _load_dataset(tmp_path / "dataset", allow_view=False)[0]
    wire = proposal_request(inputs, model="fixture", version=5)
    answer = proposed(wire)
    context = json.loads(wire["input"][1]["content"][0]["text"])
    item = next(record for record in context["records"] if record["id"] == context["coverage_units"][0])

    def mutate(payload):
        payload["overlay"]["expressions"][0]["arguments"][0]["value"]["selector"] = {
            "kind": "text", "exact": item["record"]["label"], "occurrence": 1}

    _, overlay, report = resolve_proposal(dataset, change_payload(answer, mutate), version=5)
    assert overlay["expressions"][0]["arguments"][0]["value"] == UNKNOWN
    assert report["anchor_findings"][0]["source_id"] == item["id"]
    assert report["anchor_findings"][0]["failure_reason"] == "occurrence_not_found_in_bound_unit"


def test_existing_failed_v5_response_recovers_locally_with_original_evidence(tmp_path, monkeypatch):
    import sr_adapter.semantic_proposals as proposals

    source, db = tmp_path / "source.txt", tmp_path / "memory.sqlite"
    source.write_bytes(TEXT.encode("utf-8"))
    current_resolver = proposals.resolve_proposal

    def previous_strict_resolver(dataset, answer, **kwargs):
        # V4 and v5 have identical compact aliases. This recreates the former
        # strict selector check while retaining the actual archived v5 wire.
        return current_resolver(dataset, answer, **{**kwargs, "version": 4})

    with monkeypatch.context() as previous:
        previous.setattr(proposals, "resolve_proposal", previous_strict_resolver)
        failed = ingest_ontology_source(source, db, complete=selected_answer, adapter_id="fixture", execute=True)
    assert failed["status"] == "failed" and failed["proposal_version"] == 5
    assert failed["stage"] == "semantics_validating"
    original_receipt = ontology_ingestion_status(db, failed["run_id"])
    with sqlite3.connect(db) as connection:
        original_files = dict(connection.execute("SELECT path,data FROM sro_ingestion_files WHERE run_id=?", (failed["run_id"],)))
    source.unlink()
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("Recovery made a paid call"))
    recovered = recover_ontology_ingestion(db, failed["run_id"])
    assert recovered["status"] == "complete", recovered
    assert recovered["submitted_requests"] == 0 and recovered["semantic_validation"] == "not_performed"
    assert ontology_ingestion_status(db, failed["run_id"]) == original_receipt
    with sqlite3.connect(db) as connection:
        assert dict(connection.execute("SELECT path,data FROM sro_ingestion_files WHERE run_id=?", (failed["run_id"],))) == original_files
        archive = dict(connection.execute("SELECT path,data FROM sro_files WHERE archive_id=?", (recovered["archive_id"],)))
    assert archive["request.json"] == original_files["semantics-request.json"]
    assert archive["response.json"] == original_files["semantics-response.json"]
    assert json.loads(archive["proposal.json"])["overlay"]["expressions"][0]["arguments"][0]["value"]["selector"]["exact"] == "資料A"
    report = json.loads(archive["review.json"])
    assert len(report["anchor_findings"]) == 1
    assert "解決できなかった役割の原文参照" in archive["review.md"].decode()
    tools = OntologyTools(db, ("default",))
    hit, = tools.call("search_ontology", {"collection": "default", "query": "", "kind": "claim"})["hits"]
    inspected = tools.call(hit["inspect"]["name"], hit["inspect"]["arguments"])
    assert inspected["status"] == "complete"
    value = inspected["value"]
    assert value["expressions"][0]["arguments"][0]["value"] == UNKNOWN
    assert all(record["record"]["text"].startswith("資料B") for record in value["source_records"])
    original = read_ontology_source(db, **{key: hit["reference"][key] for key in ("collection", "name", "snapshot_id")},
        package_id=value["claims"][0]["package_id"])
    assert original["data"] == TEXT.encode() and original["sha256"] == sha256(TEXT.encode())
    assert recover_ontology_ingestion(db, failed["run_id"])["outcome"] == "already_active"
