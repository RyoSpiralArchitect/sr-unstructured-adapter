"""Exact role anchors over saved readings; fixtures do not measure extraction quality."""
from copy import deepcopy
from hashlib import sha256
import json
import sqlite3

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative, verify_narrative
from sr_adapter.relation_dataset_store import (export_relation_dataset, export_relation_semantics,
    _load_dataset, verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.relation_semantics import (normalize_overlay, project_relation_semantics,
    resolve_text_selector, trace_semantic_claim, SemanticOverlayV2, LiteralValueV2, XSD)
from sr_adapter.semantic_proposals import (INSTRUCTIONS, proposal_context, proposal_schema,
    render_semantic_proposal, import_semantic_proposal, verify_semantic_proposal, resolve_proposal)
from sr_adapter.semantic_proposal_runtime import propose_semantics, main
from sr_adapter.tables import canonical_json
from test_docx_relation_sources import package, paragraph
from test_relation_dataset import visual as visual, source as source, rehash
from test_narrative import response
from test_semantic_proposals import raw

NS = "urn:anchor-test:"
TEXT = "😀審査部は申請Aを確認し、審査部は申請Aを承認する。"


def selector(exact, occurrence=0):
    return {"kind": "text", "exact": exact, "occurrence": occurrence}


@pytest.fixture
def narrative_anchors(tmp_path):
    ontology = NarrativeOntology.model_validate({"namespace": NS, "classes": {"Thing": "Thing"},
        "predicates": [{"id": "related", "label": "関係", "subject_type": "Thing", "object_type": "Thing", "description": "Fixture"}]})
    data = package(paragraph(TEXT) + paragraph("担当者の確定情報ではない。"))
    manifest, files = render_narrative(data=data, source_name="roles.docx", ontology=ontology,
        model="fixture", response=response([]))
    saved = tmp_path / "narrative"
    save_narrative(saved, manifest, files)
    base = tmp_path / "base"
    export_relation_dataset([saved], base, profile="narrative")
    return base


def anchored_declaration(dataset):
    _, _, bindings = proposal_context(dataset)
    narrative = dataset.profile.startswith("narrative")
    records = [r for r in bindings.values() if r["table"] == ("evidence" if narrative else "items")
               and (r["record"].get("text") == TEXT if narrative else r["record"].get("label") == "共通システム")]
    records.sort(key=lambda r: r["node_id"])
    nodes = [records[0]["node_id"]] * 2 if narrative else [r["node_id"] for r in records[:2]]
    arguments = [{"role": NS + role, "value": {"kind": "source", "node_id": node,
        "selector": selector(text)}} for role, node, text in zip(
            ("actor", "object"), nodes, ("審査部", "申請A") if narrative else ("システム", "システム"))]
    pid = dataset.package_ids[0]
    return {"version": 2, "source_snapshot_id": dataset.snapshot_id, "author": "handwritten fixture",
        "terms": [{"iri": NS + key, "kind": kind, "label": key}
                  for key, kind in (("Review", "relation"), ("actor", "role"), ("object", "role"))],
        "roles": [{"predicate": NS + "Review", "role": NS + role, "value_kinds": ["source"], "min_count": 1, "max_count": 1}
                  for role in ("actor", "object")],
        "expressions": [{"id": "review", "package_id": pid, "operator": "atom", "predicate": NS + "Review", "arguments": arguments}],
        "claims": [{"id": "c1", "package_id": pid, "expression_id": "review", "status": "asserted",
            "modality": "descriptive", "basis": "source_interpretation", "attributed_to": "source (speaker unspecified)",
            "evidence": nodes}]}


def as_proposal(dataset, overlay):
    context, packages, bindings = proposal_context(dataset)
    package_alias = {pid: key for key, pid in packages.items()}
    aliases = {r["node_id"]: key for key, r in bindings.items()}
    overlay = normalize_overlay(overlay)
    for field in ("version", "source_snapshot_id", "author"):
        overlay.pop(field)
    for expr in overlay["expressions"]:
        expr["package_id"] = package_alias[expr["package_id"]]
        for arg in expr["arguments"]:
            arg["value"]["node_id"] = aliases[arg["value"]["node_id"]]
    for claim in overlay["claims"]:
        claim["package_id"] = package_alias[claim["package_id"]]
        claim["evidence"] = [aliases[node] for node in claim["evidence"]]
    return {"overlay": overlay, "notes": [], "coverage": [
        {"source_id": key, "disposition": "represented" if key in overlay["claims"][0]["evidence"] else "context_only",
         "claim_ids": ["c1"] if key in overlay["claims"][0]["evidence"] else [], "note": "Authored fixture"}
        for key in context["coverage_units"]]}


@pytest.mark.parametrize("text,exact,occurrence,start,end", [
    (TEXT, "審査部", 0, 1, 4), (TEXT, "審査部", 1, 13, 16),
    ("aaaa", "aa", 2, 2, 4), ("e\u0301 / é", "é", 0, 5, 6),
    ("😀審査\n部", "審査\n部", 0, 1, 5), (" A A ", " A ", 1, 2, 5),
])
def test_literal_selection_has_unicode_offsets_and_explicit_repeated_occurrences(text, exact, occurrence, start, end):
    result = resolve_text_selector(text, selector(exact, occurrence))
    assert result == {"exact": exact, "occurrence": occurrence, "start": start, "end": end}
    assert text[result["start"]:result["end"]] == exact


@pytest.mark.parametrize("text,selection", [
    (TEXT, selector("審査部", 2)), (TEXT, selector("審査 部")), ("ABC", selector("abc")),
    ("e\u0301", selector("é")), ("申請Ａ", selector("申請A")), ("a\r\nb", selector("a\nb")),
    (TEXT, selector("")), (TEXT, selector("審査部", -1)), (TEXT, selector("審査部", True)),
    (TEXT, selector("審査部", "0")), (None, selector("審査部")),
    (TEXT, {"kind": "text", "exact": "審査部"}),
])
def test_missing_or_normalized_quotes_never_rebind_to_another_reading(text, selection):
    with pytest.raises(ValueError):
        resolve_text_selector(text, selection)


def test_distinct_roles_in_one_paragraph_and_repeated_mentions_keep_their_offsets(narrative_anchors):
    dataset = verify_relation_dataset(narrative_anchors)
    overlay = anchored_declaration(dataset)
    before = deepcopy(overlay)
    projected = project_relation_semantics(dataset, SemanticOverlayV2.model_validate(overlay))
    assert projected.profile == "relation_semantics/2" and len(projected.tables) == 12
    anchors = sorted(projected.rows()["argument_anchors"], key=lambda r: r["position"])
    assert [(a["exact"], a["start"], a["end"]) for a in anchors] == [("審査部", 1, 4), ("申請A", 5, 8)]
    assert len({a["node_id"] for a in anchors}) == 1
    overlay["expressions"][0]["arguments"][0]["value"]["selector"]["occurrence"] = 1
    changed = project_relation_semantics(dataset, overlay)
    assert changed.snapshot_id != projected.snapshot_id
    assert next(a for a in changed.rows()["argument_anchors"] if a["position"] == 0)["start"] == 13
    trace = trace_semantic_claim(dataset, before, claim_id="c1")
    assert len(trace["argument_anchors"]) == 2 and trace["source_context"] == dataset.rows()
    assert any(r["record"].get("text") == TEXT for r in trace["source_records"])


def test_visual_labels_retain_distinct_occurrences_boxes_and_uncertainty(visual, tmp_path):
    base = tmp_path / "base"
    export_relation_dataset([visual], base, profile="visual")
    dataset = verify_relation_dataset(base)
    trace = trace_semantic_claim(dataset, anchored_declaration(dataset), claim_id="c1")
    anchors = trace["argument_anchors"]
    assert {a["field"] for a in anchors} == {"label"}
    assert {a["exact"] for a in anchors} == {"システム"} and len({a["node_id"] for a in anchors}) == 2
    records = [r["record"] for r in trace["source_records"] if r["table"] == "items"]
    assert len({r["x0"] for r in records}) == 2
    assert trace["source_context"]["relations"][0]["support"] == "uncertain"


@pytest.mark.parametrize("change", ["quote", "occurrence", "node", "table", "version"])
def test_invalid_anchors_reject_before_any_export(narrative_anchors, tmp_path, change):
    dataset = verify_relation_dataset(narrative_anchors)
    overlay = anchored_declaration(dataset)
    value = overlay["expressions"][0]["arguments"][0]["value"]
    if change == "quote":
        value["selector"]["exact"] = "架空の部門"
    elif change == "occurrence":
        value["selector"]["occurrence"] = 20
    elif change == "node":
        value["node_id"] = "n_" + "0" * 64
    elif change == "table":
        _, _, bindings = proposal_context(dataset)
        value["node_id"] = next(r["node_id"] for r in bindings.values() if r["table"] == "documents")
    else:
        overlay["version"] = 1
    with pytest.raises(ValueError):
        export_relation_semantics(narrative_anchors, tmp_path / "invalid", overlay=overlay)
    assert not (tmp_path / "invalid").exists()


def test_anchor_alias_resolution_replay_and_scalar_sqlite_roundtrip(narrative_anchors, tmp_path):
    dataset = verify_relation_dataset(narrative_anchors)
    payload = as_proposal(dataset, anchored_declaration(dataset))
    out = tmp_path / "proposal"
    manifest = import_semantic_proposal(narrative_anchors, out, model="fixture", response=raw(payload))
    assert manifest["version"] == 2 and verify_semantic_proposal(out)[0] == manifest
    semantic = verify_relation_dataset(out / "semantics")
    assert semantic.profile == "relation_semantics/2"
    report = json.loads((out / "review.json").read_bytes())
    assert report["anchor_counts"] == {"selected_text": 2, "whole_record": 0}
    assert {a["role"] for a in report["argument_anchors"]} == {NS + "actor", NS + "object"}
    markdown = (out / "review.md").read_text(encoding="utf-8")
    assert "審査部" in markdown and "役割ごとの原文参照" in markdown
    db = tmp_path / "anchors.db"
    receipt = write_dataset_sqlite(out / "semantics", db)
    assert write_dataset_sqlite(out / "semantics", db)["inserted_rows"] == 0
    with sqlite3.connect(db) as con:
        name = receipt["tables"]["argument_anchors"]["name"]
        assert con.execute(f'SELECT exact, start, end FROM "{name}" ORDER BY position').fetchall() == [
            ("審査部", 1, 4), ("申請A", 5, 8)]
    assert report["semantic_validation"] == "not_performed" and report["status"] == "unreviewed"


def test_whole_record_references_remain_visible_without_fake_anchors(narrative_anchors):
    dataset = verify_relation_dataset(narrative_anchors)
    payload = as_proposal(dataset, anchored_declaration(dataset))
    payload["overlay"]["expressions"][0]["arguments"][0]["value"]["selector"] = None
    _, _, report = resolve_proposal(dataset, raw(payload))
    assert report["anchor_counts"] == {"selected_text": 1, "whole_record": 1}
    payload["overlay"] = None
    for entry in payload["coverage"]:
        entry.update(disposition="unresolved", claim_ids=[], note="No supported interpretation")
    _, overlay, report = resolve_proposal(dataset, raw(payload))
    assert overlay is None and report["anchor_counts"] == {"selected_text": 0, "whole_record": 0}


@pytest.mark.parametrize("file", ["semantics.json", "tables/argument_anchors.parquet"])
def test_rehash_cannot_hide_modified_selector_or_derived_offsets(narrative_anchors, tmp_path, file):
    dataset = verify_relation_dataset(narrative_anchors)
    out = tmp_path / "semantics"
    export_relation_semantics(narrative_anchors, out, overlay=anchored_declaration(dataset))
    path = out / file
    if file.endswith("json"):
        overlay = json.loads(path.read_bytes())
        overlay["expressions"][0]["arguments"][0]["value"]["selector"]["occurrence"] = 1
        path.write_bytes(canonical_json(overlay))
    else:
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["start"] += 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    rehash(out, file)
    with pytest.raises(ValueError):
        verify_relation_dataset(out)


def test_v1_protocol_is_frozen_and_archived_rendering_still_replays(narrative_anchors, tmp_path):
    assert sha256(INSTRUCTIONS.encode()).hexdigest() == "807d4b89cdb88378831b4df994c76785ceea46844cc2d478ab954b927d5db588"
    assert sha256(canonical_json(proposal_schema(version=1))).hexdigest() == "40181146097e58a48b619ed6137163de92776b89fd1ee81e3d2dc97a3630a5fa"
    dataset, inputs = _load_dataset(narrative_anchors, allow_view=False)
    payload = as_proposal(dataset, anchored_declaration(dataset))
    for arg in payload["overlay"]["expressions"][0]["arguments"]:
        arg["value"].pop("selector")
    legacy = tmp_path / "legacy"
    manifest = import_semantic_proposal(narrative_anchors, legacy, model="fixture", response=raw(payload), version=1)
    assert manifest["version"] == 1 and verify_semantic_proposal(legacy)[0] == manifest
    old = verify_relation_dataset(legacy / "semantics")
    assert old.profile == "relation_semantics/1" and len(old.tables) == 11
    _, new_files = render_semantic_proposal(inputs, model="fixture", response=raw(payload))
    assert json.loads(new_files["overlay.json"])["version"] == 2


def test_runtime_uses_v2_schema_and_one_call_then_replay(narrative_anchors, tmp_path, capsys):
    dataset = verify_relation_dataset(narrative_anchors)
    payload = as_proposal(dataset, anchored_declaration(dataset))
    calls = []
    def handle(request):
        wire = json.loads(request.content)
        assert "TextSelector" in wire["text"]["format"]["schema"]["$defs"]
        assert "ROLE ANCHORS" in wire["input"][0]["content"][0]["text"]
        calls.append(request)
        return httpx.Response(200, json=raw(payload))
    out = tmp_path / "run"
    receipt = propose_semantics(narrative_anchors, out, model="fixture", execute=True,
        api_key="test-only", transport=httpx.MockTransport(handle))
    assert receipt["status"] == "complete" and receipt["proposal_version"] == 2 and len(calls) == 1
    assert verify_semantic_proposal(out / "bundle")[0]["version"] == 2
    assert main(["propose", str(narrative_anchors), "--out", str(tmp_path / "v1"), "--model", "fixture",
                 "--proposal-version", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["proposal_version"] == 1


def test_literal_datatype_choices_reach_the_model_schema():
    choices = proposal_schema()["$defs"]["LiteralValueV2"]["properties"]["datatype"]["enum"]
    assert XSD + "decimal" in choices and "xsd:decimal" not in choices
    for iri in choices:
        assert iri.startswith("http://www.w3.org/")
    with pytest.raises(ValueError):
        LiteralValueV2(kind="literal", datatype="xsd:decimal", lexical="10000000.00")
    assert LiteralValueV2(kind="literal", datatype=XSD + "decimal", lexical="10000000.00").lexical == "10000000.00"


def test_cross_package_anchor_keeps_owner_and_target_distinct(narrative_anchors, tmp_path):
    _, files = verify_narrative(tmp_path / "narrative")
    manifest, payloads = render_narrative(data=files["source.bin"], source_name="second.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="fixture", response=response([]))
    save_narrative(tmp_path / "second", manifest, payloads)
    base = tmp_path / "combined"
    export_relation_dataset([tmp_path / "narrative", tmp_path / "second"], base, profile="narrative")
    dataset = verify_relation_dataset(base)
    overlay = anchored_declaration(dataset)
    _, _, bindings = proposal_context(dataset)
    pair = [next(r["node_id"] for r in bindings.values() if r["package_id"] == pid
                 and r["table"] == "evidence" and r["record"]["text"] == TEXT) for pid in dataset.package_ids]
    for arg, node in zip(overlay["expressions"][0]["arguments"], pair):
        arg["value"]["node_id"] = node
    overlay["claims"][0]["evidence"] = pair
    trace = trace_semantic_claim(dataset, overlay, claim_id="c1")
    anchors = trace["argument_anchors"]
    assert {a["package_id"] for a in anchors} == {dataset.package_ids[0]}
    assert {a["source_package_id"] for a in anchors} == set(dataset.package_ids)
    assert len(trace["source_context"]["documents"]) == 2
    out = tmp_path / "cross-proposal"
    result = import_semantic_proposal(base, out, model="fixture", response=raw(as_proposal(dataset, overlay)))
    assert verify_semantic_proposal(out)[0] == result


def test_runtime_retains_invalid_anchor_response_without_retry_or_bundle(narrative_anchors, tmp_path):
    dataset = verify_relation_dataset(narrative_anchors)
    payload = as_proposal(dataset, anchored_declaration(dataset))
    payload["overlay"]["expressions"][0]["arguments"][0]["value"]["selector"]["exact"] = "存在しない部門"
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=raw(payload))
    out = tmp_path / "invalid-run"
    receipt = propose_semantics(narrative_anchors, out, model="fixture", execute=True,
        api_key="test-only", transport=httpx.MockTransport(handle))
    assert receipt["status"] == "failed" and receipt["failure_reason"] == "proposal_validation_failed"
    assert receipt["automatic_retries"] == 0 and len(calls) == 1
    assert json.loads((out / "response.json").read_bytes()) == raw(payload)
    assert not (out / "bundle").exists()
