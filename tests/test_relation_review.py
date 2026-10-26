"""Review suggestions remain separate from the saved source and its claims."""
import json
import sqlite3

import pytest

from sr_adapter.relation_receiver import SQLiteRelationReceiver, trace_received_relation
from sr_adapter.relation_review import review_received_relations, main
from sr_adapter.narrative import render_narrative, save_narrative
from sr_adapter.relation_sources import read_source
from test_fabric_narrative import narrative as narrative
from test_narrative import ontology as ontology, response
from test_relation_receiver import delivered as delivered, revision, pair
from test_relation_dataset import visual as visual, source as source
from test_visual_relations import INVENTORY, RELATIONS, BOX, run, response as visual_response


@pytest.fixture
def reviewed_db(delivered, tmp_path):
    db = tmp_path / "review.db"
    SQLiteRelationReceiver(db).receive(delivered, collection="documents")
    return db


def test_review_reasons_preserve_unknown_denial_nonliteral_labels_and_evidence(reviewed_db):
    before = reviewed_db.read_bytes()
    report = review_received_relations(reviewed_db, collection="documents")
    assert report["reason_counts"]["unknown_object"] == 1
    assert report["reason_counts"]["nonliteral_label"] == 1
    assert "denied_assertion" not in report["reason_counts"]
    unknown = next(i for i in report["items"] if any(r["code"] == "unknown_object" for r in i["reasons"]))
    assert unknown["record"]["object_label"] is None
    denial = next(i for i in report["items"] if any(r["code"] == "nonliteral_label" for r in i["reasons"]))
    assert denial["record"]["assertion_status"] == "denied"
    traced = trace_received_relation(reviewed_db, **denial["trace_refs"][0])
    assert traced["evidence"][1]["properties"]["text"] == "B is not the owner"
    assert any(loc["properties"]["path"] == "/header_context/0/text" for e in traced["evidence"] for loc in e["locations"])
    assert reviewed_db.read_bytes() == before
    optional = review_received_relations(reviewed_db, collection="documents", include_denials=True, reason="denied_assertion")
    assert optional["total"] == 1 and optional["items"][0]["item_id"] == denial["item_id"]
    assert next(r for r in optional["items"][0]["reasons"] if r["code"] == "denied_assertion")["category"] == "information"


def test_explicit_denial_is_information_and_ambiguity_is_a_review_candidate(ontology, tmp_path):
    data = b"A is not managed by B.\n\nA has ambiguous ownership."
    units = read_source(data, "source.txt")["units"]
    claims = [{"subject": "A", "predicate": "managedBy", "object": "B" if i == 0 else None,
        "status": status, "evidence": [{"unit_id": units[i]["id"], "quote": units[i]["text"]}]}
        for i, status in enumerate(["denied", "ambiguous"])]
    manifest, files = render_narrative(data=data, source_name="source.txt", ontology=ontology, model="fixture", response=response(claims))
    save_narrative(tmp_path / "source", manifest, files)
    delivery = pair(tmp_path / "source", tmp_path / "pair")
    db = tmp_path / "db"
    SQLiteRelationReceiver(db).receive(delivery, collection="documents")
    report = review_received_relations(db, collection="documents")
    assert report["reason_counts"] == {"ambiguous_object": 1}
    assert report["items"][0]["record"]["object_id"] is None


def test_source_context_empty_extraction_is_not_a_missing_requirement(narrative, tmp_path):
    delivery = revision(narrative, tmp_path / "empty", empty=True)
    db = tmp_path / "db"
    SQLiteRelationReceiver(db).receive(delivery, collection="documents")
    report = review_received_relations(db, collection="documents", reason="no_extracted_relations")
    assert report["total"] == 1 and report["items"][0]["kind"] == "source"
    trace = trace_received_relation(db, **report["items"][0]["trace_refs"][0])
    assert trace["assertion"] is None and "Only applies" in trace["card"]["content"]
    assert "does not prove" in report["items"][0]["reasons"][0]["message"]


def test_visual_uncertainty_limitations_and_recorded_reviews_are_separate(visual, tmp_path):
    base = pair(visual, tmp_path / "base", "visual")
    review = {"version": 1, "reviewer": "synthetic", "findings": [
        {"package_id": base.source.package_ids[0], "reference_kind": "item", "reference_id": "n1", "finding": "text_mismatch", "detail": "Recheck label"},
        {"package_id": base.source.package_ids[0], "reference_kind": "image", "reference_id": None, "finding": "coverage_gap", "detail": "Inspect uncaptured content"}]}
    delivery = pair(visual, tmp_path / "reviewed", "visual", review)
    db = tmp_path / "db"
    SQLiteRelationReceiver(db).receive(delivery, collection="images")
    report = review_received_relations(db, collection="images")
    assert report["reason_counts"]["uncertain_visual_relation"] == 1
    assert report["reason_counts"]["unspecified_direction"] == 1
    assert report["reason_counts"]["recorded_review"] == 2
    records = [i for i in report["items"] if i["kind"] == "review"]
    assert {i["record"]["detail"] for i in records} == {"Recheck label", "Inspect uncaptured content"}
    item = next(i for i in records if i["record"]["reference_kind"] == "item")
    assert item["referenced_record"]["label"] == "共通システム" and item["trace_refs"]
    assert report["reason_counts"]["source_limitation"] == 1
    relation = next(i for i in report["items"] if i["kind"] == "relation")
    assert relation["record"]["support"] == "uncertain" and relation["record"]["value_text"] == "0%"
    assert relation["record"]["subject_id"] != relation["record"]["object_id"]


def test_paging_is_stable_and_old_review_items_keep_their_snapshot(reviewed_db, narrative, tmp_path):
    first = review_received_relations(reviewed_db, collection="documents", limit=1)
    second = review_received_relations(reviewed_db, collection="documents", limit=1, offset=1)
    assert first["has_more"] and first["items"][0]["item_id"] != second["items"][0]["item_id"]
    newer = revision(narrative, tmp_path / "new")
    SQLiteRelationReceiver(reviewed_db).receive(newer, collection="documents", expected_snapshot=first["snapshot_id"])
    historical = review_received_relations(reviewed_db, collection="documents", snapshot_id=first["snapshot_id"], limit=1)
    assert historical["items"] == first["items"]
    assert historical["snapshot_id"] != historical["active_snapshot_id"]


def test_review_of_unconnected_image_item_is_not_dropped(source, tmp_path):
    inventory = {**INVENTORY, "observations": [*INVENTORY["observations"],
        {"id": "n4", "label": "Unconnected", "kind": "system", "box": BOX}]}
    bundle = tmp_path / "image"
    run(source, bundle, responses=[visual_response(inventory), visual_response(RELATIONS)])
    base = pair(bundle, tmp_path / "base", "visual")
    review = {"version": 1, "reviewer": "synthetic", "findings": [{"package_id": base.source.package_ids[0],
        "reference_kind": "item", "reference_id": "n4", "finding": "text_mismatch", "detail": "Verify the isolated label"}]}
    delivery = pair(bundle, tmp_path / "reviewed", "visual", review)
    db = tmp_path / "db"
    SQLiteRelationReceiver(db).receive(delivery, collection="images")
    report = review_received_relations(db, collection="images", reason="recorded_review")
    assert report["total"] == 1
    item = report["items"][0]
    assert item["trace_refs"] == [] and item["referenced_record"]["label"] == "Unconnected"
    assert item["referenced_record"]["x0"] == 0 and item["referenced_record"]["x1"] == 500


@pytest.mark.parametrize("kwargs", [{"include_denials": 1}, {"reason": "risk_score"}, {"limit": False}, {"offset": -1}])
def test_invalid_options_do_not_open_or_create_a_database(tmp_path, kwargs):
    db = tmp_path / "absent.db"
    with pytest.raises(ValueError):
        review_received_relations(db, collection="documents", **kwargs)
    assert not db.exists()


def test_corrupt_receiver_is_not_a_clean_review(reviewed_db, delivered, capsys):
    with sqlite3.connect(reviewed_db) as con:
        con.execute(f'DELETE FROM "srd_{delivered.search.snapshot_id}_cards"')
    assert main(["--database", str(reviewed_db), "--collection", "documents"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and "items" not in result


def test_cli_filters_and_counts_are_machine_readable(reviewed_db, capsys):
    assert main(["--database", str(reviewed_db), "--collection", "documents", "--reason", "unknown_object"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total"] == 1 and result["ruleset"] == "relation_review/1"
