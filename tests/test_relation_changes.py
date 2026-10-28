"""Explicit source comparisons preserve multiplicity and evidence, not entity joins."""
from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from sr_adapter.narrative import render_narrative, save_narrative
from sr_adapter.relation_changes import compare_received_sources, main
from sr_adapter.relation_receiver import SQLiteRelationReceiver, trace_received_relation
from sr_adapter.relation_sources import read_source
from sr_adapter.visual_runtime import extract_visual
from test_fabric_narrative import narrative as narrative
from test_narrative import ontology as ontology, response
from test_relation_receiver import delivered as delivered, revision, pair
from test_visual_relations import source as source, run, INVENTORY, RELATIONS, response as visual_response


def text_version(root, ontology, *, owner="B", status="asserted", context="Approval required.", model="fixture", evidence_order=(0,)):
    data = f"A is managed by {owner}.\n\n{context}".encode()
    units = read_source(data, "source.txt")["units"]
    claim = {"subject": "A", "predicate": "managedBy", "object": owner if status in {"asserted", "denied"} else None,
        "status": status, "evidence": [{"unit_id": units[i]["id"], "quote": units[i]["text"]} for i in evidence_order]}
    manifest, files = render_narrative(data=data, source_name="source.txt", ontology=ontology, model=model, response=response([claim]))
    save_narrative(root / "source", manifest, files)
    return pair(root / "source", root)


def publish_pair(db, before, after, collection="documents"):
    receiver = SQLiteRelationReceiver(db)
    receiver.receive(before, collection=collection)
    receiver.receive(after, collection=collection, expected_snapshot=before.source.snapshot_id)
    return {"collection": collection, "before_snapshot": before.source.snapshot_id, "after_snapshot": after.source.snapshot_id,
            "before_package": before.source.package_ids[0], "after_package": after.source.package_ids[0]}


def test_changed_owner_returns_two_sourced_readings_not_an_inferred_replacement(ontology, tmp_path):
    before = text_version(tmp_path / "before", ontology)
    after = text_version(tmp_path / "after", ontology, owner="C")
    db = tmp_path / "db"
    args = publish_pair(db, before, after)
    original = db.read_bytes()
    report = compare_received_sources(db, **args)
    assert report["source_bytes_changed"] and report["source_context_changed"]
    assert report["counts"] == {"before_count": 1, "after_count": 1, "shared_count": 0, "before_only_count": 1, "after_only_count": 1}
    assert {g["classification"] for g in report["groups"]} == {"before_only", "after_only"}
    assert not report["identity_resolved"]
    for group in report["groups"]:
        refs = group["before_references"] or group["after_references"]
        trace = trace_received_relation(db, **refs[0]["trace_ref"])
        assert trace["assertion"]["properties"]["object_label"] == group["content"]["relation"]["object_label"]
        assert trace["evidence"][0]["properties"]["text"] == group["content"]["evidence"][0]["record"]["text"]
    assert db.read_bytes() == original


def test_changed_uncited_condition_is_visible_even_when_relation_values_match(ontology, tmp_path):
    before = text_version(tmp_path / "before", ontology)
    after = text_version(tmp_path / "after", ontology, context="Approval required only after June.")
    args = publish_pair(tmp_path / "db", before, after)
    report = compare_received_sources(tmp_path / "db", **args)
    assert report["total"] == 0 and report["counts"]["shared_count"] == 1
    assert report["source_context_changed"]
    context = compare_received_sources(tmp_path / "db", **args, scope="context")
    texts = {g["content"]["record"]["text"] for g in context["groups"]}
    assert texts == {"Approval required.", "Approval required only after June."}


def test_new_package_ids_and_model_metadata_do_not_invent_content_changes(delivered, narrative, tmp_path):
    after = revision(narrative, tmp_path / "after", model="different-deployment")
    args = publish_pair(tmp_path / "db", delivered, after)
    report = compare_received_sources(tmp_path / "db", **args, include_unchanged=True)
    assert report["source_metadata_changed"] and not report["source_bytes_changed"]
    assert not report["source_context_changed"] and report["counts"]["shared_count"] == 2
    assert all(g["classification"] == "unchanged_values" for g in report["groups"])
    denied = next(g for g in report["groups"] if g["content"]["relation"]["assertion_status"] == "denied")
    assert any(loc["path"] == "/header_context/0/text" for e in denied["content"]["evidence"] for loc in e["locations"])
    assert denied["before_references"][0]["graph_node_id"] != denied["after_references"][0]["graph_node_id"]
    assert denied["content"]["mentions"]["subject"]["literal_match"] is False


@pytest.mark.parametrize("status", ["denied", "unknown", "ambiguous"])
def test_status_change_is_never_normalized_into_a_positive_edge(ontology, tmp_path, status):
    # Saved candidate variation on the same text tests the reading boundary,
    # not whether either synthetic model interpretation is semantically right.
    before = text_version(tmp_path / "before", ontology)
    after = text_version(tmp_path / "after", ontology, status=status)
    args = publish_pair(tmp_path / "db", before, after)
    report = compare_received_sources(tmp_path / "db", **args)
    assert not report["source_bytes_changed"] and report["total"] == 2
    changed = next(g for g in report["groups"] if g["classification"] == "after_only")
    assert changed["content"]["relation"]["assertion_status"] == status
    if status != "denied":
        assert changed["content"]["relation"]["object_label"] is None
        assert changed["content"]["mentions"]["object"] is None


def test_evidence_order_is_part_of_the_comparison(ontology, tmp_path):
    before = text_version(tmp_path / "before", ontology, evidence_order=(0, 1))
    after = text_version(tmp_path / "after", ontology, evidence_order=(1, 0))
    args = publish_pair(tmp_path / "db", before, after)
    report = compare_received_sources(tmp_path / "db", **args)
    assert report["total"] == 2 and not report["source_context_changed"]
    orders = [g["content"]["evidence"][0]["record"]["text"] for g in report["groups"]]
    assert set(orders) == {"A is managed by B.", "Approval required."}


def test_same_label_visual_occurrences_and_duplicate_readings_keep_multiplicity(source, tmp_path):
    run(source, tmp_path / "one")
    twice = {"relations": [*RELATIONS["relations"], {**RELATIONS["relations"][0], "id": "r2"}], "limitations": []}
    run(source, tmp_path / "two", responses=[visual_response(INVENTORY), visual_response(twice)])
    before, after = pair(tmp_path / "one", tmp_path / "before", "visual"), pair(tmp_path / "two", tmp_path / "after", "visual")
    args = publish_pair(tmp_path / "db", before, after, "images")
    report = compare_received_sources(tmp_path / "db", **args)
    group, = report["groups"]
    assert group["classification"] == "count_changed"
    assert (group["before_count"], group["after_count"], group["shared_count"], group["after_only_count"]) == (1, 2, 1, 1)
    assert len({r["card_id"] for r in group["after_references"]}) == 2 and not group["identity_resolved"]
    assert group["content"]["mentions"]["subject"]["label"] == group["content"]["mentions"]["object"]["label"]
    assert group["content"]["mentions"]["subject"]["x0"] != group["content"]["mentions"]["object"]["x0"]
    assert group["content"]["relation"]["value_text"] == "0%"


def test_visual_direction_boxes_and_review_details_are_compared(source, tmp_path):
    run(source, tmp_path / "one")
    updated = deepcopy(RELATIONS)
    updated["relations"][0].update(direction="bidirectional", support="explicit", box={"x0": 0, "y0": 0, "x1": 750, "y1": 500})
    run(source, tmp_path / "two", responses=[visual_response(INVENTORY), visual_response(updated)])
    before, after = pair(tmp_path / "one", tmp_path / "before", "visual"), pair(tmp_path / "two", tmp_path / "after", "visual")
    args = publish_pair(tmp_path / "db", before, after, "images")
    report = compare_received_sources(tmp_path / "db", **args)
    change = next(g for g in report["groups"] if g["classification"] == "after_only")
    assert change["content"]["relation"]["direction"] == "bidirectional"
    assert change["content"]["relation"]["x1"] == 750 and report["total"] == 2
    review = {"version": 1, "reviewer": "synthetic", "findings": [{"package_id": after.source.package_ids[0],
        "reference_kind": "relation", "reference_id": "r1", "finding": "ambiguous_relation", "detail": "Inspect this connector"}]}
    reviewed = pair(tmp_path / "two", tmp_path / "reviewed", "visual", review)
    SQLiteRelationReceiver(tmp_path / "db").receive(reviewed, collection="images", expected_snapshot=after.source.snapshot_id)
    updated_args = {**args, "before_snapshot": after.source.snapshot_id, "before_package": after.source.package_ids[0],
                    "after_snapshot": reviewed.source.snapshot_id, "after_package": reviewed.source.package_ids[0]}
    report = compare_received_sources(tmp_path / "db", **updated_args)
    assert report["source_context_changed"] and report["total"] == 2
    change = next(g for g in report["groups"] if g["classification"] == "after_only")
    assert change["content"]["reviews"][0]["record"]["detail"] == "Inspect this connector"


def test_empty_extraction_does_not_hide_context_changes(narrative, tmp_path):
    before = revision(narrative, tmp_path / "before", empty=True)
    after = revision(narrative, tmp_path / "after", empty=True, model="different")
    args = publish_pair(tmp_path / "db", before, after)
    report = compare_received_sources(tmp_path / "db", **args)
    assert report["counts"]["before_count"] == report["counts"]["after_count"] == 0
    context = compare_received_sources(tmp_path / "db", **args, scope="context", include_unchanged=True)
    assert context["total"] > 0 and any(g["content"].get("record", {}).get("text") == "Only applies after approval." for g in context["groups"])


def test_review_moved_between_same_label_reversed_regions_changes_context(source, tmp_path):
    readings = {"relations": [*RELATIONS["relations"],
        {**RELATIONS["relations"][0], "id": "r2", "subject_id": "n2", "object_id": "n1"}], "limitations": []}
    run(source, tmp_path / "image", responses=[visual_response(INVENTORY), visual_response(readings)])
    base = pair(tmp_path / "image", tmp_path / "base", "visual")
    versions = []
    for ref in ("r1", "r2"):
        review = {"version": 1, "reviewer": "synthetic", "findings": [{"package_id": base.source.package_ids[0],
            "reference_kind": "relation", "reference_id": ref, "finding": "ambiguous_relation", "detail": "Inspect connector"}]}
        versions.append(pair(tmp_path / "image", tmp_path / ref, "visual", review))
    args = publish_pair(tmp_path / "db", *versions, collection="images")
    report = compare_received_sources(tmp_path / "db", **args, scope="context")
    assert report["source_context_changed"] and report["total"] == 2
    targets = [g["content"]["referenced_values"] for g in report["groups"]]
    assert {target["mentions"]["subject"]["x0"] for target in targets} == {0, 500}
    assert all(len(target["evidence"]) == 3 for target in targets)


def test_image_review_ignores_deployment_metadata_in_content_fingerprints(source, tmp_path):
    versions = []
    for suffix in ("old", "new"):
        responses = [visual_response(INVENTORY, model="response-" + suffix), visual_response(RELATIONS, model="response-" + suffix)]
        def handle(request):
            return httpx.Response(200, json=responses.pop(0))
        bundle = tmp_path / suffix
        extract_visual(source, bundle, endpoint=f"https://example.services.ai.azure.com/api/projects/{suffix}",
            model="deployment-" + suffix, page_label="7", execute=True,
            credential=SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="test")), transport=httpx.MockTransport(handle))
        base = pair(bundle, tmp_path / (suffix + "-base"), "visual")
        review = {"version": 1, "reviewer": "synthetic", "findings": [{"package_id": base.source.package_ids[0],
            "reference_kind": "image", "reference_id": None, "finding": "coverage_gap", "detail": "Inspect remaining content"}]}
        versions.append(pair(bundle, tmp_path / (suffix + "-reviewed"), "visual", review))
    args = publish_pair(tmp_path / "db", *versions, collection="images")
    report = compare_received_sources(tmp_path / "db", **args)
    assert report["source_metadata_changed"] and not report["source_bytes_changed"]
    assert report["total"] == 0 and report["counts"]["shared_count"] == 1
    assert not report["source_context_changed"]


def test_source_selection_paging_and_cli(ontology, tmp_path, capsys):
    before = text_version(tmp_path / "before", ontology)
    after = text_version(tmp_path / "after", ontology, owner="C")
    db = tmp_path / "db"
    args = publish_pair(db, before, after)
    page1 = compare_received_sources(db, **args, limit=1)
    page2 = compare_received_sources(db, **args, limit=1, offset=1)
    assert page1["has_more"] and not page2["has_more"]
    assert page1["groups"][0]["content_fingerprint"] != page2["groups"][0]["content_fingerprint"]
    for key, value in [("before_package", "source.txt"), ("after_package", before.source.package_ids[0])]:
        with pytest.raises(ValueError, match="does not belong"):
            compare_received_sources(db, **{**args, key: value})
    cli = ["--database", str(db)]
    for key, value in args.items():
        cli.extend(["--" + key.replace("_", "-"), value])
    assert main(cli) == 0 and json.loads(capsys.readouterr().out)["total"] == 2
    with sqlite3.connect(db) as con:
        con.execute(f'DELETE FROM "srd_{before.search.snapshot_id}_cards"')
    assert main(cli) == 1 and json.loads(capsys.readouterr().out)["error"] == "comparison_unavailable"


@pytest.mark.parametrize("change", [{"before_snapshot": None}, {"after_snapshot": "active"}, {"before_package": None},
    {"include_unchanged": 1}, {"limit": True}, {"offset": -1}, {"scope": "business_impact"}])
def test_explicit_snapshot_and_option_validation_before_reading(tmp_path, change):
    db = tmp_path / "absent.db"
    args = {"collection": "documents", "before_snapshot": "0" * 64, "after_snapshot": "1" * 64,
            "before_package": "p1", "after_package": "p2", **change}
    with pytest.raises(ValueError):
        compare_received_sources(db, **args)
    assert not db.exists()
