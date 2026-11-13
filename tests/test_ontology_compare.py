from copy import deepcopy

import pytest

from sr_adapter.local_ontology import import_ontology, search_ontology
from sr_adapter.ontology_compare import (
    compare_ontology_claims,
    review_ontology_comparison,
)
from sr_adapter.ontology_context import unpack_ontology_context
from sr_adapter.ontology_tools import OntologyTools
from sr_adapter.relation_dataset_store import export_relation_semantics
from test_local_ontology import saved as saved_fixture, narrative  # noqa: F401

saved = saved_fixture


def test_compare_keeps_primary_dissent_and_both_complete_scopes(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    refs = [h["reference"] for h in search_ontology(db, kind="claim")["hits"]]
    before = db.read_bytes()
    result = compare_ontology_claims(db, left=refs[0], right=refs[1])
    assert [c["status"] for c in result["claims"]] == ["asserted", "denied"]
    assert {d["field"] for d in result["scope_differences"]} >= {
        "status",
        "attributed_to",
    }
    assert all(
        len(v["value"]["claims"]) == 2
        for v in unpack_ontology_context(result["evidence"])
    )
    assert (
        not result["contradiction_established"] and result["precedence"] == "unresolved"
    )
    assert db.read_bytes() == before
    reviewed = review_ontology_comparison(
        result, category="different_condition", explanation="Review candidate"
    )
    assert (
        reviewed["review_status"] == "unverified"
        and not reviewed["identity_established"]
    )


def test_versions_and_shared_names_never_choose_winner(saved, tmp_path):
    base, bundle, _, overlay = saved
    db = tmp_path / "db"
    old = import_ontology(bundle, db)
    left = search_ontology(db, kind="claim")["hits"][0]["reference"]
    changed = deepcopy(overlay)
    changed["terms"][3]["description"] = "A different source-local definition"
    changed["claims"][0]["context"] = "Unapproved proposed condition"
    export_relation_semantics(base, tmp_path / "new", overlay=changed)
    import_ontology(tmp_path / "new", db, expected_snapshot=old["snapshot_id"])
    right = search_ontology(db, kind="claim")["hits"][0]["reference"]
    result = compare_ontology_claims(db, left=left, right=right)
    assert result["scope_relation"] == "same_entry_versions"
    assert any(not row["declarations_equal"] for row in result["shared_labels"])
    assert result["precedence"] == "unresolved"
    assert any(d["field"] == "context" for d in result["scope_differences"])


def test_comparison_tool_is_allowlisted_and_bounded(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    refs = [h["reference"] for h in search_ontology(db, kind="claim")["hits"]]
    args = {
        "collection": "default",
        **{
            f"{side}_{k}": ref[k]
            for side, ref in zip(("left", "right"), refs)
            for k in ("name", "snapshot_id", "id")
        },
    }
    tools = OntologyTools(db, ("default",))
    result = tools.call("compare_ontology_claims", args)
    assert result["status"] == "complete" and "evidence" not in result
    assert (
        tools.call("compare_ontology_claims", {**args, "collection": "other"})["status"]
        == "failed"
    )
    with pytest.raises(ValueError):
        compare_ontology_claims(db, left={**refs[0], "kind": "term"}, right=refs[1])
    assert (
        OntologyTools(db, ("default",), max_response_bytes=1024).call(
            "compare_ontology_claims", args
        )["error"]["code"]
        == "response_too_large"
    )


def test_expression_arguments_preserve_conditions_without_importing_linked_scope(
    saved, tmp_path
):
    from test_relation_semantics import NS

    base, _, dataset, overlay = saved
    changed = deepcopy(overlay)
    changed["expressions"].append(
        {
            "id": "about-rule",
            "package_id": dataset.package_ids[0],
            "operator": "atom",
            "predicate": NS + "Event",
            "arguments": [
                {
                    "role": NS + "proposition",
                    "value": {"kind": "expression", "expression_id": "rule"},
                }
            ],
        }
    )
    changed["expressions"].append(
        {
            "id": "dissent-only",
            "package_id": dataset.package_ids[0],
            "operator": "atom",
            "predicate": NS + "Stopped",
            "arguments": [
                {
                    "role": NS + "actor",
                    "value": {"kind": "unknown", "reason": "unidentified"},
                }
            ],
        }
    )
    changed["claims"][0]["expression_id"] = "about-rule"
    changed["claims"][1]["expression_id"] = "dissent-only"
    export_relation_semantics(base, tmp_path / "nested", overlay=changed)
    db = tmp_path / "db"
    import_ontology(tmp_path / "nested", db)
    refs = [h["reference"] for h in search_ontology(db, kind="claim")["hits"]]
    result = compare_ontology_claims(
        db, left=refs[0], right=refs[1], include_evidence=False
    )
    assert {e["id"] for e in result["expressions"][0]} == {
        "about-rule",
        "rule",
        "eligible",
        "approved",
        "not-stopped",
        "stopped",
        "handled",
    }
    assert {e["id"] for e in result["expressions"][1]} == {"dissent-only"}
