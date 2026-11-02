"""Lossless sharing must not collapse evidence scopes, types or repeated mentions."""
from copy import deepcopy
import hashlib
import json

import pytest

from sr_adapter.ontology_context import pack_ontology_context, unpack_ontology_context


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def inspection(identity="c1", name="policy.pdf", snapshot="a" * 64):
    rows = [{"unit_id": f"unit-{i}", "package_id": "p" * 64, "source_sha256": "f" * 64,
             "source_name": name, "text": "審査部\x00はAtlasを確認する。", "maybe": None,
             "typed": value} for i, value in enumerate((True, 1, 1.0, False, 0))]
    return {"status": "complete", "tool": "inspect_ontology", "tool_contract_version": 1,
        "reference": {"collection": "default", "name": name, "snapshot_id": snapshot, "kind": "claim", "id": identity},
        "active_snapshot_id": snapshot, "guidance": "Stored candidates need review.",
        "value": {"claim_id": identity, "source_context": {"evidence": rows, "locations": [], "nulls": [{"empty": None}]},
                  "terms": [{"iri": "urn:test:Atlas", "label": "Atlas", "kind": "role", "description": "definition"}],
                  "axioms": [], "roles": [], "claims": [{"id": identity, "status": "unknown"}],
                  "argument_anchors": [{"exact": "Atlas", "start": 5, "end": 10, "occurrence": 0}],
                  "vocabulary_review": {"review_required": True, "archives": ["a" * 64]}}}


def rehash(record):
    record["id"] = "ctx_" + hashlib.sha256(canonical({k: v for k, v in record.items() if k != "id"})).hexdigest()


def test_context_sharing_is_lossless_and_reduces_repeated_evidence():
    originals = [inspection(f"c{i}") for i in range(12)]
    before = deepcopy(originals)
    packed = pack_ontology_context(originals)
    assert originals == before and unpack_ontology_context(packed) == originals
    assert canonical(unpack_ontology_context(packed)) == canonical(originals)
    assert len(packed["shared"]) == 4
    assert len(canonical(packed)) < len(canonical(originals)) * 0.6
    table = next(r for r in packed["shared"] if r["field"] == "source_context")["data"]["evidence"]
    assert table["constants"]["source_sha256"] == "f" * 64
    assert "typed" in table["columns"]
    values = [row[table["columns"].index("typed")] for row in table["rows"]]
    assert [type(value) for value in values] == [bool, int, float, bool, int]
    restored = unpack_ontology_context(packed)
    assert restored[0]["value"]["source_context"]["evidence"][0]["text"] == "審査部\x00はAtlasを確認する。"
    restored[0]["value"]["terms"][0]["label"] = "changed"
    assert restored[1]["value"]["terms"][0]["label"] == "Atlas"


def test_same_iri_or_label_in_different_scopes_is_never_shared():
    a = inspection("c1")
    b = inspection("c1", name="policy.docx")
    c = inspection("c1", snapshot="b" * 64)
    d = deepcopy(a)
    d["reference"]["collection"] = "different"
    packed = pack_ontology_context([a, b, c, d])
    terms = [r for r in packed["shared"] if r["field"] == "terms"]
    assert len(terms) == 4 and len({r["id"] for r in terms}) == 4
    assert unpack_ontology_context(packed) == [a, b, c, d]


def test_irregular_json_tables_fall_back_without_losing_missing_or_null_fields():
    result = inspection()
    result["value"]["source_context"] = {"rows": [{"a": None}, {"b": None}], "extra": "metadata"}
    packed = pack_ontology_context([result])
    record = next(r for r in packed["shared"] if r["field"] == "source_context")
    assert record["encoding"] == "json"
    assert unpack_ontology_context(packed) == [result]


@pytest.mark.parametrize("change", ["missing", "duplicate", "content", "scope", "field", "unused", "row_shape", "overlap"])
def test_tampered_context_is_rejected(change):
    packed = pack_ontology_context([inspection()])
    first = packed["shared"][0]
    if change == "missing":
        packed["shared"].pop(0)
    elif change == "duplicate":
        packed["shared"].append(deepcopy(first))
    elif change == "content":
        first["data"]["evidence"]["constants"]["text"] = "changed"
    elif change in {"scope", "field", "row_shape"}:
        old = first["id"]
        if change == "scope":
            first["scope"]["name"] = "another.pdf"
        elif change == "field":
            first["field"] = "terms"
            first["encoding"] = "json"
        else:
            first["data"]["evidence"]["rows"][0].append("extra")
        rehash(first)
        packed["entries"][0]["shared_fields"]["source_context"] = first["id"]
        assert old != first["id"]
    elif change == "unused":
        extra = deepcopy(first)
        extra["scope"]["name"] = "unused"
        rehash(extra)
        packed["shared"].append(extra)
    else:
        packed["entries"][0]["inspection"]["value"]["source_context"] = {}
    with pytest.raises(ValueError):
        unpack_ontology_context(packed)


@pytest.mark.parametrize("reference_change", [{"snapshot_id": "latest"}, {"name": None}, {"kind": "all"}, {"extra": "x"}])
def test_packing_requires_complete_pinned_inspections(reference_change):
    result = inspection()
    result["reference"].update(reference_change)
    with pytest.raises(ValueError):
        pack_ontology_context([result])


def test_budgets_prevent_oversize_or_expansion_without_truncation(monkeypatch):
    import sr_adapter.ontology_context as codec

    with pytest.raises(ValueError):
        pack_ontology_context([inspection()] * 101)
    packed = pack_ontology_context([inspection("c1"), inspection("c2")])
    monkeypatch.setattr(codec, "MAX_RAW_BYTES", 1024)
    with pytest.raises(ValueError, match="byte budget"):
        unpack_ontology_context(packed)
    with pytest.raises(ValueError, match="byte budget"):
        pack_ontology_context([inspection()])


def test_empty_context_and_non_json_values():
    assert unpack_ontology_context(pack_ontology_context([])) == []
    result = inspection()
    result["value"]["tuple"] = (1, 2)
    with pytest.raises(ValueError, match="JSON"):
        pack_ontology_context([result])
    result["value"].pop("tuple")
    result["value"]["float"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pack_ontology_context([result])


def test_whole_result_digest_detects_a_removed_inspection():
    packed = pack_ontology_context([inspection("c1"), inspection("c2")])
    packed["entries"].pop()
    with pytest.raises(ValueError, match="differs from the original"):
        unpack_ontology_context(packed)
