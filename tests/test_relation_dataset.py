"""Offline, synthetic conformance tests for reusable relation handoffs."""
from dataclasses import FrozenInstanceError
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.cli import main
from sr_adapter.fabric import verify_fabric
from sr_adapter.fabric_narrative import export_narrative_fabric
from sr_adapter.fabric_visual import export_visual
from sr_adapter.knowledge import sha256
from sr_adapter.narrative import render_narrative, save_narrative, verify_narrative, NarrativeOntology
from sr_adapter.relation_dataset import Column, RelationDataset, RelationTable
from sr_adapter.relation_dataset_store import (export_relation_dataset, load_relation_dataset,
    verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.tables import canonical_json
from test_fabric_narrative import narrative as narrative
from test_narrative import response
from test_visual_relations import source as source, run, TARGET


@pytest.fixture
def visual(source, tmp_path):
    out = tmp_path / "visual"
    run(source, out)  # MockTransport only; no live model or provider.
    return out


def make_table(values):
    return RelationTable.from_rows("records", [Column("package_id"), Column("record_id"),
        Column("flag", "boolean"), Column("count", "integer"), Column("value", nullable=True)],
        ("package_id", "record_id"), values)


def test_typed_snapshot_is_immutable_and_order_independent():
    a = {"package_id": "one", "record_id": "a", "flag": False, "count": 0, "value": None}
    b = {**a, "record_id": "b", "value": ""}
    table = make_table([b, a])
    dataset = RelationDataset("test_records/1", ("one",), (table,))
    original = dataset.snapshot_id
    a["value"] = "changed"
    rows = table.rows()
    assert rows[0]["value"] is None and rows[1]["value"] == ""
    rows[0]["value"] = "changed too"
    assert dataset.snapshot_id == original
    assert make_table(reversed(table.rows())) == table
    assert RelationDataset("test_records/2", ("one",), (table,)).snapshot_id != original
    with pytest.raises(FrozenInstanceError):
        dataset.profile = "other/1"


@pytest.mark.parametrize("change", [{"flag": 0}, {"count": True}, {"count": 2**63},
    {"value": {"nested": "json"}}, {"extra": "must not be dropped"}, {"record_id": None}])
def test_invalid_scalar_contract_is_not_silently_coerced(change):
    row = {"package_id": "one", "record_id": "a", "flag": False, "count": 0, "value": None, **change}
    with pytest.raises(ValueError):
        make_table([row])


def test_duplicates_and_unknown_source_references_rejected():
    row = {"package_id": "one", "record_id": "a", "flag": True, "count": 1, "value": "text"}
    with pytest.raises(ValueError, match="duplicate"):
        make_table([row, row])
    with pytest.raises(ValueError, match="unselected"):
        RelationDataset("test_records/1", ("other",), (make_table([row]),))


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_fabric_and_portable_sink_share_exact_schemas_rows_and_evidence(profile, request, tmp_path):
    source = request.getfixturevalue(profile)
    portable = tmp_path / "portable"
    result = export_relation_dataset([source], portable, profile=profile)
    dataset = verify_relation_dataset(portable)
    assert result["snapshot_id"] == dataset.snapshot_id
    assert "target.json" not in result["files"] and "deployment-plan.json" not in result["files"]
    old = tmp_path / "fabric"
    exporter = export_narrative_fabric if profile == "narrative" else export_visual
    exporter([source], old, target=TARGET)
    _, files, _ = verify_fabric(old, target=TARGET)
    for table in dataset.tables:
        legacy = pq.read_table(pa.BufferReader(files[f"tables/{table.name}.parquet"]))
        assert legacy.schema == table.arrow_schema()
        assert legacy.to_pylist() == table.rows()
    db = tmp_path / "relations.db"
    first = write_dataset_sqlite(portable, db)
    assert first["inserted_rows"] == sum(len(t.values) for t in dataset.tables)
    assert write_dataset_sqlite(portable, db)["inserted_rows"] == 0
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute(f'SELECT * FROM "{first["tables"]["relations"]["name"]}"')]
        if profile == "narrative":
            relations = {r["assertion_status"]: r for r in rows}
            assert relations["unknown"]["object_label"] is None
            assert relations["denied"]["subject_label_match"] == 0
            evidence = first["tables"]["evidence"]["name"]
            assert con.execute(f'SELECT text FROM "{evidence}" WHERE unit_order=1').fetchone()[0] == "Only applies after approval."
            assert len({r["subject_id"] for r in rows}) == 2  # Never join mentions by name.
        else:
            assert rows[0]["subject_id"] != rows[0]["object_id"]
            assert rows[0]["subject_label"] == rows[0]["object_label"]
            assert (rows[0]["direction"], rows[0]["support"], rows[0]["value_text"]) == ("unspecified", "uncertain", "0%")
            assert rows[0]["x1"] == 500


def test_visual_review_is_preserved_without_replacing_raw_readings(visual, tmp_path):
    original, _ = load_relation_dataset([visual], profile="visual")
    review = {"version": 1, "reviewer": "synthetic reviewer", "findings": [
        {"package_id": original.package_ids[0], "reference_kind": "item", "reference_id": "n1",
         "finding": "text_mismatch", "detail": "Recheck this label."}]}
    out = tmp_path / "reviewed"
    export_relation_dataset([visual], out, profile="visual", review=review)
    dataset = verify_relation_dataset(out)
    assert dataset.snapshot_id != original.snapshot_id
    assert dataset.rows()["relations"][0]["subject_label"] == original.rows()["relations"][0]["subject_label"]
    assert dataset.rows()["relations"][0]["review_status"] == "model_candidate_needs_review"
    assert dataset.rows()["reviews"][0]["detail"] == "Recheck this label."
    old = tmp_path / "fabric-reviewed"
    export_visual([visual], old, target=TARGET, review=review)
    for table in dataset.tables:
        assert pq.read_table(old / f"tables/{table.name}.parquet").to_pylist() == table.rows()
    write_dataset_sqlite(out, tmp_path / "reviewed.db")


def test_empty_claims_still_export_context_and_typed_empty_tables(narrative, tmp_path):
    _, files = verify_narrative(narrative)
    manifest, payloads = render_narrative(data=files["source.bin"], source_name="empty.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="test-model",
        response=response([]))
    empty = tmp_path / "empty"
    save_narrative(empty, manifest, payloads)
    out = tmp_path / "empty-dataset"
    export_relation_dataset([empty], out, profile="narrative")
    dataset = verify_relation_dataset(out)
    assert dataset.rows()["relations"] == []
    assert dataset.rows()["evidence"] and dataset.rows()["documents"]
    assert write_dataset_sqlite(out, tmp_path / "empty.db")["tables"]["relations"]["rows"] == 0


def test_package_order_does_not_change_identity_and_sources_are_not_merged(narrative, tmp_path):
    _, files = verify_narrative(narrative)
    manifest, payloads = render_narrative(data=files["source.bin"], source_name="second.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="test-model",
        response=json.loads(files["response.json"]), label_policy="preserve")
    second = tmp_path / "second"
    save_narrative(second, manifest, payloads)
    a, _ = load_relation_dataset([narrative, second], profile="narrative")
    b, _ = load_relation_dataset([second, narrative], profile="narrative")
    assert a == b and a.snapshot_id == b.snapshot_id
    assert len(a.rows()["documents"]) == 2 and len(a.rows()["relations"]) == 4
    assert len({r["subject_id"] for r in a.rows()["relations"]}) == 4


def test_wrong_profile_and_duplicate_inputs_fail_before_writing(narrative, visual, tmp_path):
    for profile, bundles in [("missing", [narrative]), ("narrative", [visual]),
                             ("visual", [narrative]), ("narrative", [narrative, narrative])]:
        out = tmp_path / "invalid"
        with pytest.raises((ValueError, KeyError)):
            export_relation_dataset(bundles, out, profile=profile)
        assert not out.exists()
    with pytest.raises(ValueError, match="review"):
        export_relation_dataset([narrative], tmp_path / "bad-review", profile="narrative", review={})


def rehash(root, name):
    manifest = json.loads((root / "manifest.json").read_bytes())
    data = (root / name).read_bytes()
    manifest["files"][name] = {"bytes": len(data), "sha256": sha256(data)}
    (root / "manifest.json").write_bytes(canonical_json(manifest))


def test_rehashed_semantic_change_is_detected_by_source_replay(narrative, tmp_path):
    out = tmp_path / "portable"
    export_relation_dataset([narrative], out, profile="narrative")
    path = out / "tables/relations.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    next(row for row in rows if row["assertion_status"] == "denied")["assertion_status"] = "asserted"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    rehash(out, "tables/relations.parquet")
    with pytest.raises(ValueError, match="replayed"):
        write_dataset_sqlite(out, tmp_path / "never-created.db")
    assert not (tmp_path / "never-created.db").exists()


@pytest.mark.parametrize("field,value", [("contract_version", 2), ("contract_version", True),
    ("profile", "narrative_relations/2"), ("status", "running")])
def test_unsupported_or_incomplete_bundle_is_rejected(narrative, tmp_path, field, value):
    out = tmp_path / "portable"
    export_relation_dataset([narrative], out, profile="narrative")
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest[field] = value
    (out / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="unsupported"):
        verify_relation_dataset(out)


def test_sqlite_rollback_does_not_leave_half_a_snapshot(narrative, tmp_path):
    out, db = tmp_path / "portable", tmp_path / "existing.db"
    export_relation_dataset([narrative], out, profile="narrative")
    dataset = verify_relation_dataset(out)
    collision = f"srd_{dataset.snapshot_id}_{dataset.tables[-1].name}"
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE keep_me (value TEXT)')
        con.execute("INSERT INTO keep_me VALUES ('existing data')")
        con.execute(f'CREATE TABLE "{collision}" (value TEXT)')
    with pytest.raises(sqlite3.OperationalError):
        write_dataset_sqlite(out, db)
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT value FROM keep_me').fetchone() == ("existing data",)
        assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"keep_me", collision}


def test_sqlite_repeat_rejects_missing_rows_without_repair(narrative, tmp_path):
    out, db = tmp_path / "portable", tmp_path / "existing.db"
    export_relation_dataset([narrative], out, profile="narrative")
    result = write_dataset_sqlite(out, db)
    table = result["tables"]["relations"]["name"]
    with sqlite3.connect(db) as con:
        con.execute(f'DELETE FROM "{table}"')
    with pytest.raises(ValueError, match="complete dataset"):
        write_dataset_sqlite(out, db)
    with sqlite3.connect(db) as con:
        assert con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0


def test_cli_needs_no_cloud_target_and_preserves_existing_export(narrative, tmp_path, capsys):
    out, db = tmp_path / "portable", tmp_path / "example.db"
    args = ["knowledge", "dataset", str(narrative), "--profile", "narrative", "--out", str(out)]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "complete"
    assert main(["knowledge", "dataset-sqlite", str(out), "--database", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "committed"
    assert main(args) == 1
    assert verify_relation_dataset(out).snapshot_id == first["snapshot_id"]
