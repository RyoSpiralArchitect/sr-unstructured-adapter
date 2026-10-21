import json
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.cli import main
from sr_adapter.fabric import FabricTarget, verify_fabric
from sr_adapter.fabric_narrative import export_narrative_fabric
from sr_adapter.fabric_runtime import fabric_status, publish_fabric
from sr_adapter.knowledge import sha256
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_sources import read_source
from sr_adapter.tables import canonical_json
from test_docx_relation_sources import package, paragraph, table, row, cell
from test_fabric import FabricFixture
from test_narrative import response


@pytest.fixture
def target():
    return FabricTarget(workspace_id="11111111-1111-1111-1111-111111111111",
                        lakehouse_id="22222222-2222-2222-2222-222222222222", lakehouse_name="test")


@pytest.fixture
def narrative(tmp_path):
    data = package(paragraph("Only applies after approval.") + table([
        row([cell("Project"), cell("Owner")], '<w:tblHeader/>'),
        row([cell("A"), cell("B is not the owner")]),
    ]) + paragraph("A owner is unknown."))
    units = read_source(data, "source.docx")["units"]
    ontology = NarrativeOntology.model_validate({"namespace": "urn:test:", "classes": {"Project": "Project", "Team": "Team"},
        "predicates": [{"id": "ownedBy", "label": "Owner", "subject_type": "Project", "object_type": "Team", "description": "Owner"}]})
    candidates = [{"subject": "Project A", "predicate": "ownedBy", "object": "B", "status": "denied",
                   "evidence": [{"unit_id": u["id"], "quote": u["text"]} for u in units[3:5]]},
                  {"subject": "A", "predicate": "ownedBy", "object": None, "status": "unknown",
                   "evidence": [{"unit_id": units[-1]["id"], "quote": units[-1]["text"]}]}]
    manifest, files = render_narrative(data=data, source_name="source.docx", ontology=ontology,
        model="test-deployment", response=response(candidates), label_policy="preserve")
    out = tmp_path / "narrative"
    save_narrative(out, manifest, files)
    return out


def read(files, kind):
    return pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"])).to_pylist()


def test_docx_scalar_tables_keep_negation_context_locations_and_mentions(narrative, target, tmp_path):
    out = tmp_path / "fabric"
    export_narrative_fabric([narrative], out, target=target)
    manifest, files, plan = verify_fabric(out, target=target)
    assert manifest["mode"] == "narrative_tabular"
    for kind in manifest["tables"]:
        schema = pq.read_schema(pa.BufferReader(files[f"tables/{kind}.parquet"]))
        assert all(pa.types.is_string(f.type) or pa.types.is_integer(f.type) or pa.types.is_boolean(f.type) for f in schema)
        assert all(not f.name.endswith("_json") for f in schema)
    relations = {r["assertion_status"]: r for r in read(files, "relations")}
    assert relations["denied"]["subject_label"] == "Project A"
    assert relations["denied"]["subject_label_match"] is False
    assert relations["unknown"]["object_label"] is None
    assert relations["unknown"]["object_label_match"] is None
    evidence = {r["unit_id"]: r for r in read(files, "evidence")}
    entities = {r["entity_id"]: r for r in read(files, "entities")}
    assert len(entities) == 3  # No object entity invented for unknown.
    assert all(r["subject_id"] in entities for r in relations.values())
    assert all(r["unit_id"] in evidence for r in read(files, "evidence_links"))
    assert any(r["text"] == "Only applies after approval." for r in evidence.values())  # Uncited context survives.
    item = next(r for r in evidence.values() if r["text"] == "B is not the owner")
    assert (item["part"], item["table_index"], item["row_index"], item["column_index"], item["paragraph"]) == (
        "word/document.xml", 1, 2, 2, 1)
    locations = {(r["unit_id"], r["path"]): r["value_text"] for r in read(files, "locations")}
    assert locations[item["unit_id"], "/header_context/0/text"] == "Owner"
    assert locations[item["unit_id"], "/cell/vertical_merge"] == "none"
    assert all(t["row_count"] > 0 for t in plan["tables"])


def test_narrative_fabric_publish_and_exact_delta_readback(narrative, target, tmp_path):
    out, receipt = tmp_path / "fabric", tmp_path / "receipt.json"
    export_narrative_fabric([narrative], out, target=target)
    fixture = FabricFixture(target)
    kwargs = dict(target=target, receipt=receipt, credential=SimpleNamespace(get_token=lambda _: SimpleNamespace(token="fake")),
                  transport=httpx.MockTransport(fixture.handle))
    assert publish_fabric(out, **kwargs)["status"] == "pending"
    fixture.state = "Succeeded"
    result = fabric_status(out, **kwargs)
    assert result["status"] == "committed"
    assert all(r["verification"]["row_count"] == r["row_count"] for r in result["tables"])
    loads = len(fixture.loads)
    kwargs["receipt"] = tmp_path / "receipt2.json"
    assert publish_fabric(out, **kwargs)["status"] == "committed"
    assert len(fixture.loads) == loads


def test_rehashed_relation_tampering_rejects(narrative, target, tmp_path):
    out = tmp_path / "fabric"
    export_narrative_fabric([narrative], out, target=target)
    path = out / "tables/relations.parquet"
    stored = pq.read_table(path)
    rows = stored.to_pylist()
    rows[0]["assertion_status"] = "asserted"
    pq.write_table(pa.Table.from_pylist(rows, schema=stored.schema), path)
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest["files"]["tables/relations.parquet"] = {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}
    (out / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="table differs"):
        verify_fabric(out, target=target)


def test_narrative_cli_and_duplicate_selection(narrative, target, tmp_path, capsys):
    config = tmp_path / "target.json"
    config.write_text(target.model_dump_json())
    assert main(["knowledge", "fabric", str(narrative), "--mode", "narrative", "--target", str(config),
                 "--out", str(tmp_path / "fabric")]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "narrative_tabular"
    with pytest.raises(ValueError, match="Duplicate"):
        export_narrative_fabric([narrative, narrative], tmp_path / "duplicate", target=target)
    assert not (tmp_path / "duplicate").exists()


def test_empty_assertions_and_literal_legacy_bundle_are_supported(target, tmp_path):
    ontology = NarrativeOntology.model_validate({"namespace": "urn:test:", "classes": {"Thing": "Thing"},
        "predicates": [{"id": "relatedTo", "label": "Related", "subject_type": "Thing", "object_type": "Thing", "description": "Related"}]})
    manifest, files = render_narrative(data=b"Context with no supported relationships.", source_name="source.txt",
        ontology=ontology, model="test", response=response([]), instructions_version=1)
    save_narrative(tmp_path / "empty", manifest, files)
    export_narrative_fabric([tmp_path / "empty"], tmp_path / "fabric", target=target)
    _, files, plan = verify_fabric(tmp_path / "fabric", target=target)
    assert read(files, "relations") == []
    assert {t["kind"] for t in plan["tables"]} == {"documents", "evidence", "locations"}
    assert read(files, "documents")[0]["label_policy"] == "literal"
