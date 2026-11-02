"""Self-contained SQL memory: source/version isolation, evidence and model tools."""
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from sr_adapter.local_ontology import (import_ontology, inspect_ontology, list_ontologies,
    read_ontology_source, search_ontology)
from sr_adapter.ontology_tools import OntologyTools
from sr_adapter.relation_dataset_store import export_relation_dataset, export_relation_semantics, verify_relation_dataset
from sr_adapter.relation_receiver import ReceiverConflict
from sr_adapter.semantic_proposals import import_semantic_proposal
from test_semantic_proposals import proposed, raw
from test_relation_semantics import declaration, NS
from test_fabric_narrative import narrative as narrative
from test_relation_dataset import visual as visual, source as source
from test_semantic_anchors import narrative_anchors as narrative_anchors, anchored_declaration


@pytest.fixture
def saved(narrative, tmp_path):
    base, bundle = tmp_path / "base", tmp_path / "semantics"
    export_relation_dataset([narrative], base, profile="narrative")
    dataset = verify_relation_dataset(base)
    overlay = declaration(dataset)
    export_relation_semantics(base, bundle, overlay=overlay)
    return base, bundle, dataset, overlay


def test_db_alone_preserves_scoped_claim_vocabulary_and_original_source(saved, tmp_path):
    base, bundle, dataset, _ = saved
    db = tmp_path / "onto.db"
    first = import_ontology(bundle, db)
    assert first["outcome"] == "activated" and first["inserted_rows"] > 0
    assert import_ontology(bundle, db)["inserted_rows"] == 0
    origins = dataset.rows()["documents"]
    original = read_ontology_source(db, collection="default", name="source.docx", snapshot_id=first["snapshot_id"], package_id=origins[0]["package_id"])
    shutil.rmtree(base)
    shutil.rmtree(bundle)
    shutil.rmtree(tmp_path / "narrative")
    moved = tmp_path / "copied.db"
    shutil.copy2(db, moved)
    db.unlink()
    assert list_ontologies(moved)["entries"][0]["claim_count"] == 2
    found = search_ontology(moved, query="Handle", kind="claim")
    assert found["total"] == 2 and all(not h["scope_included"] for h in found["hits"])
    trace = inspect_ontology(moved, **found["hits"][0]["reference"])["value"]
    assert {claim["status"] for claim in trace["claims"]} == {"asserted", "denied"}
    assert {expr["operator"] for expr in trace["expressions"]} >= {"implies", "not", "and"}
    assert trace["source_context"] == dataset.rows()
    term = search_ontology(moved, query="Department", kind="term")["hits"][0]
    assert term["kind"] == "term" and term["term_kind"] == "class"
    assert inspect_ontology(moved, **term["reference"])["value"]["axioms"][0]["kind"] == "subclass"
    assert read_ontology_source(moved, collection="default", name="source.docx", snapshot_id=first["snapshot_id"], package_id=origins[0]["package_id"]) == original


def test_replacement_requires_expected_head_and_old_references_still_work(saved, tmp_path):
    base, bundle, dataset, overlay = saved
    db = tmp_path / "onto.db"
    old = import_ontology(bundle, db)
    reference = search_ontology(db, query="", kind="claim")["hits"][0]["reference"]
    overlay["author"] = "second version"
    newer = tmp_path / "new"
    export_relation_semantics(base, newer, overlay=overlay)
    with pytest.raises(ReceiverConflict):
        import_ontology(newer, db)
    current = import_ontology(newer, db, expected_snapshot=old["snapshot_id"])
    assert current["snapshot_id"] != old["snapshot_id"]
    assert inspect_ontology(db, **reference)["active_snapshot_id"] == current["snapshot_id"]
    with pytest.raises(ReceiverConflict):
        import_ontology(bundle, db)
    with pytest.raises(ValueError):
        inspect_ontology(db, **{**reference, "snapshot_id": None})
    with pytest.raises(ValueError):
        inspect_ontology(db, **{**reference, "collection": "another"})
    assert search_ontology(db, name="source.docx", snapshot_id=old["snapshot_id"], status="denied")["total"] == 1


def test_narrative_visual_and_proposal_can_coexist_without_identity_merges(saved, visual, tmp_path):
    base, bundle, dataset, _ = saved
    visual_base, visual_semantic = tmp_path / "visual-base", tmp_path / "visual-semantic"
    export_relation_dataset([visual], visual_base, profile="visual")
    export_relation_semantics(visual_base, visual_semantic, overlay=declaration(verify_relation_dataset(visual_base)))
    db = tmp_path / "onto.db"
    import_ontology(bundle, db)
    image = import_ontology(visual_semantic, db)
    proposal = tmp_path / "proposal"
    import_semantic_proposal(base, proposal, model="fixture", response=raw(proposed(dataset)))
    p = import_ontology(proposal, db, name="proposal")
    assert len(list_ontologies(db)["entries"]) == 3
    assert search_ontology(db, query=NS + "Handle", kind="term")["total"] == 3
    for result, source_dataset in [(p, dataset), (image, verify_relation_dataset(visual_base))]:
        original = read_ontology_source(db, collection="default", name=result["name"],
            snapshot_id=result["snapshot_id"], package_id=source_dataset.package_ids[0])
        assert original["data"]


def test_v2_exact_anchors_survive_db_only_retrieval(narrative_anchors, tmp_path):
    source = verify_relation_dataset(narrative_anchors)
    overlay = anchored_declaration(source)
    bundle, db = tmp_path / "semantic", tmp_path / "onto.db"
    export_relation_semantics(narrative_anchors, bundle, overlay=overlay)
    import_ontology(bundle, db)
    hit = search_ontology(db, query="審査部", kind="claim")["hits"][0]
    trace = inspect_ontology(db, **hit["reference"])["value"]
    assert {anchor["exact"] for anchor in trace["argument_anchors"]} == {"審査部", "申請A"}


def test_model_tools_are_allowlisted_bounded_and_version_pinned(saved, tmp_path):
    _, bundle, _, _ = saved
    db = tmp_path / "onto.db"
    import_ontology(bundle, db)
    tools = OntologyTools(db, ("default",))
    result = tools.call("search_ontology", {"collection": "default", "query": "Handle", "kind": "claim"})
    assert result["status"] == "complete"
    action = result["hits"][0]["inspect"]
    assert tools.call(action["name"], action["arguments"])["status"] == "complete"
    for name, args, code in [
        ("search_ontology", {"collection": "other", "query": ""}, "collection_not_allowed"),
        ("search_ontology", {"collection": "default", "query": "", "sql": "DROP TABLE sro_entries"}, "invalid_arguments"),
        ("search_ontology", {"collection": "default", "query": "", "limit": True}, "invalid_arguments"),
        ("inspect_ontology", {"collection": "default", "name": "source.docx", "kind": "claim", "id": "c1"}, "invalid_arguments"),
        ("ingest_ontology", {}, "unknown_tool")]:
        assert tools.call(name, args)["error"]["code"] == code
    small = OntologyTools(db, ("default",), max_response_bytes=1024)
    assert small.call(action["name"], action["arguments"])["error"]["code"] == "response_too_large"


@pytest.mark.parametrize("tamper", ["rows", "overlay", "source"])
def test_corruption_never_becomes_successful_evidence(saved, tmp_path, tamper):
    _, bundle, _, _ = saved
    db = tmp_path / "onto.db"
    result = import_ontology(bundle, db)
    reference = search_ontology(db, kind="claim")["hits"][0]["reference"]
    with sqlite3.connect(db) as connection:
        if tamper == "rows":
            connection.execute(f'UPDATE "srd_{result["snapshot_id"]}_claims" SET status=\'denied\'')
        elif tamper == "overlay":
            connection.execute("UPDATE sro_snapshots SET overlay='{}'")
        else:
            connection.execute("UPDATE sro_files SET data=? WHERE path LIKE '%source.bin'", (b"changed",))
    if tamper == "source":
        with pytest.raises(ValueError):
            read_ontology_source(db, collection="default", name=result["name"], snapshot_id=result["snapshot_id"], package_id=saved[2].package_ids[0])
    else:
        assert OntologyTools(db, ("default",)).call("inspect_ontology", reference)["error"]["code"] == "ontology_unavailable"


def test_archive_and_activation_roll_back_together(saved, tmp_path):
    db = tmp_path / "onto.db"
    def fail(connection, _):
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone()[0] == 1
        raise RuntimeError("receipt unavailable")
    with pytest.raises(RuntimeError):
        import_ontology(saved[1], db, _on_commit=fail)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name LIKE 'sro_%'").fetchall() == []


def test_readonly_reuse_does_not_import_arrow_or_open_external_sources(saved, tmp_path):
    db = tmp_path / "onto.db"
    import_ontology(saved[1], db)
    script = '''
import sys
class RejectArrow:
    def find_spec(self, fullname, *args):
        if fullname.startswith("pyarrow"):
            raise RuntimeError("Arrow is forbidden during reuse")
sys.meta_path.insert(0, RejectArrow())
from sr_adapter.local_ontology import search_ontology, inspect_ontology
hit = search_ontology(sys.argv[1], kind="claim")["hits"][0]
assert inspect_ontology(sys.argv[1], **hit["reference"])["status"] == "complete"
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c", script, str(db)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
