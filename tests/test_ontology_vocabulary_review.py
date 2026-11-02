"""Archive-associated declaration context remains visible to read-only model tools."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.local_ontology import import_ontology, inspect_ontology
from sr_adapter.ontology_ingest import _source_bundle, ingest_ontology_source
from sr_adapter.ontology_tools import OntologyTools
from sr_adapter.proposal_vocabulary import REGISTRY
from sr_adapter.relation_dataset_store import _save_snapshot
from sr_adapter.semantic_proposals import proposal_request, render_semantic_proposal
from sr_adapter.tables import canonical_json
from test_ontology_ingest import proposed
from test_proposal_vocabulary import reading


@pytest.fixture
def memory(tmp_path):
    source, db = tmp_path / "source.txt", tmp_path / "memory.sqlite"
    source.write_text("Review requires approval.")
    result = ingest_ontology_source(source, db, model="fixture", complete=reading, adapter_id="fixture", execute=True)
    assert result["status"] == "complete"
    source.unlink()
    return db, result


def test_model_search_and_inspection_preserve_separate_notes_and_related_claims(memory):
    db, result = memory
    tools = OntologyTools(db, ("default",))
    hit = tools.call("search_ontology", {"collection": "default", "query": "Table note.", "kind": "term"})["hits"][0]
    assert not hit["declarations_included"]
    term = tools.call(hit["inspect"]["name"], hit["inspect"]["arguments"])["value"]
    review = term["vocabulary_review"]
    assert review["review_required"] and review["snapshot_id"] == result["snapshot_id"]
    archive, = review["archives"]
    assert archive["archive_id"] == result["archive_id"] and archive["policy"] == REGISTRY
    group, = archive["terms"]
    assert group["iri"] == "urn:test:source" and group["review_required"] and group["repeated"]
    assert [(d["index"], d["description"]) for d in group["declarations"]] == [(1, "Table note."), (2, "Exception note.")]
    assert [d["pointer"] for d in group["declarations"]] == ["/overlay/terms/1", "/overlay/terms/2"]
    claim = tools.call("inspect_ontology", term["claim_references"][0])["value"]
    assert claim["vocabulary_review"]["review_required"]
    assert claim["claims"][0]["evidence"] and claim["source_context"]


def test_multiple_archives_and_other_documents_keep_distinct_scopes(tmp_path):
    db = tmp_path / "memory.sqlite"
    root = tmp_path / "inputs"
    root.mkdir()
    inputs = _source_bundle(b"Review requires approval.", "source.txt", root)
    wire = proposal_request(inputs, model="fixture", version=3)
    snapshots, archives = [], []
    for index, answer in enumerate((reading(wire), proposed(wire))):
        if index:
            content = answer["output"][0]["content"][0]
            data = json.loads(content["text"])
            data["overlay"]["terms"][1]["description"] = "Exception note.\n\nTable note."
            content["text"] = json.dumps(data)
        manifest, files = render_semantic_proposal(inputs, model="fixture", response=answer, version=3, vocabulary_policy=REGISTRY)
        bundle = tmp_path / str(index)
        _save_snapshot(bundle, files, manifest)
        result = import_ontology(bundle, db)
        snapshots.append(result["snapshot_id"])
        archives.append(result["archive_id"])
    other = tmp_path / "other.txt"
    other.write_text("An unrelated source document.")
    unrelated = ingest_ontology_source(other, db, model="fixture", complete=reading, adapter_id="fixture", execute=True)
    assert snapshots[0] == snapshots[1] != unrelated["snapshot_id"]
    term = inspect_ontology(db, collection="default", name="source.txt", snapshot_id=snapshots[0], kind="term", id="urn:test:source")["value"]
    page = term["vocabulary_review"]
    assert page["has_more"] and not page["complete"]
    records = list(page["archives"])
    action = page["next_inspect"]
    next_page = OntologyTools(db, ("default",)).call(action["name"], action["arguments"])["value"]["vocabulary_review"]
    assert not next_page["has_more"] and next_page["offset"] == 1
    records.extend(next_page["archives"])
    assert {r["archive_id"] for r in records} == set(archives)
    assert sorted(len(r["terms"][0]["declarations"]) for r in records) == [1, 2]
    assert all(r["archive_id"] != unrelated["archive_id"] for r in records)


def test_unused_vocabulary_does_not_mark_an_unrelated_claim_for_review(tmp_path):
    source, db = tmp_path / "source.txt", tmp_path / "memory.sqlite"
    source.write_text("Review requires approval.")
    def complete(wire):
        answer = proposed(wire)
        content = answer["output"][0]["content"][0]
        data = json.loads(content["text"])
        data["overlay"]["terms"].extend({"iri": "urn:test:unused", "kind": "role", "label": "unused", "description": note}
                                        for note in ("A", "B"))
        content["text"] = json.dumps(data)
        return answer
    result = ingest_ontology_source(source, db, complete=complete, adapter_id="fixture", execute=True)
    ref = {"collection": "default", "name": source.name, "snapshot_id": result["snapshot_id"]}
    assert inspect_ontology(db, **ref, kind="term", id="urn:test:unused")["value"]["vocabulary_review"]["review_required"]
    claim = inspect_ontology(db, **ref, kind="claim", id="c1")["value"]["vocabulary_review"]
    assert not claim["review_required"]
    assert all(g["iri"] != "urn:test:unused" for a in claim["archives"] for g in a["terms"])


@pytest.mark.parametrize("changed", ["manifest.json", "proposal.json", "response.json", "config.json", "descriptor",
    "rehashed_proposal", "rehashed_boundary_collapse", "rehashed_kind", "rehashed_unknown_kind"])
def test_altered_vocabulary_evidence_fails_closed(memory, changed):
    db, result = memory
    with sqlite3.connect(db) as connection:
        archive = result["archive_id"]
        if changed == "descriptor":
            connection.execute("UPDATE sro_archives SET descriptor='{}'")
        elif changed in {"rehashed_kind", "rehashed_unknown_kind"}:
            kind = "relation_semantics" if changed == "rehashed_kind" else "unrecognized"
            descriptor = json.loads(connection.execute("SELECT descriptor FROM sro_archives WHERE archive_id=?", (archive,)).fetchone()[0])
            descriptor["kind"] = kind
            raw = canonical_json(descriptor)
            new_archive = sha256(raw)
            connection.execute("UPDATE sro_archives SET archive_id=?,kind=?,descriptor=? WHERE archive_id=?", (new_archive, kind, raw.decode(), archive))
            connection.execute("UPDATE sro_files SET archive_id=? WHERE archive_id=?", (new_archive, archive))
        elif changed in {"rehashed_proposal", "rehashed_boundary_collapse"}:
            files = dict(connection.execute("SELECT path,data FROM sro_files WHERE archive_id=?", (archive,)))
            proposal = json.loads(files["proposal.json"])
            if changed == "rehashed_boundary_collapse":
                proposal["overlay"]["terms"][1]["description"] = "Exception note.\n\nTable note."
                proposal["overlay"]["terms"].pop()
            else:
                proposal["overlay"]["terms"][1]["description"] = "Changed interpretation"
            files["proposal.json"] = canonical_json(proposal)
            digest = {"bytes": len(files["proposal.json"]), "sha256": sha256(files["proposal.json"])}
            manifest = json.loads(files["manifest.json"])
            manifest["files"]["proposal.json"] = digest
            files["manifest.json"] = canonical_json(manifest)
            descriptor = json.loads(connection.execute("SELECT descriptor FROM sro_archives WHERE archive_id=?", (archive,)).fetchone()[0])
            descriptor["files"]["proposal.json"] = digest
            descriptor["files"]["manifest.json"] = {"bytes": len(files["manifest.json"]), "sha256": sha256(files["manifest.json"])}
            raw = canonical_json(descriptor)
            new_archive = sha256(raw)
            connection.execute("UPDATE sro_archives SET archive_id=?,descriptor=? WHERE archive_id=?", (new_archive, raw.decode(), archive))
            connection.execute("UPDATE sro_files SET archive_id=? WHERE archive_id=?", (new_archive, archive))
            for path in ("proposal.json", "manifest.json"):
                connection.execute("UPDATE sro_files SET data=? WHERE archive_id=? AND path=?", (files[path], new_archive, path))
        else:
            connection.execute("UPDATE sro_files SET data=? WHERE archive_id=? AND path=?", (b"{}", archive, changed))
    tools = OntologyTools(db, ("default",))
    ref = {"collection": "default", "name": "source.txt", "snapshot_id": result["snapshot_id"], "kind": "term", "id": "urn:test:source"}
    assert tools.call("inspect_ontology", ref)["error"]["code"] == "ontology_unavailable"


def test_registry_review_needs_only_copied_sqlite(memory, tmp_path):
    db, result = memory
    script = '''
import sys
class Block:
    def find_spec(self, name, *args):
        if name.startswith(("pyarrow", "pypdfium2", "sr_adapter.ocr")):
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
from sr_adapter.ontology_tools import OntologyTools
tools = OntologyTools(sys.argv[1], ("default",))
ref = {"collection": "default", "name": "source.txt", "snapshot_id": sys.argv[2], "kind": "term", "id": "urn:test:source"}
value = tools.call("inspect_ontology", ref)["value"]
assert value["vocabulary_review"]["review_required"]
assert len(value["vocabulary_review"]["archives"][0]["terms"][0]["declarations"]) == 2
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    before = sha256(db.read_bytes())
    proc = subprocess.run([sys.executable, "-c", script, str(db), result["snapshot_id"]], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert sha256(db.read_bytes()) == before
