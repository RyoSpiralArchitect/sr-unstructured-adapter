"""Proposal-local declaration resolution and recovery of an unchanged model response."""
import json
import sqlite3

import pytest

from sr_adapter.cli import main
from sr_adapter.knowledge import sha256
from sr_adapter.local_ontology import inspect_ontology, search_ontology
from sr_adapter.ontology_ingest import ingest_ontology_source, ontology_ingestion_status
from sr_adapter.ontology_recovery import recover_ontology_ingestion
from sr_adapter.proposal_vocabulary import REGISTRY, STRICT, VocabularyConflict, compile_vocabulary
from sr_adapter.relation_semantics import Term
from sr_adapter.semantic_proposals import render_semantic_proposal
from sr_adapter.tables import canonical_json
from test_ocr_ontology import Reader, image
from test_ontology_ingest import proposed
from test_visual_relations import INVENTORY, response


def reading(wire, *, conflict=None, duplicate_claim=False):
    answer = proposed(wire)
    content = answer["output"][0]["content"][0]
    payload = json.loads(content["text"])
    terms = payload["overlay"]["terms"]
    terms[1]["description"] = "Table note."
    terms.append({**terms[1], "description": "Exception note.", **(conflict or {})})
    if duplicate_claim:
        payload["overlay"]["claims"] *= 2
    content["text"] = json.dumps(payload)
    return answer


def legacy_failure(tmp_path, monkeypatch, kind="text"):
    source, db = tmp_path / ("source.txt" if kind == "text" else "source.png"), tmp_path / "memory.sqlite"
    source.write_bytes(b"A rule requires approval." if kind == "text" else image())
    options = {"ocr": "auto", "ocr_reader": Reader(), "proposal_version": 4} if kind == "ocr" else {"proposal_version": 3}

    def complete(wire):
        return reading(wire) if wire["text"]["format"]["name"] == "semantic_proposal" else response(INVENTORY)

    def previous_renderer(inputs, **kwargs):
        return render_semantic_proposal(inputs, **{**kwargs, "vocabulary_policy": STRICT})

    with monkeypatch.context() as old:
        old.setattr("sr_adapter.ontology_ingest.render_semantic_proposal", previous_renderer)
        failed = ingest_ontology_source(source, db, complete=complete, adapter_id="fixture", execute=True, **options)
    assert failed["status"] == "failed" and failed["stage"] == "semantics_validating"
    # Previous ledgers did not carry an explicit proposal version or local policy.
    failed.pop("vocabulary_policy")
    failed.pop("proposal_version")
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE sro_ingestions SET receipt=? WHERE run_id=?", (canonical_json(failed).decode(), failed["run_id"]))
    return source, db, failed


@pytest.mark.parametrize("kind", ["text", "visual", "ocr"])
def test_recovery_uses_original_wire_and_resolves_terms_for_each_source_reader(kind, tmp_path, monkeypatch, capsys):
    source, db, failed = legacy_failure(tmp_path, monkeypatch, kind)
    original = ontology_ingestion_status(db, failed["run_id"])
    source.unlink()
    for method in ("httpx.Client", "sr_adapter.ocr_sources._render", "sr_adapter.ocr_readers.local_reader"):
        monkeypatch.setattr(method, lambda *a, **kw: pytest.fail("Recovery repeated generation or OCR"))
    assert main(["knowledge", "ontology", "recover", failed["run_id"], "--database", str(db)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "complete" and out["planned_model_requests"] == out["submitted_requests"] == 0
    assert out["recovered_from_run_id"] == failed["run_id"]
    assert out["vocabulary"] == {"policy": REGISTRY, "declaration_count": 3, "term_count": 2, "repeated_iri_count": 1}
    assert ontology_ingestion_status(db, failed["run_id"]) == original
    assert ontology_ingestion_status(db, out["run_id"])["vocabulary_details"]["repetitions"][0]["declarations"][1]["index"] == 2
    for text in ("Table note.", "Exception note."):
        hit = search_ontology(db, kind="term", query=text)["hits"][0]
        term = inspect_ontology(db, **hit["reference"])["value"]
        assert term["term"]["description"] == "Exception note.\n\nTable note."
        assert term["claim_references"]
    with sqlite3.connect(db) as connection:
        bundle = dict(connection.execute("SELECT path,data FROM sro_files WHERE archive_id=?", (out["archive_id"],)))
        wire, = connection.execute("SELECT data FROM sro_ingestion_files WHERE run_id=? AND path='semantics-request.json'", (failed["run_id"],)).fetchone()
        assert bundle["request.json"] == wire and sha256(wire) == out["request_sha256"]
        assert len(json.loads(bundle["proposal.json"])["overlay"]["terms"]) == 3
        assert json.loads(bundle["manifest.json"])["vocabulary_policy"] == REGISTRY
    again = recover_ontology_ingestion(db, failed["run_id"])
    assert again["run_id"] == out["run_id"] and again["outcome"] == "already_active"


def test_registry_is_exact_local_and_order_independent():
    first = Term(iri="urn:test:a", kind="role", label="条件", description="first")
    other = first.model_copy(update={"iri": "urn:test:b"})
    declarations = [first, first, other, first.model_copy(update={"description": "second"}), first.model_copy(update={"description": ""})]
    compiled, report = compile_vocabulary(declarations)
    reversed_terms, _ = compile_vocabulary(list(reversed(declarations)))
    assert compiled == reversed_terms and len(compiled) == 2
    assert compiled[0]["description"] == "first\n\nsecond" and compiled[1] == other.model_dump()
    assert [d["index"] for d in report["repetitions"][0]["declarations"]] == [0, 1, 3, 4]
    assert report["repetitions"][0]["review_notes"]
    with pytest.raises(VocabularyConflict) as exc:
        compile_vocabulary([first, first.model_copy(update={"description": "x" * 65536})])
    assert exc.value.diagnostic["conflicts"][0]["reason"] == "description_limit"


@pytest.mark.parametrize("conflict", [{"kind": "class"}, {"label": "Source"}])
def test_conflicting_vocabulary_is_visible_and_does_not_replace_active_version(tmp_path, conflict):
    source, db = tmp_path / "source.txt", tmp_path / "memory.sqlite"
    source.write_text("The first rule.")
    options = {"execute": True, "adapter_id": "fixture"}
    first = ingest_ontology_source(source, db, complete=proposed, **options)
    source.write_text("The second rule.")
    failed = ingest_ontology_source(source, db, complete=lambda wire: reading(wire, conflict=conflict),
        expected_snapshot=first["snapshot_id"], **options)
    assert failed["status"] == "failed" and failed["failure_reason"] == "vocabulary_conflict"
    status = ontology_ingestion_status(db, failed["run_id"])
    detail = status["vocabulary_details"]["conflicts"][0]
    assert detail["fields"] == list(conflict) and detail["iri"] == "urn:test:source"
    assert [d["index"] for d in detail["declarations"]] == [1, 2]
    assert search_ontology(db, kind="claim")["hits"][0]["reference"]["snapshot_id"] == first["snapshot_id"]


def test_compatible_term_repetition_does_not_relax_claim_validation(tmp_path):
    source, db = tmp_path / "source.txt", tmp_path / "memory.sqlite"
    source.write_text("A rule requires approval.")
    result = ingest_ontology_source(source, db, complete=lambda wire: reading(wire, duplicate_claim=True), adapter_id="fixture", execute=True)
    assert result["status"] == "failed" and result["submitted_requests"] == 1
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone() == (0,)


@pytest.mark.parametrize("change", ["response", "request_with_matching_receipt", "source", "fingerprint", "unsafe_ocr_path"])
def test_recovery_rejects_changed_artifacts_before_registration(change, tmp_path, monkeypatch):
    _, db, failed = legacy_failure(tmp_path, monkeypatch, "ocr" if change == "unsafe_ocr_path" else "text")
    run_id = failed["run_id"]
    with sqlite3.connect(db) as c:
        if change in {"response", "source"}:
            path = "semantics-response.json" if change == "response" else "source.bin"
            c.execute("UPDATE sro_ingestion_files SET data=? WHERE run_id=? AND path=?", (b"changed", run_id, path))
        elif change == "request_with_matching_receipt":
            raw = b'{}'
            failed["stages"][-1]["request_sha256"] = sha256(raw)
            c.execute("UPDATE sro_ingestions SET receipt=? WHERE run_id=?", (canonical_json(failed).decode(), run_id))
            c.execute("UPDATE sro_ingestion_files SET data=? WHERE run_id=? AND path='semantics-request.json'", (raw, run_id))
        elif change == "fingerprint":
            c.execute("UPDATE sro_ingestions SET request_key=? WHERE run_id=?", ("0" * 64, run_id))
        else:
            c.execute("INSERT INTO sro_ingestion_files VALUES (?, ?, ?)", (run_id, "ocr/../escape.txt", b"bad"))
    with pytest.raises(ValueError):
        recover_ontology_ingestion(db, run_id)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM sro_ingestions").fetchone() == (1,)
        assert c.execute("SELECT count(*) FROM sro_entries").fetchone() == (0,)


def test_recovery_requires_expected_active_snapshot(tmp_path, monkeypatch):
    source, db, failed = legacy_failure(tmp_path, monkeypatch)
    source.write_text("A different policy was registered in the meantime.")
    active = ingest_ontology_source(source, db, complete=proposed, adapter_id="fixture", execute=True)
    with pytest.raises(ValueError, match="Entry changed"):
        recover_ontology_ingestion(db, failed["run_id"])
    result = recover_ontology_ingestion(db, failed["run_id"], expected_snapshot=active["snapshot_id"])
    assert result["status"] == "complete" and result["snapshot_id"] != active["snapshot_id"]
    assert search_ontology(db, name=source.name, kind="claim", snapshot_id=active["snapshot_id"])["total"] == 1
    with pytest.raises(ValueError, match="original run"):
        recover_ontology_ingestion(db, result["run_id"])
