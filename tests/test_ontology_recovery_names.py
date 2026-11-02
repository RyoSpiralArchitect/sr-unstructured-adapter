"""Recover archived source names independently of the host filesystem's name rules."""
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from sr_adapter.ontology_ingest import ingest_ontology_source, ontology_ingestion_status
from sr_adapter.ontology_recovery import recover_ontology_ingestion
from sr_adapter.proposal_vocabulary import STRICT
from sr_adapter.semantic_proposals import render_semantic_proposal
from sr_adapter.tables import canonical_json
from test_proposal_vocabulary import reading


def legacy_failure_with_source_name(tmp_path, monkeypatch, source_name):
    source, db = tmp_path / "input.txt", tmp_path / "memory.sqlite"
    source.write_text("A rule requires approval.")
    archived_source = SimpleNamespace(name=source_name, open=source.open)

    def original_renderer(inputs, **kwargs):
        return render_semantic_proposal(inputs, **{**kwargs, "vocabulary_policy": STRICT})

    with monkeypatch.context() as old:
        # A source name created on macOS must also be recoverable on Windows,
        # where that name could not be created as an actual local file.
        old.setattr("sr_adapter.ontology_ingest.Path", lambda value: archived_source if value == source else Path(value))
        old.setattr("sr_adapter.ontology_ingest.render_semantic_proposal", original_renderer)
        failed = ingest_ontology_source(source, db, name="safe-entry", complete=reading, adapter_id="fixture", execute=True)
    assert failed["status"] == "failed" and failed["stage"] == "semantics_validating"
    assert failed["source_name"] == source_name
    source.unlink()
    return db, failed


@pytest.mark.parametrize("source_name", ["source\nname.txt", "source\tname.txt", "source\rname.txt"])
def test_recovery_preserves_previously_accepted_source_names(tmp_path, monkeypatch, source_name):
    db, failed = legacy_failure_with_source_name(tmp_path, monkeypatch, source_name)
    original = ontology_ingestion_status(db, failed["run_id"])
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("Recovery called a provider"))
    result = recover_ontology_ingestion(db, failed["run_id"])
    assert result["status"] == "complete" and result["submitted_requests"] == 0
    assert result["source_name"] == source_name and result["name"] == "safe-entry"
    assert ontology_ingestion_status(db, failed["run_id"]) == original
    with sqlite3.connect(db) as connection:
        wire, = connection.execute("SELECT data FROM sro_files WHERE archive_id=? AND path='request.json'", (result["archive_id"],)).fetchone()
        source_wire, = connection.execute("SELECT data FROM sro_ingestion_files WHERE run_id=? AND path='semantics-request.json'", (failed["run_id"],)).fetchone()
        assert wire == source_wire


@pytest.mark.parametrize("source_name", [None, "", " \n\t", "source\x00.txt", "a" * 1025])
def test_recovery_rejects_names_outside_the_source_reader_contract(tmp_path, monkeypatch, source_name):
    db, failed = legacy_failure_with_source_name(tmp_path, monkeypatch, "source.txt")
    failed["source_name"] = source_name
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE sro_ingestions SET receipt=? WHERE run_id=?", (canonical_json(failed).decode(), failed["run_id"]))
    with pytest.raises(ValueError, match="Source name"):
        recover_ontology_ingestion(db, failed["run_id"])
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone() == (0,)
