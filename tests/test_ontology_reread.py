import json
import sqlite3

import pytest

from sr_adapter.local_ontology import inspect_ontology, search_ontology
from sr_adapter.ontology_ingest import ingest_ontology_source
from sr_adapter.ontology_reread import (
    prepare_ontology_reread,
    reread_ontology_source,
    read_ontology_reread,
)
from test_ocr_ontology import Reader, image
from test_ontology_ingest import proposed


@pytest.fixture
def stored(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(image())
    db = tmp_path / "onto.sqlite"
    result = ingest_ontology_source(
        source,
        db,
        ocr="auto",
        ocr_reader=Reader(),
        complete=proposed,
        adapter_id="fixture",
        execute=True,
    )
    assert result["status"] == "complete"
    ref = search_ontology(db, kind="claim")["hits"][0]["reference"]
    original = inspect_ontology(db, **ref)
    node = original["value"]["source_records"][0]["node_id"]
    source.unlink()
    return db, ref, node, original


def test_reread_uses_db_pixels_and_keeps_original_evidence(stored, tmp_path):
    db, ref, node, original = stored
    packet = prepare_ontology_reread(db, reference=ref, node_id=node)
    assert packet["descriptor"]["original_text"] == "Review requires approval."
    assert packet["descriptor"]["region"] == [0, 0, 224, 54]
    response = {
        "text": "Review requires approval.",
        "status": "read",
        "notes": "Pixels checked",
    }
    result = reread_ontology_source(
        db,
        reference=ref,
        node_id=node,
        complete=lambda request: response,
        model="fixture",
        destination=tmp_path / "run",
    )
    assert result["status"] == "complete" and result["ontology_updated"] is False
    assert inspect_ontology(db, **ref) == original
    saved = read_ontology_reread(db, result["review_id"])
    assert (
        saved["image"] == packet["image"] and saved["payload"]["response"] == response
    )
    second = reread_ontology_source(
        db,
        reference=ref,
        node_id=node,
        complete=lambda request: response,
        model="fixture",
        destination=tmp_path / "again",
    )
    assert second["review_id"] == result["review_id"]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT count(*) FROM sro_reading_reviews"
        ).fetchone() == (1,)
        connection.execute("UPDATE sro_reading_reviews SET image=?", (b"altered",))
    with pytest.raises(ValueError):
        read_ontology_reread(db, result["review_id"])


def test_foreign_unit_rejected_and_failed_reading_retained(stored, tmp_path):
    db, ref, node, _ = stored
    with pytest.raises(ValueError):
        prepare_ontology_reread(db, reference=ref, node_id="foreign")
    result = reread_ontology_source(
        db,
        reference=ref,
        node_id=node,
        complete=lambda request: {"bad": "reply"},
        model="fixture",
        destination=tmp_path / "bad",
    )
    assert result["status"] == "failed"
    assert json.loads((tmp_path / "bad/response.json").read_bytes()) == {"bad": "reply"}


def test_new_reading_is_observation_not_ontology_replacement(stored, tmp_path):
    db, ref, node, original = stored
    result = reread_ontology_source(
        db,
        reference=ref,
        node_id=node,
        complete=lambda request: {
            "text": "Different reading",
            "status": "uncertain",
            "notes": "unclear",
        },
        model="fixture",
        destination=tmp_path / "run",
    )
    assert result["status"] == "complete" and result["text_changed"]
    assert (
        read_ontology_reread(db, result["review_id"])["payload"]["review_status"]
        == "unverified"
    )
    assert inspect_ontology(db, **ref) == original


def test_linked_dissent_cannot_supply_another_claims_pixels(stored, monkeypatch):
    from copy import deepcopy
    import sr_adapter.ontology_reread as reread

    db, ref, node, original = stored
    connected = deepcopy(original)
    foreign = deepcopy(connected["value"]["source_records"][0])
    foreign["node_id"] = "foreign-dissent-node"
    connected["value"]["source_records"].append(foreign)
    other = deepcopy(connected["value"]["claims"][0])
    other.update(id="dissent-only", evidence=[foreign["node_id"]])
    connected["value"]["claims"].append(other)
    monkeypatch.setattr(reread, "inspect_ontology", lambda *a, **kw: connected)
    with pytest.raises(ValueError, match="another claim"):
        prepare_ontology_reread(db, reference=ref, node_id=foreign["node_id"])
    assert prepare_ontology_reread(db, reference=ref, node_id=node)["image"]
