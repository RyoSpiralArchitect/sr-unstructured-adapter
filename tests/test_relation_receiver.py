"""Local receiver conformance, including source replay and publication races."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
import shutil
import sqlite3

import pytest

from sr_adapter.cli import main
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative, verify_narrative
from sr_adapter.relation_dataset_store import export_relation_dataset, export_relation_view, write_dataset_sqlite
from sr_adapter.relation_receiver import (RelationDelivery, ReceiverConflict, SQLiteRelationReceiver,
    load_relation_delivery, receive_relation_views, search_received_relations, trace_received_relation)
from sr_adapter.tables import canonical_json
from test_fabric_narrative import narrative as narrative
from test_narrative import response
from test_relation_dataset import visual as visual, source as source


def pair(bundle, root, profile="narrative", review=None):
    export_relation_dataset([bundle], root / "base", profile=profile, review=review)
    export_relation_view(root / "base", root / "search", view="search")
    export_relation_view(root / "base", root / "graph", view="graph")
    return load_relation_delivery(root / "search", root / "graph")


def revision(bundle, root, *, model="second", empty=False):
    _, files = verify_narrative(bundle)
    manifest, payloads = render_narrative(data=files["source.bin"], source_name="source.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model=model,
        response=response([]) if empty else json.loads(files["response.json"]), label_policy="preserve")
    save_narrative(root / "source", manifest, payloads)
    return pair(root / "source", root)


@pytest.fixture
def delivered(narrative, tmp_path):
    return pair(narrative, tmp_path / "first")


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_complete_delivery_replay_repeat_and_trace_survive_deleted_inputs(profile, request, tmp_path):
    bundle = request.getfixturevalue(profile)
    delivery = pair(bundle, tmp_path / "pair", profile)
    db = tmp_path / "receiver.db"
    receiver = SQLiteRelationReceiver(db)
    receipt = receiver.receive(delivery, collection="demo")
    assert receipt["status"] == "committed" and receipt["readback_verified"]
    assert receipt["outcome"] == "activated" and receipt["previous_snapshot_id"] is None
    assert receipt["inserted_rows"] == sum(len(t.values) for d in delivery.datasets().values() for t in d.tables)
    repeat = receiver.receive(delivery, collection="demo")
    assert repeat["inserted_rows"] == 0 and repeat["outcome"] == "already_active"
    with pytest.raises(FrozenInstanceError):
        delivery.files = ()
    with sqlite3.connect(db) as con:
        assert dict(con.execute("SELECT path, data FROM srr_files")) == dict(delivery.files)
    shutil.rmtree(tmp_path / "pair")
    shutil.rmtree(bundle)
    result = search_received_relations(db, collection="demo")
    assert result["total"] == len(delivery.search.rows()["cards"])
    card = next(c for c in result["hits"] if c["assertion_status"] == "denied") if profile == "narrative" else result["hits"][0]
    trace = trace_received_relation(db, collection="demo", card_id=card["card_id"], snapshot_id=result["snapshot_id"])
    assert trace["card"] == card and trace["semantic_validation"] == "not_performed"
    if profile == "narrative":
        assert trace["assertion"]["properties"]["assertion_status"] == "denied"
        assert [e["properties"]["text"] for e in trace["evidence"]] == ["A", "B is not the owner"]
        assert any(loc["properties"]["path"] == "/header_context/0/text"
                   for e in trace["evidence"] for loc in e["locations"])
        assert trace["mentions"]["subject"]["properties"]["literal_match"] is False
        assert "Only applies after approval." in trace["card"]["content"]
    else:
        properties = trace["assertion"]["properties"]
        assert (properties["direction"], properties["support"], properties["value_text"]) == ("unspecified", "uncertain", "0%")
        assert trace["mentions"]["subject"]["node"]["node_id"] != trace["mentions"]["object"]["node"]["node_id"]
        assert any(e["observations"] for e in trace["evidence"])
        assert properties["x1"] == 500


def test_mismatched_or_single_views_reject_before_database_creation(narrative, delivered, tmp_path):
    revision(narrative, tmp_path / "second")
    db = tmp_path / "absent.db"
    for left, right in [("first/search", "second/graph"), ("first/graph", "first/search"), ("first/base", "first/graph")]:
        with pytest.raises(ValueError, match="same source"):
            receive_relation_views(tmp_path / left, tmp_path / right, db, collection="demo")
        assert not db.exists()
    (tmp_path / "first/search/cards.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="size/hash"):
        receive_relation_views(tmp_path / "first/search", tmp_path / "first/graph", db, collection="demo")
    assert not db.exists()


def test_activation_cas_retries_and_historical_evidence(narrative, delivered, tmp_path):
    db = tmp_path / "receiver.db"
    receiver = SQLiteRelationReceiver(db)
    first = receiver.receive(delivered, collection="demo")
    original = search_received_relations(db, collection="demo")
    second = revision(narrative, tmp_path / "second")
    with pytest.raises(ReceiverConflict):
        receiver.receive(second, collection="demo")
    updated = receiver.receive(second, collection="demo", expected_snapshot=first["snapshot_id"])
    assert updated["previous_snapshot_id"] == first["snapshot_id"]
    assert updated["snapshot_id"] != first["snapshot_id"]
    # Lost response on the current delivery is idempotent even with its old expectation.
    assert receiver.receive(second, collection="demo", expected_snapshot=first["snapshot_id"])["inserted_rows"] == 0
    with pytest.raises(ReceiverConflict):
        receiver.receive(delivered, collection="demo")
    assert search_received_relations(db, collection="demo")["snapshot_id"] == updated["snapshot_id"]
    card = original["hits"][0]["card_id"]
    with pytest.raises(ValueError, match="selected snapshot"):
        trace_received_relation(db, collection="demo", card_id=card)
    trace = trace_received_relation(db, collection="demo", card_id=card, snapshot_id=first["snapshot_id"])
    assert trace["card"] == original["hits"][0] and trace["active_snapshot_id"] == updated["snapshot_id"]
    receiver.receive(second, collection="other")
    with pytest.raises(ValueError, match="belong"):
        search_received_relations(db, collection="other", snapshot_id=first["snapshot_id"])
    with pytest.raises(ReceiverConflict):
        receiver.receive(delivered, collection="missing", expected_snapshot=updated["snapshot_id"])


def test_profile_mixing_rejected(visual, delivered, tmp_path):
    receiver = SQLiteRelationReceiver(tmp_path / "receiver.db")
    receiver.receive(delivered, collection="demo")
    other = pair(visual, tmp_path / "images", "visual")
    with pytest.raises(ReceiverConflict, match="profile"):
        receiver.receive(other, collection="demo", expected_snapshot=delivered.source.snapshot_id)
    assert receiver.receive(other, collection="images")["status"] == "committed"


def test_receiver_reuses_existing_sqlite_snapshots_without_touching_other_tables(delivered, tmp_path):
    db = tmp_path / "receiver.db"
    write_dataset_sqlite(tmp_path / "first/base", db)
    write_dataset_sqlite(tmp_path / "first/search", db)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE unrelated (value TEXT)")
        con.execute("INSERT INTO unrelated VALUES ('keep')")
    receipt = SQLiteRelationReceiver(db).receive(delivered, collection="demo")
    assert receipt["datasets"]["source"]["inserted_rows"] == 0
    assert receipt["datasets"]["search"]["inserted_rows"] == 0
    assert receipt["inserted_rows"] == sum(len(t.values) for t in delivered.graph.tables)
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT value FROM unrelated").fetchall() == [("keep",)]


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_graph_write_rolls_back_all_data_and_activation(narrative, delivered, tmp_path, monkeypatch, existing):
    import sr_adapter.relation_receiver as module
    db = tmp_path / "receiver.db"
    receiver = SQLiteRelationReceiver(db)
    if existing:
        receiver.receive(delivered, collection="demo")
    with sqlite3.connect(db) as con:
        before = list(con.iterdump())
    second = revision(narrative, tmp_path / "second")
    original = module._write_sqlite_dataset
    def fail(con, dataset):
        receipt = original(con, dataset)
        if dataset.profile == "relation_graph/1":
            raise RuntimeError("Synthetic interruption after graph write")
        return receipt
    monkeypatch.setattr(module, "_write_sqlite_dataset", fail)
    with pytest.raises(RuntimeError):
        receiver.receive(second, collection="demo", expected_snapshot=delivered.source.snapshot_id if existing else None)
    with sqlite3.connect(db) as con:
        assert list(con.iterdump()) == before
    if existing:
        assert search_received_relations(db, collection="demo")["snapshot_id"] == delivered.source.snapshot_id


def test_concurrent_updates_have_one_winner(narrative, delivered, tmp_path):
    db = tmp_path / "race.db"
    receiver = SQLiteRelationReceiver(db)
    receiver.receive(delivered, collection="demo")
    choices = [revision(narrative, tmp_path / "second", model="second"),
               revision(narrative, tmp_path / "third", model="third")]
    def update(delivery):
        try:
            return receiver.receive(delivery, collection="demo", expected_snapshot=delivered.source.snapshot_id)
        except ReceiverConflict:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, choices))
    assert sum(r is not None for r in results) == 1
    winner = next(r for r in results if r)
    assert search_received_relations(db, collection="demo")["snapshot_id"] == winner["snapshot_id"]
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM srr_snapshots").fetchone() == (2,)


@pytest.mark.parametrize("damage", ["card", "graph", "archive", "descriptor", "membership", "schema", "missing_catalog"])
def test_receiver_tampering_blocks_queries_and_repeat_without_repair(delivered, tmp_path, damage):
    db = tmp_path / "receiver.db"
    receiver = SQLiteRelationReceiver(db)
    receipt = receiver.receive(delivered, collection="demo")
    with sqlite3.connect(db) as con:
        if damage == "card":
            table = receipt["datasets"]["search"]["tables"]["cards"]["name"]
            con.execute(f'UPDATE "{table}" SET assertion_status=\'asserted\' WHERE assertion_status=\'denied\'')
        elif damage == "graph":
            table = receipt["datasets"]["graph"]["tables"]["properties"]["name"]
            con.execute(f'DELETE FROM "{table}" WHERE property_name=\'text\'')
        elif damage == "archive":
            con.execute("UPDATE srr_files SET data=? WHERE path='dataset.json'", (b"{}",))
        elif damage == "descriptor":
            con.execute("UPDATE srr_snapshots SET descriptor='{}'")
        elif damage == "membership":
            con.execute("DELETE FROM srr_members")
        elif damage == "schema":
            con.execute("ALTER TABLE srr_collections ADD COLUMN changed TEXT")
        else:
            con.execute("DROP TABLE srr_members")
    with sqlite3.connect(db) as con:
        damaged = list(con.iterdump())
    with pytest.raises(ValueError):
        search_received_relations(db, collection="demo")
    with pytest.raises(ValueError):
        receiver.receive(delivered, collection="demo")
    with sqlite3.connect(db) as con:
        assert list(con.iterdump()) == damaged


def test_queries_are_literal_paged_and_distinguish_relation_from_context(delivered, tmp_path):
    db = tmp_path / "receiver.db"
    SQLiteRelationReceiver(db).receive(delivered, collection="demo")
    page = search_received_relations(db, collection="demo", limit=1)
    assert page["total"] == 2 and page["has_more"]
    second = search_received_relations(db, collection="demo", limit=1, offset=1)
    assert not second["has_more"] and second["hits"][0]["card_id"] != page["hits"][0]["card_id"]
    unknown = search_received_relations(db, collection="demo", assertion_status="unknown")
    assert unknown["total"] == 1 and unknown["hits"][0]["object_label"] is None
    trace = trace_received_relation(db, collection="demo", card_id=unknown["hits"][0]["card_id"])
    assert "object" not in trace["mentions"]
    assert search_received_relations(db, collection="demo", query="only applies")["total"] == 0
    assert search_received_relations(db, collection="demo", query="only applies", scope="context")["total"] == 2
    assert search_received_relations(db, collection="demo", query="%' OR 1=1 --")["total"] == 0


def test_visual_review_trace_keeps_original_candidate(visual, tmp_path):
    base = pair(visual, tmp_path / "original", "visual")
    review = {"version": 1, "reviewer": "synthetic", "findings": [
        {"package_id": base.source.package_ids[0], "reference_kind": "item", "reference_id": "n1",
         "finding": "text_mismatch", "detail": "Check the label"},
        {"package_id": base.source.package_ids[0], "reference_kind": "image", "reference_id": None,
         "finding": "coverage_gap", "detail": "Incomplete extraction"}]}
    delivery = pair(visual, tmp_path / "reviewed", "visual", review)
    db = tmp_path / "receiver.db"
    SQLiteRelationReceiver(db).receive(delivery, collection="images")
    found = search_received_relations(db, collection="images", support="uncertain")
    trace = trace_received_relation(db, collection="images", card_id=found["hits"][0]["card_id"])
    assert {r["properties"]["detail"] for r in trace["reviews"]} == {"Check the label", "Incomplete extraction"}
    assert trace["assertion"]["properties"]["support"] == "uncertain"


def test_empty_extraction_context_card_and_input_validation(narrative, delivered, tmp_path):
    empty = revision(narrative, tmp_path / "empty", empty=True)
    db = tmp_path / "receiver.db"
    SQLiteRelationReceiver(db).receive(empty, collection="demo")
    found = search_received_relations(db, collection="demo")
    assert found["total"] == 1 and found["hits"][0]["card_kind"] == "source_context"
    trace = trace_received_relation(db, collection="demo", card_id=found["hits"][0]["card_id"])
    assert trace["assertion"] is None and trace["evidence"] == [] and trace["mentions"] == {}
    assert "Only applies" in trace["card"]["content"]
    for kwargs in ({"limit": True}, {"limit": 101}, {"offset": -1}, {"scope": "bad"},
                   {"assertion_status": "maybe"}, {"support": "bad"}, {"snapshot_id": "unknown"}):
        with pytest.raises(ValueError):
            search_received_relations(db, collection="demo", **kwargs)
    absent = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        search_received_relations(absent, collection="demo")
    assert not absent.exists()
    with pytest.raises(ValueError):
        SQLiteRelationReceiver(absent).receive(delivered, collection='demo"; DROP TABLE anything')
    assert not absent.exists()
    with pytest.raises(ValueError, match="distinct"):
        RelationDelivery(delivered.files + (delivered.files[0],))


def test_cli_receive_query_trace_and_conflict(narrative, delivered, tmp_path, capsys):
    db = tmp_path / "cli.db"
    common = ["--database", str(db), "--collection", "demo"]
    assert main(["knowledge", "dataset-receive", str(tmp_path / "first/search"), str(tmp_path / "first/graph"), *common]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert main(["knowledge", "dataset-query", *common, "--assertion-status", "denied"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert main(["knowledge", "dataset-trace", result["hits"][0]["card_id"], *common,
                 "--snapshot", receipt["snapshot_id"]]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["evidence"][1]["properties"]["text"] == "B is not the owner"
    revision(narrative, tmp_path / "second")
    assert main(["knowledge", "dataset-receive", str(tmp_path / "second/search"), str(tmp_path / "second/graph"), *common]) == 1
    conflict = json.loads(capsys.readouterr().out)
    assert conflict["error_type"] == "ReceiverConflict" and "--expected-snapshot" in conflict["message"]
    assert canonical_json(trace)  # Result contains only ordinary JSON scalars/containers.
