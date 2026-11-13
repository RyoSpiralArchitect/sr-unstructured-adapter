"""Raw file to SQLite and failure receipts; no real model calls in tests."""
import base64
import io
import json
import sqlite3

import httpx
import pytest
from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

from sr_adapter.cli import main
from sr_adapter.local_ontology import inspect_ontology, read_ontology_source, search_ontology
from sr_adapter.ontology_ingest import ingest_ontology_source, ontology_ingestion_status
from sr_adapter.relation_sources import read_source
from sr_adapter.semantic_proposals import proposal_schema
from test_docx_relation_sources import package, paragraph
from test_relation_sources import _pptx, _box, _image
from test_visual_relations import INVENTORY, response


def proposed(wire, *, abstain=False):
    context = json.loads(wire["input"][1]["content"][0]["text"])
    first = next(r for r in context["records"] if r["id"] in context["coverage_units"])
    overlay = {"description": "Fixture", "axioms": [], "roles": [], "claim_links": [],
        "terms": [{"iri": "urn:test:" + name, "kind": kind, "label": name} for name, kind in
        [("read", "relation"), ("source", "role")]], "expressions": [{"id": "e1", "package_id": first["package_id"],
        "operator": "atom", "predicate": "urn:test:read", "arguments": [{"role": "urn:test:source",
            "value": {"kind": "source", "node_id": first["id"]}}]}],
        "claims": [{"id": "c1", "package_id": first["package_id"], "expression_id": "e1", "status": "asserted",
            "modality": "descriptive", "basis": "source_interpretation", "attributed_to": "fixture",
            "evidence": [first["id"]]}]}
    return response({"overlay": None if abstain else overlay, "notes": ["Handwritten fixture, no accuracy claim."],
        "coverage": [{"source_id": key, "disposition": "represented" if key == first["id"] and not abstain else "context_only",
            "claim_ids": ["c1"] if key == first["id"] and not abstain else [], "note": "Fixture"}
            for key in context["coverage_units"]]})


def pdf():
    stream = io.BytesIO()
    canvas = Canvas(stream)
    canvas.drawString(40, 740, "Review requires approval.")
    canvas.showPage()
    canvas.showPage()
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize("extension,data", [
    ("txt", "審査部が申請を確認する。".encode()), ("md", b"# Policy\nReview requires approval."),
    ("csv", b"Team,Action\nReview,Approve"), ("tsv", b"Team\tAction\nReview\tApprove"),
    ("docx", package(paragraph("Review requires approval."))),
    ("pptx", _pptx([_box("1", [["Review requires approval."]])])), ("pdf", pdf())],
    ids=["txt", "md", "csv", "tsv", "docx", "pptx", "pdf"])
def test_raw_source_to_single_db_in_one_model_call(extension, data, tmp_path):
    path, db, calls = tmp_path / ("source." + extension), tmp_path / "onto.db", []
    path.write_bytes(data)
    def complete(wire):
        calls.append(wire)
        return proposed(wire)
    result = ingest_ontology_source(path, db, model="fixture", complete=complete, adapter_id="fixture-v1", execute=True)
    assert result["status"] == "complete", result
    assert result["submitted_requests"] == len(calls) == 1
    original_request = json.dumps(calls[0], ensure_ascii=False)
    assert ("Review" in original_request) if extension != "txt" else ("審査部" in original_request)
    duplicate = ingest_ontology_source(path, db, model="fixture", complete=complete, adapter_id="fixture-v1", execute=True)
    assert duplicate["submitted_requests_this_call"] == 0 and len(calls) == 1
    path.unlink()
    hit = search_ontology(db, kind="claim")["hits"][0]
    trace = inspect_ontology(db, **hit["reference"])["value"]
    source = read_ontology_source(db, collection="default", name=path.name, snapshot_id=result["snapshot_id"],
        package_id=trace["claims"][0]["package_id"])
    assert source["data"] == data
    status = ontology_ingestion_status(db, result["run_id"])
    assert status["status"] == "complete"
    assert {f["path"] for f in status["stored_files"]} >= {"source.bin", "semantics-request.json", "semantics-response.json"}
    if extension == "pdf":
        assert any("page 2" in warning for warning in status["warnings"])
        assert any(row["path"] == "/page" for row in trace["source_context"]["locations"])


def test_image_passes_original_pixels_to_both_stages_then_needs_no_original(tmp_path):
    path, db, calls = tmp_path / "image.png", tmp_path / "onto.db", []
    data = _image()
    path.write_bytes(data)
    def complete(wire):
        calls.append(wire)
        return response(INVENTORY) if len(calls) == 1 else proposed(wire)
    result = ingest_ontology_source(path, db, model="fixture", complete=complete, adapter_id="fixture-v1", execute=True)
    assert result["status"] == "complete", result
    assert len(calls) == result["submitted_requests"] == 2
    for wire in calls:
        parts = wire["input"][1]["content"]
        image = next(p for p in parts if p["type"] == "input_image")
        assert base64.b64decode(image["image_url"].split(",", 1)[1]) == data
    path.unlink()
    hit = search_ontology(db, kind="claim")["hits"][0]
    context = inspect_ontology(db, **hit["reference"])["value"]["source_context"]
    assert context["images"][0]["strategy"] == "inventory"
    assert context["relations"] == []  # General semantics follows the visual reading directly.


def test_plan_never_calls_model_and_cli_exposes_local_tools(tmp_path, capsys):
    source, db = tmp_path / "source.txt", tmp_path / "onto.db"
    source.write_text("Review requires approval.", encoding="utf-8")
    assert main(["knowledge", "ontology", "ingest", str(source), "--database", str(db), "--model", "fixture"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "prepared" and result["submitted_requests"] == 0
    assert main(["knowledge", "ontology", "status", result["run_id"], "--database", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["stored_files"]
    assert main(["knowledge", "ontology", "tools", "--database", str(db), "--collections", "default"]) == 0
    assert {tool["name"] for tool in json.loads(capsys.readouterr().out)["tools"]} == {"list_ontologies", "search_ontology", "inspect_ontology", "compare_ontology_claims"}


@pytest.mark.parametrize("failure,expected", [("timeout", "unknown"), ("429", "failed"), ("500", "unknown"),
    ("invalid_json", "failed"), ("incomplete", "failed"), ("invalid_schema", "failed"), ("abstain", "abstained")])
def test_failures_and_abstentions_keep_response_without_activation_or_retries(tmp_path, failure, expected):
    source, db, calls = tmp_path / "source.txt", tmp_path / "onto.db", []
    source.write_text("Review requires approval.", encoding="utf-8")
    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private provider error with secret")
        if failure in {"429", "500"}:
            return httpx.Response(int(failure), json={"error": "fixture"})
        if failure == "invalid_json":
            return httpx.Response(200, content=b"not-json")
        if failure == "incomplete":
            return httpx.Response(200, json=response({}, status="incomplete"))
        if failure == "invalid_schema":
            return httpx.Response(200, json=response({"wrong": "output"}))
        return httpx.Response(200, json=proposed(json.loads(request.content), abstain=True))
    result = ingest_ontology_source(source, db, model="fixture", execute=True, api_key="fixture-secret",
        transport=httpx.MockTransport(handle))
    assert result["status"] == expected, result
    assert result["submitted_requests"] == len(calls) == 1 and result["automatic_retries"] == 0
    status = ontology_ingestion_status(db, result["run_id"])
    assert ("semantics-response.json" in {f["path"] for f in status["stored_files"]}) == (failure != "timeout")
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone()[0] == 0
        assert all(b"fixture-secret" not in data for data, in connection.execute("SELECT data FROM sro_ingestion_files"))


def test_changed_source_or_adapter_requires_expected_version_before_any_call(tmp_path):
    source, db, calls = tmp_path / "source.txt", tmp_path / "onto.db", []
    source.write_text("First version.", encoding="utf-8")
    def complete(wire):
        calls.append(wire)
        return proposed(wire)
    options = dict(model="fixture", complete=complete, adapter_id="fixture-v1", execute=True)
    first = ingest_ontology_source(source, db, **options)
    assert first["status"] == "complete"
    with pytest.raises(ValueError, match="Entry changed"):
        ingest_ontology_source(source, db, **{**options, "adapter_id": "fixture-v2"})
    source.write_text("Second version.", encoding="utf-8")
    with pytest.raises(ValueError, match="Entry changed"):
        ingest_ontology_source(source, db, **options)
    assert len(calls) == 1
    second = ingest_ontology_source(source, db, **options, expected_snapshot=first["snapshot_id"])
    assert second["status"] == "complete" and second["snapshot_id"] != first["snapshot_id"]
    assert search_ontology(db, name=source.name, snapshot_id=first["snapshot_id"], kind="claim")["total"] == 1


def test_concurrent_activation_keeps_completed_response_without_overwriting(tmp_path):
    source, competing, db = tmp_path / "source.txt", tmp_path / "other.txt", tmp_path / "onto.db"
    source.write_text("First version.", encoding="utf-8")
    competing.write_text("Another writer's version.", encoding="utf-8")
    winner = []
    def complete(wire):
        winner.append(ingest_ontology_source(competing, db, name=source.name, model="fixture",
            complete=proposed, adapter_id="competing-v1", execute=True))
        return proposed(wire)
    result = ingest_ontology_source(source, db, model="fixture", complete=complete, adapter_id="fixture-v1", execute=True)
    assert result["status"] == "failed" and result["error_type"] == "ReceiverConflict"
    assert winner[0]["status"] == "complete"
    hit = search_ontology(db, kind="claim")["hits"][0]
    assert hit["reference"]["snapshot_id"] == winner[0]["snapshot_id"]
    assert "semantics-response.json" in {f["path"] for f in ontology_ingestion_status(db, result["run_id"])["stored_files"]}


def test_missing_host_adapter_identity_or_credential_does_not_submit(tmp_path, monkeypatch):
    source, db = tmp_path / "source.txt", tmp_path / "onto.db"
    source.write_text("Source text.", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter_id"):
        ingest_ontology_source(source, db, model="fixture", complete=proposed, execute=True)
    assert not db.exists()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ingest_ontology_source(source, db, model="fixture", execute=True)
    assert result["status"] == "failed" and result["submitted_requests"] == 0


def test_success_receipt_and_activation_roll_back_before_reporting_completion(tmp_path, monkeypatch):
    from sr_adapter import ontology_ingest
    source, db = tmp_path / "source.txt", tmp_path / "onto.db"
    source.write_text("Source text.", encoding="utf-8")
    original = ontology_ingest.import_ontology
    def interrupt_commit(*args, **kwargs):
        completed = kwargs.pop("_on_commit")
        def fail(connection, result):
            completed(connection, result)
            raise RuntimeError("Commit interrupted")
        return original(*args, **kwargs, _on_commit=fail)
    monkeypatch.setattr(ontology_ingest, "import_ontology", interrupt_commit)
    result = ingest_ontology_source(source, db, model="fixture", complete=proposed, adapter_id="fixture-v1", execute=True)
    assert result["status"] == "failed" and "readback_verified" not in result and "snapshot_id" not in result
    assert ontology_ingestion_status(db, result["run_id"])["status"] == "failed"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM sro_snapshots").fetchone()[0] == 0


def test_scanned_encrypted_or_broken_pdf_fails_before_model_or_database(tmp_path):
    blank = PdfWriter()
    blank.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    blank.write(stream)
    no_text = stream.getvalue()
    blank.encrypt("password")
    encrypted = io.BytesIO()
    blank.write(encrypted)
    for content in (no_text, encrypted.getvalue(), b"%PDF-broken"):
        source, db = tmp_path / "source.pdf", tmp_path / "onto.db"
        source.write_bytes(content)
        with pytest.raises(ValueError):
            ingest_ontology_source(source, db, model="fixture", execute=True)
        assert not db.exists()
    assert any("embedded reading order" in warning for warning in read_source(pdf(), "text.pdf")["warnings"])


def test_ingestion_schema_constrains_each_operator_without_changing_older_wire_contracts():
    for version in (1, 2):
        schema = proposal_schema(version=version)
        expression = schema["$defs"]["Expression" if version == 1 else "ExpressionV2"]
        assert "anyOf" not in expression
        assert "maxItems" not in expression["properties"]["operands"]
    variants = proposal_schema(version=3)["$defs"]["ExpressionV2"]["anyOf"]
    for variant, operator, bounds in zip(variants, ("atom", "not", "implies", "and", "or"),
            ((0, 0), (1, 1), (2, 2), (2, 1000), (2, 1000))):
        fields = variant["properties"]
        assert fields["operator"]["enum"] == [operator]
        assert (fields["operands"]["minItems"], fields["operands"]["maxItems"]) == bounds
        assert fields["predicate"]["type"] == ("string" if operator == "atom" else "null")
        if operator != "atom":
            assert fields["arguments"]["maxItems"] == 0
        assert set(variant["required"]) == set(fields)
