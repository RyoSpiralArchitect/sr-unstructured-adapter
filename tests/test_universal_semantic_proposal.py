"""Common v5 generation, immutable legacy wires, cache identity and local recovery."""
import json
import re
import sqlite3

import pytest

from sr_adapter.cli import main as cli_main
from sr_adapter.knowledge import sha256
from sr_adapter.ontology_ingest import _request_key, _source_bundle, ingest_ontology_source, ontology_ingestion_status
from sr_adapter.ontology_recovery import recover_ontology_ingestion
from sr_adapter.proposal_vocabulary import STRICT
from sr_adapter.relation_dataset_store import _load_dataset, _save_snapshot, export_relation_dataset
from sr_adapter.semantic_proposals import (INSTRUCTIONS_V4, VERSION, UNIVERSAL_VERSION, ProposalConfig,
    proposal_context, proposal_instructions, proposal_request, proposal_schema, render_semantic_proposal)
from sr_adapter.tables import canonical_json
from test_docx_relation_sources import package, paragraph
from test_ocr_ontology import Reader, image, source_package
from test_ontology_ingest import proposed
from test_proposal_vocabulary import reading
from test_visual_relations import INVENTORY, response


# Captured before adding v5, over this exact deterministic text source. These
# cover the complete request, including instructions, context and strict schema.
LEGACY_REQUEST_HASHES = {
    1: "99583a842b55542e5c0f28ade0ccea59084b1d63b3b76292bd29cf24c2a482ba",
    2: "cb246ce46c36a418a15084a1039902ce7d88506dbd7cb3624c8362f8516e85c2",
    3: "de500248c02ccef00274c358d090854f819fb43a569e97fa15c168d4bcac4e44",
    4: "7db09bf9e9dbb30e6b251c2546c68c3750c221f84b31a6cc783531d41c255c7c",
}


def test_legacy_complete_wire_requests_and_fingerprint_remain_frozen(tmp_path):
    data = b"Review requires approval."
    inputs = _source_bundle(data, "source.txt", tmp_path)
    for version, digest in LEGACY_REQUEST_HASHES.items():
        request = proposal_request(inputs, model="fixture", version=version)
        assert sha256(canonical_json(request)) == digest
    legacy = proposal_request(inputs, model="fixture", version=3)
    key = _request_key("source.txt", sha256(data), "fixture", "fixture", legacy, None)
    assert key == "dc6e44c913d27d960bb1863f8d87c4cd0e718980c856ebbe0e88302b66812ee0"
    assert key == _request_key("source.txt", sha256(data), "fixture", "fixture", legacy, None, proposal_version=3)
    assert key != _request_key("source.txt", sha256(data), "fixture", "fixture", legacy, None, proposal_version=5)
    # OCR identity historically hashes the plan and v4 config before rendering.
    plan = {"saved_plan": "fixture", "source_sha256": sha256(data)}
    old_fields = {"version": 2, "plan": plan, "provider": "fixture",
        "proposal_config": ProposalConfig(version=4, model="fixture", max_output_tokens=24000).model_dump(),
        "proposal_instructions": INSTRUCTIONS_V4, "proposal_schema": proposal_schema(version=4)}
    old_key = sha256(canonical_json(old_fields))
    assert old_key == _request_key("source.txt", sha256(data), "fixture", "fixture", None, plan)
    assert old_key == _request_key("source.txt", sha256(data), "fixture", "fixture", None, plan, proposal_version=4)
    assert old_key != _request_key("source.txt", sha256(data), "fixture", "fixture", None, plan, proposal_version=5)


@pytest.mark.parametrize("kind", ["native", "ocr", "visual"])
def test_v5_same_profile_retains_native_context_or_original_pixels(kind, tmp_path):
    if kind == "native":
        inputs = _source_bundle(package(paragraph("Review requires approval.")), "source.docx", tmp_path)
    elif kind == "visual":
        inputs = _source_bundle(image(), "source.png", tmp_path, visual_response=response(INVENTORY), model="fixture")
    else:
        manifest, files, _ = source_package(image(), "source.png")
        _save_snapshot(tmp_path / "source", {k: v for k, v in files.items() if k != "manifest.json"}, manifest)
        export_relation_dataset([tmp_path / "source"], tmp_path / "dataset", profile="narrative")
        inputs = _load_dataset(tmp_path / "dataset", allow_view=False)[1]
    request = proposal_request(inputs, model="fixture", version=UNIVERSAL_VERSION)
    assert request["reasoning"] == {"effort": "medium"}
    assert request["input"][0]["content"][0]["text"] == proposal_instructions(5)
    assert "Different meanings require distinct IRIs" in proposal_instructions(5)
    assert "draft/unapproved qualifier" in proposal_instructions(5)
    assert "never assume a source has page images" in proposal_instructions(5)
    schema = request["text"]["format"]["schema"]
    assert schema["$defs"]["ProposedOverlayV2"]["properties"]["terms"]["minItems"] == 1
    parts = request["input"][1]["content"]
    assert any(part["type"] == "input_image" for part in parts) == (kind != "native")
    context = json.loads(parts[0]["text"])
    if kind != "visual":
        assert not any(record["table"] == "locations" for record in context["records"])
        assert all("locations" in record for record in context["records"] if record["table"] == "evidence")
        source = _load_dataset(tmp_path / "dataset", allow_view=False)[0]
        old_context = proposal_context(source, version=3)[0]
        assert len(old_context["records"]) > len(context["records"])


def test_v5_instant_generation_rejects_date_only_without_inventing_timezone():
    schema = proposal_schema(version=5)
    for field in ("valid_from", "valid_until"):
        variants = schema["$defs"]["Claim"]["properties"][field]["anyOf"]
        assert {"type": "null"} in variants
        pattern = next(item["pattern"] for item in variants if item["type"] == "string")
        for valid in ("2026-05-01T00:00:00Z", "2026-06-30T23:59:59+09:00", "2026-05-01T00:00:00.123456-14:00"):
            assert re.fullmatch(pattern, valid)
        for invalid in ("2026-06-30", "2026-06-30T00:00:00", "2026-06-30T00:00:00+14:01", "2026-06-30T25:00:00Z"):
            assert not re.fullmatch(pattern, invalid)
        for version in (1, 2, 3, 4):
            legacy = proposal_schema(version=version)["$defs"]["Claim"]["properties"][field]
            assert not any("pattern" in item for item in legacy["anyOf"])
    assert "Date-only periods belong in claim context" in proposal_instructions(5)
    assert "Never invent a timezone" in proposal_instructions(5)


@pytest.mark.parametrize("kind", ["native", "ocr"])
def test_default_v5_cached_source_skips_model_and_ocr(kind, tmp_path, monkeypatch):
    source, db, calls = tmp_path / ("source.docx" if kind == "native" else "source.png"), tmp_path / "memory.sqlite", []
    source.write_bytes(package(paragraph("Review requires approval.")) if kind == "native" else image())
    reader = Reader()
    options = {"ocr": "auto", "ocr_reader": reader} if kind == "ocr" else {}

    def complete(wire):
        calls.append(wire)
        return proposed(wire)

    result = ingest_ontology_source(source, db, complete=complete, adapter_id="fixture", execute=True, **options)
    assert result["status"] == "complete", result
    assert result["proposal_version"] == 5
    assert len(calls) == 1
    assert calls[0]["reasoning"] == {"effort": "medium"}
    monkeypatch.setattr("sr_adapter.ocr_sources._render", lambda *a: pytest.fail("Cached source rendered again"))
    again = ingest_ontology_source(source, db, complete=lambda wire: pytest.fail("Cached source called model"),
        adapter_id="fixture", execute=True, **options)
    assert again["run_id"] == result["run_id"] and again["outcome"] == "already_active"
    assert again["submitted_requests_this_call"] == 0
    if kind == "ocr":
        assert len(reader.calls) == 1
        assert again["ocr_pages_this_call"] == again["rendered_pages_this_call"] == 0
    with pytest.raises(ValueError, match="Entry changed"):
        ingest_ontology_source(source, db, proposal_version=3 if kind == "native" else 4,
            complete=lambda wire: pytest.fail("Protocol replacement called model"), adapter_id="fixture", execute=True, **options)


@pytest.mark.parametrize("kind,version", [("native", 3), ("ocr", 4), ("native", 4), ("native", 5), ("ocr", 5), ("visual", 5)])
def test_recovery_reproduces_saved_actual_protocol_without_calls(kind, version, tmp_path, monkeypatch):
    source, db = tmp_path / ("source.txt" if kind == "native" else "source.png"), tmp_path / "memory.sqlite"
    source.write_bytes(b"Review requires approval." if kind == "native" else image())
    options = {"ocr": "auto", "ocr_reader": Reader()} if kind == "ocr" else {}

    def complete(wire):
        return reading(wire) if wire["text"]["format"]["name"] == "semantic_proposal" else response(INVENTORY)

    def strict_renderer(inputs, **kwargs):
        return render_semantic_proposal(inputs, **{**kwargs, "vocabulary_policy": STRICT})

    with monkeypatch.context() as previous:
        previous.setattr("sr_adapter.ontology_ingest.render_semantic_proposal", strict_renderer)
        failed = ingest_ontology_source(source, db, complete=complete, adapter_id="fixture", execute=True,
            proposal_version=version, **options)
    assert failed["status"] == "failed" and failed["stage"] == "semantics_validating"
    original = ontology_ingestion_status(db, failed["run_id"])
    source.unlink()
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("Recovery called model"))
    monkeypatch.setattr("sr_adapter.ocr_sources._render", lambda *a, **kw: pytest.fail("Recovery rendered source"))
    result = recover_ontology_ingestion(db, failed["run_id"])
    assert result["status"] == "complete", result
    assert result["proposal_version"] == version and result["submitted_requests"] == 0
    assert ontology_ingestion_status(db, failed["run_id"]) == original
    with sqlite3.connect(db) as connection:
        original_wire, = connection.execute("SELECT data FROM sro_ingestion_files WHERE run_id=? AND path='semantics-request.json'", (failed["run_id"],)).fetchone()
        recovered_wire, = connection.execute("SELECT data FROM sro_files WHERE archive_id=? AND path='request.json'", (result["archive_id"],)).fetchone()
        assert original_wire == recovered_wire
    assert recover_ontology_ingestion(db, failed["run_id"])["outcome"] == "already_active"


def test_raw_cli_protocol_override_and_invalid_ocr_are_explicit(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text("Review requires approval.")
    for version in (3, 4, 5):
        assert cli_main(["knowledge", "ontology", "ingest", str(source), "--database", str(tmp_path / f"v{version}.sqlite"),
            "--proposal-version", str(version)]) == 0
        assert json.loads(capsys.readouterr().out)["proposal_version"] == version
    with pytest.raises(ValueError, match="OCR ingestion requires"):
        ingest_ontology_source(source, tmp_path / "invalid.sqlite", proposal_version=3, ocr="auto", ocr_reader=Reader())


def test_standalone_cli_exposes_v5_and_retains_v2_default(tmp_path, monkeypatch, capsys):
    import sr_adapter.semantic_proposal_runtime as runtime
    versions = []

    def prepared(*args, **kwargs):
        versions.append(kwargs["version"])
        return {"status": "prepared"}

    monkeypatch.setattr(runtime, "propose_semantics", prepared)
    args = ["propose", "unused-input", "--out", str(tmp_path / "unused-output"), "--model", "fixture"]
    assert runtime.main(args) == 0
    assert runtime.main(args + ["--proposal-version", "5"]) == 0
    assert versions == [VERSION, 5] == [2, 5]
    capsys.readouterr()
