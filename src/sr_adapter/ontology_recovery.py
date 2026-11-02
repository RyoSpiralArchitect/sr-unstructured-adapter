"""Recompile a received semantic response against its exact saved request."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .knowledge import sha256
from .local_ontology import _collection, _name, _reader, _snapshot
from .ontology_ingest import (_finish_proposal, _request_key, _reserve_ingestion, _runs,
                              _save_run, _source_bundle, _validate_run_id)
from .proposal_vocabulary import REGISTRY
from .relation_dataset_store import _load_dataset, _save_snapshot, export_relation_dataset
from .semantic_proposals import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, OCR_VERSION, STRUCTURAL_VERSION, UNIVERSAL_VERSION, proposal_request
from .tables import canonical_json
from .visual_relations import INVENTORY_VERSION, visual_request


def _load_failed(database, run_id):
    _validate_run_id(run_id)
    with _reader(database) as connection:
        _runs(connection)
        row = connection.execute("SELECT request_key, collection, name, receipt FROM sro_ingestions WHERE run_id=? "
            "AND length(CAST(receipt AS BLOB)) <= ?", (run_id, 2 * 1024 * 1024)).fetchone()
        if row is None:
            raise ValueError("Missing or oversized original ingestion receipt")
        key, collection, name, raw = row
        receipt = json.loads(raw)
        _collection(collection)
        _name(name)
        # Source names are archived metadata, separate from the entry name.
        # Match read_source's contract, including names accepted on another OS.
        source_name = receipt["source_name"]
        if (not isinstance(source_name, str) or not source_name.strip()
                or "\x00" in source_name or len(source_name) > 1024):
            raise ValueError("Source name must be a nonempty filename of at most 1024 characters")
        if (receipt["run_id"] != run_id or receipt["collection"] != collection or receipt["name"] != name
                or receipt["status"] != "failed" or receipt["stage"] != "semantics_validating"
                or receipt.get("recovered_from_run_id") is not None):
            raise ValueError("Recovery requires an original run that failed after receiving a semantic response")
        count, size = connection.execute("SELECT count(*), coalesce(sum(length(data)), 0) FROM sro_ingestion_files "
                                         "WHERE run_id=?", (run_id,)).fetchone()
        if not 1 <= count <= 64 or size > 128 * 1024 * 1024:
            raise ValueError("Recorded ingestion exceeds its replay budget")
        files = dict(connection.execute("SELECT path, data FROM sro_ingestion_files WHERE run_id=?", (run_id,)))
    return receipt, key, files


def _verified_stage(parent, files, stage):
    events = [entry for entry in parent["stages"] if entry["stage"] == stage]
    if len(events) != 1 or events[0]["status"] != "received":
        raise ValueError("The original model response was not fully received")
    event = events[0]
    if parent["provider"] == "openai" and event.get("http_status") != 200:
        raise ValueError("Missing successful model acknowledgement")
    request = files[stage + "-request.json"]
    response = files[stage + "-response.json"]
    if (len(request) > MAX_REQUEST_BYTES or len(response) > MAX_RESPONSE_BYTES
            or sha256(request) != event["request_sha256"] or sha256(response) != event["response_sha256"]):
        raise ValueError("Saved request or response differs from the original receipt")
    return request, response


def recover_ontology_ingestion(database, run_id, *, expected_snapshot=None):
    """Replay saved text, OCR or visual inputs locally. The original run remains immutable."""
    if expected_snapshot is not None:
        _snapshot(expected_snapshot)
    parent, fingerprint, files = _load_failed(database, run_id)
    data, name, model = files["source.bin"], parent["source_name"], parent["model"]
    if len(data) > 16 * 1024 * 1024 or sha256(data) != parent["source_sha256"]:
        raise ValueError("Original source differs from its ingestion receipt")
    wire, raw_response = _verified_stage(parent, files, "semantics")
    reading = parent["source_reading"]
    version = parent.get("proposal_version", OCR_VERSION if reading == "local_ocr" else STRUCTURAL_VERSION)
    if (type(version) is not int or version not in (STRUCTURAL_VERSION, OCR_VERSION, UNIVERSAL_VERSION)
            or reading == "local_ocr" and version < OCR_VERSION):
        raise ValueError("Unsupported recorded ingestion protocol")
    with TemporaryDirectory(prefix="sr-ontology-recovery-") as scratch:
        root = Path(scratch)
        plan = first = None
        if reading == "local_ocr":
            from .ocr_sources import verify_ocr_payloads
            source = {k.removeprefix("ocr/"): v for k, v in files.items() if k.startswith("ocr/")}
            if "source.bin" in source:
                raise ValueError("Duplicate archived OCR source")
            source["source.bin"] = data
            manifest = verify_ocr_payloads(source)  # Verifies the exact flat file allowlist before writing.
            plan = json.loads(source["ocr-config.json"])
            if parent["ocr_plan"] != plan or plan["source_name"] != name:
                raise ValueError("Saved OCR selection differs from the receipt")
            del source["manifest.json"]
            _save_snapshot(root / "source", source, manifest)
            export_relation_dataset([root / "source"], root / "dataset", profile="narrative")
            inputs = _load_dataset(root / "dataset", allow_view=False)[1]
        elif reading == "visual_model":
            visual_wire, visual_response = _verified_stage(parent, files, "visual")
            first = visual_request(data, name, model, "inventory", version=INVENTORY_VERSION)
            if canonical_json(first) != visual_wire:
                raise ValueError("Original visual request cannot be reproduced")
            inputs = _source_bundle(data, name, root, visual_response=json.loads(visual_response), model=model)
        elif reading == "local_embedded_text":
            inputs = _source_bundle(data, name, root)
        else:
            raise ValueError("Unknown original source-reading protocol")

        request = proposal_request(inputs, model=model, version=version)
        if canonical_json(request) != wire:
            raise ValueError("Original semantic request cannot be reproduced")
        key = _request_key(name, parent["source_sha256"], parent["adapter_id"], model, first or request, plan,
                           proposal_version=version)
        if key != fingerprint:
            raise ValueError("Original generation identity differs from its recorded key")

        receipt = {k: parent[k] for k in ("collection", "name", "source_name", "source_sha256", "model", "provider",
                                          "adapter_id", "source_reading", "warnings")}
        receipt.update(version=1, run_id=uuid4().hex, status="running", stage="semantics_validating",
            recovered_from_run_id=run_id, request_sha256=sha256(wire), response_sha256=sha256(raw_response),
            vocabulary_policy=REGISTRY, proposal_version=version, planned_model_requests=0, submitted_requests=0,
            automatic_retries=0, semantic_validation="not_performed", stages=[])
        if plan is not None:
            receipt.update(ocr_plan=plan, ocr_pages_read=0, rendered_pages=0)
        reused = _reserve_ingestion(database, receipt, key, expected_snapshot)
        if reused is not None:
            return reused
        try:
            return _finish_proposal(database, root, receipt, inputs, json.loads(raw_response), version, expected_snapshot)
        except Exception as error:
            receipt.update(status="failed", failure_reason="recovery_not_completed", error_type=type(error).__name__)
            _save_run(database, receipt)
            return receipt
