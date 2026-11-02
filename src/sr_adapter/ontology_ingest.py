"""One raw source to local ontology memory; bounded, recorded model calls only.

All durable run inputs, requests, responses and receipts live in the same SQLite
file. Temporary portable bundles are removed after import or failure.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx

from .knowledge import sha256
from .local_ontology import _collection, _name, _schema, _snapshot, import_ontology
from .narrative import NarrativeOntology, render_narrative, save_narrative
from .relation_dataset_store import _hashes, _load_dataset, _save_snapshot, export_relation_dataset
from .relation_sources import read_source
from .semantic_proposals import (INSTRUCTIONS_V2, MAX_RESPONSE_BYTES, STRUCTURAL_VERSION, proposal_request,
    proposal_schema, render_semantic_proposal)
from .tables import canonical_json
from .visual_relations import INVENTORY_VERSION, project_visual, visual_request

_RUN_SCHEMA = {
    "sro_ingestions": "CREATE TABLE sro_ingestions (run_id TEXT PRIMARY KEY NOT NULL, request_key TEXT NOT NULL, collection TEXT NOT NULL, name TEXT NOT NULL, receipt TEXT NOT NULL)",
    "sro_ingestion_files": "CREATE TABLE sro_ingestion_files (run_id TEXT NOT NULL, path TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY (run_id, path))",
}


def _runs(connection, *, create=False):
    for name, sql in _RUN_SCHEMA.items():
        actual = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if actual is None and create:
            connection.execute(sql)
        elif actual != (sql,):
            raise ValueError("Unsupported ontology ingestion ledger")


def _save_run(database, receipt, files=()):
    with closing(sqlite3.connect(database, timeout=30)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        _runs(connection)
        for path, data in files:
            connection.execute("INSERT INTO sro_ingestion_files VALUES (?, ?, ?)", (receipt["run_id"], path, data))
        updated = connection.execute("UPDATE sro_ingestions SET receipt=? WHERE run_id=?",
                                     (canonical_json(receipt).decode(), receipt["run_id"]))
        if updated.rowcount != 1:
            raise ValueError("Ontology ingestion receipt disappeared")


def ontology_ingestion_status(database, run_id):
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Use the run ID returned by ingestion")
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection, connection:
        connection.execute("BEGIN")
        _runs(connection)
        row = connection.execute("SELECT receipt FROM sro_ingestions WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown ontology ingestion")
        receipt = json.loads(row[0])
        receipt["stored_files"] = [{"path": path, "bytes": size} for path, size in connection.execute(
            "SELECT path, length(data) FROM sro_ingestion_files WHERE run_id=? ORDER BY path", (run_id,))]
    return receipt


def _source_bundle(data, source_name, root, *, visual_response=None, model=None):
    """Source text parsing is local; no invented preliminary relation assertions."""
    if visual_response is None:
        vocabulary = NarrativeOntology.model_validate({"namespace": "urn:sr:source-reader:", "classes": {"Source": "Source"},
            "predicates": [{"id": "mentions", "label": "mentions", "subject_type": "Source", "object_type": "Source",
                            "description": "Unused vocabulary: this stage only reads source units."}]})
        empty = {"status": "completed", "model": "local-source-reader", "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": '{"assertions":[]}'}]}]}
        manifest, files = render_narrative(data=data, source_name=source_name, ontology=vocabulary,
            model="local-source-reader", response=empty)
        save_narrative(root / "source", manifest, files)
        profile = "narrative"
    else:
        config = {"version": INVENTORY_VERSION, "source_name": source_name, "page_label": "image", "model": model, "strategy": "inventory"}
        files = {"config.json": canonical_json(config), "source.bin": data,
            "inventory-request.json": canonical_json(visual_request(data, source_name, model, "inventory", version=INVENTORY_VERSION)),
            "inventory-response.json": canonical_json(visual_response)}
        projected = project_visual(files)
        _save_snapshot(root / "source", files, {"version": INVENTORY_VERSION, "status": "complete",
            "package_id": projected["package_id"], "files": _hashes(files)})
        profile = "visual"
    export_relation_dataset([root / "source"], root / "dataset", profile=profile)
    return _load_dataset(root / "dataset", allow_view=False)[1]


def ingest_ontology_source(source, database, *, model, collection="default", name=None,
                           expected_snapshot=None, execute=False, api_key=None, complete=None,
                           adapter_id=None, transport=None):
    """Parse, propose and import from a raw file; ``complete`` can wrap a local model.

    The injected callable receives one Responses-shaped request and returns one
    complete normalized response. Host code owns that adapter and its credentials.
    Default transport uses OpenAI; retrieval tools never invoke this function.
    """
    source = Path(source)
    if complete is not None and (not callable(complete) or not isinstance(adapter_id, str) or not adapter_id.strip()):
        raise ValueError("A host model callable requires a stable adapter_id for replay and reuse")
    if complete is None and adapter_id is not None:
        raise ValueError("adapter_id belongs to an injected host model callable")
    provider = adapter_id if complete is not None else "openai-responses"
    if len(provider) > 256:
        raise ValueError("adapter_id must contain at most 256 characters")
    _collection(collection)
    name = source.name if name is None else name
    _name(name)
    if expected_snapshot is not None:
        _snapshot(expected_snapshot)
    with source.open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    reading = read_source(data, source.name)
    visual = reading["needs_transcription"]
    with TemporaryDirectory(prefix="sr-ontology-") as scratch:
        root = Path(scratch)
        inputs = None if visual else _source_bundle(data, source.name, root)
        request = (visual_request(data, source.name, model, "inventory", version=INVENTORY_VERSION) if visual
                   else proposal_request(inputs, model=model, version=STRUCTURAL_VERSION))
        fingerprint = sha256(canonical_json({"version": 1, "source_name": source.name, "source_sha256": sha256(data),
            "provider": provider, "request": request, "proposal_instructions": INSTRUCTIONS_V2,
            "proposal_schema": proposal_schema(version=STRUCTURAL_VERSION)}))
        receipt = {"version": 1, "run_id": uuid4().hex, "status": "prepared", "stage": "prepared",
            "collection": collection, "name": name, "source_name": source.name, "source_sha256": sha256(data),
            "model": model, "provider": "host_callable" if complete is not None else "openai",
            "adapter_id": provider,
            "planned_model_requests": 2 if visual else 1, "submitted_requests": 0,
            "automatic_retries": 0, "source_reading": "visual_model" if visual else "local_embedded_text",
            "semantic_validation": "not_performed", "warnings": (
                ["Image readings and regions are unverified model candidates."] if visual else reading["warnings"]), "stages": []}
        with closing(sqlite3.connect(database, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _schema(connection, create=True)
            _runs(connection, create=True)
            current = connection.execute("SELECT active_snapshot_id FROM sro_entries WHERE collection=? AND name=?", (collection, name)).fetchone()
            active = current[0] if current else None
            if active is not None:
                for saved, in connection.execute("SELECT receipt FROM sro_ingestions WHERE request_key=? AND collection=? AND name=?",
                                                (fingerprint, collection, name)):
                    previous = json.loads(saved)
                    if previous["status"] == "complete" and previous.get("snapshot_id") == active:
                        from .local_ontology import _read_semantics
                        _read_semantics(connection, active)
                        return {**previous, "outcome": "already_active", "submitted_requests_this_call": 0}
            if active != expected_snapshot:
                raise ValueError("Entry changed; supply its expected active snapshot before generating a replacement")
            connection.execute("INSERT INTO sro_ingestions VALUES (?, ?, ?, ?, ?)",
                (receipt["run_id"], fingerprint, collection, name, canonical_json(receipt).decode()))
            connection.execute("INSERT INTO sro_ingestion_files VALUES (?, ?, ?)", (receipt["run_id"], "source.bin", data))
        first_stage = "visual" if visual else "semantics"
        _save_run(database, receipt, [(first_stage + "-request.json", canonical_json(request))])
        if not execute:
            return receipt

        def submit(stage, wire, *, saved=False):
            if not saved:
                _save_run(database, receipt, [(stage + "-request.json", canonical_json(wire))])
            receipt.update(status="running", stage=stage + "_submitting")
            receipt["submitted_requests"] += 1
            record = {"stage": stage, "request_sha256": sha256(canonical_json(wire)), "status": "submitted"}
            receipt["stages"].append(record)
            _save_run(database, receipt)
            if complete is not None:
                result = complete(wire)
                body = canonical_json(result)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Host response exceeds the bounded protocol")
            else:
                with httpx.Client(timeout=300, follow_redirects=False, transport=transport) as client:
                    with client.stream("POST", "https://api.openai.com/v1/responses", content=canonical_json(wire),
                        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"}) as reply:
                        buffer = bytearray()
                        for chunk in reply.iter_bytes(chunk_size=65536):
                            if len(buffer) + len(chunk) > MAX_RESPONSE_BYTES:
                                _save_run(database, receipt, [(stage + "-response.partial", bytes(buffer))])
                                raise ValueError("Model response exceeds 8 MiB")
                            buffer.extend(chunk)
                        body = bytes(buffer)
                        record["http_status"] = reply.status_code
                        _save_run(database, receipt, [(stage + "-response.json", body)])
                        reply.raise_for_status()
                        if reply.status_code != 200:
                            raise ValueError("Unexpected model acknowledgement")
                record.update(status="received", response_sha256=sha256(body))
                receipt["stage"] = stage + "_validating"
                _save_run(database, receipt)
                result = json.loads(body)
            if complete is not None:
                _save_run(database, receipt, [(stage + "-response.json", body)])
            record.update(status="received", response_sha256=sha256(body))
            receipt["stage"] = stage + "_validating"
            _save_run(database, receipt)
            return result

        try:
            secret = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            if complete is None and (not isinstance(secret, str) or not secret.strip()):
                raise ValueError("Credential unavailable")
            if visual:
                result = submit("visual", request, saved=True)
                inputs = _source_bundle(data, source.name, root, visual_response=result, model=model)
                request = proposal_request(inputs, model=model, version=STRUCTURAL_VERSION)
            response = submit("semantics", request, saved=not visual)
            manifest, files = render_semantic_proposal(inputs, model=model, response=response, version=STRUCTURAL_VERSION)
            _save_snapshot(root / "proposal", files, manifest)
            if manifest["proposal_status"] == "abstained":
                receipt.update(status="abstained", stage="complete", claim_count=0)
                _save_run(database, receipt, [("review.json", files["review.json"])])
                return receipt
            receipt.update(stage="activating", proposed_snapshot_id=manifest["semantic_snapshot_id"])
            _save_run(database, receipt)
            completed = {}
            def commit_receipt(connection, imported):
                completed.update(receipt)
                completed.update(status="complete", stage="complete", snapshot_id=imported["snapshot_id"],
                    claim_count=manifest["claim_count"], archive_id=imported["archive_id"], outcome=imported["outcome"],
                    readback_verified=True, review_status="unreviewed")
                updated = connection.execute("UPDATE sro_ingestions SET receipt=? WHERE run_id=?",
                    (canonical_json(completed).decode(), receipt["run_id"]))
                if updated.rowcount != 1:
                    raise ValueError("Ontology ingestion receipt disappeared")
            import_ontology(root / "proposal", database, collection=collection, name=name,
                            expected_snapshot=expected_snapshot, _on_commit=commit_receipt)
            return completed
        except httpx.HTTPStatusError as error:
            receipt.update(status="failed" if error.response.status_code < 500 else "unknown",
                failure_reason="model_request_failed", error_type=type(error).__name__)
        except (httpx.TransportError, OSError) as error:
            receipt.update(status="unknown", failure_reason="execution_outcome_unknown", error_type=type(error).__name__)
        except Exception as error:
            receipt.update(status="unknown" if receipt["stage"].endswith("_submitting") else "failed",
                failure_reason="ingestion_not_completed", error_type=type(error).__name__)
        _save_run(database, receipt)
        return receipt
