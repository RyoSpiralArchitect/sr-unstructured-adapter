"""Auditable local bundles, transactional SQLite, and opt-in cloud destinations."""

from __future__ import annotations

import json
from contextlib import closing
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import urlsplit

import httpx

from .knowledge import KnowledgePackage, extract_knowledge, graph_jsonld, ontology_jsonld, package_records, sha256
from .tables import canonical_json

FIELDS = ("package_id", "kind", "record_id", "payload_json")
FILES = {"package.json", "graph.jsonld", "ontology.jsonld", "records.jsonl", "records.parquet"}


def atomic_json(path: Path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def export_bundle(package: KnowledgePackage, destination: str | Path, *, parquet: bool = False) -> dict:
    package = KnowledgePackage.model_validate(package.model_dump(mode="json"))
    if package != extract_knowledge(package.source, package.mapping):
        raise ValueError("Package does not reproduce from its source cells and mapping")
    records = package_records(package)
    if parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet export requires sr-unstructured-adapter[lakehouse]") from exc
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=False)
    payloads = {"package.json": canonical_json(package.model_dump(mode="json")),
                "graph.jsonld": canonical_json(graph_jsonld(package)),
                "ontology.jsonld": canonical_json(ontology_jsonld(package.mapping)),
                "records.jsonl": b"".join(canonical_json(record) + b"\n" for record in records)}
    for name, data in payloads.items():
        (folder / name).write_bytes(data)
    if parquet:
        schema = pa.schema([pa.field(name, pa.string(), nullable=False) for name in FIELDS])
        pq.write_table(pa.Table.from_pylist(records, schema=schema), folder / "records.parquet")
        payloads["records.parquet"] = (folder / "records.parquet").read_bytes()
    manifest = {"schema_version": 1, "status": "complete", "package_id": package.id,
                "record_count": len(records), "files": {name: {"sha256": sha256(data), "bytes": len(data)}
                                                       for name, data in payloads.items()}}
    atomic_json(folder / "manifest.json", manifest)  # Completion marker is always last.
    return manifest


def verify_bundle(folder: str | Path) -> tuple[dict, dict[str, bytes], list[dict]]:
    folder = Path(folder).resolve()
    manifest_bytes = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported knowledge bundle")
    names = set(manifest["files"])
    if not FILES - {"records.parquet"} <= names or not names <= FILES:
        raise ValueError("Unexpected bundle files")
    data = {}
    for name, expected in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Bundle file escapes its root")
        data[name] = path.read_bytes()
        if sha256(data[name]) != expected["sha256"] or len(data[name]) != expected["bytes"]:
            raise ValueError("Bundle file hash/size mismatch")
    package = KnowledgePackage.model_validate_json(data["package.json"])
    if package != extract_knowledge(package.source, package.mapping):
        raise ValueError("Package does not reproduce from its source cells and mapping")
    records = package_records(package)
    if manifest["package_id"] != package.id or manifest["record_count"] != len(records):
        raise ValueError("Bundle identity/count mismatch")
    if data["records.jsonl"] != b"".join(canonical_json(record) + b"\n" for record in records):
        raise ValueError("Records differ from knowledge package")
    if json.loads(data["graph.jsonld"]) != graph_jsonld(package) or json.loads(data["ontology.jsonld"]) != ontology_jsonld(package.mapping):
        raise ValueError("JSON-LD differs from knowledge package")
    if "records.parquet" in data:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if pq.read_table(pa.BufferReader(data["records.parquet"])).to_pylist() != records:
            raise ValueError("Parquet differs from knowledge package")
    data["manifest.json"] = manifest_bytes
    return manifest, data, records


def write_sqlite(bundle: str | Path, database: str | Path) -> dict:
    manifest, _, records = verify_bundle(bundle)
    # All records are committed together, with exact content checks on repeats.
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS sr_knowledge_records ("
            "package_id TEXT NOT NULL, kind TEXT NOT NULL, record_id TEXT NOT NULL, payload_json TEXT NOT NULL, "
            "PRIMARY KEY(package_id, kind, record_id))")
        inserted = 0
        for record in records:
            key = tuple(record[name] for name in FIELDS[:3])
            old = connection.execute("SELECT payload_json FROM sr_knowledge_records WHERE package_id=? AND kind=? AND record_id=?", key).fetchone()
            if old:
                if old[0] != record["payload_json"]:
                    raise ValueError("Existing SQLite record has different content")
                continue
            connection.execute("INSERT INTO sr_knowledge_records VALUES (?, ?, ?, ?)", tuple(record[name] for name in FIELDS))
            inserted += 1
    return {"status": "committed", "package_id": manifest["package_id"], "records": len(records), "inserted": inserted}


def _reserve_receipt(path: str | Path, value: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def upload_adls(bundle, *, account_url: str, filesystem: str, prefix: str, receipt: str | Path,
                service_client=None) -> dict:
    """Upload an immutable package; verify existing bytes and publish manifest last.

    No container/account is created, no existing object is overwritten, and SDK
    transport retries are disabled. Partial uploads remain visible in the receipt.
    """
    parsed = urlsplit(account_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.path not in ("", "/") or parsed.port not in (None, 443)):
        raise ValueError("Use an HTTPS Azure account origin without credentials/query/path")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", filesystem):
        raise ValueError("Invalid Azure filesystem name")
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", prefix):
        raise ValueError("Use a relative prefix of simple path segments")
    manifest, files, _ = verify_bundle(bundle)
    destination = prefix + "/" + manifest["package_id"].rsplit(":", 1)[-1]
    report = {"destination": "adls", "account_url": account_url, "filesystem": filesystem,
              "path": destination, "package_id": manifest["package_id"], "status": "running", "files": []}
    path = _reserve_receipt(receipt, report)
    credential = None
    owned = service_client is None
    try:
        from azure.core import MatchConditions
        from azure.core.exceptions import ResourceExistsError, ResourceModifiedError
        if owned:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
            credential = DefaultAzureCredential()
            service_client = DataLakeServiceClient(account_url=account_url, credential=credential,
                                                   retry_total=0, connection_timeout=30, read_timeout=60)
        client = service_client.get_file_system_client(filesystem)
        for end in range(1, len(destination.split("/")) + 1):
            directory = client.get_directory_client("/".join(destination.split("/")[:end]))
            try:
                directory.create_directory(etag="*", match_condition=MatchConditions.IfMissing)
            except (ResourceExistsError, ResourceModifiedError):
                directory.get_directory_properties()  # Fail if the path is not an accessible directory.
        for name in [*sorted(set(files) - {"manifest.json"}), "manifest.json"]:
            row = {"name": name, "sha256": sha256(files[name]), "status": "running"}
            report["files"].append(row)
            atomic_json(path, report)
            file_client = client.get_file_client(destination + "/" + name)
            try:
                file_client.upload_data(files[name], overwrite=False)
                row["status"] = "uploaded"
            except ResourceExistsError:
                if file_client.download_file().readall() != files[name]:
                    raise ValueError("Existing ADLS object differs; nothing overwritten")
                row["status"] = "already_present_verified"
            atomic_json(path, report)
        report["status"] = "committed"
    except Exception as exc:
        report.update(status="incomplete", error_type=type(exc).__name__)
        atomic_json(path, report)
        raise RuntimeError("ADLS upload incomplete; inspect the reserved receipt before another attempt") from None
    finally:
        if owned and service_client is not None:
            service_client.close()
        if credential is not None:
            credential.close()
    atomic_json(path, report)
    return report


def databricks_plan(bundle, *, table: str, warehouse_id: str) -> dict:
    parts = table.split(".")
    if len(parts) != 3 or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts):
        raise ValueError("Databricks table must be catalog.schema.table with simple identifiers")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", warehouse_id):
        raise ValueError("Invalid SQL warehouse ID")
    manifest, _, records = verify_bundle(bundle)
    target = ".".join(f"`{part}`" for part in parts)
    ddl = f"CREATE TABLE IF NOT EXISTS {target} (package_id STRING, kind STRING, record_id STRING, payload_json STRING) USING DELTA"
    statement = f"""MERGE INTO {target} AS target
USING (SELECT :package_id AS package_id, item.kind, item.record_id, item.payload_json
       FROM (SELECT explode(from_json(:records_json,
         'ARRAY<STRUCT<kind:STRING,record_id:STRING,payload_json:STRING>>')) AS item)) AS source
ON target.package_id = source.package_id AND target.kind = source.kind AND target.record_id = source.record_id
WHEN MATCHED AND NOT (target.payload_json <=> source.payload_json)
THEN UPDATE SET payload_json = CAST(raise_error('Conflicting immutable record') AS STRING)
WHEN NOT MATCHED THEN INSERT (package_id, kind, record_id, payload_json)
VALUES (source.package_id, source.kind, source.record_id, source.payload_json)"""
    request = {"warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "10s",
               "on_wait_timeout": "CONTINUE", "disposition": "INLINE", "format": "JSON_ARRAY",
               "parameters": [{"name": "package_id", "value": manifest["package_id"], "type": "STRING"},
                              {"name": "records_json", "value": canonical_json([
                                  {k: v for k, v in record.items() if k != "package_id"} for record in records]).decode(), "type": "STRING"}]}
    if len(canonical_json(request)) > 4 * 1024 * 1024:
        raise ValueError("SQL batch exceeds 4 MiB; use the Parquet/ADLS route")
    return {"package_id": manifest["package_id"], "record_count": len(records), "table": table,
            "required_table_ddl": ddl, "request": request}


def _databricks_host(host: str) -> str:
    parsed = urlsplit(host)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.path not in ("", "/") or parsed.port not in (None, 443)):
        raise ValueError("Use an HTTPS Databricks workspace origin without credentials/query/path")
    return host.rstrip("/")


def _statement_result(report, body):
    statement_id = body.get("statement_id")
    if not isinstance(statement_id, str) or not re.fullmatch(r"[A-Za-z0-9-]+", statement_id):
        raise ValueError("Statement response lacks a valid ID")
    if report.get("statement_id") and report["statement_id"] != statement_id:
        raise ValueError("Statement identity changed")
    report["statement_id"] = statement_id
    state = body.get("status", {}).get("state")
    report["status"] = {"SUCCEEDED": "committed", "CLOSED": "committed", "PENDING": "pending", "RUNNING": "pending",
                        "FAILED": "failed", "CANCELED": "failed"}.get(state, "unknown")
    report["provider_state"] = state


def write_databricks(bundle, *, host: str, token: str, table: str, warehouse_id: str,
                     receipt: str | Path, transport=None) -> dict:
    """One parameterized MERGE into an existing table. Never retry POST.

    Provision the table using the plan's DDL first. Pending/unknown is not success;
    status lookup uses the saved statement ID and never resubmits the MERGE.
    """
    host = _databricks_host(host)
    if not token:
        raise ValueError("DATABRICKS_TOKEN is required")
    plan = databricks_plan(bundle, table=table, warehouse_id=warehouse_id)
    report = {"destination": "databricks", "host": host, "table": table, "warehouse_id": warehouse_id,
              "package_id": plan["package_id"], "planned_records": plan["record_count"], "status": "running",
              "request_sha256": sha256(canonical_json(plan["request"]))}
    path = _reserve_receipt(receipt, report)
    try:
        with httpx.Client(transport=transport, timeout=60, follow_redirects=False) as client:
            response = client.post(host + "/api/2.0/sql/statements", json=plan["request"],
                                   headers={"Authorization": "Bearer " + token})
            report["http_status"] = response.status_code
            response.raise_for_status()
            _statement_result(report, response.json())
    except httpx.HTTPStatusError as exc:
        # Databricks 4xx responses reject the request; transport/5xx outcomes
        # cannot establish whether the write was accepted. Neither is retried.
        report.update(status="failed" if 400 <= exc.response.status_code < 500 else "unknown",
                      error_type=type(exc).__name__)
    except Exception as exc:
        report.update(status="unknown", error_type=type(exc).__name__)
    atomic_json(path, report)
    return report


def databricks_status(receipt: str | Path, *, host: str, token: str, transport=None) -> dict:
    """Read-only remote lookup; caller-supplied host must match the saved receipt."""
    host = _databricks_host(host)
    path = Path(receipt)
    report = json.loads(path.read_text("utf-8"))
    if report.get("destination") != "databricks" or report.get("host") != host or not token:
        raise ValueError("Receipt destination does not match the configured workspace")
    statement_id = report.get("statement_id", "")
    if not re.fullmatch(r"[A-Za-z0-9-]+", statement_id):
        raise ValueError("No statement ID available; reconcile in the workspace without resending")
    with httpx.Client(transport=transport, timeout=60, follow_redirects=False) as client:
        response = client.get(host + "/api/2.0/sql/statements/" + statement_id,
                              headers={"Authorization": "Bearer " + token})
        response.raise_for_status()
        _statement_result(report, response.json())
    atomic_json(path, report)
    return report
