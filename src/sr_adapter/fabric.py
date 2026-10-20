"""Verified, source-scoped knowledge snapshots for an existing Fabric lakehouse."""

from __future__ import annotations

import io
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, field_validator

from .knowledge import KnowledgePackage, sha256
from .knowledge_store import FIELDS, atomic_json, verify_bundle
from .tables import Record, canonical_json


class FabricTarget(Record):
    workspace_id: str
    lakehouse_id: str
    lakehouse_name: str = Field(min_length=1)
    schema_name: str = Field(default="dbo", pattern=r"^[A-Za-z0-9_]{1,128}$")
    onelake_endpoint: str = "https://onelake.dfs.fabric.microsoft.com"

    @field_validator("workspace_id", "lakehouse_id")
    @classmethod
    def guid(cls, value):
        if str(UUID(value)) != value:
            raise ValueError("Use a canonical lowercase GUID")
        return value

    @field_validator("onelake_endpoint")
    @classmethod
    def storage_origin(cls, value):
        url = urlsplit(value)
        if (url.scheme != "https" or url.username or url.password or url.query or url.fragment
                or url.port not in (None, 443) or url.path not in ("", "/")
                or not re.fullmatch(r"(?:[a-z0-9]+-)?onelake\.dfs\.fabric\.microsoft\.com", url.hostname or "")):
            raise ValueError("Use a Microsoft OneLake HTTPS DFS origin")
        return value.rstrip("/")


TABLE_COLUMNS = {
    "records": list(FIELDS),
    "entities": ["package_id", "entity_id", "type_iri", "label", "identity_scope"],
    "relations": ["package_id", "relation_id", "subject_id", "predicate_iri", "object_id"],
    "observations": ["package_id", "observation_id", "subject_id", "predicate_iri", "value", "status"],
    "evidence": ["package_id", "evidence_id", "kind", "table_id", "document_sha256", "page",
                 "row_index", "column_index", "row_span", "column_span", "text", "bbox_json", "coordinate_system"],
    "evidence_links": ["package_id", "assertion_kind", "assertion_id", "evidence_id", "evidence_order"],
}
INTEGER_COLUMNS = {"page", "row_index", "column_index", "row_span", "column_span", "evidence_order"}
NULLABLE = {("observations", "subject_id"), ("observations", "value")}
NULLABLE |= {("evidence", name) for name in ("row_index", "column_index", "row_span", "column_span", "text")}
MAX_EXPORT_BYTES = 64 * 1024 * 1024


def rows_digest(rows):
    """Order-independent content digest; duplicate rows are retained and detectable."""
    return sha256(canonical_json(sorted(rows, key=canonical_json)))


def _arrow_schema(kind):
    import pyarrow as pa
    return pa.schema([pa.field(name, pa.int64() if name in INTEGER_COLUMNS else pa.string(),
                               nullable=(kind, name) in NULLABLE) for name in TABLE_COLUMNS[kind]])


def _projection(bundles):
    records, packages, sources = [], {}, {}
    for bundle in bundles:
        manifest, files, package_rows = verify_bundle(bundle)
        key = manifest["package_id"].rsplit(":", 1)[-1]
        if key in packages:
            raise ValueError("Duplicate package in Fabric snapshot")
        packages[key] = KnowledgePackage.model_validate_json(files["package.json"])
        sources[key] = files
        records.extend(package_rows)
    if not packages:
        raise ValueError("At least one verified package is required")
    rows = {kind: [] for kind in TABLE_COLUMNS}
    rows["records"] = records
    for package in packages.values():
        pid = package.id
        for entity in package.entities:
            rows["entities"].append(dict(package_id=pid, entity_id=entity.id, type_iri=entity.type,
                                         label=entity.label, identity_scope=entity.identity_scope))
        for relation in package.relations:
            rows["relations"].append(dict(package_id=pid, relation_id=relation.id, subject_id=relation.subject,
                                          predicate_iri=relation.predicate, object_id=relation.object))
        for observation in package.observations:
            rows["observations"].append(dict(package_id=pid, observation_id=observation.id,
                subject_id=observation.subject, predicate_iri=observation.predicate,
                value=observation.value, status=observation.status))
        for kind, group in (("entity", package.entities), ("relation", package.relations),
                            ("observation", package.observations)):
            for assertion in group:
                rows["evidence_links"].extend(dict(package_id=pid, assertion_kind=kind, assertion_id=assertion.id,
                    evidence_id=evidence_id, evidence_order=index) for index, evidence_id in enumerate(assertion.evidence))
        for table in package.source.tables:
            common = dict(package_id=pid, table_id=table.id, document_sha256=package.source.source_sha256,
                          page=table.page, coordinate_system=table.coordinate_system)
            rows["evidence"].append(dict(**common, evidence_id=table.id, kind="table", row_index=None,
                column_index=None, row_span=None, column_span=None, text=None,
                bbox_json=canonical_json(table.bbox.model_dump()).decode()))
            for cell in table.cells:
                rows["evidence"].append(dict(**common, evidence_id=cell.id, kind="cell", row_index=cell.row,
                    column_index=cell.column, row_span=cell.row_span, column_span=cell.column_span,
                    text=cell.text, bbox_json=canonical_json(cell.bbox.model_dump()).decode()))
    for group in rows.values():
        group.sort(key=canonical_json)
    snapshot = sha256(canonical_json({"schema_version": 1,
        "packages": sorted(p.id for p in packages.values()),
        "tables": {kind: rows_digest(group) for kind, group in rows.items()}}))
    return snapshot, sources, rows


def _plan(snapshot, rows, target):
    prefix = f"Files/sr-adapter/{snapshot}"
    tables = []
    for kind, group in rows.items():
        name = f"sr_{snapshot}_{kind}"
        tables.append({"kind": kind, "table_name": name, "row_count": len(group),
            "rows_sha256": rows_digest(group), "parquet_file": f"tables/{kind}.parquet",
            "load_url": f"https://api.fabric.microsoft.com/v1/workspaces/{target.workspace_id}"
                        f"/lakehouses/{target.lakehouse_id}/schemas/{target.schema_name}/tables/{name}/load?beta=true",
            "load_body": {"relativePath": f"{prefix}/tables/{kind}.parquet", "pathType": "File",
                          "mode": "Append", "recursive": False, "formatOptions": {"format": "Parquet"}}})
    return {"schema_version": 1, "snapshot_id": snapshot, "target": target.model_dump(mode="json"),
            "files_prefix": prefix, "tables": tables,
            "semantics": "single_writer_snapshot; append only to absent tables; existing tables require exact readback"}


def export_fabric(bundles, destination, *, target: FabricTarget):
    import pyarrow as pa
    import pyarrow.parquet as pq
    snapshot, sources, rows = _projection(bundles)
    files = {"target.json": canonical_json(target.model_dump(mode="json")),
             "deployment-plan.json": canonical_json(_plan(snapshot, rows, target))}
    for key, source_files in sources.items():
        files.update({f"bundles/{key}/{name}": data for name, data in source_files.items()})
    for kind, group in rows.items():
        output = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(group, schema=_arrow_schema(kind)), output)
        files[f"tables/{kind}.parquet"] = output.getvalue()
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Fabric snapshot exceeds the 64 MiB bounded publisher limit")
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        output = folder / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    manifest = {"schema_version": 1, "status": "complete", "snapshot_id": snapshot,
        "package_hashes": sorted(sources),
        "tables": {kind: {"rows": len(group), "rows_sha256": rows_digest(group)} for kind, group in rows.items()},
        "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in files.items()}}
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def verify_fabric(folder, *, target: FabricTarget):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(folder).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") == 2 and manifest.get("mode") == "pptx_tabular":
        from .fabric_tabular import verify_tabular
        return verify_tabular(folder, target=target)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Incomplete Fabric export")
    keys = manifest["package_hashes"]
    if not keys or any(not re.fullmatch(r"[0-9a-f]{64}", key) for key in keys):
        raise ValueError("Invalid package directory")
    files = {}
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Fabric file escapes export")
        data = path.read_bytes()
        if sha256(data) != expected["sha256"] or len(data) != expected["bytes"]:
            raise ValueError("Fabric file hash/size mismatch")
        files[name] = data
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Fabric export too large")
    snapshot, sources, rows = _projection([root / "bundles" / key for key in keys])
    expected_names = {"target.json", "deployment-plan.json"} | {f"tables/{kind}.parquet" for kind in rows}
    expected_names |= {f"bundles/{key}/{name}" for key, source in sources.items() for name in source}
    if set(files) != expected_names or snapshot != manifest["snapshot_id"]:
        raise ValueError("Fabric snapshot identity/files mismatch")
    if json.loads(files["target.json"]) != target.model_dump(mode="json"):
        raise ValueError("Pass the explicit target matching this export")
    plan = _plan(snapshot, rows, target)
    if json.loads(files["deployment-plan.json"]) != plan:
        raise ValueError("Fabric deployment plan differs from verified records")
    if manifest["tables"] != {kind: {"rows": len(group), "rows_sha256": rows_digest(group)} for kind, group in rows.items()}:
        raise ValueError("Fabric row manifest mismatch")
    for kind, group in rows.items():
        table = pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"]))
        if table.schema != _arrow_schema(kind) or table.to_pylist() != group:
            raise ValueError("Fabric Parquet differs from source package")
    files["manifest.json"] = manifest_bytes
    return manifest, files, plan
