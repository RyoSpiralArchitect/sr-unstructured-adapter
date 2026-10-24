"""Scalar Fabric tables from replayable document-relation candidates.

All source units remain available as context, including units not cited by an
edge. Locator trees use scalar path/value rows; no JSON columns or entity joins
based on names are introduced.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re

from .fabric import MAX_EXPORT_BYTES, _plan, rows_digest
from .knowledge import sha256
from .knowledge_store import atomic_json
from .narrative import verify_narrative, verify_narrative_payloads
from .tables import canonical_json


# Keep the historical Fabric schema and load-plan ordering byte-compatible.
from .relation_profiles import NARRATIVE_COLUMNS as COLUMNS, NARRATIVE_TABLES, narrative_dataset


def _schema(kind):
    return NARRATIVE_TABLES[kind].arrow_schema()


def _projection(packages):
    rows = narrative_dataset(packages).rows()
    return {kind: rows[kind] for kind in COLUMNS}


def _snapshot(packages, rows):
    return sha256(canonical_json({"mode": "narrative_tabular_v1",
        "packages": sorted(m["package_id"] for m, _ in packages),
        "tables": {k: rows_digest(v) for k, v in rows.items()}}))


def export_narrative_fabric(bundles, destination, *, target):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not 1 <= len(bundles) <= 20:
        raise ValueError("Select 1 to 20 narrative bundles")
    packages = [verify_narrative(bundle) for bundle in bundles]
    keys = [m["package_id"].rsplit(":", 1)[-1] for m, _ in packages]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate narrative package")
    rows = _projection(packages)
    snapshot = _snapshot(packages, rows)
    files = {"target.json": canonical_json(target.model_dump(mode="json")),
             "deployment-plan.json": canonical_json(_plan(snapshot, {k: v for k, v in rows.items() if v}, target))}
    for key, (_, source_files) in zip(keys, packages):
        files.update({f"sources/{key}/{name}": data for name, data in source_files.items()})
    for kind, group in rows.items():
        stream = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(group, schema=_schema(kind)), stream)
        files[f"tables/{kind}.parquet"] = stream.getvalue()
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Narrative Fabric snapshot exceeds 64 MiB")
    manifest = {"schema_version": 4, "mode": "narrative_tabular", "status": "complete", "snapshot_id": snapshot,
        "package_hashes": sorted(keys),
        "tables": {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()},
        "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in files.items()}}
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    atomic_json(root / "manifest.json", manifest)
    verify_narrative_fabric(root, target=target)
    return manifest


def verify_narrative_fabric(folder, *, target):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(folder).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 4 or manifest.get("mode") != "narrative_tabular" or manifest.get("status") != "complete":
        raise ValueError("Incomplete narrative Fabric snapshot")
    keys = manifest["package_hashes"]
    if (not 1 <= len(keys) <= 20 or len(set(keys)) != len(keys)
            or any(not re.fullmatch(r"[a-f0-9]{64}", k) for k in keys)):
        raise ValueError("Invalid narrative package selection")
    files, total = {}, 0
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root or "\\" in name or ".." in name.split("/"):
            raise ValueError("Narrative file escapes export")
        with path.open("rb") as stream:
            data = stream.read(MAX_EXPORT_BYTES - total + 1)
        total += len(data)
        if total > MAX_EXPORT_BYTES or expected != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError("Narrative snapshot size/hash mismatch")
        files[name] = data
    expected_names = {"target.json", "deployment-plan.json"} | {f"tables/{k}.parquet" for k in COLUMNS}
    packages = []
    for key in keys:
        prefix = f"sources/{key}/"
        source_files = {n[len(prefix):]: v for n, v in files.items() if n.startswith(prefix)}
        source_manifest = verify_narrative_payloads(source_files)
        if source_manifest["package_id"].rsplit(":", 1)[-1] != key:
            raise ValueError("Narrative package ID differs from directory")
        packages.append((source_manifest, source_files))
        expected_names.update(prefix + n for n in source_files)
    rows = _projection(packages)
    snapshot = _snapshot(packages, rows)
    plan = _plan(snapshot, {k: v for k, v in rows.items() if v}, target)
    if (set(files) != expected_names or manifest["snapshot_id"] != snapshot
            or json.loads(files["target.json"]) != target.model_dump(mode="json")
            or json.loads(files["deployment-plan.json"]) != plan
            or manifest["tables"] != {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()}):
        raise ValueError("Narrative Fabric projection differs from manifest/target/plan")
    for kind, group in rows.items():
        table = pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"]))
        if table.schema != _schema(kind) or table.to_pylist() != group:
            raise ValueError("Narrative table differs from saved source/candidates")
    return manifest, {**files, "manifest.json": manifest_bytes}, plan
