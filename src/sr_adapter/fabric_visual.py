"""Scalar Fabric tables for image candidates; no nested/JSON table columns."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re

from .fabric import MAX_EXPORT_BYTES, _plan, rows_digest
from .knowledge import sha256
from .knowledge_store import atomic_json
from .tables import canonical_json
from .visual_relations import verify_visual, project_visual

from .relation_profiles import (VISUAL_COLUMNS as COLUMNS, VISUAL_INTS as INTS,
                               VISUAL_REVIEW_COLUMNS as REVIEW_COLUMNS, VISUAL_TABLES, visual_dataset)

__all__ = ["COLUMNS", "INTS", "REVIEW_COLUMNS", "export_visual", "verify_visual_fabric"]


def _schema(kind):
    return VISUAL_TABLES[kind].arrow_schema()


def _projection(packages, review=None):
    rows = visual_dataset(packages, review).rows()
    kinds = [*COLUMNS, *(["reviews"] if review is not None else [])]
    return {kind: rows[kind] for kind in kinds}


def _snapshot(packages, rows):
    return sha256(canonical_json({"mode": "visual_tabular_v1", "packages": sorted(p["package_id"] for p in packages),
                                 "tables": {k: rows_digest(v) for k, v in rows.items()}}))


def export_visual(bundles, destination, *, target, review=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not 1 <= len(bundles) <= 20:
        raise ValueError("Select 1 to 20 image candidate bundles")
    packages, files = [], {}
    for bundle in bundles:
        package, sources = verify_visual(bundle)
        packages.append(package)
        files.update({f"sources/{package['package_id']}/{n}": v for n, v in sources.items()})
    rows = _projection(packages, review)
    snapshot = _snapshot(packages, rows)
    files["target.json"] = canonical_json(target.model_dump(mode="json"))
    files["deployment-plan.json"] = canonical_json(_plan(snapshot, {k: v for k, v in rows.items() if v}, target))
    if review is not None:
        files["review.json"] = canonical_json(review)
    for kind, group in rows.items():
        output = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(group, schema=_schema(kind)), output)
        files[f"tables/{kind}.parquet"] = output.getvalue()
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Visual Fabric snapshot exceeds 64 MiB")
    manifest = {"schema_version": 3, "mode": "visual_tabular", "status": "complete", "snapshot_id": snapshot,
                "package_ids": sorted(p["package_id"] for p in packages),
                "tables": {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()},
                "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in files.items()}}
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    atomic_json(root / "manifest.json", manifest)
    verify_visual_fabric(root, target=target)
    return manifest


def verify_visual_fabric(folder, *, target):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(folder).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 3 or manifest.get("mode") != "visual_tabular" or manifest.get("status") != "complete":
        raise ValueError("Incomplete visual Fabric snapshot")
    keys = manifest["package_ids"]
    if (not 1 <= len(keys) <= 20 or len(keys) != len(set(keys))
            or any(not re.fullmatch(r"[a-f0-9]{64}", k) for k in keys)):
        raise ValueError("Invalid visual package selection")
    files, total = {}, 0
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root or "\\" in name or ".." in name.split("/"):
            raise ValueError("Visual file escapes export")
        with path.open("rb") as stream:
            data = stream.read(MAX_EXPORT_BYTES - total + 1)
        total += len(data)
        if total > MAX_EXPORT_BYTES or expected != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError("Visual snapshot size/hash mismatch")
        files[name] = data
    packages, expected_names = [], {"target.json", "deployment-plan.json"} | {f"tables/{k}.parquet" for k in COLUMNS}
    review = json.loads(files["review.json"]) if "review.json" in files else None
    if "review.json" in files:
        if review is None:
            raise ValueError("A saved review cannot be null")
        expected_names.update({"review.json", "tables/reviews.parquet"})
    for key in keys:
        prefix = f"sources/{key}/"
        source_files = {n[len(prefix):]: v for n, v in files.items() if n.startswith(prefix)}
        saved = json.loads(source_files.pop("manifest.json"))
        package = project_visual(source_files)
        if (key != package["package_id"] or saved != {"version": package["config"]["version"], "status": "complete", "package_id": key,
                "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in source_files.items()}}):
            raise ValueError("Visual source receipt mismatch")
        packages.append(package)
        expected_names.update(prefix + n for n in [*source_files, "manifest.json"])
    rows = _projection(packages, review)
    snapshot = _snapshot(packages, rows)
    plan = _plan(snapshot, {k: v for k, v in rows.items() if v}, target)
    if (set(files) != expected_names or manifest["snapshot_id"] != snapshot
            or json.loads(files["target.json"]) != target.model_dump(mode="json")
            or json.loads(files["deployment-plan.json"]) != plan
            or manifest["tables"] != {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()}):
        raise ValueError("Visual Fabric projection differs from manifest/target/plan")
    for kind, group in rows.items():
        table = pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"]))
        if table.schema != _schema(kind) or table.to_pylist() != group:
            raise ValueError("Visual table differs from saved model candidates")
    return manifest, {**files, "manifest.json": manifest_bytes}, plan
