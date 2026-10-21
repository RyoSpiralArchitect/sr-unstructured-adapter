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


COMMON = "package_id source_sha256 source_name"
COLUMNS = {
    "documents": (COMMON + " media_type model_deployment response_model instructions_version label_policy "
                  "unit_count relation_count identity_scope semantic_validation").split(),
    "entities": (COMMON + " entity_id relation_id role label type_iri literal_match identity_scope").split(),
    "relations": (COMMON + " relation_id subject_id subject_label predicate_iri predicate_label object_id "
                  "object_label assertion_status subject_label_match object_label_match grounding interpretation").split(),
    "evidence": (COMMON + " unit_id unit_order text grounding source_kind part block paragraph table_index "
                 "row_index column_index line_start line_end slide shape_id").split(),
    "evidence_links": (COMMON + " relation_id unit_id evidence_order").split(),
    "locations": (COMMON + " unit_id path value_type value_text").split(),
    "limitations": (COMMON + " limitation_index description").split(),
}
INTS = set("instructions_version unit_count relation_count unit_order block paragraph table_index row_index "
           "column_index line_start line_end slide evidence_order limitation_index".split())
BOOLS = {"literal_match", "subject_label_match", "object_label_match"}
NULLABLE = {("relations", c) for c in ("object_id", "object_label", "object_label_match")}
NULLABLE |= {("evidence", c) for c in ("part block paragraph table_index row_index column_index "
                                       "line_start line_end slide shape_id").split()}
NULLABLE.add(("locations", "value_text"))


def _schema(kind):
    import pyarrow as pa
    return pa.schema([pa.field(c, pa.int64() if c in INTS else pa.bool_() if c in BOOLS else pa.string(),
                              nullable=(kind, c) in NULLABLE) for c in COLUMNS[kind]])


def _locations(value, path=""):
    """JSON Pointer paths and typed scalar values, including empty containers."""
    if isinstance(value, (dict, list)) and value:
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in entries:
            segment = str(key).replace("~", "~0").replace("/", "~1")
            yield from _locations(child, path + "/" + segment)
    else:
        kind = ("null" if value is None else "boolean" if isinstance(value, bool)
                else "integer" if isinstance(value, int) else "number" if isinstance(value, float)
                else "string" if isinstance(value, str) else "array" if isinstance(value, list) else "object")
        text = (None if value is None or isinstance(value, (dict, list))
                else str(value).lower() if isinstance(value, bool) else str(value))
        yield {"path": path, "value_type": kind, "value_text": text}


def _projection(packages):
    rows = {kind: [] for kind in COLUMNS}
    for manifest, files in packages:
        source = json.loads(files["source.json"])
        config = json.loads(files["extraction.json"])
        assertions = json.loads(files["assertions.json"])
        common = {"package_id": manifest["package_id"],
                  "source_sha256": source["source_sha256"], "source_name": source["source_name"]}
        rows["documents"].append(dict(common, media_type=source["media_type"],
            model_deployment=config["model_deployment"], response_model=config["response_model"],
            instructions_version=config.get("instructions_version", 1), label_policy=config.get("label_policy", "literal"),
            unit_count=len(source["units"]), relation_count=len(assertions),
            identity_scope=config["identity_scope"], semantic_validation=config["semantic_validation"]))
        for assertion in assertions:
            identity = assertion["assertion_id"]
            # Literal bundles predate explicit match fields and enforce both matches.
            matches = {role: assertion.get(role + "_label_match", assertion[role] is not None)
                       for role in ("subject", "object")}
            rows["relations"].append(dict(common, relation_id=identity,
                subject_id=assertion["subject_mention_id"], subject_label=assertion["subject"],
                predicate_iri=assertion["predicate"], predicate_label=assertion["predicate_label"],
                object_id=assertion["object_mention_id"], object_label=assertion["object"],
                assertion_status=assertion["status"], subject_label_match=matches["subject"],
                object_label_match=matches["object"] if assertion["object"] is not None else None,
                grounding=assertion["grounding"], interpretation=assertion["interpretation"]))
            for role in ("subject", "object"):
                if assertion[role] is not None:
                    rows["entities"].append(dict(common, entity_id=assertion[role + "_mention_id"],
                        relation_id=identity, role=role, label=assertion[role], type_iri=assertion[role + "_type"],
                        literal_match=matches[role], identity_scope="source_assertion_mention"))
            for order, evidence in enumerate(assertion["evidence"], 1):
                rows["evidence_links"].append(dict(common, relation_id=identity,
                    unit_id=evidence["unit_id"], evidence_order=order))
        for order, unit in enumerate(source["units"], 1):
            locator = unit["locator"]
            cell = locator.get("cell", {})
            row = {c: None for c in COLUMNS["evidence"]}
            row.update(common, unit_id=unit["id"], unit_order=order, text=unit["text"], grounding=unit["grounding"],
                source_kind=locator["kind"], table_index=locator.get("table"),
                row_index=cell.get("row", locator.get("row")), column_index=cell.get("column"))
            row.update({c: locator.get(c) for c in ("part", "block", "paragraph", "line_start", "line_end", "slide", "shape_id")})
            rows["evidence"].append(row)
            rows["locations"].extend(dict(common, unit_id=unit["id"], **entry) for entry in _locations(locator))
        rows["limitations"].extend(dict(common, limitation_index=i, description=warning)
                                   for i, warning in enumerate(source["warnings"], 1))
    for group in rows.values():
        group.sort(key=canonical_json)
    return rows


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
