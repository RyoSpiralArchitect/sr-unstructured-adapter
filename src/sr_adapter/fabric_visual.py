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

COMMON = "package_id image_sha256 source_name page_label strategy"
BOX = "x0 y0 x1 y1"
COLUMNS = {
    "images": (COMMON + " width_px height_px project_endpoint contract_version model_deployment response_model observation_count relation_count "
               "coordinate_system extraction_status semantic_validation").split(),
    "items": (COMMON + " item_id label kind " + BOX + " reading_status").split(),
    "relations": (COMMON + " relation_id subject_id subject_label predicate object_id object_label value_text "
                  "direction basis support evidence_description " + BOX + " review_status").split(),
    "evidence": (COMMON + " relation_id item_id evidence_order label " + BOX + " reading_status").split(),
    "limitations": (COMMON + " stage limitation_index description").split(),
}
INTS = set((BOX + " width_px height_px contract_version observation_count relation_count evidence_order limitation_index").split())
REVIEW_COLUMNS = (COMMON + " finding_id reference_kind reference_id finding detail reviewer").split()


def _schema(kind):
    import pyarrow as pa
    return pa.schema([pa.field(c, pa.int64() if c in INTS else pa.string(),
                              nullable=c in {"value_text", "project_endpoint", "reference_id"})
                      for c in (REVIEW_COLUMNS if kind == "reviews" else COLUMNS[kind])])


def _projection(packages, review=None):
    rows = {kind: [] for kind in COLUMNS}
    seen = set()
    for package in packages:
        key = package["package_id"]
        if key in seen:
            raise ValueError("Duplicate visual package")
        seen.add(key)
        info, config = package["source"], package["config"]
        common = dict(package_id=key, image_sha256=info["source_sha256"], source_name=info["source_name"],
                      page_label=config["page_label"], strategy=config["strategy"])
        nodes = {n["id"]: n for n in package["inventory"]["observations"]}
        relations = package["relations"]["relations"]
        def scoped(value):
            return key + ":" + value
        rows["images"].append(dict(common, width_px=info["width_px"], height_px=info["height_px"],
            project_endpoint=config.get("project_endpoint"), contract_version=config["version"],
            model_deployment=config["model"], response_model=";".join(package["response_models"]),
            observation_count=len(nodes), relation_count=len(relations), coordinate_system="normalized_0_1000_top_left",
            extraction_status="complete", semantic_validation="not_performed"))
        for n in nodes.values():
            rows["items"].append(dict(common, item_id=scoped(n["id"]), label=n["label"], kind=n["kind"],
                                     **n["box"], reading_status="model_proposed_unverified"))
        for r in relations:
            rows["relations"].append(dict(common, relation_id=scoped(r["id"]), subject_id=scoped(r["subject_id"]),
                subject_label=nodes[r["subject_id"]]["label"], predicate=r["predicate"], object_id=scoped(r["object_id"]),
                object_label=nodes[r["object_id"]]["label"], value_text=r["value_text"], direction=r["direction"],
                basis=r["basis"], support=r["support"], evidence_description=r["evidence_description"], **r["box"],
                review_status="unreviewed_model_candidate"))
            for order, identity in enumerate(r["evidence_ids"], 1):
                n = nodes[identity]
                rows["evidence"].append(dict(common, relation_id=scoped(r["id"]), item_id=scoped(identity),
                    evidence_order=order, label=n["label"], **n["box"], reading_status="model_proposed_unverified"))
        for stage in ("inventory", "relations"):
            for index, description in enumerate(package[stage]["limitations"], 1):
                rows["limitations"].append(dict(common, stage=stage, limitation_index=index, description=description))
    if review is not None:
        _apply_review(rows, packages, review)
    for group in rows.values():
        group.sort(key=canonical_json)
    return rows


def _apply_review(rows, packages, review):
    """Keep review findings separate; never replace a model reading or approve a claim."""
    if (not isinstance(review, dict) or set(review) != {"version", "reviewer", "findings"}
            or type(review["version"]) is not int or review["version"] != 1
            or not isinstance(review["reviewer"], str) or not 1 <= len(review["reviewer"].strip()) <= 200
            or not isinstance(review["findings"], list) or not 1 <= len(review["findings"]) <= 500):
        raise ValueError("Invalid visual review record")
    by_id = {p["package_id"]: p for p in packages}
    rows["reviews"] = []
    for index, finding in enumerate(review["findings"], 1):
        if (not isinstance(finding, dict) or set(finding) != {"package_id", "reference_kind", "reference_id", "finding", "detail"}
                or finding["package_id"] not in by_id or finding["reference_kind"] not in {"image", "item", "relation"}
                or finding["finding"] not in {"text_mismatch", "unsupported_relation", "ambiguous_relation", "coverage_gap"}
                or not isinstance(finding["detail"], str) or not 1 <= len(finding["detail"].strip()) <= 2000):
            raise ValueError("Invalid visual review finding")
        p = by_id[finding["package_id"]]
        kind, ref = finding["reference_kind"], finding["reference_id"]
        valid = ({n["id"] for n in p["inventory"]["observations"]} if kind == "item"
                 else {r["id"] for r in p["relations"]["relations"]} if kind == "relation" else {None})
        if ref not in valid:
            raise ValueError("Review reference does not exist in selected image")
        scoped = p["package_id"] + ":" + ref if ref is not None else None
        common = next({c: row[c] for c in COMMON.split()} for row in rows["images"] if row["package_id"] == p["package_id"])
        rows["reviews"].append(dict(common, finding_id=f"finding-{index}", reference_kind=kind,
            reference_id=scoped, finding=finding["finding"], detail=finding["detail"], reviewer=review["reviewer"]))
        if kind == "item":
            for row in [*rows["items"], *rows["evidence"]]:
                if row["item_id"] == scoped:
                    row["reading_status"] = "model_proposed_needs_review"
        affected = {row["relation_id"] for row in rows["evidence"] if kind == "item" and row["item_id"] == scoped}
        if kind == "relation":
            affected.add(scoped)
        for row in rows["relations"]:
            if row["relation_id"] in affected:
                row["review_status"] = "model_candidate_needs_review"


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
