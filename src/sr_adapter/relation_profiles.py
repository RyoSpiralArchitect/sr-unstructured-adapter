"""Versioned source-to-dataset profiles, shared by all destination adapters.

Narrative assertions and visual readings retain distinct schemas and semantics.
These projections accept replay-verified source packages; public loading is in
relation_dataset_store. Adding a sink does not require parsing those packages.
"""
from __future__ import annotations

import json

from .relation_dataset import Column, RelationDataset, RelationTable
from .tables import canonical_json

NARRATIVE_COMMON = "package_id source_sha256 source_name"
NARRATIVE_COLUMNS = {
    "documents": (NARRATIVE_COMMON + " media_type model_deployment response_model instructions_version label_policy "
                  "unit_count relation_count identity_scope semantic_validation").split(),
    "entities": (NARRATIVE_COMMON + " entity_id relation_id role label type_iri literal_match identity_scope").split(),
    "relations": (NARRATIVE_COMMON + " relation_id subject_id subject_label predicate_iri predicate_label object_id "
                  "object_label assertion_status subject_label_match object_label_match grounding interpretation").split(),
    "evidence": (NARRATIVE_COMMON + " unit_id unit_order text grounding source_kind part block paragraph table_index "
                 "row_index column_index line_start line_end slide shape_id").split(),
    "evidence_links": (NARRATIVE_COMMON + " relation_id unit_id evidence_order").split(),
    "locations": (NARRATIVE_COMMON + " unit_id path value_type value_text").split(),
    "limitations": (NARRATIVE_COMMON + " limitation_index description").split(),
}
NARRATIVE_INTS = set("instructions_version unit_count relation_count unit_order block paragraph table_index row_index "
           "column_index line_start line_end slide evidence_order limitation_index".split())
NARRATIVE_BOOLS = {"literal_match", "subject_label_match", "object_label_match"}
NARRATIVE_NULLABLE = {("relations", c) for c in ("object_id", "object_label", "object_label_match")}
NARRATIVE_NULLABLE |= {("evidence", c) for c in ("part block paragraph table_index row_index column_index "
                                       "line_start line_end slide shape_id").split()}
NARRATIVE_NULLABLE.add(("locations", "value_text"))


VISUAL_COMMON = "package_id image_sha256 source_name page_label strategy"
BOX = "x0 y0 x1 y1"
VISUAL_COLUMNS = {
    "images": (VISUAL_COMMON + " width_px height_px project_endpoint contract_version model_deployment response_model observation_count relation_count "
               "coordinate_system extraction_status semantic_validation").split(),
    "items": (VISUAL_COMMON + " item_id label kind " + BOX + " reading_status").split(),
    "relations": (VISUAL_COMMON + " relation_id subject_id subject_label predicate object_id object_label value_text "
                  "direction basis support evidence_description " + BOX + " review_status").split(),
    "evidence": (VISUAL_COMMON + " relation_id item_id evidence_order label " + BOX + " reading_status").split(),
    "limitations": (VISUAL_COMMON + " stage limitation_index description").split(),
}
VISUAL_INTS = set((BOX + " width_px height_px contract_version observation_count relation_count evidence_order limitation_index").split())
VISUAL_REVIEW_COLUMNS = (VISUAL_COMMON + " finding_id reference_kind reference_id finding detail reviewer").split()


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


def _narrative_rows(packages):
    rows = {kind: [] for kind in NARRATIVE_COLUMNS}
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
            row = {c: None for c in NARRATIVE_COLUMNS["evidence"]}
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


def _visual_rows(packages, review=None):
    rows = {kind: [] for kind in VISUAL_COLUMNS}
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
        common = next({c: row[c] for c in VISUAL_COMMON.split()} for row in rows["images"] if row["package_id"] == p["package_id"])
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


NARRATIVE_KEYS = {
    "documents": ("package_id",),
    "entities": ("package_id", "entity_id"),
    "relations": ("package_id", "relation_id"),
    "evidence": ("package_id", "unit_id"),
    "evidence_links": ("package_id", "relation_id", "evidence_order"),
    "locations": ("package_id", "unit_id", "path"),
    "limitations": ("package_id", "limitation_index"),
}
VISUAL_KEYS = {
    "images": ("package_id",),
    "items": ("package_id", "item_id"),
    "relations": ("package_id", "relation_id"),
    "evidence": ("package_id", "relation_id", "evidence_order"),
    "limitations": ("package_id", "stage", "limitation_index"),
    "reviews": ("package_id", "finding_id"),
}


def _tables(columns, integers, booleans, nullable, keys):
    return {kind: RelationTable(kind, tuple(Column(c,
        "integer" if c in integers else "boolean" if c in booleans else "string", (kind, c) in nullable)
        for c in names), keys[kind], ()) for kind, names in columns.items()}


NARRATIVE_TABLES = _tables(NARRATIVE_COLUMNS, NARRATIVE_INTS, NARRATIVE_BOOLS, NARRATIVE_NULLABLE, NARRATIVE_KEYS)
VISUAL_TABLES = _tables({**VISUAL_COLUMNS, "reviews": VISUAL_REVIEW_COLUMNS}, VISUAL_INTS, set(),
    {(kind, c) for kind in [*VISUAL_COLUMNS, "reviews"] for c in ("value_text", "project_endpoint", "reference_id")},
    VISUAL_KEYS)


def _dataset(profile, package_ids, schemas, rows):
    return RelationDataset(profile, tuple(package_ids), tuple(RelationTable.from_rows(
        kind, schemas[kind].columns, schemas[kind].primary_key, group) for kind, group in rows.items()))


def narrative_dataset(packages):
    """Project replay-verified narrative packages into contract v1."""
    return _dataset("narrative_relations/1", [m["package_id"] for m, _ in packages],
                    NARRATIVE_TABLES, _narrative_rows(packages))


def visual_dataset(packages, review=None):
    """Project replay-verified visual packages without merging equal labels."""
    return _dataset("visual_relations/1", [p["package_id"] for p in packages],
                    VISUAL_TABLES, _visual_rows(packages, review))
