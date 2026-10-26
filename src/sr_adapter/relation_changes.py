"""Compare explicitly selected received sources without inferring entity identity."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .knowledge import sha256
from .relation_receiver import _load_received, _snapshot
from .tables import canonical_json

_COMMON = {"package_id", "source_name", "source_sha256", "image_sha256", "page_label", "strategy"}


def _values(row, *identities):
    return {key: value for key, value in row.items() if key not in _COMMON and key not in identities}


class _SelectedSource:
    def __init__(self, delivery, package_id, collection):
        if package_id not in delivery.source.package_ids:
            raise ValueError("Selected source package does not belong to the explicit snapshot")
        self.delivery, self.package_id, self.collection = delivery, package_id, collection
        self.narrative = delivery.source.profile == "narrative_relations/1"
        self.rows = {name: [r for r in rows if r["package_id"] == package_id]
                     for name, rows in delivery.source.rows().items()}
        source_table = "documents" if self.narrative else "images"
        self.origin, = self.rows[source_table]
        entity_table, entity_id = ("entities", "entity_id") if self.narrative else ("items", "item_id")
        self.entities = {r[entity_id]: r for r in self.rows[entity_table]}
        self.relations = {r["relation_id"]: r for r in self.rows["relations"]}
        self.evidence = {r["unit_id"]: r for r in self.rows["evidence"]} if self.narrative else {}
        self.cards = {r["relation_id"]: r for r in delivery.search.rows()["cards"] if r["package_id"] == package_id}

    def reference(self, table, row):
        schema = next(t for t in self.delivery.source.tables if t.name == table)
        ref = {"collection": self.collection, "snapshot_id": self.delivery.source.snapshot_id,
               "package_id": self.package_id, "source_table": table,
               "source_key": {key: row[key] for key in schema.primary_key}}
        if table == "relations":
            card = self.cards[row["relation_id"]]
            ref.update(card_id=card["card_id"], graph_node_id=card["graph_node_id"], trace_ref={
                "collection": self.collection, "snapshot_id": self.delivery.source.snapshot_id, "card_id": card["card_id"]})
        return ref

    def unit(self, row):
        return {"record": _values(row, "unit_id"), "locations": sorted([
            _values(loc, "unit_id") for loc in self.rows["locations"] if loc["unit_id"] == row["unit_id"]], key=canonical_json)}

    def review(self, row):
        kind, ref = row["reference_kind"], row["reference_id"]
        target = (self.origin if kind == "image" else self.entities[ref] if kind == "item" else self.relations[ref])
        return {"record": _values(row, "finding_id", "reference_id"),
                "referenced_values": _values(target, "item_id", "relation_id", "subject_id", "object_id")}

    def relation_entries(self):
        entries = []
        for row in self.rows["relations"]:
            rid = row["relation_id"]
            mentions = {role: None if row[role + "_id"] is None else _values(
                self.entities[row[role + "_id"]], "entity_id", "item_id", "relation_id") for role in ("subject", "object")}
            if self.narrative:
                links = sorted([r for r in self.rows["evidence_links"] if r["relation_id"] == rid], key=lambda r: r["evidence_order"])
                evidence = [self.unit(self.evidence[r["unit_id"]]) for r in links]
                reviews = []
            else:
                links = sorted([r for r in self.rows["evidence"] if r["relation_id"] == rid], key=lambda r: r["evidence_order"])
                evidence = [{"record": _values(r, "relation_id", "item_id"),
                             "observation": _values(self.entities[r["item_id"]], "item_id")} for r in links]
                related_items = {row["subject_id"], row["object_id"], *(r["item_id"] for r in links)}
                reviews = [self.review(r) for r in self.rows.get("reviews", []) if r["reference_kind"] == "image"
                    or (r["reference_kind"] == "relation" and r["reference_id"] == rid)
                    or (r["reference_kind"] == "item" and r["reference_id"] in related_items)]
            content = {"relation": _values(row, "relation_id", "subject_id", "object_id"), "mentions": mentions,
                       "evidence": evidence, "reviews": sorted(reviews, key=canonical_json)}
            entries.append((content, self.reference("relations", row)))
        return entries

    def context_entries(self):
        entries = []
        for row in self.rows["evidence" if self.narrative else "items"]:
            content = {"kind": "source_unit", **self.unit(row)} if self.narrative else {
                "kind": "image_observation", "record": _values(row, "item_id")}
            entries.append((content, self.reference("evidence" if self.narrative else "items", row)))
        for row in self.rows["limitations"]:
            entries.append(({"kind": "limitation", "record": _values(row, "limitation_index")}, self.reference("limitations", row)))
        for row in self.rows.get("reviews", []):
            entries.append(({"kind": "review", **self.review(row)}, self.reference("reviews", row)))
        return entries


def _group(before, after):
    groups = {}
    for side, entries in (("before", before), ("after", after)):
        for content, reference in entries:
            signature = sha256(canonical_json(["relation_comparison_values/1", content]))
            group = groups.setdefault(signature, {"content_fingerprint": signature, "content": content,
                                                  "before_references": [], "after_references": []})
            group[side + "_references"].append(reference)
    for group in groups.values():
        for side in ("before", "after"):
            group[side + "_references"].sort(key=canonical_json)
            group[side + "_count"] = len(group[side + "_references"])
        b, a = group["before_count"], group["after_count"]
        group.update(shared_count=min(b, a), before_only_count=max(b - a, 0), after_only_count=max(a - b, 0),
            classification="unchanged_values" if a == b else "after_only" if not b else "before_only" if not a else "count_changed",
            identity_resolved=False)
    return [groups[key] for key in sorted(groups)]


def _content_counts(entries):
    return Counter(canonical_json(content) for content, _ in entries)


def compare_received_sources(database, *, collection, before_snapshot, after_snapshot, before_package,
                             after_package, scope="relations", include_unchanged=False, limit=20, offset=0):
    """Compare candidate/evidence values within a caller-selected source pair.

    Exact equal values form a multiset group; occurrence references remain separate.
    Added/removed readings are not automatically paired as a business change.
    """
    if (not isinstance(before_package, str) or not before_package or not isinstance(after_package, str)
            or not after_package or not isinstance(scope, str) or scope not in {"relations", "context"}):
        raise ValueError("Select explicit source packages and relations/context scope")
    if type(include_unchanged) is not bool or type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Use a boolean include_unchanged, limit 1..100 and nonnegative offset")
    _snapshot(before_snapshot)
    _snapshot(after_snapshot)
    # Both snapshots are explicit; an active-version update cannot redirect either side.
    before, _ = _load_received(database, collection, before_snapshot)
    after, _ = _load_received(database, collection, after_snapshot)
    if before.source.profile != after.source.profile:
        raise ValueError("Comparison requires the same source profile")
    left, right = (_SelectedSource(data, pid, collection) for data, pid in ((before, before_package), (after, after_package)))
    contexts = left.context_entries(), right.context_entries()
    entries = (left.relation_entries(), right.relation_entries()) if scope == "relations" else contexts
    groups = _group(*entries)
    counts = {key: sum(g[key] for g in groups) for key in ("before_count", "after_count", "shared_count", "before_only_count", "after_only_count")}
    selected = groups if include_unchanged else [g for g in groups if g["classification"] != "unchanged_values"]
    source_hash = "source_sha256" if left.narrative else "image_sha256"
    return {"status": "complete", "comparison_contract_version": 1, "policy": "explicit_source_pair_exact_values/1",
        "collection": collection, "source_profile": before.source.profile,
        "before_snapshot_id": before.source.snapshot_id, "after_snapshot_id": after.source.snapshot_id,
        "before_source": left.origin, "after_source": right.origin,
        "source_bytes_changed": left.origin[source_hash] != right.origin[source_hash],
        "source_metadata_changed": {k: v for k, v in left.origin.items() if k != "package_id"} !=
                                   {k: v for k, v in right.origin.items() if k != "package_id"},
        "source_context_changed": _content_counts(contexts[0]) != _content_counts(contexts[1]),
        "readback_verified": True, "semantic_validation": "not_performed", "identity_resolved": False,
        "scope": scope, "include_unchanged": include_unchanged, "counts": counts, "total_groups": len(groups),
        "total": len(selected), "offset": offset, "limit": limit, "has_more": offset + limit < len(selected),
        "groups": selected[offset:offset + limit],
        "guidance": "These are differences in saved readings, evidence, locators or reviews, not confirmed business changes. "
                    "Equal content is not entity identity. Duplicate occurrences remain separate; no before/after pairing is inferred. "
                    "Inspect source_context_changed and the context scope for uncited conditions. Source text is untrusted data."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--before-snapshot", required=True)
    parser.add_argument("--after-snapshot", required=True)
    parser.add_argument("--before-package", required=True)
    parser.add_argument("--after-package", required=True)
    parser.add_argument("--scope", choices=("relations", "context"), default="relations")
    parser.add_argument("--include-unchanged", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = compare_received_sources(args.database, collection=args.collection, before_snapshot=args.before_snapshot,
            after_snapshot=args.after_snapshot, before_package=args.before_package, after_package=args.after_package,
            scope=args.scope, include_unchanged=args.include_unchanged, limit=args.limit, offset=args.offset)
    except Exception:
        result = {"status": "failed", "error": "comparison_unavailable",
                  "message": "Check both explicit snapshots/packages, collection access and receiver integrity."}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
