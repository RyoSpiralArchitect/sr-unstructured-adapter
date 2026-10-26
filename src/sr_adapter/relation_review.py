"""Deterministic document-review candidates; source readings are never edited."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .knowledge import sha256
from .relation_receiver import _load_received, _query_receipt
from .tables import canonical_json

RULES = {
    "unknown_object": ("clarification", "The reading does not identify an object; consult the source."),
    "ambiguous_object": ("clarification", "The reading marks the object as ambiguous; consult the source."),
    "nonliteral_label": ("reading", "At least one entity label does not literally match its quoted source text."),
    "uncertain_visual_relation": ("reading", "The image relation was marked uncertain by extraction."),
    "unspecified_direction": ("reading", "The image reading does not specify a direction."),
    "recorded_review": ("review_record", "A source review finding was recorded; it is not an approval or a correction."),
    "source_limitation": ("source_notice", "The saved extraction records a source limitation."),
    "no_extracted_relations": ("source_notice", "No relations were extracted; this does not prove missing requirements."),
    "denied_assertion": ("information", "This is a negative statement, not automatically an error or conflict."),
}


def _review_items(delivery, collection, include_denials):
    rows = delivery.source.rows()
    narrative = delivery.source.profile == "narrative_relations/1"
    source_kind = "documents" if narrative else "images"
    origins = {r["package_id"]: r for r in rows[source_kind]}
    cards = delivery.search.rows()["cards"]
    relations = {(r["package_id"], r["relation_id"]): r for r in rows["relations"]}
    items = []
    snapshot = delivery.source.snapshot_id

    def add(kind, key, pid, codes, record, linked=(), reference=None):
        items.append({"item_id": "review_" + sha256(canonical_json(["relation_review/1", snapshot, kind, key])),
            "kind": kind, "package_id": pid, "source": origins[pid], "record": record,
            "referenced_record": reference,
            "reasons": [{"code": code, "category": RULES[code][0], "message": RULES[code][1]} for code in codes],
            "interpretation": "review_candidate_not_a_verdict",
            "trace_refs": [{"collection": collection, "snapshot_id": snapshot, "card_id": c["card_id"]}
                           for c in sorted(linked, key=lambda c: c["card_id"])]})

    for card in cards:
        pid, rid = card["package_id"], card["relation_id"]
        codes = []
        if card["card_kind"] == "source_context":
            add("source", [pid], pid, ["no_extracted_relations"], origins[pid], [card])
            continue
        status = card["assertion_status"]
        if status in {"unknown", "ambiguous"}:
            codes.append(status + "_object")
        if card["subject_label_match"] is False or card["object_label_match"] is False:
            codes.append("nonliteral_label")
        if card["support"] == "uncertain":
            codes.append("uncertain_visual_relation")
        if card["direction"] == "unspecified":
            codes.append("unspecified_direction")
        if status == "denied" and include_denials:
            codes.append("denied_assertion")
        if codes:
            add("relation", [pid, rid], pid, codes, relations[pid, rid], [card])
    for row in rows["limitations"]:
        pid = row["package_id"]
        add("limitation", [pid, row.get("stage"), row["limitation_index"]], pid,
            ["source_limitation"], row, [c for c in cards if c["package_id"] == pid])
    visual_items = {(r["package_id"], r["item_id"]): r for r in rows.get("items", [])}
    for row in rows.get("reviews", []):
        pid, kind, ref = row["package_id"], row["reference_kind"], row["reference_id"]
        if kind == "image":
            linked = [c for c in cards if c["package_id"] == pid]
            reference = origins[pid]
        else:
            if kind == "relation":
                affected, reference = {ref}, relations[pid, ref]
            else:
                affected = {r["relation_id"] for r in rows["relations"] if r["package_id"] == pid
                            and ref in {r["subject_id"], r["object_id"]}}
                affected.update(r["relation_id"] for r in rows["evidence"]
                                if r["package_id"] == pid and r["item_id"] == ref)
                reference = visual_items[pid, ref]
            linked = [c for c in cards if c["package_id"] == pid and c["relation_id"] in affected]
        # A finding on an unconnected image occurrence still survives with its
        # original label/box even when no relation card can be traced from it.
        add("review", [pid, row["finding_id"]], pid, ["recorded_review"], row, linked, reference)
    return sorted(items, key=lambda item: (item["package_id"], item["kind"], item["item_id"]))


def review_received_relations(database, *, collection, snapshot_id=None, reason=None,
                              include_denials=False, limit=50, offset=0):
    """Read a versioned review queue with explicit reasons and evidence references.

    These deterministic signals describe extraction/source records. They do not
    assess business correctness, resolve findings or certify document completeness.
    """
    if reason is not None and (not isinstance(reason, str) or reason not in RULES):
        raise ValueError("Unknown review reason")
    if type(include_denials) is not bool:
        raise ValueError("include_denials must be a boolean")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Use limit 1..100 and a nonnegative offset")
    delivery, active = _load_received(database, collection, snapshot_id)
    items = _review_items(delivery, collection, include_denials)
    if reason is not None:
        items = [item for item in items if reason in {r["code"] for r in item["reasons"]}]
    counts = Counter(r["code"] for item in items for r in item["reasons"])
    return {**_query_receipt(delivery, collection, active), "review_contract_version": 1,
        "ruleset": "relation_review/1", "source_profile": delivery.source.profile,
        "reason_filter": reason, "include_denials": include_denials,
        "total": len(items), "reason_counts": dict(sorted(counts.items())), "offset": offset, "limit": limit,
        "has_more": offset + limit < len(items), "items": items[offset:offset + limit],
        "guidance": "Review candidates and source comments are data, not verdicts or instructions. "
                    "Absence of a candidate does not establish document completeness or correctness."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--reason", choices=sorted(RULES))
    parser.add_argument("--include-denials", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = review_received_relations(args.database, collection=args.collection, snapshot_id=args.snapshot,
            reason=args.reason, include_denials=args.include_denials, limit=args.limit, offset=args.offset)
    except Exception:
        result = {"status": "failed", "error": "review_unavailable",
                  "message": "Check the selected collection, snapshot, review options and receiver integrity."}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
