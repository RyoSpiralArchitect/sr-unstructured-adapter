"""Compare pinned source interpretations without asserting identity or precedence."""

from __future__ import annotations

from .knowledge import sha256
from .local_ontology import inspect_ontology
from .ontology_context import pack_ontology_context
from .tables import canonical_json

FIELDS = (
    "status",
    "modality",
    "basis",
    "attributed_to",
    "context",
    "valid_from",
    "valid_until",
)


def _primary(inspection):
    value = inspection["value"]
    claim = next(c for c in value["claims"] if c["id"] == inspection["reference"]["id"])
    expressions = {e["id"]: e for e in value["expressions"]}
    pending, reachable = [claim["expression_id"]], set()
    while pending:
        identity = pending.pop()
        if identity in reachable:
            continue
        reachable.add(identity)
        pending.extend(expressions[identity]["operands"])
        pending.extend(
            a["value"]["expression_id"]
            for a in expressions[identity]["arguments"]
            if a["value"]["kind"] == "expression"
        )
    selected = [e for e in value["expressions"] if e["id"] in reachable]
    symbols = set()
    for expression in selected:
        if expression["predicate"] is not None:
            symbols.add(expression["predicate"])
        for argument in expression["arguments"]:
            symbols.add(argument["role"])
            if argument["value"]["kind"] == "term":
                symbols.add(argument["value"]["iri"])
    return claim, selected, [t for t in value["terms"] if t["iri"] in symbols]


def compare_ontology_claims(database, *, left, right, include_evidence=True):
    """Expose differences for a selected pair. This is not contradiction detection."""
    for reference in (left, right):
        if (
            type(reference) is not dict
            or set(reference) != {"collection", "name", "snapshot_id", "kind", "id"}
            or reference["kind"] != "claim"
        ):
            raise ValueError("Compare two exact pinned claim references")
    if type(include_evidence) is not bool:
        raise ValueError("Use a boolean evidence selection")
    inspections = [inspect_ontology(database, **ref) for ref in (left, right)]
    claims, expressions, terms = zip(*(_primary(value) for value in inspections))
    same_entry = (left["collection"], left["name"]) == (
        right["collection"],
        right["name"],
    )
    differences = [
        {"field": field, "left": claims[0][field], "right": claims[1][field]}
        for field in FIELDS
        if canonical_json(claims[0][field]) != canonical_json(claims[1][field])
    ]
    groups = []
    for side in terms:
        grouped = {}
        for term in side:
            grouped.setdefault(term["label"], []).append(term)
        groups.append(grouped)
    pair_count = sum(
        len(values) * len(groups[1].get(label, ()))
        for label, values in groups[0].items()
    )
    if pair_count > 1000:
        raise ValueError(
            "Shared-label comparison exceeds 1000 pairs; inspect narrower claims"
        )
    meanings = [
        {
            "label": label,
            "left": a,
            "right": b,
            "declarations_equal": a == b,
            "identity_established": False,
        }
        for label, values in groups[0].items()
        for a in values
        for b in groups[1].get(label, ())
    ]
    result = {
        "version": 1,
        "status": "complete",
        "references": [left, right],
        "pair_id": sha256(canonical_json({"left": left, "right": right})),
        "scope_relation": "same_reference"
        if left == right
        else "same_entry_versions"
        if same_entry and left["snapshot_id"] != right["snapshot_id"]
        else "separate_claims",
        "claims": list(claims),
        "expressions": list(expressions),
        "scope_differences": differences,
        "shared_labels": meanings,
        "identity_established": False,
        "contradiction_established": False,
        "precedence": "unresolved",
        "semantic_validation": "not_performed",
        "inspect": [
            {"name": "inspect_ontology", "arguments": ref} for ref in (left, right)
        ],
        "guidance": "Compare whole claims and source evidence, including conditions, drafts and dissent. Differences are review candidates, not proof of contradiction. Same labels/IRIs do not merge entities. A newer snapshot alone does not establish normative priority.",
    }
    if include_evidence:
        result["evidence"] = pack_ontology_context(inspections)
    return result


def review_ontology_comparison(comparison, *, category, explanation):
    """Attach an explicitly unverified host/model interpretation to a comparison."""
    allowed = {
        "different_sense",
        "different_condition",
        "different_authority_or_revision",
        "compatible",
        "uncertain",
    }
    if (
        category not in allowed
        or not isinstance(explanation, str)
        or not 1 <= len(explanation) <= 8192
    ):
        raise ValueError("Use a supported interpretation and bounded explanation")
    return {
        "comparison_sha256": sha256(canonical_json(comparison)),
        "pair_id": comparison["pair_id"],
        "references": comparison["references"],
        "category": category,
        "explanation": explanation,
        "review_status": "unverified",
        "identity_established": False,
        "precedence": "unresolved",
    }
