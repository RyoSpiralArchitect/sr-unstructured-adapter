"""Shared response limits and exact, source-scoped citation handles."""

from copy import deepcopy
import hashlib

from .tables import canonical_json

MAX_ACTIONS_PER_TURN = 8


def citation_catalog(inspections):
    """Issue deterministic handles only for evidence actually provided to the model."""
    references = {canonical_json(v["reference"]): v["reference"] for v in inspections}
    return {
        "ref_" + hashlib.sha256(key).hexdigest(): deepcopy(references[key])
        for key in sorted(references)
    }


def response_contract(inspections, *, remaining_tool_calls):
    """A provider-neutral contract; hosts may additionally enforce its JSON schema.

    The schema describes the preferred handle response. Exact legacy reference
    objects remain accepted by the receiver for existing callbacks and replays.
    """
    catalog = citation_catalog(inspections)
    allowed = min(MAX_ACTIONS_PER_TURN, remaining_tool_calls)
    return {
        "version": 1,
        "max_actions_per_turn": MAX_ACTIONS_PER_TURN,
        "allowed_actions_this_turn": allowed,
        "citation_catalog": catalog,
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["actions", "answer"],
            "properties": {
                "actions": {
                    "type": "array",
                    "maxItems": allowed,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "arguments"],
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                    },
                },
                "answer": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "citations"],
                            "properties": {
                                "text": {"type": "string", "minLength": 1},
                                "citations": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": list(catalog)}
                                    if catalog
                                    else {"not": {}},
                                },
                            },
                        },
                    ]
                },
            },
        },
    }


def resolve_answer(answer, inspections):
    """Expand issued handles without guessing; retain readable, unlinked answers.

    A linked citation establishes reference integrity, not semantic support.
    The host still needs to review whether that source supports the answer.
    """
    catalog = citation_catalog(inspections)
    allowed = {canonical_json(ref) for ref in catalog.values()}
    normalized = deepcopy(answer)
    linked, unlinked, issues = [], [], []
    normalized["citations"] = []
    for raw in answer["citations"]:
        ref = catalog.get(raw) if isinstance(raw, str) else raw
        if ref is not None and canonical_json(ref) in allowed:
            resolved = deepcopy(ref)
            normalized["citations"].append(resolved)
            linked.append(resolved)
        else:
            normalized["citations"].append(deepcopy(raw))
            unlinked.append(deepcopy(raw))
            issues.append(
                {"citation": deepcopy(raw), "code": "unknown_citation_handle"}
                if isinstance(raw, str)
                else {"citation": deepcopy(raw), "code": "reference_not_provided"}
            )
    return {
        "status": "complete" if linked and not unlinked else "needs_review",
        "answer": normalized,
        "raw_answer": deepcopy(answer),
        "linked_citations": linked,
        "unlinked_citations": unlinked,
        "citation_issues": issues,
        "evidence_status": "linked" if linked and not unlinked else "unlinked",
    }
