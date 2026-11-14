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


def response_contract(inspections, *, remaining_tool_calls, target_sources=None):
    """A provider-neutral contract; hosts may additionally enforce its JSON schema.

    The schema describes the preferred handle response. Exact legacy reference
    objects remain accepted by the receiver for existing callbacks and replays.
    """
    target_sources = validate_target_sources(target_sources)
    catalog = citation_catalog(inspections)
    primary = {
        h: r
        for h, r in catalog.items()
        if target_sources is None or source_scope(r) in target_sources
    }
    allowed = min(MAX_ACTIONS_PER_TURN, remaining_tool_calls)
    contract = {
        "version": 1,
        "max_actions_per_turn": MAX_ACTIONS_PER_TURN,
        "allowed_actions_this_turn": allowed,
        "citation_catalog": primary,
        "target_sources": target_sources,
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
                                    "items": {"type": "string", "enum": list(primary)}
                                    if primary
                                    else {"not": {}},
                                },
                            },
                        },
                    ]
                },
            },
        },
    }

    if target_sources is not None:
        supplemental = {h: r for h, r in catalog.items() if h not in primary}
        contract["supplemental_catalog"] = supplemental
        props = contract["response_schema"]["properties"]["answer"]["anyOf"][1][
            "properties"
        ]
        props["supplemental_citations"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(supplemental)}
            if supplemental
            else {"not": {}},
        }
    return contract


def source_scope(reference):
    return {k: reference[k] for k in ("collection", "name", "snapshot_id")}


def validate_target_sources(sources):
    if sources is None:
        return None
    if not isinstance(sources, list) or not sources:
        raise ValueError("Use a nonempty list of exact target source scopes")
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != {"collection", "name", "snapshot_id"}
            or any(not isinstance(v, str) or not v.strip() for v in source.values())
        ):
            raise ValueError("Target sources require collection, name and snapshot_id")
    return deepcopy(sources)


def resolve_answer(answer, inspections, *, target_sources=None):
    """Expand issued handles without guessing; retain readable, unlinked answers.

    A linked citation establishes reference integrity, not semantic support.
    The host still needs to review whether that source supports the answer.
    """
    target_sources = validate_target_sources(target_sources)
    inspections = list(inspections)
    catalog = citation_catalog(inspections)
    allowed = {canonical_json(ref) for ref in catalog.values()}
    normalized = deepcopy(answer)
    linked, unlinked, issues = [], [], []
    normalized["citations"] = []
    for raw in answer["citations"]:
        ref = catalog.get(raw) if isinstance(raw, str) else raw
        if (
            ref is not None
            and canonical_json(ref) in allowed
            and (target_sources is None or source_scope(ref) in target_sources)
        ):
            resolved = deepcopy(ref)
            normalized["citations"].append(resolved)
            linked.append(resolved)
        else:
            normalized["citations"].append(deepcopy(raw))
            unlinked.append(deepcopy(raw))
            issues.append(
                {
                    "citation": deepcopy(raw),
                    "code": "primary_outside_target"
                    if ref is not None and canonical_json(ref) in allowed
                    else "unknown_citation_handle",
                }
                if isinstance(raw, str)
                else {
                    "citation": deepcopy(raw),
                    "code": "primary_outside_target"
                    if ref is not None and canonical_json(ref) in allowed
                    else "reference_not_provided",
                }
            )
    supplemental = []
    if target_sources is not None:
        for raw in answer.get("supplemental_citations", []):
            ref = catalog.get(raw) if isinstance(raw, str) else raw
            if (
                ref is not None
                and canonical_json(ref) in allowed
                and source_scope(ref) not in target_sources
            ):
                supplemental.append(deepcopy(ref))
            else:
                unlinked.append(deepcopy(raw))
                issues.append(
                    {"citation": deepcopy(raw), "code": "invalid_supplemental_citation"}
                )
        normalized["supplemental_citations"] = supplemental
        missing = [
            scope
            for scope in target_sources
            if not any(source_scope(ref) == scope for ref in linked)
        ]
        issues.extend(
            {"code": "target_source_not_cited", "source": scope} for scope in missing
        )
    return {
        "supplemental_citations": supplemental,
        "status": "complete"
        if linked and not unlinked and not issues
        else "needs_review",
        "answer": normalized,
        "raw_answer": deepcopy(answer),
        "linked_citations": linked,
        "unlinked_citations": unlinked,
        "citation_issues": issues,
        "evidence_status": "linked"
        if linked and not unlinked and not issues
        else "unlinked",
    }
