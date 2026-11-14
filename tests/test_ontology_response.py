from copy import deepcopy

import pytest

from sr_adapter.ontology_response import (
    citation_catalog,
    resolve_answer,
    response_contract,
)


def inspection(**overrides):
    return {
        "reference": {
            "collection": "local",
            "name": "same.pdf",
            "snapshot_id": "a" * 64,
            "kind": "claim",
            "id": "same-id",
            **overrides,
        }
    }


@pytest.mark.parametrize(
    "changed",
    [
        {"collection": "other"},
        {"name": "same.docx"},
        {"snapshot_id": "b" * 64},
        {"kind": "entity"},
        {"id": "other-id"},
    ],
)
def test_handles_keep_entire_scope_and_do_not_retarget(changed):
    a, b = inspection(), inspection(**changed)
    handle_a = next(iter(citation_catalog([a])))
    handle_b = next(iter(citation_catalog([b])))
    assert handle_a != handle_b
    answer = {
        "text": "Readable result",
        "citations": [handle_a, handle_b, b["reference"]],
    }
    result = resolve_answer(answer, [a])
    assert result["status"] == "needs_review"
    assert result["linked_citations"] == [a["reference"]]
    assert result["unlinked_citations"] == [handle_b, b["reference"]]
    assert result["raw_answer"] == answer
    assert result["answer"]["text"] == answer["text"]
    assert [i["code"] for i in result["citation_issues"]] == [
        "unknown_citation_handle",
        "reference_not_provided",
    ]


def test_catalog_order_duplicates_and_mutation_cannot_change_scope():
    a, b = inspection(), inspection(name="other.pdf")
    before = deepcopy([a, b])
    catalog = citation_catalog([a, b, a])
    assert catalog == citation_catalog([b, a])
    assert len(catalog) == 2
    next(iter(catalog.values()))["name"] = "mutated"
    assert [a, b] == before
    result = resolve_answer(
        {"text": "x", "citations": list(citation_catalog([a]))}, [a]
    )
    result["linked_citations"][0]["name"] = "mutated"
    assert [a, b] == before


@pytest.mark.parametrize(
    "remaining,expected", [(0, 0), (1, 1), (7, 7), (8, 8), (24, 8)]
)
def test_contract_tracks_remaining_budget_and_only_exposes_inspected_handles(
    remaining, expected
):
    a = inspection()
    contract = response_contract([a], remaining_tool_calls=remaining)
    assert contract["allowed_actions_this_turn"] == expected
    schema = contract["response_schema"]["properties"]
    assert schema["actions"]["maxItems"] == expected
    assert schema["answer"]["anyOf"][1]["properties"]["citations"]["items"][
        "enum"
    ] == list(citation_catalog([a]))
    empty = response_contract([], remaining_tool_calls=remaining)
    assert not empty["citation_catalog"]
    assert (
        resolve_answer({"text": "x", "citations": list(citation_catalog([a]))}, [])[
            "status"
        ]
        == "needs_review"
    )
