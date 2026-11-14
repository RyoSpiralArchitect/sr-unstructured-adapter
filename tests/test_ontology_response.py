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


def test_target_scope_keeps_counterevidence_separate():
    from sr_adapter.ontology_response import source_scope

    a, b = inspection(), inspection(name="other.docx")
    scope = [source_scope(a["reference"])]
    contract = response_contract([a, b], remaining_tool_calls=8, target_sources=scope)
    primary = next(iter(contract["citation_catalog"]))
    other = next(iter(contract["supplemental_catalog"]))
    good = resolve_answer(
        {
            "text": "A says X; B disagrees",
            "citations": [primary],
            "supplemental_citations": [other],
        },
        [a, b],
        target_sources=scope,
    )
    assert good["status"] == "complete"
    assert good["linked_citations"] == [a["reference"]]
    assert good["supplemental_citations"] == [b["reference"]]
    wrong = resolve_answer(
        {"text": "A says X", "citations": [other]}, [a, b], target_sources=scope
    )
    assert wrong["status"] == "needs_review"
    assert not wrong["linked_citations"]
    assert wrong["answer"]["text"] == "A says X"
    missing = resolve_answer(
        {"text": "only B", "citations": [], "supplemental_citations": [other]},
        [a, b],
        target_sources=scope,
    )
    assert missing["status"] == "needs_review"
    assert any(
        i["code"] == "target_source_not_cited" for i in missing["citation_issues"]
    )


@pytest.mark.parametrize("scope", [[], [{}], [{"name": "x"}]])
def test_incomplete_target_scopes_are_rejected(scope):
    with pytest.raises(ValueError):
        response_contract([], remaining_tool_calls=8, target_sources=scope)
