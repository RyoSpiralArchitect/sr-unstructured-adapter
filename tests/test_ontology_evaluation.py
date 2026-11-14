from copy import deepcopy
import json
from pathlib import Path

import pytest

from sr_adapter.local_ontology import import_ontology, list_ontologies, search_ontology
from sr_adapter.ontology_evaluation import (
    MODEL,
    ProviderRefusal,
    prepare_evaluation,
    run_evaluation_cell,
    score_evaluation,
)
from test_local_ontology import saved as saved_fixture, narrative  # noqa: F401

saved = saved_fixture


@pytest.fixture
def prepared(saved, tmp_path):
    db = tmp_path / "source.sqlite"
    import_ontology(saved[1], db)
    entry = list_ontologies(db)["entries"][0]
    refs = [h["reference"] for h in search_ontology(db, kind="claim")["hits"]]
    suite = {
        "version": 1,
        "id": "test-suite",
        "collection": "default",
        "corpus": [
            {
                "name": entry["name"],
                "snapshot_id": entry["snapshot_id"],
                "source_sha256": entry["sources"][0]["source_sha256"],
            }
        ],
        "source_units": {"X1": "Fixture source unit"},
        "questions": [
            {
                "id": f"q-{i}",
                "category": category,
                "question": "Who handles it?",
                "evidence_groups": [
                    {
                        "label": ref["id"],
                        "source_unit": "X1",
                        "acceptable_references": [ref],
                    }
                    for ref in refs
                ],
                "rubric": [
                    {"id": "r1", "kind": "required", "description": "GOLD_SENTINEL"},
                    {
                        "id": "f1",
                        "kind": "forbidden",
                        "description": "FALSE_MERGE_SENTINEL",
                    },
                    {
                        "id": "grounding",
                        "kind": "required",
                        "description": "Cited context supports material assertions",
                    },
                ],
            }
            for i, category in enumerate(
                ["identity"] * 3
                + ["condition"] * 3
                + ["revision"] * 3
                + ["unknown"] * 3
            )
        ],
    }
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(suite), encoding="utf-8")
    root = tmp_path / "evaluation"
    before = db.read_bytes()
    result = prepare_evaluation(db, source, root)
    assert db.read_bytes() == before
    return root, result["protocol_id"], refs, suite


def envelope(response, usage=True):
    return {
        "model": MODEL,
        "response": response,
        "usage": {"input_tokens": 100, "output_tokens": 10} if usage else None,
    }


def final(refs):
    return {"actions": [], "answer": {"text": "Fixture answer", "citations": refs}}


def test_frozen_pair_is_gold_blind_ordered_and_preserves_complete_denominator(prepared):
    root, identity, refs, suite = prepared
    calls = []

    def complete(wire):
        assert wire["model"] == MODEL and wire["reasoning"] == {"effort": "medium"}
        assert "GOLD_SENTINEL" not in json.dumps(wire)
        assert "FALSE_MERGE_SENTINEL" not in json.dumps(wire)
        calls.append(wire)
        request = wire["input"]
        if not request["tools"]:
            assert request["evidence"]["entries"]
            return envelope(final(refs))
        if not request["history"]:
            return envelope(
                {
                    "actions": [
                        {
                            "name": "search_ontology",
                            "arguments": {
                                "collection": "default",
                                "query": "",
                                "kind": "claim",
                            },
                        }
                    ],
                    "answer": None,
                }
            )
        if len(request["history"]) == 1:
            return envelope(
                {
                    "actions": [
                        h["inspect"] for h in request["history"][0]["result"]["hits"]
                    ],
                    "answer": None,
                }
            )
        return envelope(final(refs))

    with pytest.raises(ValueError, match="order"):
        run_evaluation_cell(root, identity, "q-0", "retrieval", complete=complete)
    a = run_evaluation_cell(root, identity, "q-0", "all_evidence", complete=complete)
    b = run_evaluation_cell(root, identity, "q-0", "retrieval", complete=complete)
    assert a["result"]["status"] == b["result"]["status"] == "complete"
    assert (a["model_calls"], b["model_calls"]) == (1, 3)
    scores = score_evaluation(root, identity)
    assert len(scores["cells"]) == 24
    assert scores["arms"]["retrieval"]["statuses"] == {"complete": 1, "not_run": 11}
    assert all(r["semantic_result"] == "unreviewed" for r in scores["cells"])
    assert all(p["retrieval_minus_all_input_tokens"] is None for p in scores["pairs"])
    assert [r["cited_reference_groups"] for r in scores["cells"][:2]] == [2, 2]
    reviews = {
        "protocol_id": identity,
        "judgments": [
            {
                "question_id": r["question_id"],
                "arm": r["arm"],
                "receipt_sha256": r["receipt_sha256"],
                "checks": {"r1": "pass", "f1": "pass", "grounding": "pass"},
            }
            for r in scores["cells"][:2]
        ],
    }
    checked = score_evaluation(root, identity, reviews)
    assert checked["pairs"][0]["retrieval_minus_all_input_tokens"] == 200
    reviews["judgments"][1]["checks"]["grounding"] = "fail"
    assert not score_evaluation(root, identity, reviews)["pairs"][0]["quality_matched"]
    reviews["judgments"][1]["checks"]["grounding"] = "pass"
    reviews["judgments"][1]["checks"]["f1"] = "fail"
    assert (
        score_evaluation(root, identity, reviews)["pairs"][0][
            "retrieval_minus_all_input_tokens"
        ]
        is None
    )
    reviews["judgments"][0]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale"):
        score_evaluation(root, identity, reviews)
    with pytest.raises(FileExistsError):
        run_evaluation_cell(root, identity, "q-0", "all_evidence", complete=complete)
    assert len(calls) == 4


@pytest.mark.parametrize(
    "filename", ["suite.json", "ontology.sqlite", "all-evidence.json", "protocol.json"]
)
def test_changed_input_is_rejected_before_any_callback(prepared, filename):
    root, identity, _, _ = prepared
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        run_evaluation_cell(
            root,
            identity,
            "q-0",
            "all_evidence",
            complete=lambda _: pytest.fail("called"),
        )


def test_unknown_usage_and_invented_citations_are_not_success_or_zero_tokens(prepared):
    root, identity, refs, _ = prepared
    wrong = {**refs[0], "snapshot_id": "0" * 64}
    r = run_evaluation_cell(
        root,
        identity,
        "q-0",
        "all_evidence",
        complete=lambda _: envelope(final([wrong]), usage=False),
    )
    assert r["result"]["status"] == "needs_review" and r["result"]["answer"]["text"]
    assert r["usage"] is None and r["calls_with_unknown_usage"] == 1
    scores = score_evaluation(root, identity)
    assert scores["cells"][0]["cited_reference_groups"] == 0
    assert scores["arms"]["all_evidence"]["cells_with_known_usage"] == 0


def test_provider_refusal_records_failure_and_stops_later_cells(prepared):
    root, identity, _, _ = prepared
    calls = []

    def reject(wire):
        calls.append(wire)
        raise ProviderRefusal("rejected fixture")

    receipt = run_evaluation_cell(
        root, identity, "q-0", "all_evidence", complete=reject
    )
    assert receipt["result"]["status"] == "failed" and receipt["model_calls"] == 1
    assert (root / "STOP.json").exists()
    with pytest.raises(ValueError, match="stopped"):
        run_evaluation_cell(root, identity, "q-0", "retrieval", complete=reject)
    scores = score_evaluation(root, identity)
    assert scores["arms"]["all_evidence"]["statuses"] == {"failed": 1, "not_run": 11}
    assert len(calls) == 1


def test_unfinished_search_keeps_round_budget_failure_and_token_counts(prepared):
    root, identity, refs, _ = prepared
    run_evaluation_cell(
        root, identity, "q-0", "all_evidence", complete=lambda _: envelope(final(refs))
    )
    receipt = run_evaluation_cell(
        root,
        identity,
        "q-0",
        "retrieval",
        complete=lambda _: envelope(
            {
                "actions": [
                    {
                        "name": "search_ontology",
                        "arguments": {"collection": "default", "query": ""},
                    }
                ],
                "answer": None,
            }
        ),
    )
    assert receipt["result"]["reason"] == "round_budget" and receipt["model_calls"] == 6
    assert receipt["usage"] == {"input_tokens": 600, "output_tokens": 60}
    assert receipt["provided_references"] == []
    assert score_evaluation(root, identity)["cells"][1]["status"] == "incomplete"


def test_interruption_retains_submitted_calls_and_partial_usage(prepared):
    root, identity, refs, _ = prepared
    run_evaluation_cell(
        root, identity, "q-0", "all_evidence", complete=lambda _: envelope(final(refs))
    )
    attempts = []

    def interrupt(wire):
        attempts.append(wire)
        if len(attempts) == 2:
            raise KeyboardInterrupt()
        return envelope(
            {
                "actions": [
                    {
                        "name": "search_ontology",
                        "arguments": {"collection": "default", "query": ""},
                    }
                ],
                "answer": None,
            }
        )

    with pytest.raises(KeyboardInterrupt):
        run_evaluation_cell(root, identity, "q-0", "retrieval", complete=interrupt)
    scores = score_evaluation(root, identity)
    row = scores["cells"][1]
    assert row["status"] == "interrupted" and row["model_calls"] == 2
    assert row["usage"] is None and row["calls_with_unknown_usage"] == 1
    assert row["known_usage"] == {"input_tokens": 100, "output_tokens": 10}
    assert row["tool_calls"] is None and row["known_tool_calls_lower_bound"] == 1
    assert row["receipt_sha256"] is None
    assert scores["arms"]["retrieval"]["statuses"] == {"interrupted": 1, "not_run": 11}
    with pytest.raises(FileExistsError):
        run_evaluation_cell(root, identity, "q-0", "retrieval", complete=interrupt)
    assert len(attempts) == 2


def test_handwritten_suite_matches_original_fixture_units_and_balances_formats():
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "fixtures", repo / "scripts/cross_format_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = json.loads((repo / "evaluations/ontology_questions_v1.json").read_bytes())
    expected = {
        f"{p}{i}": text
        for p, profile in module.PROFILES.items()
        for i, text in enumerate(profile["units"], 1)
    }
    assert suite["source_units"] == expected
    assert len(suite["questions"]) == 12 and len(suite["corpus"]) == 6
    from sr_adapter.ontology_evaluation import _validate_suite

    _validate_suite(suite)
    for q in suite["questions"]:
        assert any(r["kind"] == "forbidden" for r in q["rubric"])
        assert any(
            r["id"] == "grounding" and r["kind"] == "required" for r in q["rubric"]
        )
        names = {
            r["name"] for g in q["evidence_groups"] for r in g["acceptable_references"]
        }
        assert len(names) == 2 and len({Path(n).suffix for n in names}) == 2
        assert all(n in q["question"] for n in names)
    duplicate = deepcopy(suite)
    duplicate["questions"][1]["id"] = duplicate["questions"][0]["id"]
    with pytest.raises(ValueError):
        _validate_suite(duplicate)
