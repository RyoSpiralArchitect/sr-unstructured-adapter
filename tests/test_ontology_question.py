import json

import pytest

from sr_adapter.local_ontology import import_ontology
from sr_adapter.ontology_question import answer_ontology_question
from test_local_ontology import saved as saved_fixture, narrative  # noqa: F401


saved = saved_fixture


def test_question_uses_discovery_inspection_and_keeps_db_unchanged(saved, tmp_path):
    db = tmp_path / "onto.db"
    import_ontology(saved[1], db)
    original = db.read_bytes()
    calls = []

    def complete(request):
        calls.append(request)
        if len(calls) == 1:
            return {
                "actions": [
                    {
                        "name": "search_ontology",
                        "arguments": {
                            "collection": "default",
                            "query": "Handle",
                            "kind": "claim",
                        },
                    }
                ],
                "answer": None,
            }
        if len(calls) == 2:
            hit = request["history"][0]["result"]["hits"][0]
            return {"actions": [hit["inspect"]], "answer": None}
        ref = request["evidence"]["entries"][0]["inspection"]["reference"]
        return {
            "actions": [],
            "answer": {
                "text": "Source-local candidate with dissent.",
                "citations": [ref],
            },
        }

    result = answer_ontology_question(
        db,
        collections=("default",),
        question="Who handles it?",
        complete=complete,
        destination=tmp_path / "run",
    )
    assert result["status"] == "complete" and result["inspection_count"] == 1
    assert result["model_calls"] == 3 and result["tool_calls"] == 2
    assert db.read_bytes() == original
    assert (
        len(
            json.loads((tmp_path / "run/inspections.json").read_bytes())[0]["value"][
                "claims"
            ]
        )
        == 2
    )
    assert all(not r["evidence"]["entries"] for r in calls[:2])
    with pytest.raises(FileExistsError):
        answer_ontology_question(
            db,
            collections=("default",),
            question="Again?",
            complete=complete,
            destination=tmp_path / "run",
        )


def test_uninspected_citation_keeps_readable_answer_but_is_unlinked(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        destination=tmp_path / "run",
        complete=lambda request: {
            "actions": [],
            "answer": {"text": "Tentative answer", "citations": [{"id": "invented"}]},
        },
    )
    assert (
        result["status"] == "needs_review"
        and result["answer"]["text"] == "Tentative answer"
    )
    assert result["evidence_status"] == "unlinked"


def test_budget_stops_without_retry_and_failure_retains_response(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    action = {
        "name": "search_ontology",
        "arguments": {"collection": "default", "query": ""},
    }
    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        destination=tmp_path / "run",
        max_tool_calls=1,
        complete=lambda request: {"actions": [action, action], "answer": None},
    )
    assert (
        result["reason"] == "tool_budget"
        and result["model_calls"] == 1
        and result["tool_calls"] == 0
    )
    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        destination=tmp_path / "bad",
        complete=lambda request: {"unexpected": "response"},
    )
    assert result["status"] == "failed" and (tmp_path / "bad/response-00.json").exists()


def test_session_pins_catalog_before_model_changes_original(saved, tmp_path):
    import sqlite3

    db = tmp_path / "db"
    import_ontology(saved[1], db)
    seen = []

    def complete(request):
        seen.append(request)
        if len(seen) == 1:
            with sqlite3.connect(db) as connection:
                connection.execute("DELETE FROM sro_entries")
            return {
                "actions": [
                    {
                        "name": "search_ontology",
                        "arguments": {"collection": "default", "query": ""},
                    }
                ],
                "answer": None,
            }
        assert request["history"][0]["result"]["hits"]
        return {
            "actions": [],
            "answer": {"text": "Uninspected candidates exist", "citations": []},
        }

    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        complete=complete,
        destination=tmp_path / "run",
    )
    assert result["status"] == "needs_review"


def test_loop_cannot_inspect_invented_references_or_exceed_rounds(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    action = {
        "name": "inspect_ontology",
        "arguments": {
            "collection": "default",
            "name": "x",
            "snapshot_id": "a" * 64,
            "kind": "claim",
            "id": "x",
        },
    }
    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        destination=tmp_path / "run",
        max_rounds=1,
        complete=lambda request: {"actions": [action], "answer": None},
    )
    assert result["reason"] == "round_budget" and result["model_calls"] == 1
    history = json.loads((tmp_path / "run/history.json").read_bytes())
    assert history[0]["result"]["error"]["code"] == "reference_not_discovered"


def test_final_phase_selected_without_running_commentary_actions():
    from sr_adapter.ontology_question import question_response

    def message(phase, value):
        return {
            "type": "message",
            "role": "assistant",
            "phase": phase,
            "content": [{"type": "output_text", "text": json.dumps(value)}],
        }

    final = message(
        "final_answer", {"actions": [], "answer": {"text": "done", "citations": []}}
    )
    result = question_response(
        {
            "status": "completed",
            "output": [message("commentary", {"actions": ["ignore"]}), final],
        }
    )
    assert result["answer"]["text"] == "done"
    with pytest.raises(ValueError):
        question_response({"status": "completed", "output": [final, final]})


def test_malformed_later_action_retains_completed_tool_output(saved, tmp_path):
    db = tmp_path / "db"
    import_ontology(saved[1], db)
    result = answer_ontology_question(
        db,
        collections=("default",),
        question="q",
        destination=tmp_path / "run",
        complete=lambda request: {
            "actions": [
                {"name": "list_ontologies", "arguments": {"collection": "default"}},
                {"name": "broken"},
            ],
            "answer": None,
        },
    )
    assert result["status"] == "failed" and result["tool_calls"] == 1
    history = json.loads((tmp_path / "run/history.json").read_bytes())
    assert len(history) == 1 and history[0]["result"]["entries"]
