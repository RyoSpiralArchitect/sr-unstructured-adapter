"""Agent tools expose only explicitly bound, replay-verified receiver data."""
import json
import sqlite3

import pytest

from sr_adapter.relation_agent import RelationEvidenceTools, main
from sr_adapter.relation_receiver import SQLiteRelationReceiver
from test_fabric_narrative import narrative as narrative
from test_relation_receiver import delivered as delivered, revision, pair
from test_relation_dataset import visual as visual, source as source


@pytest.fixture
def tools(delivered, tmp_path):
    db = tmp_path / "tools.db"
    SQLiteRelationReceiver(db).receive(delivered, collection="documents")
    return RelationEvidenceTools(db, ("documents",))


def test_search_summary_links_to_complete_evidence_and_never_writes(tools):
    before = tools.database.read_bytes()
    found = tools.call("search_relations", {"collection": "documents", "query": "", "assertion_status": "denied"})
    assert found["total"] == 1 and found["hits"][0]["assertion_status"] == "denied"
    hit = found["hits"][0]
    assert "content" not in hit and hit["content_included"] is False
    traced = tools.call(hit["trace"]["name"], hit["trace"]["arguments"])
    assert traced["status"] == "complete" and traced["snapshot_id"] == found["snapshot_id"]
    assert [e["properties"]["text"] for e in traced["evidence"]] == ["A", "B is not the owner"]
    assert "Only applies after approval." in traced["card"]["content"]
    assert traced["assertion"]["properties"]["subject_label_match"] is False
    assert tools.database.read_bytes() == before


def test_search_trace_stays_on_original_snapshot_after_update(tools, narrative, tmp_path):
    found = tools.call("search_relations", {"collection": "documents", "query": ""})
    new = revision(narrative, tmp_path / "new")
    SQLiteRelationReceiver(tools.database).receive(new, collection="documents", expected_snapshot=found["snapshot_id"])
    hit = found["hits"][0]
    trace = tools.call("trace_relation", hit["trace"]["arguments"])
    assert trace["snapshot_id"] == found["snapshot_id"] != trace["active_snapshot_id"]
    assert trace["card"]["card_id"] == hit["card_id"]


@pytest.mark.parametrize("arguments", [
    {"collection": "documents", "query": "", "database": "/other.db"},
    {"collection": "documents", "query": "", "sql": "DROP TABLE example"},
    {"collection": "documents", "query": "", "limit": True},
    {"collection": "documents", "query": "", "limit": 21},
    {"collection": "documents", "query": "", "offset": -1},
    {"collection": "documents", "query": "", "snapshot_id": "latest"},
    {"collection": "documents", "query": []},
    {"collection": "documents", "query": "", "assertion_status": {}},
])
def test_strict_argument_validation_precedes_receiver_access(tmp_path, arguments):
    tool = RelationEvidenceTools(tmp_path / "not-created.db", ("documents",))
    assert tool.call("search_relations", arguments)["error"]["code"] == "invalid_arguments"
    assert not tool.database.exists()


def test_collection_allowlist_unknown_tools_and_required_snapshot(tools):
    assert tools.call("search_relations", {"collection": "other", "query": ""})["error"]["code"] == "collection_not_allowed"
    assert tools.call("execute_sql", {})["error"]["code"] == "unknown_tool"
    assert tools.call("trace_relation", {"collection": "documents", "card_id": "c_" + "0" * 64})["error"]["code"] == "invalid_arguments"
    schema = tools.definitions()
    assert schema[0]["input_schema"]["properties"]["collection"]["enum"] == ["documents"]
    schema[0]["input_schema"]["properties"]["collection"]["enum"].append("other")
    assert tools.definitions()[0]["input_schema"]["properties"]["collection"]["enum"] == ["documents"]


def test_oversize_trace_has_explicit_error_without_partial_source_text(tools):
    hit = tools.call("search_relations", {"collection": "documents", "query": ""})["hits"][0]
    small = RelationEvidenceTools(tools.database, tools.collections, max_response_bytes=1024)
    result = small.call("trace_relation", hit["trace"]["arguments"])
    assert result["error"]["code"] == "response_too_large" and "evidence" not in result
    assert result["error"]["response_bytes"] > result["error"]["limit_bytes"]
    assert "Only applies" not in json.dumps(result)


def test_corruption_returns_error_not_empty_success_or_source_exception(tools, delivered):
    table = f"srd_{delivered.search.snapshot_id}_cards"
    with sqlite3.connect(tools.database) as con:
        con.execute(f'DELETE FROM "{table}"')
    result = tools.call("search_relations", {"collection": "documents", "query": ""})
    assert result["status"] == "failed" and result["error"]["code"] == "receiver_unavailable"
    assert "hits" not in result and str(tools.database) not in json.dumps(result)


def test_visual_uncertainty_and_coordinates_survive_tool_dispatch(visual, tmp_path):
    delivery = pair(visual, tmp_path / "visual-pair", "visual")
    db = tmp_path / "visual.db"
    SQLiteRelationReceiver(db).receive(delivery, collection="images")
    tool = RelationEvidenceTools(db, ("images",))
    found = tool.call("search_relations", {"collection": "images", "query": "", "support": "uncertain"})
    assert found["total"] == 1 and found["hits"][0]["assertion_status"] is None
    traced = tool.call("trace_relation", found["hits"][0]["trace"]["arguments"])
    assert traced["assertion"]["properties"]["direction"] == "unspecified"
    assert traced["assertion"]["properties"]["x1"] == 500


def test_cli_definitions_and_local_tool_call(tools, capsys):
    host = ["--database", str(tools.database), "--collection", "documents"]
    assert main([*host, "definitions"]) == 0
    assert len(json.loads(capsys.readouterr().out)["tools"]) == 2
    assert main([*host, "call", "search_relations", "--arguments", '{"collection":"documents","query":""}']) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2
    assert main([*host, "call", "search_relations", "--arguments", '{bad-json']) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_configuration"
