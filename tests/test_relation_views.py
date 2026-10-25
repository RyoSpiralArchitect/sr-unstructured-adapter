"""Search/graph conformance uses saved synthetic sources, never live providers."""
from dataclasses import replace
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.cli import main
from sr_adapter.narrative import render_narrative, save_narrative, verify_narrative, NarrativeOntology
from sr_adapter.relation_dataset import RelationTable
from sr_adapter.relation_dataset_store import (export_relation_dataset, export_relation_view,
    load_relation_dataset, verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.relation_sources import read_source
from sr_adapter.relation_views import project_relation_view
from sr_adapter.tables import canonical_json
from test_fabric_narrative import narrative as narrative
from test_narrative import ontology as ontology, response
from test_relation_dataset import visual as visual, source as source, rehash


@pytest.fixture
def four_statuses(ontology, tmp_path):
    data = b"A owner B.\n\nA not owner B.\n\nA owner unknown.\n\nA owner ambiguous.\n\nOnly applies after approval."
    units = read_source(data, "scope.txt")["units"]
    rows = [{"subject": "A", "predicate": "managedBy", "object": "B" if i < 2 else None,
             "status": status, "evidence": [{"unit_id": units[i]["id"], "quote": units[i]["text"]}]}
            for i, status in enumerate(("asserted", "denied", "unknown", "ambiguous"))]
    manifest, files = render_narrative(data=data, source_name="scope.txt", ontology=ontology,
                                     model="test-model", response=response(rows))
    out = tmp_path / "four-statuses"
    save_narrative(out, manifest, files)
    return load_relation_dataset([out], profile="narrative")[0]


def property_map(graph):
    result = {}
    for row in graph.rows()["properties"]:
        field = {"string": "text_value", "integer": "integer_value", "boolean": "boolean_value"}.get(row["value_kind"])
        result.setdefault(row["node_id"], {})[row["property_name"]] = row[field] if field else None
    return result


def test_all_assertion_statuses_remain_statements_not_direct_fact_edges(four_statuses):
    graph = project_relation_view(four_statuses, view="graph")
    search = project_relation_view(four_statuses, view="search")
    rows, properties = graph.rows(), property_map(graph)
    nodes = {r["node_id"]: r for r in rows["nodes"]}
    cards = {r["assertion_status"]: r for r in search.rows()["cards"]}
    assert set(cards) == {"asserted", "denied", "unknown", "ambiguous"}
    for status, card in cards.items():
        node = card["graph_node_id"]
        assert nodes[node]["node_kind"] == "assertion"
        assert nodes[node]["assertion_status"] == properties[node]["assertion_status"] == status
        assert nodes[node]["label"].startswith(f"[{status}]")
        assert f"Assertion status: {status}" in card["content"]
        assert card["title"].startswith(f"[{status}]")
        assert "Only applies after approval." in card["content"]  # Uncited source qualification survives.
        assert "Full saved source context" in card["content"]
        roles = {e["role"] for e in rows["edges"] if e["source_id"] == node}
        assert {"subject", "cites"} <= roles
        assert ("object" in roles) == (status in {"asserted", "denied"})
        assert (properties[node]["object_id"] is None) == (status in {"unknown", "ambiguous"})
    # Semantic predicates live on assertion nodes; no business edge joins two mentions.
    assert not any(nodes[e["source_id"]]["node_kind"] == nodes[e["target_id"]]["node_kind"] == "mention"
                   for e in rows["edges"])
    assert "managedBy" not in {e["role"] for e in rows["edges"]}
    assert len({e["target_id"] for e in rows["edges"] if e["role"] == "subject"}) == 4


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_graph_retains_every_scalar_row_and_joins_back_to_source(profile, request):
    dataset, _ = load_relation_dataset([request.getfixturevalue(profile)], profile=profile)
    graph = project_relation_view(dataset, view="graph")
    properties = property_map(graph)
    restored = {table.name: [] for table in dataset.tables}
    nodes = {row["node_id"]: row for row in graph.rows()["nodes"]}
    for row in nodes.values():
        assert row["source_snapshot_id"] == dataset.snapshot_id
        restored[row["source_table"]].append(properties[row["node_id"]])
    for table in dataset.tables:
        assert sorted(restored[table.name], key=canonical_json) == table.rows()
    for edge in graph.rows()["edges"]:
        assert nodes[edge["source_id"]]["package_id"] == nodes[edge["target_id"]]["package_id"] == edge["package_id"]
    if profile == "narrative":
        assert any(p.get("literal_match") is False for p in properties.values())
        assert any(p.get("path") == "/header_context/0/text" and p["value_text"] == "Owner" for p in properties.values())
    else:
        claim = next(p for p in properties.values() if "predicate" in p)
        assert (claim["direction"], claim["support"], claim["value_text"]) == ("unspecified", "uncertain", "0%")
        assert claim["subject_id"] != claim["object_id"] and claim["subject_label"] == claim["object_label"]
        assert claim["x1"] == 500


@pytest.mark.parametrize("profile", ["narrative", "visual"])
@pytest.mark.parametrize("view", ["search", "graph"])
def test_view_export_replays_and_existing_sqlite_sink_reads_every_table(profile, view, request, tmp_path):
    base, out, db = tmp_path / "base", tmp_path / "view", tmp_path / "view.db"
    export_relation_dataset([request.getfixturevalue(profile)], base, profile=profile)
    result = export_relation_view(base, out, view=view)
    dataset = verify_relation_dataset(out)
    assert result["input_snapshot_id"] == verify_relation_dataset(base).snapshot_id
    assert dataset == project_relation_view(verify_relation_dataset(base), view=view)
    assert not any("target" in name or "deployment" in name for name in result["files"])
    if view == "search":
        assert [json.loads(line) for line in (out / "cards.jsonl").read_bytes().splitlines()] == dataset.rows()["cards"]
        if profile == "narrative":
            cards = {r["assertion_status"]: r for r in dataset.rows()["cards"]}
            assert cards["denied"]["subject_label_match"] is False
            assert cards["unknown"]["object_label_match"] is None
        else:
            assert dataset.rows()["cards"][0]["assertion_status"] is None
    written = write_dataset_sqlite(out, db)
    assert written["inserted_rows"] == sum(len(t.values) for t in dataset.tables)
    assert write_dataset_sqlite(out, db)["inserted_rows"] == 0
    for table in dataset.tables:
        assert pq.read_table(out / f"tables/{table.name}.parquet").schema == table.arrow_schema()


def test_query_from_search_hit_to_negative_assertion_and_original_evidence(narrative, tmp_path):
    base, search, graph, db = (tmp_path / n for n in ("base", "search", "graph", "demo.db"))
    export_relation_dataset([narrative], base, profile="narrative")
    export_relation_view(base, search, view="search")
    export_relation_view(base, graph, view="graph")
    s = write_dataset_sqlite(search, db)["tables"]["cards"]["name"]
    g = write_dataset_sqlite(graph, db)["tables"]
    with sqlite3.connect(db) as con:
        result = con.execute(f'''SELECT c.assertion_status, p.text_value
            FROM "{s}" c JOIN "{g['edges']['name']}" e ON c.graph_node_id=e.source_id AND e.role='cites'
            JOIN "{g['properties']['name']}" p ON e.target_id=p.node_id AND p.property_name='text'
            WHERE c.assertion_status='denied' ORDER BY e.position''').fetchall()
    assert result == [("denied", "A"), ("denied", "B is not the owner")]


def test_visual_review_and_context_are_in_both_views(visual):
    original, _ = load_relation_dataset([visual], profile="visual")
    review = {"version": 1, "reviewer": "synthetic", "findings": [
        {"package_id": original.package_ids[0], "reference_kind": "item", "reference_id": "n1",
         "finding": "text_mismatch", "detail": "Recheck the visible label."},
        {"package_id": original.package_ids[0], "reference_kind": "image", "reference_id": None,
         "finding": "coverage_gap", "detail": "Some items were not extracted."}]}
    dataset, _ = load_relation_dataset([visual], profile="visual", review=review)
    search = project_relation_view(dataset, view="search")
    card = search.rows()["cards"][0]
    assert card["direction"] == "unspecified" and card["support"] == "uncertain"
    assert card["review_status"] == "model_candidate_needs_review"
    assert "Recheck the visible label." in card["content"] and "Some items were not extracted." in card["content"]
    graph = project_relation_view(dataset, view="graph")
    nodes = {r["node_id"]: r for r in graph.rows()["nodes"]}
    assert {nodes[e["target_id"]]["node_kind"] for e in graph.rows()["edges"] if e["role"] == "reviews"} == {"mention", "source"}


def test_multiple_sources_never_cross_contaminate_cards_or_node_identity(narrative, tmp_path):
    _, files = verify_narrative(narrative)
    manifest, payload = render_narrative(data=files["source.bin"], source_name="separate-scope.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="different-model",
        response=json.loads(files["response.json"]), label_policy="preserve")
    second = tmp_path / "second"
    save_narrative(second, manifest, payload)
    a, _ = load_relation_dataset([narrative, second], profile="narrative")
    b, _ = load_relation_dataset([second, narrative], profile="narrative")
    for view in ("search", "graph"):
        assert project_relation_view(a, view=view) == project_relation_view(b, view=view)
    cards = project_relation_view(a, view="search").rows()["cards"]
    assert len({c["graph_node_id"] for c in cards}) == 4
    for card in cards:
        other = "separate-scope.docx" if card["source_name"] == "source.docx" else '"source.docx"'
        assert other not in card["content"]


def test_empty_extraction_has_context_card_and_no_assertion_nodes(narrative, tmp_path):
    _, files = verify_narrative(narrative)
    manifest, payload = render_narrative(data=files["source.bin"], source_name="empty.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="test", response=response([]))
    empty = tmp_path / "empty"
    save_narrative(empty, manifest, payload)
    dataset, _ = load_relation_dataset([empty], profile="narrative")
    card, = project_relation_view(dataset, view="search").rows()["cards"]
    assert card["card_kind"] == "source_context" and card["assertion_status"] is None
    assert card["relation_id"] is None and card["evidence_count"] == 0
    assert "Only applies after approval." in card["content"]
    graph = project_relation_view(dataset, view="graph")
    assert not any(row["node_kind"] == "assertion" for row in graph.rows()["nodes"])


@pytest.mark.parametrize("view", ["search", "graph"])
def test_wrong_profile_schema_and_dangling_reference_are_rejected(narrative, view):
    source, _ = load_relation_dataset([narrative], profile="narrative")
    with pytest.raises(ValueError, match="supported"):
        project_relation_view(replace(source, profile="narrative_relations/2"), view=view)
    with pytest.raises(ValueError, match="tables"):
        project_relation_view(replace(source, tables=tuple(t for t in source.tables if t.name != "locations")), view=view)
    table = next(t for t in source.tables if t.name == "relations")
    rows = table.rows()
    rows[0]["subject_id"] = "absent"
    bad = RelationTable.from_rows(table.name, table.columns, table.primary_key, rows)
    with pytest.raises(ValueError, match="Dangling"):
        project_relation_view(replace(source, tables=tuple(bad if t.name == bad.name else t for t in source.tables)), view=view)


@pytest.mark.parametrize("view", ["search", "graph"])
def test_rehashed_view_tamper_cannot_promote_a_denial(narrative, tmp_path, view):
    base, out = tmp_path / "base", tmp_path / "view"
    export_relation_dataset([narrative], base, profile="narrative")
    export_relation_view(base, out, view=view)
    name = f"tables/{'cards' if view == 'search' else 'nodes'}.parquet"
    original = pq.read_table(out / name)
    rows = original.to_pylist()
    next(r for r in rows if r["assertion_status"] == "denied")["assertion_status"] = "asserted"
    pq.write_table(pa.Table.from_pylist(rows, schema=original.schema), out / name)
    rehash(out, name)
    with pytest.raises(ValueError, match="replayed"):
        verify_relation_dataset(out)


def test_search_jsonl_and_embedded_source_hashes_are_verified(narrative, tmp_path):
    base, out = tmp_path / "base", tmp_path / "view"
    export_relation_dataset([narrative], base, profile="narrative")
    export_relation_view(base, out, view="search")
    path = out / "cards.jsonl"
    saved = path.read_bytes()
    path.write_bytes(saved.replace(b'"assertion_status":"denied"', b'"assertion_status":"asserted"'))
    rehash(out, "cards.jsonl")
    with pytest.raises(ValueError, match="Search cards"):
        verify_relation_dataset(out)
    path.write_bytes(saved)
    rehash(out, "cards.jsonl")
    inner = json.loads((out / "input/manifest.json").read_bytes())
    inner["files"]["dataset.json"]["sha256"] = "0" * 64
    (out / "input/manifest.json").write_bytes(canonical_json(inner))
    rehash(out, "input/manifest.json")
    with pytest.raises(ValueError, match="contract"):
        verify_relation_dataset(out)


@pytest.mark.parametrize("view", ["search", "graph"])
def test_projection_budget_fails_before_writing_without_truncation(narrative, tmp_path, monkeypatch, view):
    base = tmp_path / "base"
    export_relation_dataset([narrative], base, profile="narrative")
    monkeypatch.setattr("sr_adapter.relation_views.MAX_VIEW_BYTES", 10)
    with pytest.raises(ValueError, match="no truncation"):
        export_relation_view(base, tmp_path / "absent", view=view)
    assert not (tmp_path / "absent").exists()


def test_cli_and_nested_view_rejection(narrative, tmp_path, capsys):
    base, out = tmp_path / "base", tmp_path / "view"
    export_relation_dataset([narrative], base, profile="narrative")
    assert main(["knowledge", "dataset-view", str(base), "--view", "search", "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "relation_view" and result["profile"] == "relation_search/1"
    assert main(["knowledge", "dataset-sqlite", str(out), "--database", str(tmp_path / "cli.db")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "committed"
    with pytest.raises(ValueError, match="unsupported"):
        export_relation_view(out, tmp_path / "nested", view="graph")
