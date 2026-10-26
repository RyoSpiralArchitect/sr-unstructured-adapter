"""Semantic scope and transport tests use saved synthetic source packages only."""
from copy import deepcopy
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.relation_dataset_store import (export_relation_dataset, export_relation_semantics,
    export_relation_view, load_relation_dataset, verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.relation_semantics import (LANG_STRING, XSD, SemanticOverlay, LiteralValue, main,
    project_relation_semantics, trace_semantic_claim)
from sr_adapter.relation_views import project_relation_view
from sr_adapter.tables import canonical_json
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative, verify_narrative
from test_fabric_narrative import narrative as narrative
from test_narrative import ontology as ontology
from test_relation_dataset import visual as visual, source as source, rehash

NS = "urn:semantic-test:"


def declaration(source):
    nodes = project_relation_view(source, view="graph").rows()["nodes"]
    entity_table = "entities" if source.profile.startswith("narrative") else "items"
    mentions = [n["node_id"] for n in nodes if n["source_table"] == entity_table
                and (entity_table == "entities" or n["label"] == "共通システム")]
    evidence = next(n["node_id"] for n in nodes if n["source_table"] == "evidence")
    pid = source.package_ids[0]
    terms = [{"iri": NS + name, "kind": kind, "label": name} for name, kind in [
        ("Org", "class"), ("Department", "class"), ("System", "class"), ("Handle", "relation"), ("Event", "relation"),
        ("Approved", "relation"), ("Stopped", "relation"), ("actor", "role"), ("object", "role"), ("amount", "role"),
        ("proposition", "role"), ("Request", "individual")]]

    def atom(key, predicate, arguments):
        return {"id": key, "package_id": pid, "operator": "atom", "predicate": NS + predicate,
                "arguments": [{"role": NS + role, "value": value} for role, value in arguments]}

    expressions = [atom("handled", "Handle", [
        ("actor", {"kind": "source", "node_id": mentions[0]}),
        ("object", {"kind": "source", "node_id": mentions[1]}),
        ("amount", {"kind": "literal", "lexical": "000.00", "datatype": XSD + "decimal"})]),
        atom("approved", "Approved", [("object", {"kind": "term", "iri": NS + "Request"})]),
        atom("stopped", "Stopped", [("object", {"kind": "source", "node_id": mentions[0]})]),
        {"id": "not-stopped", "package_id": pid, "operator": "not", "operands": ["stopped"]},
        {"id": "eligible", "package_id": pid, "operator": "and", "operands": ["approved", "not-stopped"]},
        {"id": "rule", "package_id": pid, "operator": "implies", "operands": ["eligible", "handled"]}]
    claims = [{"id": key, "package_id": pid, "expression_id": "rule", "status": status, "modality": "required",
        "basis": "source_interpretation", "attributed_to": person, "context": "Synthetic declaration, not a model result",
        "valid_from": "2027-04-01T00:00:00+09:00", "valid_until": "2028-04-01T00:00:00+09:00", "evidence": [evidence]}
        for key, status, person in [("c1", "asserted", "operations"), ("c2", "denied", "audit")]]
    return {"version": 1, "source_snapshot_id": source.snapshot_id, "author": "synthetic", "terms": terms,
        "axioms": [{"id": "hierarchy", "kind": "subclass", "left": NS + "Department", "right": NS + "Org"},
            {"id": "disjoint", "kind": "disjoint", "left": NS + "System", "right": NS + "Org"}],
        "roles": [{"predicate": NS + "Handle", "role": NS + role, "min_count": 1, "max_count": 1,
                   "value_kinds": [kind]} for role, kind in [("actor", "source"), ("object", "source"), ("amount", "literal")]],
        "expressions": expressions, "claims": claims,
        "claim_links": [{"id": "opposes", "source_claim": "c2", "target_claim": "c1", "kind": "challenges"}]}


@pytest.fixture
def case(narrative):
    dataset = load_relation_dataset([narrative], profile="narrative")[0]
    return dataset, declaration(dataset)


def test_conditional_nary_claim_is_not_split_into_positive_facts(case):
    source, overlay = case
    before, declaration_before = source.describe(), deepcopy(overlay)
    dataset = project_relation_semantics(source, overlay)
    rows = dataset.rows()
    assert dataset.profile == "relation_semantics/1" and len(dataset.tables) == 11
    assert {row["expression_id"] for row in rows["claims"]} == {"rule"}
    assert {row["status"] for row in rows["claims"]} == {"asserted", "denied"}
    assert {row["modality"] for row in rows["claims"]} == {"required"}
    assert len([r for r in rows["arguments"] if r["expression_id"] == "handled"]) == 3
    assert next(r for r in rows["arguments"] if r["value_kind"] == "literal")["lexical"] == "000.00"
    assert [r["target_expression_id"] for r in sorted(rows["operands"], key=lambda r: r["position"])
            if r["expression_id"] == "rule"] == ["eligible", "handled"]
    assert all(row["source_snapshot_id"] == source.snapshot_id for table in rows.values() for row in table)
    assert source.describe() == before and overlay == declaration_before
    # The existing source views do not acquire the new interpretation.
    assert all("semantic-test" not in c["content"] for c in project_relation_view(source, view="search").rows()["cards"])


def test_order_independence_but_argument_and_operand_order_are_retained(case):
    source, overlay = case
    before = project_relation_semantics(source, overlay)
    reordered = deepcopy(overlay)
    for key in ("terms", "axioms", "roles", "expressions", "claims", "claim_links"):
        reordered[key].reverse()
    assert project_relation_semantics(source, SemanticOverlay.model_validate(reordered)) == before
    reordered["expressions"][0]["operands"].reverse()
    assert project_relation_semantics(source, reordered).snapshot_id != before.snapshot_id


def test_trace_retains_dissent_full_context_and_negated_original_readings(case):
    source, overlay = case
    trace = trace_semantic_claim(source, overlay, claim_id="c1")
    assert {c["id"] for c in trace["claims"]} == {"c1", "c2"}
    assert len(trace["expressions"]) == 6 and trace["claim_links"][0]["kind"] == "challenges"
    assert trace["source_context"] == source.rows()
    assert "denied" in {r["assertion_status"] for r in trace["source_context"]["relations"]}
    assert "unknown" in {r["assertion_status"] for r in trace["source_context"]["relations"]}
    assert any(row["path"] == "/header_context/0/text" for row in trace["source_context"]["locations"])
    assert not trace["inference_performed"] and trace["semantic_validation"] == "not_performed"
    assert len(trace["source_context"]["evidence"]) > len(overlay["claims"][0]["evidence"])
    with pytest.raises(ValueError, match="Unknown semantic claim"):
        trace_semantic_claim(source, overlay, claim_id="missing")


def test_source_identity_and_image_regions_survive_equal_labels(visual):
    source = load_relation_dataset([visual], profile="visual")[0]
    overlay = declaration(source)
    dataset = project_relation_semantics(source, overlay)
    args = [row for row in dataset.rows()["arguments"] if row["expression_id"] == "handled" and row["value_kind"] == "source"]
    assert len({row["node_id"] for row in args}) == 2
    trace = trace_semantic_claim(source, overlay, claim_id="c1")
    mentions = [item["record"] for item in trace["source_records"] if item["table"] == "items"]
    assert len({m["label"] for m in mentions}) == 1
    assert len({m["x0"] for m in mentions}) == 2
    assert trace["source_context"] == source.rows()
    assert trace["source_context"]["relations"][0]["support"] == "uncertain"


def test_explicit_cross_document_references_preserve_package_boundaries(narrative, tmp_path):
    _, files = verify_narrative(narrative)
    manifest, payloads = render_narrative(data=files["source.bin"], source_name="second.docx",
        ontology=NarrativeOntology.model_validate_json(files["ontology.json"]), model="test-model",
        response=json.loads(files["response.json"]), label_policy="preserve")
    second = tmp_path / "second"
    save_narrative(second, manifest, payloads)
    dataset = load_relation_dataset([narrative, second], profile="narrative")[0]
    overlay = declaration(dataset)
    first_pid, second_pid = dataset.package_ids
    nodes = project_relation_view(dataset, view="graph").rows()["nodes"]
    pair = [next(n["node_id"] for n in nodes if n["package_id"] == pid and n["source_table"] == "entities")
            for pid in (first_pid, second_pid)]
    for arg, node in zip(overlay["expressions"][0]["arguments"], pair):
        arg["value"] = {"kind": "source", "node_id": node}
    overlay["expressions"][1]["package_id"] = second_pid
    overlay["claims"][1]["package_id"] = second_pid
    rows = project_relation_semantics(dataset, overlay).rows()
    assert len(rows["overlays"]) == 2 and len(rows["terms"]) == 2 * len(overlay["terms"])
    args = [row for row in rows["arguments"] if row["expression_id"] == "handled" and row["value_kind"] == "source"]
    assert {row["source_package_id"] for row in args} == {first_pid, second_pid}
    assert {row["package_id"] for row in args} == {first_pid}
    trace = trace_semantic_claim(dataset, overlay, claim_id="c1")
    assert len(trace["source_context"]["documents"]) == 2
    assert {c["package_id"] for c in trace["claims"]} == {first_pid, second_pid}


def test_class_cycles_and_unasserted_expressions_are_stored_without_entailment(case):
    source, overlay = case
    overlay["axioms"].append({"id": "reverse", "kind": "subclass", "left": NS + "Org", "right": NS + "Department"})
    overlay["expressions"].append({"id": "unasserted", "package_id": source.package_ids[0],
        "operator": "or", "operands": ["approved", "stopped"]})
    rows = project_relation_semantics(source, overlay).rows()
    assert len(rows["axioms"]) == 3 and len(rows["claims"]) == 2 and len(rows["expressions"]) == 7
    assert "unasserted" not in {e["id"] for e in trace_semantic_claim(source, overlay, claim_id="c1")["expressions"]}


def test_expression_arguments_and_explicit_unknowns_keep_their_scope(case):
    source, overlay = case
    expr = {"id": "about-rule", "package_id": source.package_ids[0], "operator": "atom", "predicate": NS + "Event",
        "arguments": [{"role": NS + "proposition", "value": {"kind": "expression", "expression_id": "rule"}},
                      {"role": NS + "actor", "value": {"kind": "unknown", "reason": "not identified"}}]}
    overlay["expressions"].append(expr)
    overlay["claims"][0]["expression_id"] = "about-rule"
    rows = project_relation_semantics(source, overlay).rows()["arguments"]
    assert next(r for r in rows if r["value_kind"] == "expression")["expression_id_ref"] == "rule"
    unknown = next(r for r in rows if r["value_kind"] == "unknown")
    assert unknown["reason"] == "not identified" and unknown["node_id"] is None
    assert len(trace_semantic_claim(source, overlay, claim_id="c1")["expressions"]) == 7


@pytest.mark.parametrize("datatype,lexical,language", [
    ("string", "", None), ("boolean", "false", None), ("integer", "+00000", None),
    ("integer", "123456789012345678901234567890", None), ("decimal", "+001.000", None),
    ("date", "2028-02-29", None), ("dateTime", "2027-10-27T12:34:56Z", None),
    ("langString", "審査部", "ja-JP"),
])
def test_literal_lexical_values_are_not_coerced(datatype, lexical, language, case):
    source, overlay = case
    value = {"kind": "literal", "datatype": LANG_STRING if datatype == "langString" else XSD + datatype,
             "lexical": lexical, "language": language}
    overlay["expressions"][0]["arguments"][2]["value"] = value
    row = next(r for r in project_relation_semantics(source, overlay).rows()["arguments"] if r["value_kind"] == "literal")
    assert (row["lexical"], row["datatype"], row["language"]) == (lexical, value["datatype"], language)


@pytest.mark.parametrize("datatype,lexical,language", [
    ("boolean", "False", None), ("integer", "1.0", None), ("decimal", "NaN", None),
    ("decimal", "1e3", None), ("date", "2027-02-29", None),
    ("dateTime", "2027-10-27T10:00:00", None), ("float", "1.5", None),
    ("dateTime", "2027-10-27T10:00:00+09:60", None),
    ("dateTime", "2027-10-27T10:00:00.1234567Z", None),
    ("string", "日本語", "ja"), ("langString", "日本語", None),
])
def test_invalid_or_unsupported_typed_literals_are_rejected(datatype, lexical, language):
    with pytest.raises(ValueError):
        LiteralValue(kind="literal", datatype=LANG_STRING if datatype == "langString" else XSD + datatype,
                     lexical=lexical, language=language)


@pytest.mark.parametrize("change", ["snapshot", "duplicate", "dangling_operand", "missing_source", "missing_term",
    "axiom_kind", "empty_citation", "wrong_package", "missing_claim", "wrong_role", "arity", "extra", "version_bool"])
def test_invalid_declarations_fail_before_output(case, change):
    source, overlay = case
    if change == "snapshot":
        overlay["source_snapshot_id"] = "0" * 64
    elif change == "duplicate":
        overlay["expressions"].append(deepcopy(overlay["expressions"][0]))
    elif change == "dangling_operand":
        overlay["expressions"][-1]["operands"][0] = "missing"
    elif change == "missing_source":
        overlay["expressions"][0]["arguments"][0]["value"]["node_id"] = "missing"
    elif change == "missing_term":
        overlay["expressions"][0]["predicate"] = NS + "missing"
    elif change == "axiom_kind":
        overlay["axioms"][0]["right"] = NS + "Handle"
    elif change == "empty_citation":
        overlay["claims"][0]["evidence"] = []
    elif change == "wrong_package":
        overlay["claims"][0]["package_id"] = "other"
    elif change == "missing_claim":
        overlay["claim_links"][0]["target_claim"] = "absent"
    elif change == "wrong_role":
        overlay["expressions"][0]["arguments"][0]["role"] = NS + "Department"
    elif change == "arity":
        overlay["expressions"][-1]["operands"].pop()
    elif change == "extra":
        overlay["claims"][0]["automatic_truth"] = True
    elif change == "version_bool":
        overlay["version"] = True
    with pytest.raises(ValueError):
        project_relation_semantics(source, overlay)


@pytest.mark.parametrize("change", ["missing", "too_many", "kind", "undeclared_role", "bounds", "bool", "duplicate"])
def test_declared_role_constraints_are_checked(case, change):
    source, overlay = case
    args = overlay["expressions"][0]["arguments"]
    if change == "missing":
        args.pop()
    elif change == "too_many":
        args.append(deepcopy(args[0]))
    elif change == "kind":
        args[0]["value"] = {"kind": "unknown", "reason": "unknown actor"}
    elif change == "undeclared_role":
        args[0]["role"] = NS + "proposition"
    elif change == "bounds":
        overlay["roles"][0]["max_count"] = 0
    elif change == "bool":
        overlay["roles"][0]["min_count"] = True
    elif change == "duplicate":
        overlay["roles"].append(deepcopy(overlay["roles"][0]))
    with pytest.raises(ValueError):
        project_relation_semantics(source, overlay)


@pytest.mark.parametrize("start,end", [("2027-04-01", None), ("2028-04-01T00:00:00Z", "2027-04-01T00:00:00Z"),
    ("2027-04-01T09:00:00+09:00", "2027-04-01T00:00:00Z")])
def test_validity_interval_is_timezone_aware_and_half_open(case, start, end):
    source, overlay = case
    overlay["claims"][0].update(valid_from=start, valid_until=end)
    with pytest.raises(ValueError):
        project_relation_semantics(source, overlay)


def test_undated_unknown_hypothesis_and_cyclic_disagreement_are_preserved(case):
    source, overlay = case
    overlay["claims"][0].update(basis="hypothesis", evidence=[], status="unknown", modality="possible",
                                 valid_from=None, valid_until=None)
    overlay["claim_links"].append({"id": "reverse", "source_claim": "c1", "target_claim": "c2", "kind": "challenges"})
    row = next(r for r in project_relation_semantics(source, overlay).rows()["claims"] if r["id"] == "c1")
    assert (row["status"], row["basis"], row["valid_from"]) == ("unknown", "hypothesis", None)
    assert len(trace_semantic_claim(source, overlay, claim_id="c1")["claims"]) == 2


@pytest.mark.parametrize("cycle", ["operand", "argument"])
def test_cycles_through_logical_operands_or_quoted_propositions_fail(case, cycle):
    source, overlay = case
    if cycle == "operand":
        overlay["expressions"][-1]["operands"][0] = "rule"
    else:
        overlay["expressions"][1]["arguments"][0]["value"] = {"kind": "expression", "expression_id": "rule"}
    with pytest.raises(ValueError, match="Cyclic"):
        project_relation_semantics(source, overlay)


def test_nesting_bound_and_shared_subexpressions(case):
    source, overlay = case
    overlay["expressions"][-1]["operands"] = ["eligible", "eligible"]
    assert project_relation_semantics(source, overlay)
    for index in range(64):
        overlay["expressions"].append({"id": f"n{index}", "package_id": source.package_ids[0],
            "operator": "not", "operands": ["approved" if index == 0 else f"n{index - 1}"]})
    with pytest.raises(ValueError, match="64 levels"):
        project_relation_semantics(source, overlay)


def test_budget_rejects_instead_of_truncating(case, monkeypatch):
    from sr_adapter import relation_semantics as module
    source, overlay = case
    monkeypatch.setattr(module, "MAX_RESULT_BYTES", 100)
    with pytest.raises(ValueError, match="budget"):
        project_relation_semantics(source, overlay)
    with pytest.raises(ValueError, match="budget"):
        trace_semantic_claim(source, overlay, claim_id="c1")
    monkeypatch.setattr(module, "MAX_OVERLAY_BYTES", 100)
    with pytest.raises(ValueError, match="budget"):
        project_relation_semantics(source, overlay)


@pytest.mark.parametrize("profile", ["narrative", "visual"])
def test_portable_replay_native_scalar_tables_and_sqlite_repeat(profile, request, tmp_path):
    base, out, db = tmp_path / "base", tmp_path / "semantics", tmp_path / "db"
    export_relation_dataset([request.getfixturevalue(profile)], base, profile=profile)
    source = verify_relation_dataset(base)
    overlay = declaration(source)
    manifest = export_relation_semantics(base, out, overlay=overlay)
    dataset = verify_relation_dataset(out)
    assert dataset == project_relation_semantics(source, overlay)
    assert manifest["input_snapshot_id"] == source.snapshot_id
    receipt = write_dataset_sqlite(out, db)
    assert write_dataset_sqlite(out, db)["inserted_rows"] == 0
    for table in dataset.tables:
        assert pq.read_table(out / f"tables/{table.name}.parquet").schema == table.arrow_schema()
    with sqlite3.connect(db) as connection:
        claims = receipt["tables"]["claims"]["name"]
        assert connection.execute(f'SELECT COUNT(*) FROM "{claims}"').fetchone()[0] == 2
        connection.execute(f'UPDATE "{claims}" SET status=?', ("asserted",))
    with pytest.raises(ValueError, match="differs"):
        write_dataset_sqlite(out, db)


@pytest.mark.parametrize("target", ["table", "declaration"])
def test_rehashed_changes_cannot_bypass_semantic_replay(narrative, tmp_path, target):
    base, out = tmp_path / "base", tmp_path / "semantics"
    export_relation_dataset([narrative], base, profile="narrative")
    export_relation_semantics(base, out, overlay=declaration(verify_relation_dataset(base)))
    if target == "table":
        path = out / "tables/claims.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["modality"] = "permitted"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    else:
        path = out / "semantics.json"
        changed = json.loads(path.read_bytes())
        changed["claims"][0]["status"] = "unknown"
        path.write_bytes(canonical_json(changed))
    rehash(out, path.relative_to(out).as_posix())
    with pytest.raises(ValueError, match="differs"):
        verify_relation_dataset(out)


def test_semantic_bundles_cannot_be_silently_flattened_or_stacked(narrative, tmp_path):
    base, out = tmp_path / "base", tmp_path / "semantics"
    export_relation_dataset([narrative], base, profile="narrative")
    overlay = declaration(verify_relation_dataset(base))
    export_relation_semantics(base, out, overlay=overlay)
    for action in [lambda: export_relation_view(out, tmp_path / "bad", view="graph"),
                   lambda: export_relation_semantics(out, tmp_path / "bad", overlay=overlay)]:
        with pytest.raises(ValueError, match="unsupported"):
            action()
        assert not (tmp_path / "bad").exists()
    overlay["source_snapshot_id"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot"):
        export_relation_semantics(base, tmp_path / "bad", overlay=overlay)
    assert not (tmp_path / "bad").exists()


def test_cli_export_and_trace(narrative, tmp_path, capsys):
    base, out = tmp_path / "base", tmp_path / "semantics"
    export_relation_dataset([narrative], base, profile="narrative")
    path = tmp_path / "overlay.json"
    path.write_bytes(canonical_json(declaration(verify_relation_dataset(base))))
    assert main(["export", "--source", str(base), "--overlay", str(path), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "relation_semantics"
    assert main(["trace", "--bundle", str(out), "--claim", "c1"]) == 0
    assert len(json.loads(capsys.readouterr().out)["claims"]) == 2
    assert main(["trace", "--bundle", str(out), "--claim", "missing"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "semantic_overlay_unavailable"
