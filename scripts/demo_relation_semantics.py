"""Synthetic, handwritten semantic declarations. No model or cloud calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_dataset_store import (export_relation_dataset, export_relation_semantics,
    verify_relation_dataset, write_dataset_sqlite)
from sr_adapter.relation_semantics import XSD, trace_semantic_claim
from sr_adapter.relation_sources import read_source
from sr_adapter.relation_views import project_relation_view

NS = "urn:sr:semantic-demo:"


def declaration(source):
    """Create a reviewable example; this is not automatic semantic extraction."""
    graph = project_relation_view(source, view="graph").rows()
    props = {}
    for prop in graph["properties"]:
        props.setdefault(prop["node_id"], {})[prop["property_name"]] = prop["text_value"]
    nodes = graph["nodes"]
    pid = source.package_ids[0]
    actor = next(node["node_id"] for node in nodes if node["source_table"] == "entities" and props[node["node_id"]]["role"] == "object")
    system = next(node["node_id"] for node in nodes if node["source_table"] == "entities" and props[node["node_id"]]["role"] == "subject")
    evidence = [node["node_id"] for node in nodes if node["source_table"] == "evidence"]
    terms = []
    for kind, names in {"class": ["Organization", "Department", "System"],
        "relation": ["Process", "Operate", "Approved", "EmergencyStop"],
        "role": ["actor", "request", "system", "ceiling", "currency"], "individual": ["RequestA", "JPY"]}.items():
        terms.extend({"iri": NS + name, "kind": kind, "label": name} for name in names)

    def atom(identity, predicate, arguments):
        return {"id": identity, "package_id": pid, "operator": "atom", "predicate": NS + predicate,
                "arguments": [{"role": NS + role, "value": value} for role, value in arguments]}

    request = {"kind": "term", "iri": NS + "RequestA"}
    expressions = [
        atom("approved", "Approved", [("request", request)]),
        atom("stopped", "EmergencyStop", [("system", {"kind": "source", "node_id": system})]),
        {"id": "not-stopped", "package_id": pid, "operator": "not", "operands": ["stopped"]},
        {"id": "eligible", "package_id": pid, "operator": "and", "operands": ["approved", "not-stopped"]},
        atom("process", "Process", [("actor", {"kind": "source", "node_id": actor}), ("request", request),
            ("system", {"kind": "source", "node_id": system}),
            ("ceiling", {"kind": "literal", "lexical": "10000000.00", "datatype": XSD + "decimal"}),
            ("currency", {"kind": "term", "iri": NS + "JPY"})]),
        {"id": "rule", "package_id": pid, "operator": "implies", "operands": ["eligible", "process"]},
    ]
    claims = [{"id": key, "package_id": pid, "expression_id": "rule", "status": status,
        "modality": "required", "basis": "source_interpretation", "attributed_to": person,
        "context": "合成資料の解釈例。対象は申請A。", "valid_from": "2027-04-01T00:00:00+09:00",
        "valid_until": "2028-04-01T00:00:00+09:00", "evidence": evidence}
        for key, status, person in [("operations-reading", "asserted", "合成資料・審査担当"),
                                    ("audit-reading", "denied", "合成資料・監査担当")]]
    return {"version": 1, "source_snapshot_id": source.snapshot_id, "author": "handwritten synthetic fixture",
        "description": "条件・例外・期間・多者役割・異なる解釈を保存する例。業務ルールは実行しない。",
        "terms": terms, "axioms": [
            {"id": "department-type", "kind": "subclass", "left": NS + "Department", "right": NS + "Organization"},
            {"id": "distinct-types", "kind": "disjoint", "left": NS + "System", "right": NS + "Organization"},
            {"id": "process-type", "kind": "subrelation", "left": NS + "Process", "right": NS + "Operate"}],
        "roles": [{"predicate": NS + "Process", "role": NS + role, "min_count": 1, "max_count": 1,
                   "value_kinds": [kind]} for role, kind in [("actor", "source"), ("request", "term"),
                    ("system", "source"), ("ceiling", "literal"), ("currency", "term")]],
        "expressions": expressions, "claims": claims,
        "claim_links": [{"id": "disagreement", "source_claim": "audit-reading", "target_claim": "operations-reading",
                         "kind": "challenges", "note": "解釈の異議を記録。正誤や論理的矛盾の判定ではない。"}]}


def run(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    text = "\n\n".join([
        "融資システムの主管は審査部である。",
        "申請Aが承認済み、かつ緊急停止中でない場合、審査部は融資システムで申請Aを10000000.00円まで処理しなければならない。",
        "審査担当はこの条件を2027年4月1日から2028年4月1日直前までのルールと解釈している。",
        "監査担当はその解釈に異議を記録している。どちらの解釈を採るかは未決定。",
        "外部委託の場合は別途承認が必要。これは本デモの意味表現にまだ割り当てていない。",
    ])
    data = text.encode()
    ontology = NarrativeOntology.model_validate({"namespace": NS,
        "classes": {"System": "システム", "Department": "部門"}, "predicates": [
            {"id": "ownedBy", "label": "主管", "subject_type": "System", "object_type": "Department", "description": "資料上の主管"}]})
    unit = read_source(data, "synthetic-rules.txt")["units"][0]
    readings = {"assertions": [{"subject": "融資システム", "predicate": "ownedBy", "object": "審査部", "status": "asserted",
        "evidence": [{"unit_id": unit["id"], "quote": unit["text"]}]}]}
    response = {"status": "completed", "model": "handwritten-fixture", "output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": json.dumps(readings, ensure_ascii=False)}]}]}
    manifest, files = render_narrative(data=data, source_name="synthetic-rules.txt", ontology=ontology,
        model="handwritten-fixture", response=response)
    save_narrative(root / "extracted", manifest, files)
    export_relation_dataset([root / "extracted"], root / "source", profile="narrative")
    source = verify_relation_dataset(root / "source")
    overlay = declaration(source)
    (root / "overlay.json").write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    export_relation_semantics(root / "source", root / "semantics", overlay=overlay)
    dataset = verify_relation_dataset(root / "semantics")
    db = root / "semantics.db"
    receipt = write_dataset_sqlite(root / "semantics", db)
    repeat = write_dataset_sqlite(root / "semantics", db)
    trace = trace_semantic_claim(source, overlay, claim_id="operations-reading")
    with sqlite3.connect(db) as connection:
        table = receipt["tables"]["claims"]["name"]
        statuses = connection.execute(f'SELECT status, modality FROM "{table}" ORDER BY status').fetchall()
    assert statuses == [("asserted", "required"), ("denied", "required")] and repeat["inserted_rows"] == 0
    report = {"status": "complete", "model_calls": 0, "cloud_calls": 0, "fixture": "synthetic handwritten declarations",
              "snapshot_id": dataset.snapshot_id, "receipt": receipt, "repeat": repeat, "trace": trace}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text("""# 条件・役割・解釈を保持する意味層のデモ

`承認済み AND NOT 緊急停止 → 審査部が申請Aを上限1,000万円で処理する`
という合成資料を、条件付き・期間付きの義務として記述しています。
申請Aが実際に承認済みである、といった事実は追加しません。

- 組織・部門・システムの型と階層、多者の役割、金額と通貨を表に保存。
- 審査担当の肯定と監査担当の否定を同じ式への別々の主張として保存。
- 文書内の未割当の条件も元資料の文脈に保持。
- 元資料からの再構成、Parquet/SQLiteの読み戻し、同じ保存の再実行を確認。

`overlay.json` が今回明示的に書いた解釈、`semantics/` が元資料を含む保存物、
`semantics.db` が表、`report.json` が検証と根拠追跡の結果です。
自動抽出精度・論理推論・実運用上の正しさは、このデモでは検証していません。
""", encoding="utf-8")
    return {"status": "complete", "snapshot_id": dataset.snapshot_id, "tables": len(dataset.tables),
            "claims": len(trace["claims"]), "model_calls": 0, "cloud_calls": 0,
            "report": str((root / "report.json").resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False, indent=2))
