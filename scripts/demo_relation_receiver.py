"""Synthetic, provider-free examples of receiver applications.

The extraction responses below are handwritten fixtures, not model predictions
or an extraction-quality benchmark. All output stays in a new local directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_dataset_store import export_relation_dataset, export_relation_view
from sr_adapter.relation_receiver import (ReceiverConflict, receive_relation_views,
    search_received_relations, trace_received_relation)
from sr_adapter.relation_sources import read_source


def prepare(root, owner):
    sentences = [f"融資システムの主管は{owner}である。", "決済システムの主管は融資企画部ではない。",
                 "照会APIの主管は未確定である。", "この資料は検討用。主管変更は承認後のみ適用する。"]
    data = "\n\n".join(sentences).encode()
    ontology = NarrativeOntology.model_validate({"namespace": "urn:receiver-demo:",
        "classes": {"System": "システム", "Team": "部門"}, "predicates": [
            {"id": "ownedBy", "label": "主管", "subject_type": "System", "object_type": "Team", "description": "資料に記載された主管"}]})
    units = read_source(data, "synthetic-system-owners.txt")["units"]
    assertions = [{"subject": subject, "predicate": "ownedBy", "object": target, "status": status,
                   "evidence": [{"unit_id": units[i]["id"], "quote": units[i]["text"]}]}
                  for i, (subject, target, status) in enumerate([
                      ("融資システム", owner, "asserted"), ("決済システム", "融資企画部", "denied"), ("照会API", None, "unknown")])]
    response = {"status": "completed", "model": "handwritten-fixture-no-model-call",
        "output": [{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": json.dumps({"assertions": assertions}, ensure_ascii=False)}]}]}
    manifest, files = render_narrative(data=data, source_name="synthetic-system-owners.txt", ontology=ontology,
        model="handwritten-fixture-no-model-call", response=response)
    save_narrative(root / "extracted", manifest, files)
    export_relation_dataset([root / "extracted"], root / "dataset", profile="narrative")
    export_relation_view(root / "dataset", root / "search", view="search")
    export_relation_view(root / "dataset", root / "graph", view="graph")


def run(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    prepare(root / "v1", "融資企画部")
    db = root / "receiver.db"
    args = (root / "v1/search", root / "v1/graph", db)
    first = receive_relation_views(*args, collection="systems")
    repeat = receive_relation_views(*args, collection="systems")
    found = search_received_relations(db, collection="systems", query="融資システム")
    trace = trace_received_relation(db, collection="systems", card_id=found["hits"][0]["card_id"],
                                   snapshot_id=found["snapshot_id"])
    denials = search_received_relations(db, collection="systems", assertion_status="denied")
    review_queue = search_received_relations(db, collection="systems", assertion_status="unknown")
    context = search_received_relations(db, collection="systems", query="承認後", scope="context")
    prepare(root / "v2", "業務企画部")
    second = receive_relation_views(root / "v2/search", root / "v2/graph", db, collection="systems",
                                    expected_snapshot=first["snapshot_id"])
    current = search_received_relations(db, collection="systems", query="融資システム")
    previous = search_received_relations(db, collection="systems", query="融資システム", snapshot_id=first["snapshot_id"])
    stale_rejected = False
    try:
        receive_relation_views(*args, collection="systems")
    except ReceiverConflict:
        stale_rejected = True
    assert repeat["inserted_rows"] == 0 and stale_rejected
    assert trace["evidence"][0]["properties"]["text"] == "融資システムの主管は融資企画部である。"
    assert (found["total"], denials["total"], review_queue["total"], context["total"]) == (1, 1, 1, 3)
    assert previous["hits"][0]["object_label"] == "融資企画部"
    assert current["hits"][0]["object_label"] == "業務企画部"
    result = {"status": "complete", "fixture": "synthetic handwritten responses", "model_calls": 0,
        "cloud_calls": 0, "receipts": {"first": first, "repeat": repeat, "updated": second},
        "examples": {"relation_search": found, "evidence_trace": trace, "negative_claims": denials,
            "review_queue": review_queue, "uncited_context": context, "previous_revision": previous, "current_revision": current},
        "stale_delivery_rejected": stale_rejected}
    (root / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text("""# 関係データ受け取りの応用デモ

合成資料と手書きの抽出応答を使った、受け取り・利用経路の確認です。
モデル呼び出しもクラウド接続も行っていません。抽出精度の評価ではありません。

| 利用例 | この実行で確認したこと |
| --- | --- |
| 案件の初動調査 | 「融資システム」で関係カード1件を検索し、同じ版の原文へ戻る |
| 資料レビュー | 主管不明1件を取り出す。否定1件は肯定の主管関係に変えない |
| 条件の確認 | 関係の引用に入っていない「承認後」の条件も本文検索で3件に届く |
| 版の比較 | 旧版は融資企画部、新版は業務企画部。利用中の版を切替後も旧版の根拠が残る |
| 再送・更新 | 同じ受け取りは追加0行。古い再送は新版を上書きしない |

全件の受領結果と検索・追跡結果は [report.json](report.json)、
受け取り済みデータと元資料は `receiver.db` に保存されています。
`v1/` と `v2/` に元バンドル・共通表・検索カード・グラフ表も残しています。

この版比較は利用者が同じ論理資料だと指定した2版の表示例です。
文書横断の同一実体判定、変更の自動マッチング、業務上の変更確定は行いません。
""", encoding="utf-8")
    return {"status": "complete", "report": str((root / "report.json").resolve()),
            "database": str(db.resolve()), "model_calls": 0, "cloud_calls": 0,
            "stale_delivery_rejected": stale_rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out), ensure_ascii=False, indent=2))
