#!/usr/bin/env python3
"""Author the fixed column/row correspondence probe; no inference or PDF I/O."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def specification():
    cases = []
    headers = ["案件", "主担当", "副担当", "予算(万円)"]
    families = [
        ("column-order", ["朝凪", "夕凪", "星凪"], ["東部室", "西部室", "南部室"], ["125", "240", "未定"]),
        ("header-binding", ["白樺", "青葉", "若竹"], ["企画室", "審査室", "運用室"], ["315", "480", "未定"]),
        ("row-order", ["港", "丘", "森"], ["海班", "山班", "川班"], ["165", "290", "未定"]),
    ]
    for family, projects, offices, budgets in families:
        original = [headers[:]] + [[projects[i], offices[i], offices[(i + 1) % 3], budgets[i]] for i in range(3)]
        for variant in ("base", "changed"):
            rows = copy.deepcopy(original)
            if variant == "changed":
                if family == "column-order":
                    rows = [[row[i] for i in (3, 2, 0, 1)] for row in rows]
                elif family == "header-binding":
                    rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
                else:
                    rows = [rows[0], *reversed(rows[1:])]
            records = {row[rows[0].index("案件")]: dict(zip(rows[0], row)) for row in rows[1:]}
            fields = []
            for term, value in (("smallest", budgets[0]), ("largest", budgets[1])):
                word = "最小" if term == "smallest" else "最大"
                fields.append({"id": term, "task": "value_recognition",
                    "question": f"予算(万円)列の数値のうち{word}のものは何か。数値のみ。",
                    "accepted": [value], "page": 1, "evidence": [value], "anchors": [value]})
            for index, (project, column) in enumerate([
                (projects[0], "主担当"), (projects[1], "主担当"), (projects[0], "副担当"),
                (projects[1], "副担当"), (projects[0], "予算(万円)"), (projects[2], "予算(万円)"),
            ]):
                value = records[project][column]
                fields.append({"id": f"binding_{index}", "task": "association",
                    "question": f"案件「{project}」の「{column}」は何か。名称または数値のみ。",
                    "accepted": ["UNKNOWN" if value == "未定" else value], "page": 1,
                    "evidence": [value, " ".join(records[project].values())], "anchors": [value]})
            cases.append({"id": f"{family}-{variant}", "kind": "synthetic", "source_pages": [1],
                "tags": [family, variant], "fields": fields,
                "document": {"pages": [{"title": "担当・予算一覧", "tables": [{"rows": rows, "widths": [180] * 4}],
                    "notes": ["予算の単位は万円。未定は確定値ではない。", "主担当と副担当を区別して読むこと。"]}]}})
    pilot = json.loads((ROOT / "data/vlm_benchmark/pilot-v1.json").read_text("utf-8"))
    for case in pilot["cases"]:
        if case["id"] in {"da-roles", "da-story-policy"}:
            case = copy.deepcopy(case)
            for field in case["fields"]:
                field["task"] = "real_document"
            cases.append(case)
    return {"annotation_status": "agent-authored; visual source check required; not independent human gold",
            "study": "table-correspondence-v1", "cases": cases}


if __name__ == "__main__":
    output = ROOT / "data/vlm_benchmark/table-correspondence-v1.json"
    output.write_text(json.dumps(specification(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
