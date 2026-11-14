"""Prepare and score the paired evaluation offline; no credentials or HTTP calls."""

import argparse
import json
from pathlib import Path

from sr_adapter.ontology_evaluation import prepare_evaluation, score_evaluation
from sr_adapter.knowledge_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--database", type=Path, required=True)
    prepare.add_argument("--suite", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--protocol-id", required=True)
    report.add_argument("--reviews", type=Path)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_evaluation(args.database, args.suite, args.out)
    else:
        reviews = json.loads(args.reviews.read_bytes()) if args.reviews else None
        result = score_evaluation(args.run, args.protocol_id, reviews)
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "scores.json", result)
        suite = json.loads((args.run / "suite.json").read_bytes())
        questions = {q["id"]: q for q in suite["questions"]}
        checklist = []
        for cell in result["cells"]:
            path = (
                args.run / "cells" / cell["question_id"] / cell["arm"] / "receipt.json"
            )
            receipt = json.loads(path.read_bytes()) if path.exists() else {}
            if not receipt.get("result", {}).get("answer", {}).get("text"):
                continue
            checklist.append(
                {
                    "question_id": cell["question_id"],
                    "arm": cell["arm"],
                    "receipt_sha256": cell["receipt_sha256"],
                    "checks": {
                        r["id"]: "uncertain"
                        for r in questions[cell["question_id"]]["rubric"]
                    },
                    "notes": "",
                }
            )
        atomic_json(
            args.out / "review-template.json",
            {"protocol_id": args.protocol_id, "judgments": checklist},
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
