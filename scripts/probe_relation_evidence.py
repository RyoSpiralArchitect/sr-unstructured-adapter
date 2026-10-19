#!/usr/bin/env python3
"""Score saved narrative bundles against fixed, source-scoped development gold.

No authentication or model calls. Tuple correctness and context evidence coverage
are separate measurements; literal values alone are also counted to expose
misassociation. Gold is never included in an extraction request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from sr_adapter.knowledge import sha256
from sr_adapter.narrative import NarrativeOntology, verify_narrative
from sr_adapter.relation_sources import read_source


FIELDS = ("subject", "predicate", "object", "status")


def _key(row, namespace=""):
    predicate = row["predicate"]
    if namespace and predicate.startswith(namespace):
        predicate = predicate[len(namespace):]
    return row["subject"], predicate, row["object"], row["status"]


def _rows(keys):
    return [dict(zip(FIELDS, key)) for key in sorted(keys, key=lambda x: json.dumps(x, ensure_ascii=False))]


def _metrics(actual, expected):
    actual, expected = Counter(actual), Counter(expected)
    tp = sum((actual & expected).values())
    return {"tp": tp, "fp": sum((actual - expected).values()), "fn": sum((expected - actual).values()),
            "predicted": sum(actual.values()), "expected": sum(expected.values())}


def score(gold_path: str | Path, bundles: dict[str, str | Path]) -> dict:
    path = Path(gold_path)
    gold = json.loads(path.read_bytes())
    if (gold.get("schema_version") != 1 or gold.get("gold_origin") != "manually_defined_synthetic"
            or gold.get("gold_must_not_be_sent_to_model") is not True):
        raise ValueError("Unsupported development gold")
    ontology = NarrativeOntology.model_validate_json((path.parent / gold["ontology_path"]).read_bytes())
    documents = gold["documents"]
    if len({d["path"] for d in documents}) != len(documents) or set(bundles) != {d["path"] for d in documents}:
        raise ValueError("Bundle bindings must match every gold source exactly once")
    results = []
    for doc in documents:
        original = (path.parent / doc["path"]).read_bytes()
        source = read_source(original, doc["path"])
        manifest, files = verify_narrative(bundles[doc["path"]])
        if files["source.bin"] != original or json.loads(files["source.json"]) != source:
            raise ValueError("Bundle does not bind the exact original source")
        if json.loads(files["ontology.json"]) != ontology.model_dump(mode="json"):
            raise ValueError("Ontology differs from fixed gold")
        expected = [_key(row) for row in doc["assertions"]]
        if len(set(expected)) != len(expected):
            raise ValueError("Gold tuples must be unique within a document")
        by_line = {u["locator"].get("line_start"): u for u in source["units"]}
        for row in doc["assertions"]:
            if not row["required_lines"] or any(n not in by_line for n in row["required_lines"]):
                raise ValueError("Gold context must point to original source units")
        observed = json.loads(files["assertions.json"])
        actual = [_key(row, ontology.namespace) for row in observed]
        context = []
        for row in doc["assertions"]:
            matches = [item for item in observed if _key(item, ontology.namespace) == _key(row)]
            required = {by_line[n]["id"] for n in row["required_lines"]}
            # Each row must stand on its own; combining evidence across duplicate
            # model assertions must not turn incomplete rows into a passing one.
            covered = len(matches) == 1 and required <= {e["unit_id"] for e in matches[0]["evidence"]}
            context.append({**dict(zip(FIELDS, _key(row))), "required_lines": row["required_lines"],
                            "exact_tuple_occurrences": len(matches), "context_complete": covered})
        expected_values = [row["object"] for row in doc["assertions"] if row["object"] is not None]
        actual_values = [row["object"] for row in observed if row["object"] is not None]
        results.append({"source": doc["path"], "source_sha256": sha256(original),
            "bundle_package_id": manifest["package_id"], "tuple_metrics": _metrics(actual, expected),
            "literal_value_metrics": _metrics(actual_values, expected_values),
            "unexpected_tuples": _rows((Counter(actual) - Counter(expected)).elements()),
            "missing_tuples": _rows((Counter(expected) - Counter(actual)).elements()),
            "context_complete": sum(item["context_complete"] for item in context),
            "context_expected": len(context), "context_rows": context})
    return {"schema_version": 1, "dataset_id": gold["dataset_id"], "gold_sha256": sha256(path.read_bytes()),
        "documents": results, "tuple_totals": {key: sum(r["tuple_metrics"][key] for r in results)
            for key in ("tp", "fp", "fn", "predicted", "expected")},
        "context_complete": sum(r["context_complete"] for r in results),
        "context_expected": sum(r["context_expected"] for r in results),
        "limitations": ["Authored synthetic development set, not held-out or real-world accuracy.",
            "Context coverage checks fixed required source units, not every semantic implication.",
            "Literal-value matches alone do not establish correct row, subject, predicate or status association.",
            "No Foundry managed evaluation, retrieval, publication or downstream agent QA is performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--bundle", required=True, action="append", help="SOURCE_NAME=BUNDLE_DIRECTORY")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bindings = [value.split("=", 1) for value in args.bundle]
    if any(len(row) != 2 for row in bindings) or len(dict(bindings)) != len(bindings):
        parser.error("Each source requires exactly one bundle binding")
    result = score(args.gold, dict(bindings))
    with Path(args.out).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("tuple_totals", "context_complete", "context_expected")}))


if __name__ == "__main__":
    main()
