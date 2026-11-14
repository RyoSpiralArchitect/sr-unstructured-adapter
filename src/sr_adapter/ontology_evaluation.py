"""Frozen paired evaluation; provider calls and semantic judgments remain explicit."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from .knowledge_store import atomic_json
from .local_ontology import (
    inspect_ontology,
    list_ontologies,
    read_ontology_source,
    search_ontology,
)
from .ontology_context import pack_ontology_context, unpack_ontology_context
from .ontology_question import GUIDANCE, answer_ontology_question
from .tables import canonical_json

MODEL = "gpt-6-luna"
ARMS = ("all_evidence", "retrieval")
BUDGETS = {
    "max_rounds": 6,
    "max_tool_calls": 24,
    "max_inspections": 12,
    "max_context_bytes": 2 * 1024 * 1024,
}
MODEL_SETTINGS = {
    "model": MODEL,
    "reasoning": {"effort": "medium"},
    "max_output_tokens": 6000,
}


class ProviderRefusal(RuntimeError):
    """A host transport can signal rejection: stop this protocol, without retry."""


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(value):
    return canonical_json(value)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _implementation():
    package = Path(__file__).parent
    paths = sorted([*package.rglob("*.py"), *package.rglob("*.cpp")])
    return {str(p.relative_to(package)): _hash(p) for p in paths}


def _validate_suite(suite):
    questions = suite["questions"]
    if (
        suite["version"] != 1
        or len(questions) != 12
        or Counter(q["category"] for q in questions)
        != Counter({"identity": 3, "condition": 3, "revision": 3, "unknown": 3})
        or len({q["id"] for q in questions}) != 12
    ):
        raise ValueError("Use twelve distinct questions, three per frozen category")
    if not suite["corpus"] or len({c["name"] for c in suite["corpus"]}) != len(
        suite["corpus"]
    ):
        raise ValueError("Use distinct pinned corpus entries")
    for q in questions:
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", q["id"])
            or not isinstance(q["question"], str)
            or not 1 <= len(q["question"]) <= 8192
            or not q["evidence_groups"]
            or not q["rubric"]
        ):
            raise ValueError("Invalid question or missing gold")
        if len({r["id"] for r in q["rubric"]}) != len(q["rubric"]):
            raise ValueError("Duplicate rubric criterion")
        for r in q["rubric"]:
            if (
                r["kind"] not in {"required", "forbidden"}
                or not r["description"].strip()
            ):
                raise ValueError("Invalid rubric criterion")
        for g in q["evidence_groups"]:
            if (
                not g["acceptable_references"]
                or g["source_unit"] not in suite["source_units"]
            ):
                raise ValueError("Missing source unit or reference alternatives")


def prepare_evaluation(database, suite_file, destination):
    """Freeze a private DB, gold and full evidence. No network or model invocation."""
    suite = _read(suite_file)
    _validate_suite(suite)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    copied = root / "ontology.sqlite"
    with closing(
        sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    ) as src:
        with closing(sqlite3.connect(copied)) as dst:
            src.backup(dst)
    catalog = list_ontologies(copied, collection=suite["collection"])["entries"]
    if {(e["name"], e["snapshot_id"]) for e in catalog} != {
        (e["name"], e["snapshot_id"]) for e in suite["corpus"]
    }:
        raise ValueError("The active corpus differs from the frozen gold")
    inspections = []
    for entry in suite["corpus"]:
        scope = {
            "collection": suite["collection"],
            "name": entry["name"],
            "snapshot_id": entry["snapshot_id"],
        }
        found = search_ontology(copied, **scope, kind="claim", limit=100)
        if not found["hits"] or found["has_more"]:
            raise ValueError("Corpus exceeds the bounded full-evidence baseline")
        values = [inspect_ontology(copied, **h["reference"]) for h in found["hits"]]
        if any(v["value"]["vocabulary_review"]["has_more"] for v in values):
            raise ValueError(
                "Full-evidence baseline requires complete vocabulary history"
            )
        packages = {c["package_id"] for v in values for c in v["value"]["claims"]}
        if len(packages) != 1:
            raise ValueError("Each benchmark entry must have one original source")
        original = read_ontology_source(
            copied, **scope, package_id=next(iter(packages))
        )
        if original["sha256"] != entry["source_sha256"]:
            raise ValueError("Original source differs from gold corpus")
        inspections.extend(values)
    allowed = {_key(v["reference"]) for v in inspections}
    for q in suite["questions"]:
        for group in q["evidence_groups"]:
            if any(_key(r) not in allowed for r in group["acceptable_references"]):
                raise ValueError("Gold refers to an unavailable primary claim")
    packed = pack_ontology_context(inspections)
    if unpack_ontology_context(packed) != inspections:
        raise ValueError("Full evidence did not round-trip exactly")
    atomic_json(root / "suite.json", suite)
    atomic_json(root / "all-evidence.json", packed)
    protocol = {
        "version": 1,
        "suite_id": suite["id"],
        "model_settings": MODEL_SETTINGS,
        "budgets": BUDGETS,
        "automatic_retries": 0,
        "attempts_per_cell": 1,
        "max_model_calls": 12 * (1 + BUDGETS["max_rounds"]),
        "guidance": GUIDANCE,
        "implementation": _implementation(),
        "order": [
            {"question_id": q["id"], "arm": arm}
            for i, q in enumerate(suite["questions"])
            for arm in (ARMS if i % 2 == 0 else ARMS[::-1])
        ],
        "files": {
            name: _hash(root / name)
            for name in ("suite.json", "ontology.sqlite", "all-evidence.json")
        },
        "claim_ceiling": "Synthetic development comparison; unequal interaction budgets, not equal-compute efficacy.",
    }
    atomic_json(root / "protocol.json", protocol)
    return {
        "status": "prepared",
        "protocol_id": _hash(root / "protocol.json"),
        "questions": 12,
        "cells": 24,
        "inspections": len(inspections),
        "model_calls": 0,
        "max_model_calls": protocol["max_model_calls"],
    }


def _verify(root, protocol_id):
    root = Path(root)
    if _hash(root / "protocol.json") != protocol_id:
        raise ValueError("Evaluation protocol changed")
    protocol = _read(root / "protocol.json")
    if protocol["implementation"] != _implementation():
        raise ValueError("Evaluation implementation changed; retain the old run")
    for name, digest in protocol["files"].items():
        if _hash(root / name) != digest:
            raise ValueError("Frozen evaluation input changed")
    return protocol, _read(root / "suite.json")


def _tokens(envelope):
    usage = envelope.get("usage")
    if type(usage) is not dict or any(
        type(usage.get(k)) is not int or usage[k] < 0
        for k in ("input_tokens", "output_tokens")
    ):
        return None
    return {k: usage[k] for k in ("input_tokens", "output_tokens")}


def run_evaluation_cell(root, protocol_id, question_id, arm, *, complete):
    """One ordered cell, with a trusted fixed-model host callback; no built-in HTTP.

    complete receives model_settings plus input (never gold), and returns
    {model, response: {actions, answer}, usage: {input_tokens, output_tokens}}.
    Missing usage stays unknown. ProviderRefusal stops all future protocol cells.
    """
    root = Path(root)
    protocol, suite = _verify(root, protocol_id)
    if (root / "STOP.json").exists():
        raise ValueError("Protocol stopped after a provider refusal")
    cell = {"question_id": question_id, "arm": arm}
    if cell not in protocol["order"] or not callable(complete):
        raise ValueError("Unknown cell or missing callback")
    index = protocol["order"].index(cell)
    for previous in protocol["order"][:index]:
        if not (
            root / "cells" / previous["question_id"] / previous["arm"] / "receipt.json"
        ).exists():
            raise ValueError("Execute cells in the frozen counterbalanced order")
    directory = root / "cells" / question_id / arm
    directory.mkdir(parents=True, exist_ok=False)
    question = next(q for q in suite["questions"] if q["id"] == question_id)
    events, provided = [], {}
    start = time.monotonic()

    def invoke(request):
        if len(events) >= (
            1 if arm == "all_evidence" else protocol["budgets"]["max_rounds"]
        ):
            raise ValueError("Model-call budget exhausted")
        wire = {**deepcopy(protocol["model_settings"]), "input": deepcopy(request)}
        if len(canonical_json(request)) > protocol["budgets"]["max_context_bytes"]:
            raise ValueError("Context budget exceeded")
        number = len(events)
        event = {"status": "submitted", "usage": None}
        events.append(event)
        atomic_json(directory / f"request-{number:02d}.json", wire)
        atomic_json(directory / "events.json", events)
        for e in request["evidence"]["entries"]:
            ref = e["inspection"]["reference"]
            provided[_key(ref)] = ref
        try:
            envelope = complete(deepcopy(wire))
            raw = canonical_json(envelope)
            if len(raw) > 1024 * 1024:
                (directory / f"response-{number:02d}.partial").write_bytes(
                    raw[: 1024 * 1024]
                )
                raise ValueError("Response budget exceeded")
            atomic_json(directory / f"response-{number:02d}.json", envelope)
            if type(envelope) is not dict or envelope.get("model") != MODEL:
                raise ValueError("Host must report the fixed model identity")
            event.update(status="returned", usage=_tokens(envelope))
            return envelope["response"]
        except Exception as error:
            event.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, ProviderRefusal):
                atomic_json(root / "STOP.json", {"reason": "provider_refusal", **cell})
            raise
        finally:
            atomic_json(directory / "events.json", events)

    try:
        if arm == "retrieval":
            result = answer_ontology_question(
                root / "ontology.sqlite",
                collections=(suite["collection"],),
                question=question["question"],
                complete=invoke,
                destination=directory / "loop",
                **protocol["budgets"],
            )
        else:
            packed = _read(root / "all-evidence.json")
            response = invoke(
                {
                    "guidance": protocol["guidance"],
                    "question": question["question"],
                    "tools": [],
                    "history": [],
                    "evidence": packed,
                    "remaining": {"rounds": 1, "tool_calls": 0, "inspections": 0},
                }
            )
            if (
                type(response) is not dict
                or set(response) != {"actions", "answer"}
                or response["actions"] != []
                or type(response["answer"]) is not dict
            ):
                raise ValueError("Full-evidence baseline requires one final answer")
            answer = response["answer"]
            if (
                set(answer) != {"text", "citations"}
                or not isinstance(answer["text"], str)
                or not answer["text"].strip()
                or type(answer["citations"]) is not list
            ):
                raise ValueError("Invalid final answer")
            linked = [r for r in answer["citations"] if _key(r) in provided]
            unlinked = [r for r in answer["citations"] if _key(r) not in provided]
            result = {
                "status": "complete" if linked and not unlinked else "needs_review",
                "answer": answer,
                "linked_citations": linked,
                "unlinked_citations": unlinked,
                "tool_calls": 0,
            }
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__}
    usage = (
        {
            k: sum(e["usage"][k] for e in events)
            for k in ("input_tokens", "output_tokens")
        }
        if events and all(e["usage"] is not None for e in events)
        else None
    )
    receipt = {
        "protocol_id": protocol_id,
        **cell,
        "result": result,
        "model_calls": len(events),
        "usage": usage,
        "elapsed_seconds": time.monotonic() - start,
        "known_usage": {
            k: sum(e["usage"][k] for e in events if e["usage"] is not None)
            for k in ("input_tokens", "output_tokens")
        },
        "calls_with_unknown_usage": sum(e["usage"] is None for e in events),
        "provided_references": list(provided.values()),
        "semantic_review": "pending",
    }
    atomic_json(directory / "receipt.json", receipt)
    return receipt


def _interrupted(directory, protocol_id, cell):
    """Recover observations, never resume or promote an interrupted attempt."""
    events_path = directory / "events.json"
    events = _read(events_path) if events_path.exists() else []
    usages = [_tokens(e) for e in events]
    known = {
        k: sum(u[k] for u in usages if u is not None)
        for k in ("input_tokens", "output_tokens")
    }
    provided = {}
    for i in range(len(events)):
        request = _read(directory / f"request-{i:02d}.json")
        for entry in request["input"]["evidence"]["entries"]:
            reference = entry["inspection"]["reference"]
            provided[_key(reference)] = reference
    tool_lower_bound = 0
    loop_receipt = directory / "loop" / "receipt.json"
    history = directory / "loop" / "history.json"
    if loop_receipt.exists():
        tool_lower_bound = _read(loop_receipt).get("tool_calls", 0)
    if history.exists():
        tool_lower_bound = max(tool_lower_bound, len(_read(history)))
    return {
        "protocol_id": protocol_id,
        **cell,
        "result": {
            "status": "interrupted",
            "tool_calls": 0 if cell["arm"] == "all_evidence" else None,
            "known_tool_calls_lower_bound": tool_lower_bound,
        },
        "model_calls": len(events),
        "usage": known if events and all(u is not None for u in usages) else None,
        "known_usage": known,
        "calls_with_unknown_usage": sum(u is None for u in usages),
        "elapsed_seconds": None,
        "provided_references": list(provided.values()),
    }


def score_evaluation(root, protocol_id, reviews=None):
    """Keep all 24 planned cells; semantic scores require receipt-bound review."""
    root = Path(root)
    protocol, suite = _verify(root, protocol_id)
    questions = {q["id"]: q for q in suite["questions"]}
    judgments = {}
    if reviews is not None:
        if reviews["protocol_id"] != protocol_id:
            raise ValueError("Review belongs to another protocol")
        for review in reviews["judgments"]:
            key = (review["question_id"], review["arm"])
            if (
                key in judgments
                or {"question_id": key[0], "arm": key[1]} not in protocol["order"]
            ):
                raise ValueError("Duplicate or unknown review cell")
            judgments[key] = review
    rows = []
    for cell in protocol["order"]:
        q = questions[cell["question_id"]]
        path = root / "cells" / cell["question_id"] / cell["arm"] / "receipt.json"
        receipt = (
            _read(path)
            if path.exists()
            else _interrupted(path.parent, protocol_id, cell)
            if path.parent.exists()
            else None
        )
        if receipt is not None and (
            receipt["protocol_id"] != protocol_id
            or any(receipt[k] != cell[k] for k in cell)
        ):
            raise ValueError("Cell receipt belongs to another evaluation")
        result = receipt["result"] if receipt else {}
        linked = {_key(r) for r in result.get("linked_citations", [])}
        provided = (
            {_key(r) for r in receipt["provided_references"]} if receipt else set()
        )
        groups = [
            {_key(r) for r in g["acceptable_references"]} for g in q["evidence_groups"]
        ]
        row = {
            **cell,
            "category": q["category"],
            "status": result.get("status", "not_run"),
            "model_calls": receipt["model_calls"] if receipt else 0,
            "usage": receipt["usage"] if receipt else None,
            "known_usage": receipt["known_usage"]
            if receipt
            else {"input_tokens": 0, "output_tokens": 0},
            "calls_with_unknown_usage": receipt["calls_with_unknown_usage"]
            if receipt
            else 0,
            "elapsed_seconds": receipt["elapsed_seconds"] if receipt else None,
            "tool_calls": result.get("tool_calls", 0),
            "known_tool_calls_lower_bound": result.get(
                "known_tool_calls_lower_bound", result.get("tool_calls", 0)
            ),
            "required_reference_groups": len(groups),
            "provided_reference_groups": sum(bool(g & provided) for g in groups),
            "cited_reference_groups": sum(bool(g & linked) for g in groups),
            "semantic_result": "unreviewed",
            "checks": None,
            "receipt_sha256": _hash(path) if path.exists() else None,
        }
        review = judgments.get((cell["question_id"], cell["arm"]))
        if review is not None:
            if (
                not receipt
                or not result.get("answer", {}).get("text")
                or review["receipt_sha256"] != row["receipt_sha256"]
                or set(review["checks"]) != {r["id"] for r in q["rubric"]}
                or any(
                    v not in {"pass", "fail", "uncertain"}
                    for v in review["checks"].values()
                )
            ):
                raise ValueError("Review is stale, incomplete, or has no answer")
            row["checks"] = review["checks"]
            values = set(review["checks"].values())
            row["semantic_result"] = (
                "fail"
                if "fail" in values
                else "uncertain"
                if "uncertain" in values
                else "pass"
            )
        rows.append(row)
    summaries = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {
            "planned": len(selected),
            "statuses": dict(Counter(r["status"] for r in selected)),
            "semantic_results": dict(Counter(r["semantic_result"] for r in selected)),
            "model_calls": sum(r["model_calls"] for r in selected),
            "known_usage_lower_bound": {
                k: sum(r["known_usage"][k] for r in selected)
                for k in ("input_tokens", "output_tokens")
            },
            "calls_with_unknown_usage": sum(
                r["calls_with_unknown_usage"] for r in selected
            ),
            "cells_with_known_usage": sum(r["usage"] is not None for r in selected),
        }
    paired = []
    for q in questions:
        a, b = [
            next(r for r in rows if r["question_id"] == q and r["arm"] == arm)
            for arm in ARMS
        ]
        eligible = all(
            r["status"] == "complete"
            and r["semantic_result"] == "pass"
            and r["usage"] is not None
            for r in (a, b)
        )
        paired.append(
            {
                "question_id": q,
                "quality_matched": eligible,
                "retrieval_minus_all_input_tokens": b["usage"]["input_tokens"]
                - a["usage"]["input_tokens"]
                if eligible
                else None,
            }
        )
    return {
        "protocol_id": protocol_id,
        "cells": rows,
        "arms": summaries,
        "pairs": paired,
        "claim_ceiling": "Reference coverage is not semantic recall. Unreviewed, failed and unrun cells remain visible; token deltas require two completed, reviewed passing answers.",
    }
