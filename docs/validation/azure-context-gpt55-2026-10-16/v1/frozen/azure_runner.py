#!/usr/bin/env python3
"""Frozen eight-request Azure GPT-5.5 context comparison; no retrieval or images."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
from time import monotonic
from urllib.parse import urlsplit

import httpx

from sr_adapter.knowledge import sha256
from sr_adapter.knowledge_store import atomic_json
from sr_adapter.tables import canonical_json

ARMS = ["normalized_records", "row_contexts"]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _endpoint(value):
    url = urlsplit(value)
    if (url.scheme != "https" or url.username or url.password or url.query or url.fragment
            or url.port not in (None, 443)
            or not re.fullmatch(r"[a-z0-9-]+\.services\.ai\.azure\.com", url.hostname or "")
            or not re.fullmatch(r"/api/projects/[A-Za-z0-9._-]+/openai/v1/responses", url.path)):
        raise ValueError("Use the explicit existing Azure project Responses endpoint")
    return value


def _verify_files(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()) != expected:
            raise ValueError("Frozen file hash mismatch")


def prepare(source, destination, endpoint):
    source = Path(source)
    original = json.loads((source / "spec.json").read_bytes())
    _verify_files(source, original["files"])
    if (original["arms"] != ARMS or original["planned_requests"] != 8 or original["repeats"] != 2
            or [(case["name"], case["questions"]) for case in original["cases"]] != [("synthetic", 8), ("roles", 4)]):
        raise ValueError("Only the frozen two-case, eight-request probe is supported")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name in [*original["files"], "spec.json"]:
        output = root / "inputs" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((source / name).read_bytes())
    (root / "azure_runner.py").write_bytes(Path(__file__).read_bytes())
    order = []
    for repeat in (1, 2):
        cases = original["cases"] if repeat == 1 else original["cases"][::-1]
        for case in cases:
            for arm in ARMS if repeat == 1 else ARMS[::-1]:
                order.append({"id": f"{case['name']}-{arm}-{repeat}", "case": case["name"], "arm": arm, "repeat": repeat})
    spec = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "paired context consumption only; no images, native MCP, retrieval or Fabric reads",
        "api": "azure_project_responses_v1", "endpoint_sha256": sha256(_endpoint(endpoint).encode()),
        "model": "gpt-5.5", "response_model_pattern": r"gpt-5\.5(?:-[0-9]{4}-[0-9]{2}-[0-9]{2})?",
        "reasoning": {"effort": "low"}, "store": False, "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 4096, "temperature": "omitted", "timeout_seconds": 120, "automatic_retries": 0,
        "auth": "AzureCliCredential; https://ai.azure.com/.default; credentials never saved",
        "arms": ARMS, "repeats": 2, "planned_requests": 8, "planned_answers_per_arm": 24,
        "cases": original["cases"], "order": order,
        "source_spec_sha256": sha256((source / "spec.json").read_bytes()),
        "files": {path.relative_to(root).as_posix(): sha256(path.read_bytes())
                  for path in sorted(root.rglob("*")) if path.is_file()}}
    _write(root / "spec.json", spec)
    return spec


def _verified_spec(folder):
    root = Path(folder)
    spec = json.loads((root / "spec.json").read_bytes())
    _verify_files(root, spec["files"])
    if (spec["schema_version"] != 1 or spec["model"] != "gpt-5.5" or spec["planned_requests"] != 8
            or spec["repeats"] != 2 or spec["arms"] != ARMS or spec["max_output_tokens"] != 4096
            or spec["reasoning"] != {"effort": "low"} or spec["store"] is not False
            or spec["text"] != {"format": {"type": "json_object"}} or spec["temperature"] != "omitted"
            or spec["automatic_retries"] != 0 or spec["timeout_seconds"] != 120 or len(spec["order"]) != 8
            or [(case["name"], case["questions"]) for case in spec["cases"]] != [("synthetic", 8), ("roles", 4)]
            or sha256(Path(__file__).read_bytes()) != spec["files"]["azure_runner.py"]):
        raise ValueError("Unsupported or different frozen runner/specification")
    expected = {(case["name"], arm, repeat) for case in spec["cases"] for arm in ARMS for repeat in (1, 2)}
    if len(expected) != 8 or {(row["case"], row["arm"], row["repeat"]) for row in spec["order"]} != expected:
        raise ValueError("Missing or duplicate planned request slot")
    return spec


def _request(folder, spec, row):
    root = Path(folder) / "inputs"
    prompt = "Source context:\n" + (root / row["case"] / (row["arm"] + ".json")).read_text(encoding="utf-8")
    prompt += "\nQuestions:\n" + (root / row["case"] / "questions.json").read_text(encoding="utf-8")
    return {"model": spec["model"], "reasoning": spec["reasoning"], "store": False,
        "text": spec["text"], "max_output_tokens": spec["max_output_tokens"],
        "input": [{"role": "system", "content": (root / "system.txt").read_text(encoding="utf-8")},
                  {"role": "user", "content": prompt}]}


def _scorer(folder):
    # Reuse the original frozen score function, rather than changing its metric.
    definition = importlib.util.spec_from_file_location("frozen_relation_scorer", Path(folder) / "inputs" / "runner.py")
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module.score


def _result(folder, spec, row, raw):
    if raw.get("status") != "completed" or raw.get("error") or not re.fullmatch(spec["response_model_pattern"], raw.get("model", "")):
        raise ValueError("Incomplete response or unexpected model")
    messages = [item for item in raw.get("output", []) if item.get("type") == "message"]
    if len(messages) != 1 or messages[0].get("role") != "assistant":
        raise ValueError("Expected one assistant output")
    content = messages[0].get("content", [])
    if len(content) != 1 or content[0].get("type") != "output_text":
        raise ValueError("Missing JSON text or refusal")
    payload = json.loads(content[0]["text"])
    case = next(case for case in spec["cases"] if case["name"] == row["case"])
    scores = _scorer(folder)(payload, json.loads((Path(folder) / "inputs" / row["case"] / "gold.json").read_bytes()), case["assertion_ids"])
    usage = raw["usage"]
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if not isinstance(usage.get(name), int) or usage[name] < 0:
            raise ValueError("Missing token usage")
    return scores, usage


def run(frozen, destination, *, endpoint, execute=False, credential=None, transport=None):
    folder = Path(frozen)
    spec = _verified_spec(folder)
    if sha256(_endpoint(endpoint).encode()) != spec["endpoint_sha256"]:
        raise ValueError("Endpoint differs from the frozen target")
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256((folder / "spec.json").read_bytes()), "planned_requests": 8,
        "requests": [{**row, "status": "not_executed"} for row in spec["order"]]}
    atomic_json(out / "receipt.json", receipt)
    if not execute:
        receipt["status"] = "not_executed"
        atomic_json(out / "receipt.json", receipt)
        return receipt
    owned = credential is None
    failed = False
    try:
        if owned:
            from azure.identity import AzureCliCredential
            credential = AzureCliCredential()
        with httpx.Client(timeout=spec["timeout_seconds"], transport=transport, follow_redirects=False) as client:
            for row in receipt["requests"]:
                if failed:
                    row["status"] = "skipped_previous_failure"
                    continue
                request = _request(folder, spec, row)
                _write(out / (row["id"] + "-request.json"), request)
                row.update(status="submitting", request_sha256=sha256(canonical_json(request)))
                atomic_json(out / "receipt.json", receipt)
                started = monotonic()
                try:
                    token = credential.get_token("https://ai.azure.com/.default").token
                    response = client.post(endpoint, content=canonical_json(request),
                        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                    row.update(http_status=response.status_code, elapsed_seconds=round(monotonic() - started, 4),
                               wire_response_sha256=sha256(response.content))
                    # Preserve successful and failed raw bodies before interpreting them.
                    (out / (row["id"] + "-response.json")).write_bytes(response.content)
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError("Unexpected response acknowledgement")
                    raw = response.json()
                    row["response_sha256"] = sha256(canonical_json(raw))
                    row["completion_status"] = raw.get("status")
                    row["response_model"] = raw.get("model")
                    row["scores"], row["usage"] = _result(folder, spec, row, raw)
                    row["status"] = "completed"
                except Exception as exc:
                    failed = True
                    row.update(status="failed", error_type=type(exc).__name__,
                               elapsed_seconds=round(monotonic() - started, 4))
                atomic_json(out / "receipt.json", receipt)
    except Exception as exc:
        receipt["setup_error_type"] = type(exc).__name__
        failed = True
    finally:
        if owned and credential is not None:
            credential.close()
    completed = [row for row in receipt["requests"] if row["status"] == "completed"]
    same_model = len({row["response_model"] for row in completed}) == 1
    receipt["status"] = "complete" if not failed and len(completed) == 8 and same_model else "incomplete"
    receipt["completed_requests"] = len(completed)
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(out / "receipt.json", receipt)
    return receipt


def replay(frozen, run_folder):
    folder, out = Path(frozen), Path(run_folder)
    spec = _verified_spec(folder)
    receipt = json.loads((out / "receipt.json").read_bytes())
    if receipt["spec_sha256"] != sha256((folder / "spec.json").read_bytes()):
        raise ValueError("Receipt belongs to different frozen inputs")
    if [{key: row[key] for key in ("id", "case", "arm", "repeat")} for row in receipt["requests"]] != spec["order"]:
        raise ValueError("Receipt request order differs")
    result = {"spec_sha256": receipt["spec_sha256"], "status": receipt["status"], "planned_requests": 8,
        "completed_requests": 0, "response_models": [], "arms": {}}
    for arm in ARMS:
        result["arms"][arm] = {"planned_answers": 24, "completed_requests": 0, "answer_match": 0,
            "evidence_match": 0, "strict": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_input_tokens": 0, "reasoning_tokens": 0}
    for row in receipt["requests"]:
        if row["status"] != "completed":
            continue
        request = json.loads((out / (row["id"] + "-request.json")).read_bytes())
        raw_bytes = (out / (row["id"] + "-response.json")).read_bytes()
        raw = json.loads(raw_bytes)
        if (request != _request(folder, spec, row) or sha256(canonical_json(request)) != row["request_sha256"]
                or sha256(raw_bytes) != row["wire_response_sha256"] or sha256(canonical_json(raw)) != row["response_sha256"]):
            raise ValueError("Saved request/response differs from the frozen experiment")
        scores, usage = _result(folder, spec, row, raw)
        if scores != row["scores"] or usage != row["usage"] or raw["model"] != row["response_model"]:
            raise ValueError("Receipt score/usage/model differs from raw response")
        result["completed_requests"] += 1
        result["response_models"].append(raw["model"])
        arm = result["arms"][row["arm"]]
        arm["completed_requests"] += 1
        for metric in ("answer_match", "evidence_match", "strict"):
            arm[metric] += sum(score[metric] for score in scores)
        for metric in ("input_tokens", "output_tokens", "total_tokens"):
            arm[metric] += usage[metric]
        arm["cached_input_tokens"] += usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        arm["reasoning_tokens"] += usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    result["response_models"] = sorted(set(result["response_models"]))
    if result["status"] == "complete" and (result["completed_requests"] != 8 or len(result["response_models"]) != 1):
        raise ValueError("Incomplete or heterogeneous responses cannot be complete")
    first, second = (result["arms"][arm] for arm in ARMS)
    result["winner"] = None
    if result["status"] == "complete":
        deltas = [second[key] - first[key] for key in ("answer_match", "strict")]
        result["comparison"] = "tie" if deltas == [0, 0] else "mixed"
        if min(deltas) >= 0 and max(deltas) > 0:
            result.update(winner="row_contexts", comparison="row_contexts_higher")
        if max(deltas) <= 0 and min(deltas) < 0:
            result.update(winner="normalized_records", comparison="normalized_records_higher")
    else:
        result["comparison"] = "incomplete"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source-frozen", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--endpoint", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--frozen", required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--endpoint", required=True)
    execute.add_argument("--execute", action="store_true")
    audit = commands.add_parser("replay")
    audit.add_argument("--frozen", required=True)
    audit.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_frozen, args.out, args.endpoint)
        print(json.dumps({"status": "frozen", "planned_requests": result["planned_requests"]}))
    elif args.command == "run":
        result = run(args.frozen, args.out, endpoint=args.endpoint, execute=args.execute)
        print(json.dumps({"status": result["status"], "planned_requests": 8,
                          "completed_requests": result.get("completed_requests", 0)}))
        return int(result["status"] == "incomplete")
    else:
        print(json.dumps(replay(args.frozen, args.run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
