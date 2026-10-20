"""Explicit, bounded Foundry image calls with durable receipts and no retries."""
from __future__ import annotations

from pathlib import Path
import time

import httpx

from .knowledge import sha256
from .knowledge_store import atomic_json
from .narrative import _json_response
from .narrative_runtime import project_endpoint
from .tables import canonical_json
from .visual_relations import (Inventory, MAX_BYTES, VERSION, image_info, project_visual,
                               visual_request)


def extract_visual(source, destination, *, endpoint, model, page_label, strategy="grounded",
                   execute=False, credential=None, transport=None):
    endpoint = project_endpoint(endpoint)
    path = Path(source)
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    info = image_info(data, path.name)
    if strategy not in {"grounded", "direct"} or not isinstance(page_label, str) or not 1 <= len(page_label) <= 100:
        raise ValueError("Specify a strategy and an explicit page label")
    stages = ["inventory", "relations"] if strategy == "grounded" else ["direct"]
    first = visual_request(data, path.name, model, stages[0])
    config = {"version": VERSION, "source_name": path.name, "page_label": page_label,
              "model": model, "strategy": strategy, "project_endpoint": endpoint}
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    files = {"config.json": canonical_json(config), "source.bin": data}
    for name, content in files.items():
        (root / name).write_bytes(content)
    report = {"version": VERSION, "status": "prepared", "project_endpoint": endpoint, **info,
              "strategy": strategy, "model_deployment": model, "planned_requests": len(stages),
              "automatic_retries": 0, "requests": [], "publication": "not_performed"}
    receipt = root / "receipt.json"
    atomic_json(receipt, report)
    atomic_json(root / "planned-request.json", first)
    if not execute:
        return report
    owned, active, inventory = credential is None, None, None
    try:
        if owned:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        with httpx.Client(timeout=300, transport=transport, follow_redirects=False) as client:
            for stage in stages:
                request = visual_request(data, path.name, model, stage, inventory)
                request_bytes = canonical_json(request)
                files[stage + "-request.json"] = request_bytes
                (root / (stage + "-request.json")).write_bytes(request_bytes)
                active = {"stage": stage, "status": "submitting", "request_sha256": sha256(request_bytes)}
                report["requests"].append(active)
                report["status"] = "running"
                atomic_json(receipt, report)
                token = credential.get_token("https://ai.azure.com/.default").token
                started = time.monotonic()
                with client.stream("POST", endpoint + "/openai/v1/responses", content=request_bytes,
                        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}) as response:
                    active["http_status"] = response.status_code
                    buffer = bytearray()
                    for chunk in response.iter_bytes(chunk_size=65536):
                        if len(buffer) + len(chunk) > MAX_BYTES:
                            buffer.extend(chunk[:MAX_BYTES - len(buffer)])
                            (root / (stage + "-response.partial")).write_bytes(buffer)
                            active["response_truncated_locally"] = True
                            active["saved_bytes"] = len(buffer)
                            raise ValueError("Inference response exceeds local limit")
                        buffer.extend(chunk)
                    body = bytes(buffer)
                    (root / (stage + "-response.json")).write_bytes(body)
                    active["response_sha256"] = sha256(body)
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError("Unexpected inference acknowledgement")
                active["elapsed_seconds"] = round(time.monotonic() - started, 3)
                # Save failures too, but never treat a refusal/truncation as a candidate package.
                import json
                raw = json.loads(body)
                parsed = _json_response(raw)
                if stage == "inventory":
                    inventory = Inventory.model_validate(parsed, strict=True)
                files[stage + "-response.json"] = body
                active.update(status="completed", response_model=raw["model"], usage=raw.get("usage"))
                atomic_json(receipt, report)
        execution = {k: report[k] for k in ("project_endpoint", "model_deployment", "source_sha256", "requests")}
        files["execution.json"] = canonical_json(execution)
        (root / "execution.json").write_bytes(files["execution.json"])
        projected = project_visual(files)
        manifest = {"version": VERSION, "status": "complete", "package_id": projected["package_id"],
                    "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in files.items()}}
        atomic_json(root / "manifest.json", manifest)
        report.update(status="complete", package_id=projected["package_id"],
                      observation_count=len(projected["inventory"]["observations"]),
                      relation_count=len(projected["relations"]["relations"]),
                      semantic_validation="not_performed")
    except httpx.HTTPStatusError as exc:
        report.update(status="failed" if 400 <= exc.response.status_code < 500 else "unknown",
                      error_type=type(exc).__name__)
        if active:
            active["status"] = report["status"]
    except (httpx.TransportError, OSError) as exc:
        report.update(status="unknown", error_type=type(exc).__name__)
        if active:
            active["status"] = "unknown"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if active and active["status"] != "completed":
            active["status"] = "failed"
    finally:
        if owned and credential is not None:
            credential.close()
    atomic_json(receipt, report)
    return report
