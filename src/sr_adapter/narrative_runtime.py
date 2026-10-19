"""Explicit, receipt-first Foundry extraction; no hidden retries or data publication."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit

import httpx

from .knowledge import sha256
from .knowledge_store import atomic_json
from .narrative import (EXTRACTION_INSTRUCTIONS_VERSION, NarrativeOntology, _json_response, extraction_request,
                        render_narrative, save_narrative, transcription_request)
from .relation_sources import read_source
from .tables import canonical_json


def project_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None, 443)
            or not re.fullmatch(r"[A-Za-z0-9-]+\.services\.ai\.azure\.com", parsed.hostname or "")
            or not re.fullmatch(r"/api/projects/[A-Za-z0-9][A-Za-z0-9_-]*/?", parsed.path)):
        raise ValueError("Use an explicit Azure public-cloud Foundry project endpoint")
    return value.rstrip("/")


def extract_narrative(source_path: str | Path, *, ontology: NarrativeOntology, endpoint: str,
                      model: str, destination: str | Path, execute: bool = False,
                      credential=None, transport=None,
                      instructions_version: int = EXTRACTION_INSTRUCTIONS_VERSION) -> dict:
    endpoint = project_endpoint(endpoint)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise ValueError("Invalid model deployment name")
    path = Path(source_path)
    data = path.read_bytes()
    source = read_source(data, path.name)
    planned_extraction = extraction_request(source, ontology, model, instructions_version=instructions_version)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "prepared", "project_endpoint": endpoint,
        "model_deployment": model, "source_name": path.name, "source_sha256": sha256(data),
        "ontology_sha256": sha256(canonical_json(ontology.model_dump(mode="json"))),
        "instructions_version": instructions_version,
        "planned_model_requests": 2 if source["needs_transcription"] else 1,
        "automatic_retries": 0, "requests": [], "publication": "not_performed"}
    receipt_path = root / "receipt.json"
    atomic_json(receipt_path, receipt)
    (root / "source.bin").write_bytes(data)
    atomic_json(root / "ontology.json", ontology.model_dump(mode="json"))
    first = (transcription_request(data, source, model) if source["needs_transcription"]
             else planned_extraction)
    atomic_json(root / "planned-request.json", first)
    if not execute:
        return receipt
    owned = credential is None
    active = None
    try:
        if owned:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        with httpx.Client(timeout=180, transport=transport, follow_redirects=False) as client:
            def send(stage, request):
                nonlocal active
                active = {"stage": stage, "status": "submitting", "request_sha256": sha256(canonical_json(request))}
                receipt["requests"].append(active)
                receipt["status"] = "running"
                atomic_json(root / (stage + "-request.json"), request)
                atomic_json(receipt_path, receipt)
                token = credential.get_token("https://ai.azure.com/.default").token
                response = client.post(endpoint + "/openai/v1/responses", content=canonical_json(request),
                    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                (root / (stage + "-response.json")).write_bytes(response.content)
                active.update(http_status=response.status_code, wire_response_sha256=sha256(response.content))
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("Unexpected inference acknowledgement")
                raw = response.json()
                _json_response(raw)
                active.update(status="completed", response_model=raw["model"], usage=raw.get("usage"))
                atomic_json(receipt_path, receipt)
                return raw
            transcription_response = None
            if source["needs_transcription"]:
                transcription_response = send("transcription", first)
                source = read_source(data, path.name, transcription=_json_response(transcription_response))
            raw = send("extraction", extraction_request(source, ontology, model, instructions_version=instructions_version))
        manifest, files = render_narrative(data=data, source_name=path.name, ontology=ontology, model=model,
            response=raw, transcription_response=transcription_response, instructions_version=instructions_version)
        save_narrative(root / "bundle", manifest, files)
        receipt.update(status="complete", package_id=manifest["package_id"], assertion_count=manifest["assertion_count"],
            bundle_manifest_sha256=sha256((root / "bundle/manifest.json").read_bytes()), semantic_validation="not_performed")
    except httpx.HTTPStatusError as exc:
        receipt.update(status="failed" if 400 <= exc.response.status_code < 500 else "unknown",
                       error_type=type(exc).__name__)
        if active:
            active["status"] = receipt["status"]
    except (httpx.TransportError, OSError) as exc:
        receipt.update(status="unknown", error_type=type(exc).__name__)
        if active:
            active["status"] = "unknown"
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        if active and active["status"] != "completed":
            active["status"] = "failed"
    finally:
        if owned and credential is not None:
            credential.close()
    atomic_json(receipt_path, receipt)
    return receipt
