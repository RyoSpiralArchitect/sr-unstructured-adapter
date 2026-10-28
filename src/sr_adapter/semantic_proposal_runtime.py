"""One explicit OpenAI Responses call; no hidden retries or automatic publication."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import httpx

from .knowledge import sha256
from .knowledge_store import atomic_json
from .relation_dataset_store import _load_dataset, _save_snapshot
from .semantic_proposals import (MAX_RESPONSE_BYTES, proposal_request,
    render_semantic_proposal, verify_semantic_proposal)
from .tables import canonical_json

RESPONSES_URL = "https://api.openai.com/v1/responses"


def propose_semantics(bundle, destination, *, model, execute=False, api_key=None,
                      max_output_tokens=24000, transport=None):
    source, inputs = _load_dataset(bundle, allow_view=False)
    request = proposal_request(inputs, model=model, max_output_tokens=max_output_tokens)
    request_bytes = canonical_json(request)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    (root / "request.json").write_bytes(request_bytes)
    for name, data in inputs.items():
        path = root / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    receipt = {"version": 1, "status": "prepared", "stage": "prepared", "provider": "openai", "model": model,
        "source_snapshot_id": source.snapshot_id, "request_sha256": sha256(request_bytes),
        "planned_model_requests": 1, "submitted_requests": 0, "automatic_retries": 0,
        "publication": "not_performed", "semantic_validation": "not_performed"}
    receipt_path = root / "receipt.json"
    atomic_json(receipt_path, receipt)
    if not execute:
        return receipt
    try:
        secret = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not isinstance(secret, str) or not secret.strip():
            receipt["failure_reason"] = "credential_unavailable"
            raise ValueError("OpenAI API key unavailable")
        receipt.update(status="running", stage="submitting", submitted_requests=1)
        atomic_json(receipt_path, receipt)
        with httpx.Client(timeout=300, follow_redirects=False, transport=transport) as client:
            with client.stream("POST", RESPONSES_URL, content=request_bytes,
                    headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"}) as response:
                receipt["http_status"] = response.status_code
                buffer = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(buffer) + len(chunk) > MAX_RESPONSE_BYTES:
                        buffer.extend(chunk[:MAX_RESPONSE_BYTES - len(buffer)])
                        (root / "response.partial").write_bytes(buffer)
                        receipt["response_truncated_locally"] = True
                        raise ValueError("Response exceeds local limit")
                    buffer.extend(chunk)
                body = bytes(buffer)
                (root / "response.json").write_bytes(body)
                receipt.update(stage="response_received", wire_response_sha256=sha256(body))
                if response.status_code != 200:
                    try:
                        error = json.loads(body).get("error", {})
                        for field in ("type", "code"):
                            value = error.get(field)
                            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
                                receipt["provider_error_" + field] = value
                    except (ValueError, AttributeError):
                        pass
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("Unexpected inference acknowledgement")
        raw = json.loads(body)
        receipt["stage"] = "validating"
        manifest, files = render_semantic_proposal(inputs, model=model, response=raw, max_output_tokens=max_output_tokens)
        receipt["stage"] = "saving"
        _save_snapshot(root / "bundle", files, manifest)
        receipt.update(status="complete", stage="complete", response_model=raw["model"], usage=raw.get("usage"),
                       proposal_status=manifest["proposal_status"], claim_count=manifest["claim_count"],
                       semantic_snapshot_id=manifest["semantic_snapshot_id"], review_status="unreviewed",
                       bundle_manifest_sha256=sha256((root / "bundle/manifest.json").read_bytes()))
    except httpx.HTTPStatusError as exc:
        code = receipt.get("provider_error_code")
        quota = receipt.get("provider_error_type") == "insufficient_quota" or code == "credit_balance_exhausted"
        reason = ("provider_quota_exhausted" if quota else "provider_authentication_failed" if exc.response.status_code == 401
                  else "provider_rate_limited" if exc.response.status_code == 429 else "provider_request_failed")
        receipt.update(status="failed" if 400 <= exc.response.status_code < 500 else "unknown",
                       error_type=type(exc).__name__, failure_reason=reason)
    except (httpx.TransportError, OSError) as exc:
        receipt.update(status="unknown", error_type=type(exc).__name__, failure_reason="execution_outcome_unknown")
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        receipt.setdefault("failure_reason", "proposal_validation_failed" if receipt["stage"] == "validating" else "proposal_execution_failed")
    atomic_json(receipt_path, receipt)
    return receipt


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose")
    propose.add_argument("source", help="Replayable narrative or visual base dataset")
    propose.add_argument("--out", required=True)
    propose.add_argument("--model", required=True)
    propose.add_argument("--max-output-tokens", type=int, default=24000)
    propose.add_argument("--execute", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("bundle")
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            result = propose_semantics(args.source, args.out, model=args.model, execute=args.execute,
                                       max_output_tokens=args.max_output_tokens)
        else:
            result = verify_semantic_proposal(args.bundle)[0]
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] in ("complete", "prepared") else 1
    except (ValueError, TypeError, OSError, KeyError):
        print(json.dumps({"status": "failed", "error": "semantic_proposal_unavailable"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
