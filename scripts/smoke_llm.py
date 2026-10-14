#!/usr/bin/env python3
"""Opt-in, bounded live checks of the actual LLM drivers using synthetic data."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from sr_adapter.drivers.multicloud_driver import GoogleAIDriver, MistralDriver
from sr_adapter.drivers.openai_driver import OpenAIDriver
from sr_adapter.normalizer import LLMNormalizer


PROMPT = 'Return exactly this JSON object and nothing else: {"status":"ok","value":42}'
EXPECTED = {"status": "ok", "value": 42}
MODES = ("sync", "async", "stream", "async_stream")
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", OpenAIDriver),
    "mistral": ("MISTRAL_API_KEY", MistralDriver),
    "gemini": ("GEMINI_API_KEY", GoogleAIDriver),
}


def check_normalized(result):
    matches = False
    if len(result.choices) == 1:
        try:
            matches = json.loads(result.choices[0].text) == EXPECTED
        except (ValueError, TypeError):
            pass
    return {
        "status": "passed" if result.is_complete and matches else "failed",
        "completion_status": result.status,
        "content_matches": matches,
        "finish_reasons": [choice.finish_reason for choice in result.choices],
        "usage": result.usage,
        "response_model": result.model,
    }


def check_result(provider, raw):
    return check_normalized(LLMNormalizer().normalize(provider, raw, prompt=PROMPT))


def check_stream(provider, events):
    """Keep candidate boundaries, terminal reasons, and trailing usage events."""
    choices = {}
    usage = {}
    model = None
    count = 0
    blocked = False
    for event in events:
        count += 1
        normalized = LLMNormalizer().normalize(provider, event, prompt=PROMPT)
        model = normalized.model or model
        usage.update(normalized.usage)
        blocked = blocked or normalized.status == "blocked"
        for position, choice in enumerate(normalized.choices):
            index = choice.metadata.get("index", position)
            if type(index) is not int or index < 0:
                raise ValueError("Invalid stream candidate index")
            if index not in choices:
                choices[index] = {"text": "", "finish_reason": None, "metadata": {}}
            current = choices[index]
            if current["finish_reason"] is not None and choice.text:
                raise ValueError("Stream contained text after candidate completion")
            current["text"] += choice.text
            if choice.finish_reason is not None:
                current["finish_reason"] = choice.finish_reason
            # Refusals and tool calls can arrive before the terminal chunk.
            for key, value in choice.metadata.items():
                if value:
                    current["metadata"][key] = value
    raw = {"model": model, "usage": usage, "choices": [
        {"index": index, "finish_reason": value["finish_reason"],
         "message": {"content": value["text"], **value["metadata"]}}
        for index, value in sorted(choices.items())
    ]}
    result = LLMNormalizer().normalize(provider, raw, prompt=PROMPT)
    if blocked:
        result.status = "blocked"
    receipt = check_normalized(result)
    receipt["events"] = count
    return receipt


async def collect_async(driver):
    return [event async for event in driver.async_stream_generate(PROMPT)]


def run_provider(provider, model, on_receipt=lambda receipt: None):
    key_name, driver_type = PROVIDERS[provider]
    if not os.environ.get(key_name):
        receipts = [{"provider": provider, "model": model, "mode": mode,
                     "status": "missing_credential", "attempted": False, "key_env": key_name}
                    for mode in MODES]
        for receipt in receipts:
            on_receipt(receipt)
        return receipts
    config = {"api_key": os.environ[key_name], "model": model, "max_retries": 0,
              "timeout": 30, "max_tokens": 96, "temperature": 0}
    if provider == "gemini":
        config["generation_config"] = {"responseMimeType": "application/json"}
    else:
        config["response_format"] = {"type": "json_object"}
    driver = driver_type(provider, config)
    checks = (
        lambda: check_result(provider, driver.generate(PROMPT, metadata={"purpose": "adapter-smoke"})),
        lambda: check_result(provider, asyncio.run(driver.async_generate(PROMPT))),
        lambda: check_stream(provider, driver.stream_generate(PROMPT)),
        lambda: check_stream(provider, asyncio.run(collect_async(driver))),
    )
    receipts = []
    failed = False
    for mode, check in zip(MODES, checks):
        receipt = {"provider": provider, "model": model, "mode": mode, "attempted": False}
        if failed:
            receipt.update(status="skipped", reason="previous_failure")
        else:
            # Persist intent before crossing the provider boundary. A killed
            # process leaves an explicit running/unknown receipt, never a pass.
            receipt.update(status="running", attempted=True)
            on_receipt(dict(receipt))
            start = time.perf_counter()
            try:
                receipt.update(check())
            except KeyboardInterrupt:
                receipt.update(status="interrupted")
                on_receipt(receipt)
                raise
            except Exception as exc:
                # Messages, request URLs and response bodies can contain secrets.
                receipt.update(status="failed", error_type=type(exc).__name__)
                response = getattr(exc.__cause__, "response", None)
                if response is not None:
                    receipt["http_status"] = response.status_code
            receipt["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
            failed = receipt["status"] != "passed"
        receipts.append(receipt)
        on_receipt(receipt)
        print(json.dumps(receipt), flush=True)
    return receipts


def _save_report(path, report):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live-api", action="store_true", help="Permit up to four billable requests per provider")
    parser.add_argument("--provider", choices=(*PROVIDERS, "all"), default="all")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--mistral-model", default="mistral-small-latest")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--output", type=Path, help="New receipt file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if not args.allow_live_api:
        parser.error("Live requests require --allow-live-api")
    providers = tuple(PROVIDERS) if args.provider == "all" else (args.provider,)
    report = {
        "schema_version": 1, "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(), "synthetic_data_only": True,
        "max_requests": 4 * len(providers), "max_output_tokens_per_request": 96,
        "results": [{"provider": provider, "model": getattr(args, f"{provider}_model"),
                     "mode": mode, "status": "pending", "attempted": False}
                    for provider in providers for mode in MODES],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
        except OSError:
            parser.error("Cannot reserve output file; choose a new writable receipt path")

    def record(receipt):
        for index, previous in enumerate(report["results"]):
            if (previous["provider"], previous["mode"]) == (receipt["provider"], receipt["mode"]):
                report["results"][index] = receipt
                break
        if args.output:
            _save_report(args.output, report)

    try:
        for provider in providers:
            run_provider(provider, getattr(args, f"{provider}_model"), record)
    except KeyboardInterrupt:
        return 130
    return 0 if all(r["status"] == "passed" for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
