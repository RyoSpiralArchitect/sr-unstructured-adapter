#!/usr/bin/env python3
"""Opt-in, bounded live checks of the actual LLM drivers using synthetic data."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from sr_adapter.drivers.multicloud_driver import MistralDriver
from sr_adapter.drivers.openai_driver import OpenAIDriver
from sr_adapter.normalizer import LLMNormalizer


PROMPT = 'Return exactly this JSON object and nothing else: {"status":"ok","value":42}'
EXPECTED = {"status": "ok", "value": 42}


def check_result(provider, raw):
    result = LLMNormalizer().normalize(provider, raw, prompt=PROMPT)
    text = "".join(choice.text for choice in result.choices)
    return json.loads(text) == EXPECTED, result.usage, result.model


def check_stream(events):
    result = ""
    count = 0
    for event in events:
        count += 1
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str):
                result += content
            elif isinstance(content, list):
                result += "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return json.loads(result) == EXPECTED, {"events": count}, None


async def collect_async(driver):
    return [event async for event in driver.async_stream_generate(PROMPT)]


def run_provider(provider, model):
    key_name = "OPENAI_API_KEY" if provider == "openai" else "MISTRAL_API_KEY"
    if not os.environ.get(key_name):
        return [{"provider": provider, "status": "missing_credential", "key_env": key_name}]
    driver_type = OpenAIDriver if provider == "openai" else MistralDriver
    config = {"api_key": os.environ[key_name], "model": model, "max_retries": 0, "timeout": 30,
              "max_tokens": 96, "temperature": 0, "response_format": {"type": "json_object"}}
    driver = driver_type(provider, config)
    checks = (
        ("sync", lambda: check_result(provider, driver.generate(PROMPT, metadata={"purpose": "adapter-smoke"}))),
        ("async", lambda: check_result(provider, asyncio.run(driver.async_generate(PROMPT)))),
        ("stream", lambda: check_stream(driver.stream_generate(PROMPT))),
        ("async_stream", lambda: check_stream(asyncio.run(collect_async(driver)))),
    )
    receipts = []
    for mode, check in checks:
        start = time.perf_counter()
        receipt = {"provider": provider, "model": model, "mode": mode}
        try:
            passed, usage, actual_model = check()
            receipt.update(status="passed" if passed else "failed", usage=usage)
            if actual_model:
                receipt["response_model"] = actual_model
        except Exception as exc:
            # Provider exception messages may contain URLs, headers, or credentials.
            receipt.update(status="failed", error_type=type(exc).__name__)
            response = getattr(exc.__cause__, "response", None)
            if response is not None:
                receipt["http_status"] = response.status_code
                try:
                    error = response.json().get("error", {})
                    code = error.get("code") if isinstance(error, dict) else None
                    if isinstance(code, str) and code.replace("_", "").isalnum():
                        receipt["provider_error_code"] = code
                except (ValueError, AttributeError):
                    pass
        receipt["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
        if receipt["status"] == "failed":
            break  # Never spend on additional modes after the first failure.
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live-api", action="store_true", help="Permit up to four billable requests per provider")
    parser.add_argument("--provider", choices=("openai", "mistral", "all"), default="all")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--mistral-model", default="mistral-small-latest")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_live_api:
        parser.error("Live requests require --allow-live-api")
    providers = ("openai", "mistral") if args.provider == "all" else (args.provider,)
    receipts = []
    for provider in providers:
        receipts.extend(run_provider(provider, getattr(args, f"{provider}_model")))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "synthetic_data_only": True,
              "max_requests": 4 * len(providers), "results": receipts}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if receipts and all(r["status"] == "passed" for r in receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
