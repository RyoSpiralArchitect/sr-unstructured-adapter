# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip(
    "pytest_benchmark",
    reason="perf tests require pytest-benchmark; install extras[perf] to enable",
)

from pytest_benchmark.fixture import BenchmarkFixture

from sr_adapter.pipeline import convert

pytestmark = pytest.mark.perf


def _write_json_records(path: Path, *, records: int = 250) -> None:
    payload = [{"id": idx, "message": f"record-{idx}"} for idx in range(records)]
    path.write_text("\n".join(json.dumps(row) for row in payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SR_ADAPTER_NO_LLM", "1")
    monkeypatch.setenv(
        "SR_ADAPTER_DISABLE_NATIVE_RUNTIME",
        os.environ.get("SR_ADAPTER_DISABLE_NATIVE_RUNTIME", "0"),
    )


def test_convert_txt_small(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Example paragraph for throughput measurement." * 4, encoding="utf-8")

    def _run() -> None:
        convert(source, recipe="default", llm_ok=False)

    benchmark(_run)


def test_convert_json_medium(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    _write_json_records(source, records=300)

    def _run() -> None:
        convert(source, recipe="default", llm_ok=False)

    benchmark(_run)
