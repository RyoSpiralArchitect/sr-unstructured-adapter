from __future__ import annotations

import io
import json
from pathlib import Path

from sr_adapter.schema import Document
from sr_adapter.writer import write_jsonl


def test_write_jsonl_accepts_mappings(tmp_path: Path) -> None:
    destination = tmp_path / "out.jsonl"
    payloads = [
        {"source": "file.txt", "ok": True},
        {"source": "file2.txt", "ok": False, "error": "boom"},
    ]

    write_jsonl(payloads, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == payloads


def test_write_jsonl_supports_file_like_handles() -> None:
    buffer = io.StringIO()
    document = Document(blocks=[], meta={})

    write_jsonl([document], buffer)

    buffer.seek(0)
    content = buffer.read().strip()
    assert json.loads(content)["blocks"] == []
