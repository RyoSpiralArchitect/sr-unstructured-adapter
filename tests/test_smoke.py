from __future__ import annotations

import json
from pathlib import Path

from sr_adapter import build_payload, to_llm_messages, to_unified_payload


def test_payload_basic(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")

    payload_dict = to_unified_payload(path)

    assert payload_dict["source"].endswith("a.txt")
    assert payload_dict["mime"].startswith("text/")
    assert payload_dict["text"] == "hello"
    assert payload_dict["meta"]["size"] == 5
    assert payload_dict["meta"]["line_count"] == 1
    assert payload_dict["meta"]["word_count"] == 1


def test_json_payload_includes_schema(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": 1, "b": {"c": 2}}), encoding="utf-8")

    payload = build_payload(path)

    assert payload.mime == "application/json"
    assert payload.meta["json_valid"] is True
    assert payload.meta["json_top_level_type"] == "dict"
    assert payload.meta["json_top_level_keys"] == ["a", "b"]


def test_to_llm_messages_chunking(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("a" * 5000, encoding="utf-8")

    payload = build_payload(path)
    messages = to_llm_messages(payload, chunk_size=2000)

    assert len(messages) == 3
    assert "part 1" in messages[0]["content"]
    assert "part 3" in messages[-1]["content"]
