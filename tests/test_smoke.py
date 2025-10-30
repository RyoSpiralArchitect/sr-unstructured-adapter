from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from sr_adapter import build_payload, to_llm_messages, to_unified_payload


def test_rtf_payload_extraction(tmp_path: Path) -> None:
    path = tmp_path / "note.rtf"
    path.write_text("{\\rtf1\\ansi This is \\b bold\\b0 text}", encoding="utf-8")

    payload = build_payload(path)

    assert "bold" in payload.text
    assert payload.mime == "application/rtf"
    assert payload.meta.get("rtf_extracted") in {True, False}


def test_eml_payload_handles_basic_headers(tmp_path: Path) -> None:
    source = tmp_path / "mail.eml"
    source.write_text(
        dedent(
            """
            From: Alice <alice@example.com>
            To: Bob <bob@example.com>
            Subject: Greetings
            Date: Tue, 1 Jan 2024 10:00:00 +0000
            Content-Type: multipart/alternative; boundary=123

            --123
            Content-Type: text/plain; charset=utf-8

            Hello Bob!
            --123--
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    payload = build_payload(source)

    assert "Hello Bob" in payload.text
    assert payload.meta["email_subject"] == "Greetings"
    assert payload.meta["email_has_body"] is True


def test_payload_basic(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")

    payload_dict = to_unified_payload(path)

    assert payload_dict["source"].endswith("a.txt")
    assert payload_dict["mime"].startswith("text/")
    assert payload_dict["text"] == "hello"
    assert payload_dict["meta"]["size"] == 5
    assert payload_dict["meta"]["line_count"] == 1


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
    assert "part 1/3" in messages[0]["content"]
    assert "part 3/3" in messages[-1]["content"]
