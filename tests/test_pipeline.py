from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import Workbook
from pypdf import PdfWriter

from sr_adapter.pipeline import convert
from sr_adapter.parsers import parse_txt
from sr_adapter.recipe import apply_recipe
from sr_adapter.schema import Block
from sr_adapter.sniff import detect_type


def _create_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def _create_docx(path: Path) -> None:
    doc = DocxDocument()
    doc.add_heading("Example Document", level=1)
    doc.add_paragraph("Body text")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Key"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "1"
    doc.save(str(path))


def _create_xlsx(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Key"
    ws["B1"] = "Value"
    ws["A2"] = "Temp"
    ws["B2"] = 42
    wb.save(str(path))


def test_detect_type_handles_various_formats(tmp_path: Path) -> None:
    txt = tmp_path / "sample.txt"
    txt.write_text("plain", encoding="utf-8")

    md = tmp_path / "sample.md"
    md.write_text("# Title\n- item", encoding="utf-8")

    pdf = tmp_path / "sample.pdf"
    _create_pdf(pdf)

    docx = tmp_path / "sample.docx"
    _create_docx(docx)

    xlsx = tmp_path / "sample.xlsx"
    _create_xlsx(xlsx)

    assert detect_type(txt) == "text"
    assert detect_type(md) == "md"
    assert detect_type(pdf) == "pdf"
    assert detect_type(docx) == "docx"
    assert detect_type(xlsx) == "xlsx"


def test_convert_with_recipe_applies_patterns(tmp_path: Path) -> None:
    log = tmp_path / "call.log"
    log.write_text(
        "[2024-01-01 09:30] Call started\nCaller: Alice\n- greeted\n",
        encoding="utf-8",
    )

    document = convert(log, recipe="call_log")
    assert document.meta["type"] == "text"
    assert len(document.blocks) >= 3
    assert document.blocks[0].type == "meta"
    assert document.blocks[1].type == "header"
    assert any(block.type == "list" for block in document.blocks)


def test_parse_txt_extracts_kv_and_log(tmp_path: Path) -> None:
    source = tmp_path / "structured.txt"
    source.write_text(
        "\n".join(
            [
                "Subject: Update",
                "2024-01-01 09:30 status ok",
                "- bullet item",
            ]
        ),
        encoding="utf-8",
    )

    blocks = parse_txt(source)

    types = {block.type for block in blocks}
    assert "kv" in types
    assert "log" in types
    log_block = next(block for block in blocks if block.type == "log")
    assert log_block.attrs["timestamp"].startswith("2024-01-01")
    assert "status ok" in log_block.attrs["message"]


def test_recipe_fallback_preserves_attrs(tmp_path: Path) -> None:
    block = Block(type="paragraph", text="unknown content", attrs={"a": "1"})
    processed = apply_recipe([block], "default")
    assert processed[0].type == "paragraph"
    assert processed[0].attrs["a"] == "1"


def test_cli_convert_produces_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "sensor.log"
    source.write_text(
        "Sensor: A1\n2024-01-01T00:00:00\nTemp: 21.5\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

    result = subprocess.run(
        [sys.executable, "-m", "sr_adapter.cli", "convert", str(source), "--recipe", "sensor_log", "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["meta"]["type"] == "text"
    assert any(block["type"] == "kv" for block in payload["blocks"])


def test_convert_eml_extracts_headers(tmp_path: Path) -> None:
    eml = tmp_path / "mail.eml"
    eml.write_text(
        "\n".join(
            [
                "From: Example <sender@example.com>",
                "To: Recipient <user@example.com>",
                "Subject: Hello",
                "MIME-Version: 1.0",
                "Content-Type: text/plain; charset=utf-8",
                "",
                "Hello world",
            ]
        ),
        encoding="utf-8",
    )

    document = convert(eml, recipe="default")

    assert document.meta["type"] == "eml"
    assert document.blocks[0].type == "meta"
    assert document.blocks[0].attrs["subject"] == "Hello"
    assert any(block.type in {"paragraph", "kv"} for block in document.blocks[1:])


def test_convert_ics_yields_event_blocks(tmp_path: Path) -> None:
    ics = tmp_path / "event.ics"
    ics.write_text(
        "\n".join(
            [
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "BEGIN:VEVENT",
                "SUMMARY:Sync",
                "DTSTART:20240201T120000Z",
                "DTEND:20240201T123000Z",
                "LOCATION:Video",
                "DESCRIPTION:Weekly sync",
                "END:VEVENT",
                "END:VCALENDAR",
            ]
        ),
        encoding="utf-8",
    )

    document = convert(ics, recipe="default")

    assert document.meta["type"] == "ics"
    assert any(block.type == "event" for block in document.blocks)
    event = next(block for block in document.blocks if block.type == "event")
    assert event.attrs["summary"] == "Sync"


def test_parse_txt_refines_large_paragraphs(tmp_path: Path) -> None:
    body = " ".join(["Sentence number %d." % i for i in range(1, 80)])
    path = tmp_path / "long.txt"
    path.write_text(body, encoding="utf-8")

    blocks = parse_txt(path)

    assert len(blocks) > 1
    assert all(len(block.text) <= 600 for block in blocks)

