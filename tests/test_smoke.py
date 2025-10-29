from __future__ import annotations

import base64
import json
from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import Workbook

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


def test_payload_counts_ignore_trailing_newlines(tmp_path: Path) -> None:
    path = tmp_path / "multiline.txt"
    path.write_text("first line\nsecond line\n", encoding="utf-8")

    payload_dict = to_unified_payload(path)
    meta = payload_dict["meta"]

    assert meta["line_count"] == 2
    assert meta["word_count"] == 4
    assert meta["char_count"] == len("first line\nsecond line\n")


def test_json_payload_includes_schema(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": 1, "b": {"c": 2}}), encoding="utf-8")

    payload = build_payload(path)

    assert payload.mime == "application/json"
    assert payload.meta["json_valid"] is True
    assert payload.meta["json_top_level_type"] == "dict"
    assert payload.meta["json_top_level_keys"] == ["a", "b"]
    assert "word_count" not in payload.meta


def test_to_llm_messages_chunking(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("a" * 5000, encoding="utf-8")

    payload = build_payload(path)
    messages = to_llm_messages(payload, chunk_size=2000)

    assert len(messages) == 3
    assert "part 1" in messages[0]["content"]
    assert "part 3" in messages[-1]["content"]


def test_html_payload_extracts_text(tmp_path: Path) -> None:
    html = """
    <html>
      <head><title>Sample Page</title></head>
      <body>
        <h1>Heading One</h1>
        <p>Hello <strong>world</strong>!</p>
      </body>
    </html>
    """
    path = tmp_path / "page.html"
    path.write_text(html, encoding="utf-8")

    payload = build_payload(path)

    assert "Hello" in payload.text and "world" in payload.text
    assert payload.meta["html_title"] == "Sample Page"
    assert "Heading One" in payload.meta["html_headings"]
    assert payload.meta["word_count"] >= 3


def test_docx_payload_includes_tables(tmp_path: Path) -> None:
    path = tmp_path / "report.docx"
    document = DocxDocument()
    document.add_paragraph("Docx hello world")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Cell 1"
    table.cell(0, 1).text = "Cell 2"
    document.save(path)

    payload = build_payload(path)

    assert "Docx hello world" in payload.text
    assert payload.meta["docx_paragraphs"] >= 1
    assert payload.meta["docx_tables"] == 1
    assert payload.meta["word_count"] >= 3


def test_xlsx_payload_renders_cells(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "SheetA"
    sheet.append(["Name", "Value"])
    sheet.append(["Alpha", 1])
    workbook.save(path)

    payload = build_payload(path)

    assert "# SheetA" in payload.text
    assert "Alpha" in payload.text
    assert payload.meta["workbook_sheet_count"] == 1
    assert payload.meta["workbook_cell_values"] == 4
    assert payload.meta["word_count"] >= 2


def test_pdf_payload_reports_pages(tmp_path: Path) -> None:
    pdf_bytes = base64.b64decode(
        "JVBERi0xLjMKJcTl8uXrp/Og0MTGCjQgMCBvYmoKPDwgL0xlbmd0aCAxMTIgPj4Kc3RyZWFtCkJUIC9GMSAyNCBUZiA1MCAxNTAgVGQKKEhlbGxvIFBERikgVGoKRVQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8IC9UeXBlIC9Gb250IC9TdWJ0eXBlIC9UeXBlMSAvTmFtZSAvRjEgL0Jhc2VGb250IC9IZWx2ZXRpY2EgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWyAwIDAgMjAwIDIwMCBdIC9Db250ZW50cyA0IDAgUiAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA1IDAgUiA+PiA+PiA+PgplbmRvYmoKMiAwIG9iago8PCAvVHlwZSAvUGFnZXMgL0tpZHMgWyAzIDAgUiBdIC9Db3VudCAxID4+CmVuZG9iagoxIDAgb2JqCjw8IC9UeXBlIC9DYXRhbG9nIC9QYWdlcyAyIDAgUiA+PgplbmRvYmoKeHJlZg0KMCA2DQowMDAwMDAwMDAwIDY1NTM1IGYgDQowMDAwMDAwMDEwIDAwMDAwIG4gDQowMDAwMDAwMDYwIDAwMDAwIG4gDQowMDAwMDAwMTE3IDAwMDAwIG4gDQowMDAwMDAwMjIwIDAwMDAwIG4gDQowMDAwMDAwMzAzIDAwMDAwIG4gDQp0cmFpbGVyDQo8PCAvU2l6ZSA2IC9Sb290IDEgMCBSID4+DQplbmR0cmFpbGVyDQpzdGFydHhyZWYNCjM0OQ0KJSVFT0YN"
    )
    path = tmp_path / "hello.pdf"
    path.write_bytes(pdf_bytes)

    payload = build_payload(path)

    assert "Hello PDF" in payload.text
    assert payload.meta["pdf_page_count"] == 1
    assert payload.meta["pdf_has_text"] is True
    assert payload.meta["word_count"] >= 2
