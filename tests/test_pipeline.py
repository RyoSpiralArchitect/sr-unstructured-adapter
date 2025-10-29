from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sr_adapter import convert, detect_type


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_convert_produces_document(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Meeting Notes\n\n- item one\n- item two", encoding="utf-8")
    out = tmp_path / "out.jsonl"

    document = convert(source, recipe="default", out=out)

    payloads = read_jsonl(out)
    assert len(payloads) == 1
    assert payloads[0]["meta"]["source"].endswith("note.txt")
    assert document.meta["type"] == "text"
    assert any(block["type"] in {"paragraph", "list"} for block in payloads[0]["blocks"])


def test_recipe_patterns_are_applied(tmp_path: Path) -> None:
    source = tmp_path / "call.log"
    source.write_text(
        "[2023-12-01 09:15] Call started\nCaller: Alice\n- Asked about pricing",
        encoding="utf-8",
    )
    out = tmp_path / "call.jsonl"

    convert(source, recipe="call_log", out=out)

    payload = read_jsonl(out)[0]
    types = [block["type"] for block in payload["blocks"]]
    assert "meta" in types
    assert "header" in types
    assert "list" in types


def test_detect_type_uses_extension(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2", encoding="utf-8")

    assert detect_type(source) == "csv"


def test_xlsx_is_parsed_as_table(tmp_path: Path) -> None:
    source = tmp_path / "table.xlsx"
    _write_minimal_xlsx(source)
    out = tmp_path / "table.jsonl"

    document = convert(source, recipe="default", out=out)

    assert document.meta["type"] == "xlsx"
    payload = read_jsonl(out)[0]
    assert payload["blocks"][0]["type"] == "table"
    assert payload["blocks"][0]["attrs"]["rows"] == "2"


def _write_minimal_xlsx(path: Path) -> None:
    content_types = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n  <Default Extension=\"xml\" ContentType=\"application/xml\"/>\n  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>\n  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>\n  <Override PartName=\"/xl/sharedStrings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>\n</Types>\n"""

    root_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>\n</Relationships>\n"""

    workbook_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>\n  <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>\n</Relationships>\n"""

    workbook = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">\n  <sheets>\n    <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/>\n  </sheets>\n</workbook>\n"""

    shared_strings = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"3\" uniqueCount=\"3\">\n  <si><t>Header 1</t></si>\n  <si><t>Header 2</t></si>\n  <si><t>Cell B2</t></si>\n</sst>\n"""

    worksheet = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\n  <sheetData>\n    <row r=\"1\">\n      <c r=\"A1\" t=\"s\"><v>0</v></c>\n      <c r=\"B1\" t=\"s\"><v>1</v></c>\n    </row>\n    <row r=\"2\">\n      <c r=\"A2\"><v>42</v></c>\n      <c r=\"B2\" t=\"s\"><v>2</v></c>\n    </row>\n  </sheetData>\n</worksheet>\n"""

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
