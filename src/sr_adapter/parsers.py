"""File-type specific parsers that produce :class:`~sr_adapter.schema.Block` objects."""

from __future__ import annotations

import csv
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List

from .schema import Block

_XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_txt(path: str | Path) -> List[Block]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _blocks_from_plaintext(text, source=str(path))


def parse_md(path: str | Path) -> List[Block]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: List[Block] = []
    buffer: List[str] = []
    block_type = "paragraph"
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if buffer:
                blocks.append(Block(type=block_type, text="\n".join(buffer), source=str(path)))
                buffer = []
                block_type = "paragraph"
            continue
        if stripped.startswith("#"):
            if buffer:
                blocks.append(Block(type=block_type, text="\n".join(buffer), source=str(path)))
                buffer = []
            level = len(stripped) - len(stripped.lstrip("#"))
            block_type = "title" if level == 1 and not blocks else "header"
            blocks.append(
                Block(
                    type=block_type,
                    text=stripped[level:].strip(),
                    source=str(path),
                    confidence=0.9,
                )
            )
            block_type = "paragraph"
            continue
        if stripped.startswith(('- ', '* ')):
            if buffer:
                blocks.append(Block(type=block_type, text="\n".join(buffer), source=str(path)))
                buffer = []
            blocks.append(Block(type="list", text=stripped[2:].strip(), source=str(path), confidence=0.7))
            continue
        buffer.append(stripped)
    if buffer:
        blocks.append(Block(type=block_type, text="\n".join(buffer), source=str(path)))
    if not blocks:
        blocks.append(Block(type="paragraph", text="", source=str(path)))
    return blocks


def parse_html(path: str | Path) -> List[Block]:
    parser = _HTMLBlockExtractor(str(path))
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    parser.feed(text)
    parser.close()
    return parser.blocks


def parse_csv(path: str | Path) -> List[Block]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(content))
    rows = [row for row in reader]
    text_rows = [", ".join(cell.strip() for cell in row) for row in rows]
    block_text = "\n".join(text_rows)
    return [
        Block(
            type="table",
            text=block_text,
            attrs={"rows": str(len(rows)), "columns": str(len(rows[0]) if rows else 0)},
            source=str(path),
            confidence=0.8,
        )
    ]


def parse_pdf(path: str | Path) -> List[Block]:
    try:
        from pdfminer.high_level import extract_text
    except ImportError:  # pragma: no cover - optional dependency guard
        return parse_txt(path)

    text = extract_text(str(path)) or ""
    return _blocks_from_plaintext(text, source=str(path), base_confidence=0.6)


def parse_docx(path: str | Path) -> List[Block]:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        try:
            xml = archive.read("word/document.xml")
        except KeyError:
            return parse_txt(path)
    root = ET.fromstring(xml)
    texts: List[str] = []
    for elem in root.iter():
        if elem.tag.endswith("}t") and elem.text:
            texts.append(elem.text)
        if elem.tag.endswith("}tab"):
            texts.append("\t")
        if elem.tag.endswith("}br"):
            texts.append("\n")
    joined = "".join(texts)
    return _blocks_from_plaintext(joined, source=str(path), base_confidence=0.65)


def parse_json(path: str | Path) -> List[Block]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [Block(type="code", text=content.strip(), source=str(path), confidence=0.7)]


def parse_xlsx(path: str | Path) -> List[Block]:
    try:
        with zipfile.ZipFile(path) as archive:
            sheet_name = _pick_worksheet(archive)
            if sheet_name is None:
                return parse_txt(path)
            shared_strings = _read_shared_strings(archive)
            sheet_xml = archive.read(sheet_name)
    except zipfile.BadZipFile:
        return parse_txt(path)

    root = ET.fromstring(sheet_xml)
    rows: List[List[str]] = []
    max_cols = 0
    for row_elem in root.findall(".//a:row", _XLSX_NS):
        cells: List[str] = []
        expected_col = 0
        for cell_elem in row_elem.findall("a:c", _XLSX_NS):
            ref = cell_elem.get("r")
            column_index = _column_index(ref) if ref else expected_col
            while len(cells) < column_index:
                cells.append("")
            value_elem = cell_elem.find("a:v", _XLSX_NS)
            value = _resolve_cell_value(cell_elem, value_elem.text if value_elem is not None else "", shared_strings)
            cells.append(value)
            expected_col = column_index + 1
        max_cols = max(max_cols, len(cells))
        rows.append(cells)

    if not rows:
        return parse_txt(path)

    for row in rows:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))

    text_rows = [", ".join(cell.strip() for cell in row) for row in rows]
    return [
        Block(
            type="table",
            text="\n".join(text_rows),
            attrs={"rows": str(len(rows)), "columns": str(max_cols)},
            source=str(path),
            confidence=0.75,
        )
    ]


def parse_unknown(path: str | Path) -> List[Block]:
    return parse_txt(path)


def _blocks_from_plaintext(
    text: str,
    *,
    source: str,
    base_confidence: float = 0.5,
) -> List[Block]:
    text = text.replace("\r\n", "\n")
    segments = re.split(r"\n{2,}", text.strip() or "")
    blocks: List[Block] = []
    first_block = True
    for segment in segments:
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            cleaned = lines[0]
            block_type = "paragraph"
            confidence = base_confidence
            if first_block and len(cleaned.split()) <= 10:
                block_type = "title"
                confidence = min(1.0, base_confidence + 0.2)
            blocks.append(
                Block(type=block_type, text=cleaned, source=source, confidence=confidence)
            )
            first_block = False
            continue

        for line in lines:
            cleaned = line
            confidence = base_confidence
            block_type = "paragraph"
            if cleaned.startswith("- ") or cleaned.startswith("* "):
                block_type = "list"
                confidence = min(1.0, base_confidence + 0.1)
            blocks.append(Block(type=block_type, text=cleaned, source=source, confidence=confidence))
        first_block = False
    if not blocks:
        blocks.append(Block(type="paragraph", text=text.strip(), source=source, confidence=base_confidence))
    return blocks


class _HTMLBlockExtractor(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source
        self.blocks: List[Block] = []
        self._buffer: List[str] = []
        self._current_type = "paragraph"

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"h1", "h2", "h3"}:
            self._flush()
            self._current_type = "title" if tag_lower == "h1" and not self.blocks else "header"
        elif tag_lower in {"li"}:
            self._flush()
            self._current_type = "list"
        elif tag_lower in {"code", "pre"}:
            self._flush()
            self._current_type = "code"

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"p", "h1", "h2", "h3", "li", "code", "pre"}:
            self._flush()
            self._current_type = "paragraph"

    def handle_data(self, data: str) -> None:
        self._buffer.append(data)

    def handle_startendtag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self._buffer.append("\n")

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        if not text:
            return
        confidence = 0.6 if self._current_type == "paragraph" else 0.75
        self.blocks.append(
            Block(
                type=self._current_type,
                text=text,
                source=self.source,
                confidence=confidence,
            )
        )


def _pick_worksheet(archive: zipfile.ZipFile) -> str | None:
    for name in archive.namelist():
        if name.startswith("xl/worksheets/") and name.endswith(".xml"):
            return name
    return None


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    strings: List[str] = []
    for item in root.findall("a:si", _XLSX_NS):
        parts = [node.text or "" for node in item.findall(".//a:t", _XLSX_NS)]
        strings.append("".join(parts))
    return strings


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(0, index - 1)


def _resolve_cell_value(cell_elem: ET.Element, raw: str, shared_strings: List[str]) -> str:
    if cell_elem.get("t") == "s":
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return raw
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return raw
    return raw
