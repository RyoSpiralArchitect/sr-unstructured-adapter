"""Collection of lightweight parsers that emit :class:`Block` objects."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pypdf import PdfReader

from .schema import Block, clone_model


_LIST_MARKERS = re.compile(r"^(?:[-*\u2022\u30fb]|\d+[.)])\s+")
_TIMESTAMP_PREFIX = re.compile(r"^\[?\d{4}[-/]\d{2}[-/]\d{2}")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?=[\w\"'(])")
_MAX_CHARS_PER_CHUNK = 600


def _new_block(block_type: str, text: str, source: Path, confidence: float = 0.5) -> Block:
    return Block(type=block_type, text=text, source=str(source), confidence=confidence)


def _split_paragraphs(text: str) -> Iterable[str]:
    buf: List[str] = []
    for line in text.splitlines():
        if line.strip():
            buf.append(line)
        elif buf:
            yield "\n".join(buf)
            buf.clear()
    if buf:
        yield "\n".join(buf)


def _classify_chunk(chunk: str) -> str:
    stripped = chunk.strip()
    if not stripped:
        return "other"
    if stripped.startswith("#") and stripped.count("#") <= 6:
        return "heading"
    if stripped.upper() == stripped and len(stripped.split()) <= 6:
        return "heading"
    if _TIMESTAMP_PREFIX.match(stripped):
        return "metadata"
    lines = stripped.splitlines()
    if len(lines) == 1 and len(stripped) <= 80:
        if any(stripped.startswith(prefix) for prefix in ("title:", "subject:")):
            return "metadata"
    if all(_LIST_MARKERS.match(line) for line in lines):
        return "list_item"
    if all(":" in line for line in lines) and len(lines) <= 6:
        return "metadata"
    return "paragraph"


def _explode_structured_chunk(chunk: str) -> List[str]:
    if "\n" not in chunk:
        return [chunk]
    lines = [line for line in chunk.splitlines() if line.strip()]
    # Treat short structured segments (logs, bullet lists) line-by-line.
    if lines and all(len(line) < 200 for line in lines) and len(lines) <= 16:
        return lines
    return [chunk]


def _shatter_chunk(chunk: str) -> List[str]:
    chunk = chunk.strip()
    if len(chunk) <= _MAX_CHARS_PER_CHUNK:
        return [chunk]
    sentences = [seg.strip() for seg in _SENTENCE_BOUNDARY.split(chunk) if seg.strip()]
    if len(sentences) > 1:
        grouped: List[str] = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue
            tentative = f"{current} {sentence}" if current else sentence
            if len(tentative) <= _MAX_CHARS_PER_CHUNK:
                current = tentative
            else:
                grouped.append(current)
                current = sentence
        if current:
            grouped.append(current)
        return grouped
    return [chunk[i : i + _MAX_CHARS_PER_CHUNK] for i in range(0, len(chunk), _MAX_CHARS_PER_CHUNK)]


def _iter_refined_chunks(text: str) -> Iterable[str]:
    for chunk in _split_paragraphs(text):
        for piece in _explode_structured_chunk(chunk):
            for refined in _shatter_chunk(piece):
                cleaned = refined.strip()
                if cleaned:
                    yield cleaned


def parse_txt(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    blocks = [
        _new_block(_classify_chunk(chunk), chunk, source) for chunk in _iter_refined_chunks(text)
    ]
    return blocks or [_new_block("other", text, source, confidence=0.3)]


def parse_md(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    blocks: List[Block] = []
    in_code = False
    code_buffer: List[str] = []
    for line in text.splitlines():
        stripped = line.strip("\n")
        if stripped.startswith("```"):
            if in_code:
                blocks.append(
                    _new_block("code", "\n".join(code_buffer), source, confidence=0.7)
                )
                code_buffer.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if stripped.startswith("#"):
            blocks.append(_new_block("heading", stripped.lstrip("# "), source, confidence=0.8))
            continue
        if _LIST_MARKERS.match(stripped):
            blocks.append(_new_block("list_item", stripped, source, confidence=0.7))
            continue
        if stripped:
            for chunk in _shatter_chunk(stripped):
                blocks.append(_new_block("paragraph", chunk, source))
    if code_buffer:
        blocks.append(_new_block("code", "\n".join(code_buffer), source, confidence=0.7))
    return blocks or [_new_block("other", text, source, confidence=0.3)]


def parse_html(path: str | Path) -> List[Block]:
    source = Path(path)
    html = source.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    blocks: List[Block] = []
    for element in soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        name = element.name or "p"
        if name == "title":
            blocks.append(_new_block("title", text, source, confidence=0.9))
        elif name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            blocks.append(_new_block("heading", text, source, confidence=0.85))
        elif name == "li":
            blocks.append(_new_block("list_item", text, source, confidence=0.7))
        elif name in {"pre", "code"}:
            blocks.append(_new_block("code", text, source, confidence=0.75))
        elif name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            blocks.append(
                Block(
                    type="table",
                    text="\n".join([", ".join(row) for row in rows]),
                    attrs={"rows": json.dumps(rows, ensure_ascii=False)},
                    source=str(source),
                    confidence=0.8,
                )
            )
        else:
            for chunk in _shatter_chunk(text):
                blocks.append(_new_block("paragraph", chunk, source, confidence=0.6))
    return blocks or [_new_block("paragraph", soup.get_text("\n", strip=True), source, confidence=0.4)]


def parse_csv(path: str | Path) -> List[Block]:
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader]
    text = "\n".join([", ".join(row) for row in rows])
    return [
        Block(
            type="table",
            text=text,
            attrs={"rows": json.dumps(rows, ensure_ascii=False)},
            source=str(source),
            confidence=0.8,
        )
    ]


def parse_pdf(path: str | Path) -> List[Block]:
    source = Path(path)
    reader = PdfReader(str(source))
    blocks: List[Block] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in _iter_refined_chunks(text):
            base = _new_block(
                _classify_chunk(chunk),
                chunk,
                source,
                confidence=0.55,
            )
            blocks.append(clone_model(base, attrs={"page": str(index + 1)}))
    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_docx(path: str | Path) -> List[Block]:
    source = Path(path)
    doc = DocxDocument(str(source))
    blocks: List[Block] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        block_type = "paragraph"
        if paragraph.style and paragraph.style.name:
            style = paragraph.style.name.lower()
            if "heading" in style:
                block_type = "heading"
        for chunk in _shatter_chunk(text):
            blocks.append(_new_block(block_type, chunk, source, confidence=0.65))
    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        blocks.append(
            Block(
                type="table",
                text="\n".join([", ".join(row) for row in rows]),
                attrs={"rows": json.dumps(rows, ensure_ascii=False)},
                source=str(source),
                confidence=0.75,
            )
        )
    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_xlsx(path: str | Path) -> List[Block]:
    source = Path(path)
    workbook = load_workbook(filename=str(source), read_only=True, data_only=True)
    sheet = workbook.active
    rows = [
        ["" if cell.value is None else str(cell.value) for cell in row]
        for row in sheet.iter_rows(values_only=True)
    ]
    text = "\n".join([", ".join(row) for row in rows])
    workbook.close()
    return [
        Block(
            type="table",
            text=text,
            attrs={"rows": json.dumps(rows, ensure_ascii=False)},
            source=str(source),
            confidence=0.75,
        )
    ]


def parse_json(path: str | Path) -> List[Block]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8", errors="ignore"))
    return [
        Block(
            type="code",
            text=json.dumps(data, ensure_ascii=False, indent=2),
            source=str(source),
            confidence=0.6,
        )
    ]

