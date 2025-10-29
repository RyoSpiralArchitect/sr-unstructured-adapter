"""Collection of lightweight parsers that emit :class:`Block` objects."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

from email import policy
from email.parser import BytesParser
from email.utils import getaddresses

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pypdf import PdfReader

from .logs import LogEntry, iter_log_entries
from .schema import Block, clone_model


try:
    import extract_msg
except Exception:  # pragma: no cover - optional dependency guard
    extract_msg = None  # type: ignore[assignment]


_LIST_MARKERS = re.compile(r"^(?:[-*\u2022\u30fb]|\d+[.)])\s+")


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
        return "header"
    if stripped.upper() == stripped and len(stripped.split()) <= 6:
        return "header"
    lines = stripped.splitlines()
    if len(lines) == 1 and len(stripped) <= 80:
        if any(stripped.startswith(prefix) for prefix in ("title:", "subject:")):
            return "meta"
    if all(_LIST_MARKERS.match(line) for line in lines):
        return "list"
    if all(":" in line for line in lines) and len(lines) <= 6:
        return "kv"
    return "paragraph"


def _explode_structured_chunk(chunk: str) -> List[str]:
    if "\n" not in chunk:
        return [chunk]
    lines = [line for line in chunk.splitlines() if line.strip()]
    # Treat short structured segments (logs, bullet lists) line-by-line.
    if lines and all(len(line) < 200 for line in lines) and len(lines) <= 16:
        return lines
    return [chunk]


def parse_txt(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    chunks: List[str] = []
    for chunk in _split_paragraphs(text):
        chunks.extend(_explode_structured_chunk(chunk))
    blocks = [
        _new_block(_classify_chunk(chunk), chunk.strip(), source) for chunk in chunks if chunk.strip()
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
            blocks.append(_new_block("header", stripped.lstrip("# "), source, confidence=0.8))
            continue
        if _LIST_MARKERS.match(stripped):
            blocks.append(_new_block("list", stripped, source, confidence=0.7))
            continue
        if stripped:
            blocks.append(_new_block("paragraph", stripped, source))
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
            blocks.append(_new_block("header", text, source, confidence=0.85))
        elif name == "li":
            blocks.append(_new_block("list", text, source, confidence=0.7))
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
            blocks.append(_new_block("paragraph", text, source, confidence=0.6))
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


def parse_log(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    entries = list(iter_log_entries(lines))

    consumed = [False] * len(lines)
    by_start: Dict[int, LogEntry] = {}
    by_end: Dict[int, int] = {}
    for entry in entries:
        if entry.start_line is None:
            continue
        start = entry.start_line
        end = entry.end_line if entry.end_line is not None else start + 1
        by_start[start] = entry
        by_end[start] = end
        for idx in range(start, end):
            if 0 <= idx < len(consumed):
                consumed[idx] = True

    blocks: List[Block] = []
    index = 0

    def _emit_paragraph(start: int, end: int) -> None:
        for line in lines[start:end]:
            stripped = line.strip()
            if not stripped:
                continue
            blocks.append(_new_block("paragraph", stripped, source, confidence=0.45))

    while index < len(lines):
        if index in by_start:
            entry = by_start[index]
            end = by_end.get(index, index + 1)
            attrs: Dict[str, object] = {}
            if entry.timestamp:
                attrs["timestamp"] = entry.timestamp
                if entry.raw_timestamp and entry.raw_timestamp != entry.timestamp:
                    attrs["raw_timestamp"] = entry.raw_timestamp
            elif entry.raw_timestamp:
                attrs["raw_timestamp"] = entry.raw_timestamp
            if entry.level:
                attrs["level"] = entry.level
            attrs["line_start"] = str(index + 1)
            attrs["line_end"] = str(end)

            message = entry.message or entry.raw.strip()
            confidence = 0.55
            if entry.level:
                confidence += 0.15
            if entry.timestamp:
                confidence += 0.15
            if "\n" in message:
                confidence -= 0.05
            confidence = max(0.4, min(confidence, 0.9))

            blocks.append(
                Block(
                    type="log",
                    text=message,
                    attrs=attrs,
                    source=str(source),
                    confidence=confidence,
                )
            )
            index = end
            continue

        start = index
        while index < len(lines) and not consumed[index]:
            index += 1
        _emit_paragraph(start, index)

    if not blocks:
        stripped = text.strip()
        if stripped:
            return [_new_block("paragraph", stripped, source, confidence=0.3)]
        return [_new_block("other", "", source, confidence=0.1)]

    return blocks


def parse_pdf(path: str | Path) -> List[Block]:
    source = Path(path)
    reader = PdfReader(str(source))
    blocks: List[Block] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in _split_paragraphs(text):
            base = _new_block(
                "paragraph",
                chunk.strip(),
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
                block_type = "header"
        blocks.append(_new_block(block_type, text, source, confidence=0.65))
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
    rows: List[List[str]] = []
    for row in sheet.iter_rows(values_only=True):
        values = ["" if value is None else str(value) for value in row]
        rows.append(values)
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


def parse_email(path: str | Path) -> List[Block]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".msg" and extract_msg is not None:
        message = extract_msg.Message(str(source))
        try:
            blocks: List[Block] = []
            if message.subject:
                blocks.append(
                    _new_block("title", message.subject.strip(), source, confidence=0.9)
                )

            address_lines = []
            if message.sender:
                address_lines.append(f"From: {message.sender}")
            if message.to:
                tos = [addr.strip() for addr in message.to.split(";") if addr.strip()]
                if tos:
                    address_lines.append(f"To: {', '.join(tos)}")
            if message.cc:
                ccs = [addr.strip() for addr in message.cc.split(";") if addr.strip()]
                if ccs:
                    address_lines.append(f"Cc: {', '.join(ccs)}")
            if address_lines:
                blocks.append(
                    _new_block("kv", "\n".join(address_lines), source, confidence=0.7)
                )

            if message.body:
                for chunk in _split_paragraphs(message.body):
                    blocks.append(
                        _new_block("paragraph", chunk.strip(), source, confidence=0.55)
                    )
            for attachment in message.attachments:
                name = attachment.longFilename or attachment.shortFilename or "attachment"
                blocks.append(_new_block("attachment", name, source, confidence=0.4))
            return blocks or [_new_block("other", "", source, confidence=0.1)]
        finally:
            message.close()

    with source.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    blocks = []

    subject = (message.get("subject") or "").strip()
    if subject:
        blocks.append(_new_block("title", subject, source, confidence=0.9))

    address_lines = []
    for header in ("from", "to", "cc"):
        values = [addr for _, addr in getaddresses([message.get(header, "")]) if addr]
        if values:
            address_lines.append(f"{header.title()}: {', '.join(values)}")
    if address_lines:
        blocks.append(_new_block("kv", "\n".join(address_lines), source, confidence=0.7))

    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disposition = part.get_content_disposition()
        if disposition == "attachment":
            filename = part.get_filename() or "attachment"
            blocks.append(_new_block("attachment", filename, source, confidence=0.4))
            continue
        if part.get_content_type().startswith("text/"):
            content = part.get_content().strip()
            for chunk in _split_paragraphs(content):
                blocks.append(_new_block("paragraph", chunk.strip(), source, confidence=0.55))

    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_zip(path: str | Path) -> List[Block]:
    source = Path(path)
    blocks: List[Block] = []

    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            description = f"{member.filename} ({member.file_size} bytes)"
            blocks.append(_new_block("attachment", description, source, confidence=0.45))
            suffix = Path(member.filename).suffix.lower()
            if member.file_size <= 64 * 1024 and suffix in {".txt", ".log", ".csv", ".tsv"}:
                with archive.open(member) as handle:
                    content = handle.read().decode("utf-8", errors="ignore")
                for chunk in _split_paragraphs(content):
                    blocks.append(_new_block("paragraph", chunk.strip(), source, confidence=0.5))

    return blocks or [_new_block("other", "", source, confidence=0.2)]


def parse_image(path: str | Path) -> List[Block]:
    source = Path(path)
    return [_new_block("image", source.name, source, confidence=0.3)]

