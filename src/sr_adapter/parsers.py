"""Collection of lightweight parsers that emit :class:`Block` objects."""

from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List
from xml.dom import minidom
from xml.etree import ElementTree as ET

from email import policy
from email.parser import BytesParser
from email.utils import getaddresses

from .loaders import extract_pptx_slides
from .logs import LogEntry, iter_log_entries
from .schema import Block, Span


try:
    import extract_msg
except Exception:  # pragma: no cover - optional dependency guard
    extract_msg = None  # type: ignore[assignment]


_LIST_MARKERS = re.compile(r"^(?:[-*\u2022\u30fb]|\d+[.)])\s+")
_FENCE = re.compile(r"^```(\w+)?\s*$")
_URL = re.compile(r"https?://[^\s)]+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_DATE = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")


def _annotate_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    for regex, label in ((_URL, "url"), (_EMAIL, "email"), (_DATE, "date")):
        for match in regex.finditer(text):
            spans.append(Span(start=match.start(), end=match.end(), label=label))
    return spans


def _new_block(
    block_type: str,
    text: str,
    source: Path,
    confidence: float = 0.5,
    **attrs: Any,
) -> Block:
    block = Block(type=block_type, text=text, source=str(source), confidence=confidence)
    if attrs:
        block.attrs = dict(attrs)
    if text:
        block.spans = _annotate_spans(text)
    return block


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


def _is_md_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = stripped.strip("|").split("|")
    else:
        cells = stripped.split("|")
    return len(cells) >= 2


def _is_md_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if set(stripped) - {"|", "-", ":", " "}:
        return False
    parts = [part.strip() for part in stripped.strip("|").split("|") if part.strip()]
    if not parts:
        return False
    return all(set(part) <= {"-", ":"} for part in parts)


def _prettify_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    try:
        parsed = minidom.parseString(rough)
        return parsed.toprettyxml(indent="  ")
    except Exception:
        return rough.decode("utf-8", errors="ignore")


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
    code_lang: str | None = None
    code_buffer: List[str] = []
    buf_list: List[str] = []

    def flush_list() -> None:
        nonlocal buf_list
        if buf_list:
            blocks.append(
                _new_block("list", "\n".join(buf_list), source, confidence=0.75)
            )
            buf_list = []

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        fence_match = _FENCE.match(stripped)
        if fence_match:
            if in_code:
                blocks.append(
                    _new_block(
                        "code",
                        "\n".join(code_buffer),
                        source,
                        confidence=0.8,
                        lang=code_lang,
                    )
                )
                code_buffer.clear()
                in_code = False
                code_lang = None
            else:
                in_code = True
                code_buffer = []
                code_lang = fence_match.group(1) or None
            index += 1
            continue

        if in_code:
            code_buffer.append(line)
            index += 1
            continue

        if stripped and _is_md_table_row(stripped):
            # Require a separator line to confirm table structure
            if index + 1 < len(lines) and _is_md_table_separator(lines[index + 1]):
                flush_list()
                table_lines = [stripped, lines[index + 1].strip()]
                index += 2
                while index < len(lines) and _is_md_table_row(lines[index]):
                    table_lines.append(lines[index].strip())
                    index += 1
                rows = []
                for line_idx, row in enumerate(table_lines):
                    if line_idx == 1 and _is_md_table_separator(row):
                        continue
                    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                    rows.append(cells)
                table_text = "\n".join([" | ".join(row) for row in rows])
                blocks.append(
                    _new_block(
                        "table",
                        table_text,
                        source,
                        confidence=0.85,
                        rows=rows,
                    )
                )
                continue

        if not stripped:
            flush_list()
            index += 1
            continue

        if stripped.startswith("#"):
            flush_list()
            blocks.append(
                _new_block(
                    "header", stripped.lstrip("# ").strip(), source, confidence=0.9
                )
            )
            index += 1
            continue

        if _LIST_MARKERS.match(stripped):
            buf_list.append(stripped)
            index += 1
            continue

        flush_list()
        blocks.append(_new_block("paragraph", stripped, source, confidence=0.65))
        index += 1

    flush_list()
    if in_code:
        blocks.append(
            _new_block(
                "code",
                "\n".join(code_buffer),
                source,
                confidence=0.8,
                lang=code_lang,
            )
        )
    return blocks or [_new_block("other", text, source, confidence=0.3)]


def parse_html(path: str | Path) -> List[Block]:
    source = Path(path)
    html = source.read_text(encoding="utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except Exception:
        return [_new_block("paragraph", html, source, confidence=0.4)]

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    blocks: List[Block] = []
    for element in soup.find_all(
        ["title", "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table", "a"]
    ):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        name = element.name or "p"
        if name == "title":
            blocks.append(_new_block("title", text, source, confidence=0.95))
        elif name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            blocks.append(
                _new_block("header", text, source, confidence=0.9, level=name)
            )
        elif name == "li":
            blocks.append(_new_block("list", text, source, confidence=0.75))
        elif name in {"pre", "code"}:
            blocks.append(_new_block("code", text, source, confidence=0.8))
        elif name == "a":
            href = element.get("href", "")
            block = _new_block(
                "paragraph", text, source, confidence=0.65, href=href
            )
            if href:
                block.spans.append(Span(start=0, end=len(block.text), label="anchor"))
            blocks.append(block)
        elif name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            table_text = "\n".join([", ".join(row) for row in rows])
            blocks.append(
                _new_block(
                    "table",
                    table_text,
                    source,
                    confidence=0.85,
                    rows=rows,
                )
            )
        else:
            blocks.append(_new_block("paragraph", text, source, confidence=0.6))
    if blocks:
        return blocks
    fallback = soup.get_text("\n", strip=True)
    return [_new_block("paragraph", fallback, source, confidence=0.45)]


def parse_csv(path: str | Path) -> List[Block]:
    source = Path(path)
    with source.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        rows = [row for row in reader]

    has_header = False
    try:
        has_header = csv.Sniffer().has_header(sample)
    except Exception:
        has_header = False

    text = "\n".join([", ".join(row) for row in rows])
    return [
        _new_block(
            "table",
            text,
            source,
            confidence=0.85,
            has_header=has_header,
            rows=rows,
            n_rows=len(rows),
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
            attrs: Dict[str, Any] = {}
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
                _new_block(
                    "log",
                    message,
                    source,
                    confidence=confidence,
                    **attrs,
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
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception:
        return [_new_block("other", "", source, confidence=0.1, note="pypdf_missing")]

    try:
        reader = PdfReader(str(source))
    except Exception as exc:
        return [_new_block("other", "", source, confidence=0.1, error=type(exc).__name__)]

    try:
        max_pages = int(os.getenv("SR_ADAPTER_PDF_MAX_PAGES", "400"))
    except ValueError:
        max_pages = 400

    pages = list(reader.pages)
    total_pages = len(pages)
    blocks: List[Block] = []
    for index, page in enumerate(pages[:max_pages], start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        stripped = text.strip()
        if not stripped:
            blocks.append(_new_block("other", "", source, confidence=0.2, page=index))
            continue
        for chunk in _split_paragraphs(text):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            blocks.append(
                _new_block(
                    "paragraph",
                    cleaned,
                    source,
                    confidence=0.6,
                    page=index,
                )
            )

    if total_pages > max_pages:
        blocks.append(
            _new_block(
                "other",
                "",
                source,
                confidence=0.2,
                pages_read=max_pages,
                pages_total=total_pages,
            )
        )

    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_docx(path: str | Path) -> List[Block]:
    source = Path(path)
    try:
        from docx import Document as DocxDocument  # type: ignore[import-not-found]
    except Exception:
        return [_new_block("other", "", source, confidence=0.2, note="python_docx_missing")]

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
        confidence = 0.65
        if block_type == "header":
            confidence = 0.8
        blocks.append(_new_block(block_type, text, source, confidence=confidence))
    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        table_text = "\n".join([", ".join(row) for row in rows])
        blocks.append(
            _new_block("table", table_text, source, confidence=0.75, rows=rows)
        )
    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_pptx(path: str | Path) -> List[Block]:
    source = Path(path)
    slides = extract_pptx_slides(source)
    blocks: List[Block] = []

    for index, slide in enumerate(slides, start=1):
        title = slide.get("title")
        bullets = slide.get("bullets", [])
        paragraphs = slide.get("paragraphs", []) or []

        if title:
            blocks.append(
                _new_block(
                    "header",
                    f"Slide {index}: {title}",
                    source,
                    confidence=0.75,
                )
            )

        if bullets:
            formatted = "\n".join(f"- {bullet}" for bullet in bullets if bullet)
            if formatted:
                blocks.append(_new_block("list", formatted, source, confidence=0.65))

        remaining = paragraphs[1:] if title else paragraphs
        extra_text = [line for line in remaining if line and line not in bullets]
        if extra_text:
            blocks.append(
                _new_block("paragraph", "\n".join(extra_text), source, confidence=0.55)
            )

    return blocks or [_new_block("other", "", source, confidence=0.1)]


def parse_xlsx(path: str | Path) -> List[Block]:
    source = Path(path)
    try:
        from openpyxl import load_workbook  # type: ignore[import-not-found]
    except Exception:
        return [_new_block("table", "", source, confidence=0.3, note="openpyxl_missing")]

    try:
        max_rows = int(os.getenv("SR_ADAPTER_XLSX_MAX_ROWS", "10000"))
    except ValueError:
        max_rows = 10000
    try:
        max_sheets = int(os.getenv("SR_ADAPTER_XLSX_MAX_SHEETS", "10"))
    except ValueError:
        max_sheets = 10

    workbook = load_workbook(filename=str(source), read_only=True, data_only=True)
    blocks: List[Block] = []
    try:
        worksheets = list(workbook.worksheets)
        for ws in worksheets[:max_sheets]:
            rows: List[List[Any]] = []
            truncated = False
            for index, row in enumerate(ws.iter_rows(values_only=True)):
                if index >= max_rows:
                    truncated = True
                    break
                rows.append(list(row))
            text_lines = []
            for row in rows:
                line = ", ".join("" if cell is None else str(cell) for cell in row)
                text_lines.append(line)
            text = "\n".join(text_lines)
            blocks.append(
                _new_block(
                    "table",
                    text,
                    source,
                    confidence=0.8,
                    sheet=ws.title,
                    rows=rows,
                    rows_read=len(rows),
                    truncated=truncated,
                )
            )
        if len(worksheets) > max_sheets:
            blocks.append(
                _new_block(
                    "other",
                    "",
                    source,
                    confidence=0.25,
                    sheets_read=max_sheets,
                    sheets_total=len(worksheets),
                )
            )
    finally:
        workbook.close()

    return blocks or [_new_block("table", "", source, confidence=0.3)]


def parse_pptx(path: str | Path) -> List[Block]:
    source = Path(path)
    blocks: List[Block] = []
    with zipfile.ZipFile(source) as archive:
        slide_members = sorted(
            member
            for member in archive.namelist()
            if member.startswith("ppt/slides/") and member.endswith(".xml")
        )
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        for index, member in enumerate(slide_members, start=1):
            try:
                xml_data = archive.read(member)
            except KeyError:  # pragma: no cover - corrupted archive guard
                continue
            try:
                tree = ET.fromstring(xml_data)
            except Exception:  # pragma: no cover - malformed slide guard
                continue
            texts = [node.text.strip() for node in tree.findall(".//a:t", namespace) if node.text]
            if not texts:
                continue
            title = texts[0]
            body = "\n".join(texts)
            blocks.append(
                _new_block(
                    "slide",
                    body,
                    source,
                    confidence=0.6,
                    title=title,
                    slide=index,
                )
            )
    return blocks or [_new_block("other", "", source, confidence=0.2)]


def parse_json(path: str | Path) -> List[Block]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8", errors="ignore")
    stripped = raw.strip()
    if not stripped:
        return [_new_block("other", "", source, confidence=0.1)]

    try:
        data = json.loads(stripped)
    except Exception:
        lines = [line for line in stripped.splitlines() if line.strip()]
        valid = 0
        for line in lines[:50]:
            try:
                json.loads(line)
            except Exception:
                continue
            else:
                valid += 1
        return [
            _new_block(
                "code",
                stripped,
                source,
                confidence=0.55,
                jsonl_valid=valid,
                n_lines=len(lines),
            )
        ]

    formatted = json.dumps(data, ensure_ascii=False, indent=2)
    return [
        _new_block(
            "code",
            formatted,
            source,
            confidence=0.7,
            top_type=type(data).__name__,
        )
    ]


def parse_jsonl(path: str | Path) -> List[Block]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    records: List[Any] = []
    errors = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except Exception:
            errors += 1
    text = "\n".join(line for line in lines if line.strip())
    confidence = 0.6 if errors == 0 else 0.45
    return [
        _new_block(
            "code",
            text,
            source,
            confidence=confidence,
            records_valid=len(records),
            records_invalid=errors,
        )
    ]


def parse_yaml(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    stripped = text.strip()
    if not stripped:
        return [_new_block("other", "", source, confidence=0.1)]

    blocks = [_new_block("code", chunk, source, confidence=0.55) for chunk in _split_paragraphs(text)]
    return blocks or [_new_block("code", stripped, source, confidence=0.55)]


def parse_xml(path: str | Path) -> List[Block]:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="ignore")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        stripped = text.strip()
        if not stripped:
            return [_new_block("other", "", source, confidence=0.1)]
        return [_new_block("code", stripped, source, confidence=0.45)]

    blocks: List[Block] = []
    root_tag = root.tag.split("}")[-1]
    blocks.append(_new_block("meta", f"XML root: {root_tag}", source, confidence=0.7))
    pretty = _prettify_xml(root).strip()
    if pretty:
        blocks.append(_new_block("code", pretty, source, confidence=0.6))
    text_content = " ".join(segment.strip() for segment in root.itertext() if segment.strip())
    if text_content:
        blocks.append(_new_block("paragraph", text_content, source, confidence=0.5))
    return blocks or [_new_block("code", text.strip(), source, confidence=0.55)]


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

