"""Utilities for reading various unstructured file formats."""

from __future__ import annotations

import base64
import json
import zipfile
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup

try:  # Optional dependencies are declared but lazily imported to keep import cost low.
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover - dependency import guard
    DocxDocument = None  # type: ignore[assignment]

try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - dependency import guard
    load_workbook = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - dependency import guard
    PdfReader = None  # type: ignore[assignment]

try:
    import extract_msg
except Exception:  # pragma: no cover - dependency import guard
    extract_msg = None  # type: ignore[assignment]

try:  # Pillow is optional – only used for richer image metadata when available.
    from PIL import Image
except Exception:  # pragma: no cover - dependency import guard
    Image = None  # type: ignore[assignment]

_TEXT_EXTENSIONS = {
    ".csv",
    ".log",
    ".md",
    ".rst",
    ".text",
    ".txt",
}

_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_DOCX_EXTENSIONS = {".docx"}
_XLSX_EXTENSIONS = {".xlsx"}
_EMAIL_EXTENSIONS = {".eml", ".msg"}
_ZIP_EXTENSIONS = {".zip"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

_HTML_MIMES = {"text/html", "application/xhtml+xml"}
_DOCX_MIMES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_XLSX_MIMES = {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
_EMAIL_MIMES = {"message/rfc822", "application/vnd.ms-outlook"}
_ZIP_MIMES = {"application/zip"}
_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
}


def _read_html(path: Path) -> Tuple[str, Dict[str, object]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    text = soup.get_text("\n", strip=True)
    extra: Dict[str, object] = {"extracted_as_text": True}

    if soup.title and soup.title.string:
        extra["html_title"] = soup.title.string.strip()

    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    if headings:
        extra["html_headings"] = [h for h in headings if h]

    return text, extra


def _read_docx(path: Path) -> Tuple[str, Dict[str, object]]:
    if DocxDocument is None:  # pragma: no cover - defensive guard
        raise RuntimeError("python-docx is not available")

    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    tables = document.tables

    buffer = StringIO()
    for paragraph in paragraphs:
        buffer.write(paragraph)
        buffer.write("\n")

    for table in tables:
        buffer.write("\n")
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            buffer.write("\t".join(cells))
            buffer.write("\n")

    text = buffer.getvalue().strip()
    extra: Dict[str, object] = {
        "extracted_as_text": True,
        "docx_paragraphs": len(document.paragraphs),
        "docx_tables": len(tables),
    }
    return text, extra


def _read_pdf(path: Path) -> Tuple[str, Dict[str, object]]:
    if PdfReader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("pypdf is not available")

    reader = PdfReader(str(path))
    pages_text: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages_text.append(extracted.strip())

    joined = "\n\n".join(filter(None, pages_text))
    has_text = any(pages_text)
    extra: Dict[str, object] = {
        "extracted_as_text": has_text,
        "pdf_page_count": len(reader.pages),
        "pdf_has_text": has_text,
    }
    return joined, extra


def _read_xlsx(path: Path) -> Tuple[str, Dict[str, object]]:
    if load_workbook is None:  # pragma: no cover - defensive guard
        raise RuntimeError("openpyxl is not available")

    workbook = load_workbook(path, data_only=True)
    try:
        sheet_names = list(workbook.sheetnames)
        rendered_sheets: List[str] = []
        populated_cells = 0

        for sheet in workbook.worksheets:
            rows: List[str] = []
            for row in sheet.iter_rows(values_only=True):
                if all(value is None for value in row):
                    continue
                rendered_row = []
                for value in row:
                    if value is None:
                        rendered_row.append("")
                    else:
                        rendered_row.append(str(value))
                        populated_cells += 1
                rows.append("\t".join(rendered_row))

            if rows:
                rendered_sheets.append(f"# {sheet.title}\n" + "\n".join(rows))

        text = "\n\n".join(rendered_sheets)
        extra: Dict[str, object] = {
            "extracted_as_text": bool(text),
            "workbook_sheet_count": len(sheet_names),
            "workbook_sheet_names": sheet_names,
            "workbook_cell_values": populated_cells,
        }
        return text, extra
    finally:
        workbook.close()


def _read_eml(path: Path) -> Tuple[str, Dict[str, object]]:
    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    text_parts: List[str] = []
    attachments: List[Dict[str, object]] = []
    for part in message.walk():
        maintype = part.get_content_maintype()
        disposition = part.get_content_disposition()
        if maintype == "multipart":
            continue
        if disposition == "attachment":
            payload = part.get_payload(decode=True) or b""
            attachments.append(
                {
                    "filename": part.get_filename() or "",
                    "mime": part.get_content_type(),
                    "size": len(payload),
                }
            )
            continue
        if part.get_content_type().startswith("text/"):
            text_parts.append(part.get_content().strip())

    text = "\n\n".join(part for part in text_parts if part).strip()

    extra: Dict[str, object] = {"extracted_as_text": bool(text), "email_attachments": attachments}

    subject = (message.get("subject") or "").strip()
    if subject:
        extra["email_subject"] = subject

    addresses = getaddresses([message.get("from", "")])
    if addresses:
        extra["email_from"] = [addr for _, addr in addresses if addr]

    tos = getaddresses([message.get("to", "")])
    if tos:
        extra["email_to"] = [addr for _, addr in tos if addr]

    ccs = getaddresses([message.get("cc", "")])
    if ccs:
        extra["email_cc"] = [addr for _, addr in ccs if addr]

    date_header = message.get("date")
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
        except (TypeError, ValueError):  # pragma: no cover - malformed date guard
            parsed = None
        if parsed is not None:
            extra["email_date"] = parsed.isoformat()

    return text, extra


def _read_msg(path: Path) -> Tuple[str, Dict[str, object]]:
    if extract_msg is None:  # pragma: no cover - defensive guard
        raise RuntimeError("extract_msg is not available")

    message = extract_msg.Message(str(path))
    try:
        text = (message.body or "").strip()
        attachments_meta: List[Dict[str, object]] = []
        for attachment in message.attachments:
            attachments_meta.append(
                {
                    "filename": attachment.longFilename or attachment.shortFilename or "",
                    "mime": getattr(attachment, "mimeType", ""),
                    "size": getattr(attachment, "size", 0),
                }
            )

        extra: Dict[str, object] = {
            "extracted_as_text": bool(text),
            "email_subject": message.subject or "",
            "email_from": [message.sender] if message.sender else [],
            "email_to": [addr.strip() for addr in (message.to or "").split(";") if addr.strip()],
            "email_cc": [addr.strip() for addr in (message.cc or "").split(";") if addr.strip()],
            "email_attachments": attachments_meta,
        }

        if message.date:
            try:
                extra["email_date"] = message.date.isoformat()  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - fallback when date is str
                extra["email_date"] = str(message.date)

        return text, extra
    finally:
        message.close()


def _read_zip(path: Path) -> Tuple[str, Dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        lines: List[str] = []
        entries: List[Dict[str, object]] = []
        contains_nested = False
        text_snippets: List[str] = []

        for member in members:
            member_path = member.filename
            lines.append(f"{member_path} ({member.file_size} bytes)")
            entries.append(
                {
                    "name": member_path,
                    "size": member.file_size,
                    "compressed": member.compress_size,
                }
            )
            if member_path.lower().endswith(".zip"):
                contains_nested = True
            if member.file_size <= 64 * 1024 and not member.is_dir():
                if Path(member_path).suffix.lower() in _TEXT_EXTENSIONS | {".json", ".csv", ".tsv", ".log"}:
                    with archive.open(member) as handle:
                        try:
                            snippet = handle.read().decode("utf-8", errors="ignore")
                        except OSError:
                            continue
                        text_snippets.append(snippet.strip())

    summary_text = "\n".join(lines)
    combined_snippet = "\n\n".join(filter(None, text_snippets))
    text = "\n\n".join(filter(None, [summary_text, combined_snippet]))

    extra: Dict[str, object] = {
        "extracted_as_text": bool(text_snippets),
        "zip_entry_count": len(members),
        "zip_entries": entries,
    }
    if contains_nested:
        extra["zip_contains_nested"] = True
    return text, extra


def _read_image(path: Path) -> Tuple[str, Dict[str, object]]:
    meta: Dict[str, object] = {"extracted_as_text": False}
    if Image is not None:
        try:
            with Image.open(path) as img:
                meta.update(
                    {
                        "image_format": img.format,
                        "image_width": img.width,
                        "image_height": img.height,
                        "image_mode": img.mode,
                    }
                )
        except Exception:  # pragma: no cover - image parsing fallback
            pass
    return "", meta


def read_file_contents(path: Path, mime: str) -> Tuple[str, Dict[str, object]]:
    """Return a textual representation and extra metadata for *path*.

    Parameters
    ----------
    path:
        File to read.
    mime:
        MIME type determined by the caller.
    """

    suffix = path.suffix.lower()
    extra: Dict[str, object] = {}

    if mime == "application/json" or suffix == ".json":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            extra.update({"json_valid": False})
            return raw, extra

        extra.update(
            {
                "json_valid": True,
                "json_top_level_type": type(data).__name__,
            }
        )
        if isinstance(data, dict):
            extra["json_top_level_keys"] = sorted(map(str, data.keys()))
        return json.dumps(data, ensure_ascii=False, indent=2), extra

    if mime in _HTML_MIMES or suffix in _HTML_EXTENSIONS:
        text, html_extra = _read_html(path)
        extra.update(html_extra)
        return text, extra

    if mime in _DOCX_MIMES or suffix in _DOCX_EXTENSIONS:
        text, docx_extra = _read_docx(path)
        extra.update(docx_extra)
        return text, extra

    if mime == "application/pdf" or suffix == ".pdf":
        text, pdf_extra = _read_pdf(path)
        extra.update(pdf_extra)
        return text, extra

    if mime in _XLSX_MIMES or suffix in _XLSX_EXTENSIONS:
        text, xlsx_extra = _read_xlsx(path)
        extra.update(xlsx_extra)
        return text, extra

    if mime in _EMAIL_MIMES or suffix in _EMAIL_EXTENSIONS:
        try:
            if suffix == ".msg" or mime == "application/vnd.ms-outlook":
                text, email_extra = _read_msg(path)
            else:
                text, email_extra = _read_eml(path)
        except RuntimeError:
            # Surface binary preview when specialised reader is unavailable.
            blob = path.read_bytes()
            extra.update(
                {
                    "binary_preview_bytes": min(len(blob), 32),
                    "binary_preview_base64": base64.b64encode(blob[:32]).decode("ascii"),
                }
            )
            return "", extra
        extra.update(email_extra)
        return text, extra

    if mime in _ZIP_MIMES or suffix in _ZIP_EXTENSIONS:
        text, zip_extra = _read_zip(path)
        extra.update(zip_extra)
        return text, extra

    if mime in _IMAGE_MIMES or suffix in _IMAGE_EXTENSIONS:
        text, image_extra = _read_image(path)
        extra.update(image_extra)
        return text, extra

    if mime.startswith("text/") or suffix in _TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, extra

    # Fallback for binary files – provide a short base64 preview.
    blob = path.read_bytes()
    extra.update(
        {
            "binary_preview_bytes": min(len(blob), 32),
            "binary_preview_base64": base64.b64encode(blob[:32]).decode("ascii"),
        }
    )
    return "", extra
