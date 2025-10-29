"""Utilities for reading various unstructured file formats."""

from __future__ import annotations

import base64
import json
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

_HTML_MIMES = {"text/html", "application/xhtml+xml"}
_DOCX_MIMES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_XLSX_MIMES = {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


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
