# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded source units for relation extraction, without OCR or model calls.

All line, slide, row and column locators are one-based. CSV ``row`` counts
logical records including the header; ``line_start/end`` count physical lines.
PPTX text boxes retain paragraph boundaries, and table units retain physical
cell positions and merge flags without inferring headers. Image transcript
locators identify supplied transcript lines, never pixel bounding boxes.

Limits reject the whole source rather than returning a truncated success.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import stat
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from PIL import Image


_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024
_MAX_UNITS = 500
_MAX_ZIP_BYTES = 32 * 1024 * 1024
_MAX_ZIP_ENTRIES = 2048
_MAX_XML_BYTES = 4 * 1024 * 1024
_MAX_XML_ELEMENTS = 50_000
_MAX_XML_DEPTH = 128
_MAX_IMAGE_PIXELS = 16_000_000
_MAX_IMAGE_SIDE = 16_384

_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_R = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
_STRICT_R = "http://purl.oclc.org/ooxml/officeDocument/relationships"
_NS = {"p": _P, "a": _A, "r": _R}
_STRICT_NAMESPACES = {
    "http://purl.oclc.org/ooxml/presentationml/main": _P,
    "http://purl.oclc.org/ooxml/drawingml/main": _A,
    _STRICT_R: _R,
}
_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}


class _Units:
    def __init__(self, source_hash: str):
        self.source_hash = source_hash
        self.units: list[dict] = []
        self.warnings: list[str] = []
        self.text_bytes = 0

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add(self, text: str, locator: dict, grounding: str = "source_text") -> None:
        size = len(text.encode("utf-8"))
        if len(self.units) >= _MAX_UNITS or self.text_bytes + size > _MAX_TEXT_BYTES:
            raise ValueError("Source exceeds 500 units or 64 KiB of extracted UTF-8 text")
        identity = json.dumps(
            [self.source_hash, text, locator, grounding],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        self.units.append({
            "id": "source-unit-" + hashlib.sha256(identity).hexdigest(),
            "text": text, "locator": locator, "grounding": grounding,
        })
        self.text_bytes += size


def _utf8(data: bytes, units: _Units) -> str:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Text source must be valid UTF-8") from exc
    if "\x00" in text:
        raise ValueError("Text source contains a NUL character")
    if text.startswith("\ufeff"):
        units.warn("The UTF-8 byte order mark is omitted from extracted text.")
        text = text[1:]
    return text


def _read_text(data: bytes, units: _Units) -> None:
    text = _utf8(data, units).replace("\r\n", "\n").replace("\r", "\n")
    for number, line in enumerate(text.split("\n"), 1):
        if line.strip():
            units.add(line, {"kind": "text", "line_start": number, "line_end": number})


def _read_csv(data: bytes, delimiter: str, units: _Units) -> None:
    reader = csv.reader(io.StringIO(_utf8(data, units), newline=""), delimiter=delimiter, strict=True)
    try:
        headers = next(reader, None)
        if not headers:
            raise ValueError("CSV requires a nonempty header record")
        line_end = reader.line_num
        for row_number, row in enumerate(reader, 2):
            line_start = line_end + 1
            line_end = reader.line_num
            if len(row) != len(headers):
                raise ValueError(f"CSV row {row_number} has a different width from its header")
            cells = [
                {"column": column, "header": header, "value": value}
                for column, (header, value) in enumerate(zip(headers, row), 1)
            ]
            units.add(
                json.dumps({"cells": cells}, ensure_ascii=False, separators=(",", ":")),
                {"kind": "csv", "row": row_number, "line_start": line_start,
                 "line_end": line_end, "cells": cells},
            )
    except csv.Error as exc:
        raise ValueError("Malformed CSV/TSV source") from exc


class _BoundedTreeBuilder(ET.TreeBuilder):
    def __init__(self):
        super().__init__()
        self.elements = 0
        self.depth = 0

    def start(self, tag, attrs):
        self.elements += 1
        self.depth += 1
        if self.elements > _MAX_XML_ELEMENTS or self.depth > _MAX_XML_DEPTH:
            raise ValueError("PPTX XML exceeds the element or nesting limit")
        return super().start(tag, attrs)

    def end(self, tag):
        self.depth -= 1
        return super().end(tag)

    def doctype(self, name, pubid, system):
        raise ValueError("PPTX XML document type declarations are not allowed")


def _xml(data: bytes) -> ET.Element:
    if len(data) > _MAX_XML_BYTES:
        raise ValueError("PPTX XML part exceeds 4 MiB")
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(target=_BoundedTreeBuilder()))
    except ET.ParseError as exc:
        raise ValueError("Malformed PPTX XML") from exc
    # Strict and transitional OOXML use different namespaces for the same parts.
    for element in root.iter():
        for old, new in _STRICT_NAMESPACES.items():
            if element.tag.startswith("{" + old + "}"):
                element.tag = "{" + new + "}" + element.tag.split("}", 1)[1]
            for key in list(element.attrib):
                if key.startswith("{" + old + "}"):
                    element.attrib["{" + new + "}" + key.split("}", 1)[1]] = element.attrib.pop(key)
    return root


def _zip_parts(data: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ZIP_ENTRIES:
                raise ValueError("PPTX ZIP contains too many entries")
            if sum(entry.file_size for entry in entries) > _MAX_ZIP_BYTES:
                raise ValueError("PPTX ZIP expands beyond 32 MiB")
            names: set[str] = set()
            parts = {}
            for entry in entries:
                name = entry.filename.rstrip("/")
                if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                        or any(part in ("", ".", "..") for part in name.split("/"))):
                    raise ValueError("PPTX ZIP contains an unsafe entry name")
                if name in names:
                    raise ValueError("PPTX ZIP contains a duplicate entry")
                names.add(name)
                if entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Encrypted entries and symlinks are not allowed in PPTX")
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("Unsupported PPTX ZIP compression")
                if not entry.is_dir():
                    # Reading also verifies the ZIP CRC; no paths are extracted.
                    parts[name] = archive.read(entry)
            return parts
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError) as exc:
        raise ValueError("Invalid PPTX ZIP package") from exc


def _relationship_target(owner: str, target: str) -> str:
    if not target or "\\" in target or "\x00" in target:
        raise ValueError("Invalid PPTX relationship target")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("External or fragment PPTX relationships are not allowed")
    decoded = unquote(parsed.path, errors="strict")
    if "\\" in decoded or "\x00" in decoded:
        raise ValueError("Invalid encoded PPTX relationship target")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(owner), decoded))
    if path.startswith("/"):
        path = path[1:]
    if not path or path == ".." or path.startswith("../"):
        raise ValueError("PPTX relationship escapes its package")
    return path


def _relationships(parts: dict[str, bytes]) -> dict[str, dict[str, tuple[str, str]]]:
    result = {}
    for name, data in parts.items():
        if not name.endswith(".rels"):
            continue
        if name == "_rels/.rels":
            owner = ""
        elif "/_rels/" in name:
            prefix, leaf = name.rsplit("/_rels/", 1)
            if "/" in leaf:
                raise ValueError("Invalid PPTX relationships part name")
            owner = prefix + "/" + leaf[:-5]
        else:
            raise ValueError("Invalid PPTX relationships part name")
        root = _xml(data)
        if root.tag != "{" + _PACKAGE_R + "}Relationships":
            raise ValueError("Invalid PPTX relationships root")
        relationships = {}
        for element in root:
            if element.tag != "{" + _PACKAGE_R + "}Relationship":
                raise ValueError("Invalid PPTX relationship element")
            if element.get("TargetMode", "Internal") != "Internal":
                raise ValueError("External PPTX relationships are not allowed")
            identity, kind = element.get("Id"), element.get("Type")
            if not identity or not kind or identity in relationships:
                raise ValueError("Missing or duplicate PPTX relationship identifier")
            target = _relationship_target(owner, element.get("Target", ""))
            if target not in parts:
                raise ValueError("PPTX relationship target is missing")
            relationships[identity] = (kind, target)
        result[owner] = relationships
    return result


def _is_relationship(kind: str, name: str) -> bool:
    return kind in (_R + "/" + name, _STRICT_R + "/" + name)


def _paragraph_text(body: ET.Element, units: _Units) -> tuple[str, int]:
    paragraphs = []
    for paragraph in body.findall("a:p", _NS):
        chunks = []
        for child in paragraph:
            if child.tag in ("{" + _A + "}r", "{" + _A + "}fld"):
                chunks.extend(node.text or "" for node in child.findall("a:t", _NS))
            elif child.tag == "{" + _A + "}br":
                chunks.append("\n")
            elif child.tag not in ("{" + _A + "}pPr", "{" + _A + "}endParaRPr"):
                units.warn("PPTX unsupported paragraph elements are not extracted.")
        paragraphs.append("".join(chunks))
    return "\n".join(paragraphs), len(paragraphs)


def _shape_id(shape: ET.Element) -> str:
    identity = shape.find(".//p:cNvPr", _NS)
    if identity is None or not identity.get("id"):
        raise ValueError("PPTX shape has no identifier")
    return identity.attrib["id"]


def _positive_int(value: str) -> int:
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("Invalid PPTX table span")
    return int(value)


def _merge_flag(value: str) -> bool:
    if value not in ("0", "1", "false", "true"):
        raise ValueError("Invalid PPTX table merge flag")
    return value in ("1", "true")


def _read_table(table: ET.Element, slide: int, shape_id: str, units: _Units) -> None:
    columns = table.findall("a:tblGrid/a:gridCol", _NS)
    rows = table.findall("a:tr", _NS)
    if not columns or not rows:
        raise ValueError("PPTX table has no cell grid")
    for row_number, row in enumerate(rows, 1):
        cells = row.findall("a:tc", _NS)
        if len(cells) != len(columns):
            raise ValueError("PPTX table row does not match its column grid")
        for column, cell in enumerate(cells, 1):
            body = cell.find("a:txBody", _NS)
            if body is None:
                raise ValueError("PPTX table cell has no text body")
            text, paragraphs = _paragraph_text(body, units)
            row_span = _positive_int(cell.get("rowSpan", "1"))
            column_span = _positive_int(cell.get("gridSpan", "1"))
            if row_number + row_span - 1 > len(rows) or column + column_span - 1 > len(columns):
                raise ValueError("PPTX table span exceeds its grid")
            units.add(text, {
                "kind": "pptx", "slide": slide, "shape_id": shape_id,
                "cell": {"row": row_number, "column": column},
                "row_span": row_span, "column_span": column_span,
                "horizontal_merge": _merge_flag(cell.get("hMerge", "0")),
                "vertical_merge": _merge_flag(cell.get("vMerge", "0")),
                "paragraph_start": 1, "paragraph_end": paragraphs,
            })


def _read_shapes(tree: ET.Element, slide: int, units: _Units) -> None:
    for shape in tree:
        if shape.tag == "{" + _P + "}grpSp":
            _read_shapes(shape, slide, units)
        elif shape.tag == "{" + _P + "}sp":
            body = shape.find("p:txBody", _NS)
            if body is not None:
                text, paragraphs = _paragraph_text(body, units)
                if text.strip():
                    units.add(text, {"kind": "pptx", "slide": slide, "shape_id": _shape_id(shape),
                                     "paragraph_start": 1, "paragraph_end": paragraphs})
        elif shape.tag == "{" + _P + "}graphicFrame":
            graphic = shape.find("a:graphic/a:graphicData", _NS)
            table = graphic.find("a:tbl", _NS) if graphic is not None else None
            if table is not None:
                _read_table(table, slide, _shape_id(shape), units)
            else:
                units.warn("PPTX SmartArt, charts and other graphic frames are not extracted.")
        elif shape.tag == "{" + _P + "}pic":
            units.warn("PPTX images are not transcribed.")
        elif shape.tag not in ("{" + _P + "}nvGrpSpPr", "{" + _P + "}grpSpPr", "{" + _P + "}extLst"):
            units.warn("PPTX connectors, alternate or unsupported shapes are not extracted.")


def _read_pptx(data: bytes, units: _Units) -> None:
    parts = _zip_parts(data)
    relationships = _relationships(parts)
    office_parts = [target for kind, target in relationships.get("", {}).values()
                    if _is_relationship(kind, "officeDocument")]
    if len(office_parts) != 1 or "[Content_Types].xml" not in parts:
        raise ValueError("PPTX requires exactly one presentation part and content types")
    presentation_name = office_parts[0]
    content_types = _xml(parts["[Content_Types].xml"])
    expected_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
    if content_types.tag != "{" + _CONTENT_TYPES + "}Types" or not any(
        item.get("PartName") == "/" + presentation_name and item.get("ContentType") == expected_type
        for item in content_types
    ):
        raise ValueError("PPTX presentation content type is missing or unsupported")
    presentation = _xml(parts[presentation_name])
    if presentation.tag != "{" + _P + "}presentation":
        raise ValueError("PPTX presentation root is invalid")
    slides = presentation.findall("p:sldIdLst/p:sldId", _NS)
    if not slides or len(slides) > _MAX_UNITS:
        raise ValueError("PPTX must contain between 1 and 500 slides")
    for links in relationships.values():
        for kind, _ in links.values():
            if _is_relationship(kind, "notesSlide"):
                units.warn("PPTX speaker notes are not extracted.")
            if any(_is_relationship(kind, item) for item in ("image", "audio", "video")):
                units.warn("PPTX images, audio and video are not transcribed.")
            if any(_is_relationship(kind, item) for item in ("chart", "diagramData", "diagramDrawing")):
                units.warn("PPTX SmartArt and charts are not extracted.")
            if any(_is_relationship(kind, item) for item in ("slideMaster", "slideLayout")):
                units.warn("PPTX master and layout text is not extracted; only slide text is included.")
    seen_slides: set[str] = set()
    seen_slide_ids: set[str] = set()
    for slide_number, reference in enumerate(slides, 1):
        relationship_id = reference.get("{" + _R + "}id", "")
        link = relationships.get(presentation_name, {}).get(relationship_id)
        slide_id = reference.get("id", "")
        if not link or not _is_relationship(link[0], "slide"):
            raise ValueError("PPTX slide relationship is missing or invalid")
        if not slide_id or slide_id in seen_slide_ids or link[1] in seen_slides:
            raise ValueError("PPTX contains missing or duplicate slide identifiers")
        seen_slides.add(link[1])
        seen_slide_ids.add(slide_id)
        slide = _xml(parts[link[1]])
        if slide.tag != "{" + _P + "}sld":
            raise ValueError("PPTX slide root is invalid")
        if slide.get("show") in ("0", "false"):
            units.warn("PPTX hidden slides are included in source order.")
        tree = slide.find("p:cSld/p:spTree", _NS)
        if tree is None:
            raise ValueError("PPTX slide has no shape tree")
        shape_ids = [node.get("id") for node in tree.findall(".//p:cNvPr", _NS)]
        if any(not identity for identity in shape_ids) or len(set(shape_ids)) != len(shape_ids):
            raise ValueError("PPTX contains missing or duplicate shape identifiers")
        _read_shapes(tree, slide_number, units)


def _read_image(data: bytes, extension: str, transcription: dict | None, units: _Units) -> bool:
    expected = _IMAGE_FORMATS[extension]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[expected]) as image:
                width, height = image.size
                if image.format != expected:
                    raise ValueError("Image format does not match its filename")
                if (width < 1 or height < 1 or max(width, height) > _MAX_IMAGE_SIDE
                        or width * height > _MAX_IMAGE_PIXELS):
                    raise ValueError("Image exceeds the supported dimension limit")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Only single-frame images are supported")
                image.verify()
            # verify() does not decode pixels; load a fresh image to reject damaged data.
            with Image.open(io.BytesIO(data), formats=[expected]) as image:
                image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or oversized image source") from exc
    if transcription is None:
        units.warn("Image text has not been transcribed.")
        return True
    if not isinstance(transcription, dict) or set(transcription) != {"lines"}:
        raise ValueError("Image transcription must contain only a lines list")
    lines = transcription["lines"]
    if not isinstance(lines, list) or not lines:
        raise ValueError("Image transcription requires a nonempty lines list")
    for number, text in enumerate(lines, 1):
        if not isinstance(text, str) or not text.strip() or any(char in text for char in ("\r", "\n", "\x00")):
            raise ValueError("Image transcription lines must be nonempty single-line strings")
        units.add(text, {"kind": "image", "frame": 1, "transcript_line": number},
                  "model_transcription_unverified")
    units.warn("Image transcription is unverified; transcript line locators are not pixel bounding boxes.")
    return False


def read_source(data: bytes, source_name: str, *, transcription: dict | None = None) -> dict:
    """Read supported source bytes, or raise ``ValueError`` without partial output.

    Images without a transcription return no units and ``needs_transcription=True``.
    Supplying a transcription does not verify it against the image. Other formats
    require extractable text and reject the transcription argument. Raw bytes are
    represented by their SHA-256; the caller is responsible for storing them.
    """
    if not isinstance(data, bytes) or not data or len(data) > _MAX_INPUT_BYTES:
        raise ValueError("Source must be nonempty bytes of at most 16 MiB")
    if not isinstance(source_name, str) or not source_name.strip() or "\x00" in source_name or len(source_name) > 1024:
        raise ValueError("Source name must be a nonempty filename of at most 1024 characters")
    extension = PurePosixPath(source_name.replace("\\", "/")).suffix.lower()
    if extension not in _MEDIA_TYPES:
        raise ValueError("Unsupported source format")
    if transcription is not None and extension not in _IMAGE_FORMATS:
        raise ValueError("Transcription is accepted only for image sources")
    source_hash = hashlib.sha256(data).hexdigest()
    units = _Units(source_hash)
    needs_transcription = False
    if extension in (".md", ".txt"):
        _read_text(data, units)
    elif extension in (".csv", ".tsv"):
        _read_csv(data, "," if extension == ".csv" else "\t", units)
    elif extension == ".pptx":
        _read_pptx(data, units)
    else:
        needs_transcription = _read_image(data, extension, transcription, units)
    if not needs_transcription and not any(unit["text"].strip() for unit in units.units):
        raise ValueError("Source contains no extractable text units")
    return {
        "schema_version": 1, "source_name": source_name, "source_sha256": source_hash,
        "media_type": _MEDIA_TYPES[extension], "units": units.units,
        "warnings": units.warnings, "needs_transcription": needs_transcription,
    }
