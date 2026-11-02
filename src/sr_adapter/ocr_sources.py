"""Replayable OCR source packages, with page images and unverified line regions.

Reading uses local engines. Verification replays saved readings, not OCR, and
does not claim that an OCR string or a recorded rendered view is ground truth.
"""
from __future__ import annotations

from contextlib import closing
import io
import json
import math
from pathlib import Path
import threading

from PIL import Image, ImageOps
from pypdf import PdfReader
from pypdf.generic import ContentStream

from .knowledge import sha256
from .relation_sources import _Units
from .tables import canonical_json, identifier

VERSION = 1
MAX_PAGES = 12
MAX_PIXELS = 16_000_000
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_FRAMES_BYTES = 20 * 1024 * 1024
MAX_PACKAGE_BYTES = 28 * 1024 * 1024
_PDF_LOCK = threading.Lock()  # PDFium calls must not overlap between threads.


def _pdf(data):
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 100:
        raise ValueError("OCR PDF must be unencrypted and have 1..100 pages")
    return reader


def _has_raster(resources, depth=0, seen=None, stream=None):
    if depth > 12:
        raise ValueError("PDF resource nesting exceeds the OCR routing limit")
    seen = set() if seen is None else seen
    if stream is not None and any(op == b"INLINE IMAGE" for _, op in ContentStream(stream, None).operations):
        return True
    resources = resources.get_object() if hasattr(resources, "get_object") else resources or {}
    objects = resources.get("/XObject", {})
    objects = objects.get_object() if hasattr(objects, "get_object") else objects
    for reference in objects.values():
        obj = reference.get_object()
        identity = id(obj)
        if identity in seen:
            continue
        seen.add(identity)
        if obj.get("/Subtype") == "/Image":
            return True
        if obj.get("/Subtype") == "/Form" and _has_raster(obj.get("/Resources"), depth + 1, seen, obj):
            return True
    return False


def _selection(data, source_name, mode, dpi, pages):
    if mode not in {"auto", "always"} or type(dpi) is not int or not 72 <= dpi <= 300:
        raise ValueError("Use OCR mode auto/always and DPI 72..300")
    if not isinstance(source_name, str) or not 1 <= len(source_name) <= 256 or any(ord(c) < 32 for c in source_name):
        raise ValueError("Use a bounded source filename")
    if not isinstance(data, bytes) or not 1 <= len(data) <= 16 * 1024 * 1024:
        raise ValueError("OCR source must contain 1 byte..16 MiB")
    extension = Path(source_name).suffix.lower()
    is_pdf = extension == ".pdf"
    if not is_pdf and extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("OCR accepts PDF, PNG, JPEG or WebP")
    reader = _pdf(data) if is_pdf else None
    if not is_pdf:
        with Image.open(io.BytesIO(data)) as image:
            expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[extension]
            if (image.format != expected or getattr(image, "n_frames", 1) != 1
                    or image.width * image.height > MAX_PIXELS):
                raise ValueError("OCR images require one PNG/JPEG/WebP frame of at most 16 million pixels")
    total = len(reader.pages) if is_pdf else 1
    selected = list(range(1, total + 1)) if pages is None else list(pages)
    if (not 1 <= len(selected) <= MAX_PAGES or any(type(p) is not int or not 1 <= p <= total for p in selected)
            or selected != sorted(set(selected))):
        raise ValueError("Select 1..12 distinct physical pages in increasing order")
    plans = []
    for number in selected:
        text = (reader.pages[number - 1].extract_text() or "") if is_pdf else ""
        usable = len(text.strip()) >= 16 and "\ufffd" not in text and "\x00" not in text
        page = reader.pages[number - 1] if is_pdf else None
        raster = _has_raster(page.get("/Resources"), stream=page.get_contents()) if is_pdf else True
        native = mode == "auto" and is_pdf and usable and not raster
        plans.append({"page": number, "method": "embedded_text" if native else "ocr",
                      "reason": "usable_text_without_raster" if native else "forced_or_raster_or_sparse_text"})
    media_type = "application/pdf" if is_pdf else "image/" + ("jpeg" if extension in {".jpg", ".jpeg"} else extension[1:])
    return total, plans, media_type


def _signature(reader_signature):
    signature = json.loads(canonical_json(reader_signature))
    if (not isinstance(signature, dict) or not isinstance(signature.get("engine"), str)
            or not signature["engine"] or not isinstance(signature.get("version"), str)
            or not signature["version"] or len(canonical_json(signature)) > 65536):
        raise ValueError("OCR readers require an explicit stable engine/version signature")
    return signature


def plan_ocr(data, source_name, *, mode="auto", dpi=144, pages=None, reader_signature):
    total, plans, media_type = _selection(data, source_name, mode, dpi, pages)
    import PIL
    renderer = {"pillow": PIL.__version__}
    if media_type == "application/pdf":
        import pypdfium2 as pdfium
        import pypdf
        renderer.update(pdfium=str(pdfium.PDFIUM_INFO), pypdfium=str(pdfium.PYPDFIUM_INFO), pypdf=pypdf.__version__)
    return {"version": VERSION, "source_name": source_name, "source_sha256": sha256(data),
        "media_type": media_type,
        "page_count": total, "pages": plans, "mode": mode, "dpi": dpi,
        "max_edge": 2500, "reader": _signature(reader_signature), "renderer": renderer,
        "coordinate_system": "rendered_pixels_top_left", "rendered_view_verification": "recorded_not_independently_rerendered"}


def _verify_plan(data, plan):
    keys = {"version", "source_name", "source_sha256", "media_type", "page_count", "pages", "mode", "dpi",
            "max_edge", "reader", "renderer", "coordinate_system", "rendered_view_verification"}
    if (not isinstance(plan, dict) or set(plan) != keys or type(plan["version"]) is not int
            or plan["version"] != VERSION or plan["source_sha256"] != sha256(data)
            or not isinstance(plan["pages"], list) or len(canonical_json(plan)) > 1024 * 1024):
        raise ValueError("Invalid OCR source plan")
    total, selected, media = _selection(data, plan["source_name"], plan["mode"], plan["dpi"], [p["page"] for p in plan["pages"]])
    if (type(plan["page_count"]) is not int or total != plan["page_count"] or selected != plan["pages"]
            or media != plan["media_type"] or type(plan["max_edge"]) is not int or plan["max_edge"] != 2500
            or plan["coordinate_system"] != "rendered_pixels_top_left"
            or plan["rendered_view_verification"] != "recorded_not_independently_rerendered"):
        raise ValueError("Saved OCR plan differs from its source and routing policy")
    _signature(plan["reader"])
    renderer = plan["renderer"]
    if (not isinstance(renderer, dict) or set(renderer) != ({"pillow", "pdfium", "pypdfium", "pypdf"} if media == "application/pdf" else {"pillow"})
            or any(not isinstance(v, str) or not 1 <= len(v) <= 256 for v in renderer.values())):
        raise ValueError("Missing OCR renderer provenance")


def _png(image):
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("OCR frame exceeds 16 million pixels")
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    data = output.getvalue()
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError("OCR frame exceeds 8 MiB")
    return data


def _render(data, plan, number):
    if plan["media_type"] != "application/pdf":
        with Image.open(io.BytesIO(data)) as raw:
            if getattr(raw, "n_frames", 1) != 1 or raw.width * raw.height > MAX_PIXELS:
                raise ValueError("OCR images require one frame of at most 16 million pixels")
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((plan["max_edge"], plan["max_edge"]))
            return _png(image)
    import pypdfium2 as pdfium
    with _PDF_LOCK, closing(pdfium.PdfDocument(data)) as document:
        with closing(document[number - 1]) as page:
            width, height = page.get_size()
            if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                raise ValueError("Invalid PDF page geometry")
            scale = min(plan["dpi"] / 72, plan["max_edge"] / max(width, height))
            if math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS:
                raise ValueError("PDF render exceeds pixel budget")
            with closing(page.render(scale=scale, rotation=0, draw_annots=True)) as bitmap:
                return _png(bitmap.to_pil())


def _validate_lines(value, width, height):
    if not isinstance(value, dict) or set(value) != {"lines"} or not isinstance(value["lines"], list) or len(value["lines"]) > 500:
        raise ValueError("OCR must return a bounded lines list")
    for line in value["lines"]:
        if not isinstance(line, dict) or set(line) != {"text", "box", "confidence", "words"}:
            raise ValueError("OCR lines require text, box, confidence and words")
        if not isinstance(line["text"], str) or not line["text"].strip() or len(line["text"]) > 65536 or "\x00" in line["text"]:
            raise ValueError("Invalid OCR line text")
        box = line["box"]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("OCR line is outside its rendered page")
        confidence = line["confidence"]
        if confidence is not None and (type(confidence) not in {float, int} or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("OCR confidence must be null or a finite value between 0 and 1")
        if not isinstance(line["words"], list) or len(line["words"]) > 10000:
            raise ValueError("OCR word receipt exceeds its bound")
        for word in line["words"]:
            if not isinstance(word, dict) or set(word) != {"text", "box", "confidence"}:
                raise ValueError("Invalid OCR word receipt")
            _validate_lines({"lines": [{**word, "words": []}]}, width, height)
            word_box = word["box"]
            if not (box[0] <= word_box[0] < word_box[2] <= box[2] and box[1] <= word_box[1] < word_box[3] <= box[3]):
                raise ValueError("OCR word lies outside its line")
    return value["lines"]


def _project(data, plan, pages):
    _verify_plan(data, plan)
    if len(pages) != len(plan["pages"]) or len(canonical_json(plan)) > 1024 * 1024:
        raise ValueError("OCR page selection differs")
    units = _Units(sha256(data))
    units.warn("OCR text and line boxes are unverified readings; confidence is an engine score, not a calibrated probability.")
    units.warn("OCR line/word order is not table structure. Check headers, columns and conditions against the saved page image.")
    if len(pages) != plan["page_count"]:
        units.warn("Only selected physical pages were read: " + ", ".join(str(p["page"]) for p in plan["pages"]))
    for selected, page in zip(plan["pages"], pages):
        number = selected["page"]
        if (not isinstance(page, dict) or set(page) != {"page", "method", "width", "height", "frame_sha256", "reading"}
                or type(page["page"]) is not int or page["page"] != number or page["method"] != selected["method"]
                or any(type(page[d]) is not int or not 1 <= page[d] <= plan["max_edge"] for d in ("width", "height"))):
            raise ValueError("OCR reading was assigned to a different page or method")
        common = {"kind": "ocr_page", "page": number, "method": page["method"],
            "frame_sha256": page["frame_sha256"], "frame_width": page["width"], "frame_height": page["height"],
            "coordinate_system": plan["coordinate_system"]}
        if page["method"] == "embedded_text":
            if page["reading"] != {"text": _pdf(data).pages[number - 1].extract_text() or ""}:
                raise ValueError("Embedded PDF text differs from the original page")
            units.add(page["reading"]["text"], common, "source_text")
        elif page["method"] == "ocr":
            lines = _validate_lines(page["reading"], page["width"], page["height"])
            if not lines:
                units.warn(f"Page {number}: OCR returned no text; this is not proof the page is blank.")
            for index, line in enumerate(lines, 1):
                units.add(line["text"], {**common, "line": index, "box": line["box"],
                    "confidence": line["confidence"], "engine": plan["reader"]["engine"]}, "ocr_unverified")
        else:
            raise ValueError("Unsupported page reading method")
    return {"schema_version": 1, "source_name": plan["source_name"], "source_sha256": sha256(data),
        "media_type": plan["media_type"], "units": units.units, "warnings": units.warnings, "needs_transcription": False}


def render_ocr(data, plan, reader, *, record=None):
    _verify_plan(data, plan)
    files, pages, total = {"source.bin": data, "ocr-config.json": canonical_json(plan)}, [], 0
    pdf = _pdf(data) if plan["media_type"] == "application/pdf" else None
    for selected in plan["pages"]:
        number = selected["page"]
        frame = _render(data, plan, number)
        total += len(frame)
        if total > MAX_FRAMES_BYTES:
            raise ValueError("Rendered OCR pages exceed 20 MiB")
        frame_name = f"frame-{number:04d}.png"
        files[frame_name] = frame
        if record:
            record(frame_name, frame)
        with Image.open(io.BytesIO(frame)) as image:
            width, height = image.size
        reading = {"text": pdf.pages[number - 1].extract_text() or ""} if selected["method"] == "embedded_text" else reader(frame)
        page = {"page": number, "method": selected["method"], "width": width, "height": height,
            "frame_sha256": sha256(frame), "reading": reading}
        raw = canonical_json(page)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Page OCR response exceeds 4 MiB")
        files[f"page-{number:04d}.json"] = raw
        if record:
            record(f"page-{number:04d}.json", raw)
        pages.append(page)
    return _finish(data, plan, pages, files)


def _finish(data, plan, pages, files):
    source = _project(data, plan, pages)
    # Fit the established narrative scalar profile without fabricating a model response.
    files.update({"source.json": canonical_json(source), "assertions.json": canonical_json([]),
        "extraction.json": canonical_json({"source_name": plan["source_name"], "model_deployment": "local-source-reader",
            "response_model": "local-ocr-reader", "instructions_version": 1, "label_policy": "literal",
            "identity_scope": "source_assertion_mention", "semantic_validation": "not_performed"})})
    hashes = {name: {"bytes": len(value), "sha256": sha256(value)} for name, value in files.items()}
    if sum(map(len, files.values())) > MAX_PACKAGE_BYTES:
        raise ValueError("OCR package exceeds 28 MiB")
    manifest = {"schema_version": VERSION, "kind": "ocr_source", "status": "complete",
        "package_id": identifier("ocr", hashes), "assertion_count": 0, "warnings": source["warnings"], "files": hashes}
    return manifest, files


def verify_ocr_payloads(files):
    manifest = json.loads(files["manifest.json"])
    plan = json.loads(files["ocr-config.json"])
    if (manifest.get("kind") != "ocr_source" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != VERSION
            or manifest.get("status") != "complete" or sum(map(len, files.values())) > MAX_PACKAGE_BYTES + 1024 * 1024):
        raise ValueError("Unsupported OCR source package")
    _verify_plan(files["source.bin"], plan)
    selected = [p["page"] for p in plan["pages"]]
    if not 1 <= len(selected) <= MAX_PAGES or selected != sorted(set(selected)) or any(type(p) is not int or not 1 <= p <= 100 for p in selected):
        raise ValueError("Invalid saved OCR page selection")
    expected_names = {"source.bin", "ocr-config.json", "source.json", "assertions.json", "extraction.json", "manifest.json"}
    expected_names |= {f"{prefix}-{p:04d}.{ext}" for p in selected for prefix, ext in [("frame", "png"), ("page", "json")]}
    if set(files) != expected_names:
        raise ValueError("OCR source files differ from the page selection")
    pages, raw_files = [], {k: files[k] for k in ("source.bin", "ocr-config.json")}
    if sum(len(files[f"frame-{p:04d}.png"]) for p in selected) > MAX_FRAMES_BYTES:
        raise ValueError("Saved OCR frames exceed 20 MiB")
    for number in selected:
        frame_name, page_name = f"frame-{number:04d}.png", f"page-{number:04d}.json"
        frame = files[frame_name]
        if len(files[page_name]) > 4 * 1024 * 1024:
            raise ValueError("Saved OCR reading exceeds 4 MiB")
        page = json.loads(files[page_name])
        with Image.open(io.BytesIO(frame)) as image:
            if image.format != "PNG" or image.width * image.height > MAX_PIXELS or image.size != (page["width"], page["height"]):
                raise ValueError("Saved OCR frame geometry differs")
            image.load()
        if len(frame) > MAX_FRAME_BYTES or sha256(frame) != page["frame_sha256"]:
            raise ValueError("Saved OCR frame differs from its reading")
        raw_files.update({frame_name: frame, page_name: files[page_name]})
        pages.append(page)
    expected, rebuilt = _finish(files["source.bin"], plan, pages, raw_files)
    if expected != manifest or rebuilt != {k: v for k, v in files.items() if k != "manifest.json"}:
        raise ValueError("OCR source does not replay from saved readings and page views")
    return manifest
