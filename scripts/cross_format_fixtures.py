"""Deterministic, synthetic Japanese banking documents for a bounded live probe.

This module only creates raw PDF, DOCX and PNG files. It performs no extraction,
OCR, model invocation or network access. Expectations are handwritten gold, not
claims that any extraction model has satisfied them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas


PROFILES = {
    "A": {
        "title": "資料A：融資運用手順（合成資料）",
        "units": [
            "資料Aは融資運用部の現行手順である。対象はLoanSystem L-01の申請A。適用対象期間は2026-05-01から2026-06-30まで。",
            "資料Aの「Atlas」は融資審査システムの名称であり、人の組織ではない。",
            "融資運用部の記載：融資額が100万円以下の場合、審査部による審査が必要である。",
            "融資運用部の記載：反社確認が未完了の場合、融資を実行してはならない。",
            "資料Bとの審査担当および未完了時の実行可否の不一致は未解決であり、優先資料は未決定である。最終承認者の氏名は記載されていない。",
        ],
        "expected": {
            "atlas_kind": "loan_review_system",
            "source_attribution": "融資運用部",
            "document_status": "current_procedure",
            "threshold": "100万円以下",
            "reviewer": "審査部",
            "incomplete_check_execution": "prohibited",
            "emergency_condition": False,
            "draft_qualifier": False,
            "final_approver_name": None,
        },
    },
    "B": {
        "title": "資料B：監査部改定案（未承認・合成資料）",
        "units": [
            "資料Bは監査部の改定案であり、未承認である。対象はLoanSystem L-01の申請A。適用対象期間は2026-05-01から2026-06-30まで。",
            "資料Bの「Atlas」は監査審査チームの名称であり、システムではない。",
            "監査部の改定案：融資額が100万円以下の場合、部長による審査が必要である。この案は未承認である。",
            "監査部の改定案：緊急時に限り、反社確認が未完了でも融資実行を許可する。この例外案は未承認である。",
            "資料Aとの審査担当および未完了時の実行可否の不一致は未解決であり、優先資料は未決定である。最終承認者の氏名は記載されていない。",
        ],
        "expected": {
            "atlas_kind": "audit_review_team",
            "source_attribution": "監査部",
            "document_status": "unapproved_draft",
            "threshold": "100万円以下",
            "reviewer": "部長",
            "incomplete_check_execution": "permitted_in_draft_only",
            "emergency_condition": True,
            "draft_qualifier": True,
            "final_approver_name": None,
        },
    },
}

EXPECTATIONS = [
    {
        "id": "same_name_nonidentity",
        "requirement": "Atlas is a system in A and a team in B; its shared label does not establish shared identity.",
        "evidence_units": {"A": ["A2"], "B": ["B2"]},
    },
    {
        "id": "same_target_conflicting_reviewers",
        "requirement": "For LoanSystem L-01/application A in the same period and at 100万円以下, retain A's 審査部 and B's 部長 with their separate source attributions.",
        "evidence_units": {"A": ["A1", "A3"], "B": ["B1", "B3"]},
    },
    {
        "id": "source_local_conditions",
        "requirement": "A prohibits execution when the check is incomplete; B proposes permission only in an emergency. Do not drop, exchange or globally apply these conditions.",
        "evidence_units": {"A": ["A4"], "B": ["B4"]},
    },
    {
        "id": "conflict_no_winner",
        "requirement": "The sources explicitly leave the disagreement and document priority unresolved; do not select a winner or invent a reconciliation.",
        "evidence_units": {"A": ["A5"], "B": ["B5"]},
    },
    {
        "id": "draft_qualifier",
        "requirement": "B's reviewer rule and emergency exception remain attributed, unapproved draft proposals; do not promote either to an approved operative policy.",
        "evidence_units": {"B": ["B1", "B3", "B4"]},
    },
    {
        "id": "missing_final_approver",
        "requirement": "The named final approver is unknown in both documents; neither the reviewer nor the document author supplies that missing name.",
        "evidence_units": {"A": ["A5"], "B": ["B5"]},
    },
    {
        "id": "evidence_survives_reuse",
        "requirement": "Every reused conclusion must retain the source entry, immutable snapshot, claim/evidence references and original locator; exact anchors refer to saved readings, not global entity IDs.",
        "evidence_units": {"A": ["A1", "A2", "A3", "A4", "A5"], "B": ["B1", "B2", "B3", "B4", "B5"]},
    },
]

PDF_FONT = "HeiseiKakuGo-W5"
IMAGE_SIZE = (1600, 2263)
FIXED_TIME = datetime(2026, 1, 1)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _wrap(text, measure, max_width):
    """Wrap without changing, normalizing or dropping any source characters."""
    lines, line = [], ""
    for char in text:
        if line and measure(line + char) > max_width:
            lines.append(line)
            line = ""
        line += char
    if line:
        lines.append(line)
    return lines


def _docx_bytes(title, units):
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin = section.bottom_margin = Mm(20)
    section.left_margin = section.right_margin = Mm(20)
    properties = document.core_properties
    properties.title = title
    properties.subject = "Synthetic cross-format ontology evaluation"
    properties.author = "Synthetic fixture generator"
    properties.last_modified_by = "Synthetic fixture generator"
    properties.created = properties.modified = FIXED_TIME
    properties.revision = 1
    normal = document.styles["Normal"]
    normal.font.name = "Hiragino Kaku Gothic ProN"
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(16)
    normal.paragraph_format.line_spacing = 1.4
    fonts = normal.element.get_or_add_rPr().find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        normal.element.get_or_add_rPr().append(fonts)
    fonts.set(qn("w:eastAsia"), "Hiragino Kaku Gothic ProN")
    paragraph = document.add_paragraph(title)
    paragraph.runs[0].bold = True
    paragraph.runs[0].font.size = Pt(16)
    for text in units:
        document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    # python-docx's package timestamps follow wall-clock time. Canonicalize both
    # entry order and metadata, leaving every generated XML/asset byte intact.
    result = BytesIO()
    with ZipFile(BytesIO(buffer.getvalue())) as original, ZipFile(result, "w") as stable:
        for name in sorted(original.namelist()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            stable.writestr(info, original.read(name), compress_type=ZIP_DEFLATED, compresslevel=9)
    return result.getvalue()


def _pdf_bytes(title, units):
    if PDF_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(PDF_FONT))
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle(title)
    canvas.setAuthor("Synthetic fixture generator")
    width, height = A4
    left, right, y = 46, width - 46, height - 54
    for text, size, gap in [(title, 16, 24)] + [(text, 12, 22) for text in units]:
        canvas.setFont(PDF_FONT, size)
        lines = _wrap(text, lambda value: pdfmetrics.stringWidth(value, PDF_FONT, size), right - left)
        for line in lines:
            canvas.drawString(left, y, line)
            y -= size * 1.6
        y -= gap
    if y < 46:
        raise ValueError("Synthetic PDF content exceeds its single page")
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _png_bytes(title, units, font_path):
    font = ImageFont.truetype(str(font_path), 36)
    title_font = ImageFont.truetype(str(font_path), 46)
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    left, y = 84, 88
    for text, selected_font, line_height, gap in [(title, title_font, 68, 48)] + [
        (text, font, 58, 48) for text in units
    ]:
        lines = _wrap(text, lambda value: draw.textlength(value, font=selected_font), IMAGE_SIZE[0] - 2 * left)
        for line in lines:
            draw.text((left, y), line, font=selected_font, fill="#17212a")
            y += line_height
        y += gap
    if y > IMAGE_SIZE[1] - 70:
        raise ValueError("Synthetic image content exceeds its single page")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def build_fixtures(root, font_path):
    """Write six raw inputs and a gold manifest; return absolute source paths.

    Source bytes are repeatable with the same generator, dependencies and image
    font. The manifest records renderer versions and font hash to make that
    dependency explicit. It is not a promise of byte parity across library/font
    versions. Existing files at these fixed fixture names are replaced.
    """
    import docx
    import PIL
    import reportlab

    root = Path(root).resolve()
    font_path = Path(font_path).resolve(strict=True)
    root.mkdir(parents=True, exist_ok=True)
    profiles = deepcopy(PROFILES)
    sources = []
    for profile, definition in profiles.items():
        texts = definition.pop("units")
        definition["units"] = [{"id": f"{profile}{i}", "text": text} for i, text in enumerate(texts, 1)]
        definition["expected"].update(
            target_system="LoanSystem L-01", application="申請A",
            period={"from": "2026-05-01", "until_inclusive": "2026-06-30", "timezone": None},
            conflict_status="unresolved", selected_winner=None,
        )
        payloads = {
            "pdf": _pdf_bytes(definition["title"], texts),
            "docx": _docx_bytes(definition["title"], texts),
            "png": _png_bytes(definition["title"], texts, font_path),
        }
        for format, data in payloads.items():
            path = root / f"atlas-policy-{profile.lower()}.{format}"
            path.write_bytes(data)
            sources.append({
                "profile": profile, "format": format, "path": str(path), "source_name": path.name,
                "sha256": _sha256(data), "bytes": len(data),
                "title": definition["title"], "text_units": deepcopy(definition["units"]),
                "expected": deepcopy(definition["expected"]),
            })
    manifest = {
        "version": 1, "corpus_id": "atlas-cross-format-source-scope-v1", "synthetic": True,
        "profiles": profiles, "sources": sources, "expectations": deepcopy(EXPECTATIONS),
        "rendering": {
            "reportlab_version": reportlab.Version, "python_docx_version": docx.__version__,
            "pillow_version": PIL.__version__, "pdf_font": PDF_FONT,
            "image_font_path": str(font_path), "image_font_sha256": _sha256(font_path.read_bytes()),
            "image_dimensions": list(IMAGE_SIZE), "pdf_pages": 1,
            "docx_zip_timestamp": "1980-01-01T00:00:00", "model_calls": 0,
        },
        "claim_ceiling": "Handwritten synthetic expectations; no extraction or semantic accuracy has been measured.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="Japanese-capable font for the PNG sources")
    args = parser.parse_args()
    result = build_fixtures(args.out, args.font)
    print(json.dumps({"manifest": str(args.out.resolve() / "manifest.json"), "sources": len(result["sources"]), "model_calls": 0}))
