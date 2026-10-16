# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from html import escape

import pytest
from PIL import Image, features

from sr_adapter import relation_sources
from sr_adapter.relation_sources import read_source


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_R = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"


def _zip(parts: dict[str, bytes | str], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return stream.getvalue()


def _parts(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rels(items: list[tuple[str, str, str]]) -> str:
    return f'<Relationships xmlns="{PACKAGE_R}">' + "".join(
        f'<Relationship Id="{identity}" Type="{R}/{kind}" Target="{target}"/>'
        for identity, kind, target in items
    ) + "</Relationships>"


def _paragraph(runs: list[str]) -> str:
    return "<a:p>" + "".join(f'<a:r><a:t xml:space="preserve">{escape(text)}</a:t></a:r>' for text in runs) + "</a:p>"


def _box(identity: str, paragraphs: list[list[str]]) -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{identity}" name="box"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        "<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>"
        + "".join(_paragraph(runs) for runs in paragraphs) + "</p:txBody></p:sp>"
    )


def _slide(body: str) -> str:
    return (
        f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="0" name="root"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        "<p:grpSpPr/>" + body + "</p:spTree></p:cSld></p:sld>"
    )


def _pptx(bodies: list[str], order: list[int] | None = None) -> bytes:
    order = order if order is not None else list(range(len(bodies)))
    parts = {
        "[Content_Types].xml": (
            f'<Types xmlns="{CONTENT_TYPES}"><Override PartName="/ppt/presentation.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/></Types>'
        ),
        "_rels/.rels": _rels([("rId1", "officeDocument", "ppt/presentation.xml")]),
        "ppt/presentation.xml": (
            f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>'
            + "".join(f'<p:sldId id="{256 + index}" r:id="rId{index + 1}"/>' for index in order)
            + "</p:sldIdLst></p:presentation>"
        ),
        "ppt/_rels/presentation.xml.rels": _rels([
            (f"rId{index + 1}", "slide", f"slides/slide{index + 1}.xml")
            for index in reversed(range(len(bodies)))
        ]),
    }
    for index, body in enumerate(bodies, 1):
        parts[f"ppt/slides/slide{index}.xml"] = _slide(body)
    return _zip(parts)


def _cell(text: str, attrs: str = "") -> str:
    return f"<a:tc {attrs}><a:txBody><a:bodyPr/><a:lstStyle/>{_paragraph([text])}</a:txBody><a:tcPr/></a:tc>"


def _table(rows: list[list[str]], columns: int = 2, identity: str = "8") -> str:
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{identity}" name="table"/>'
        '<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm/><a:graphic>'
        f'<a:graphicData uri="{A}/table"><a:tbl><a:tblPr/><a:tblGrid>'
        + '<a:gridCol w="100"/>' * columns + "</a:tblGrid>"
        + "".join('<a:tr h="100">' + "".join(row) + "</a:tr>" for row in rows)
        + "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )


def _image(format_name: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "white").save(stream, format=format_name)
    return stream.getvalue()


def test_text_preserves_nonempty_physical_lines_and_source_receipt():
    source = "# 組織\r\n\r\n  A は B を所有。  \rC は D を支援。\n".encode()
    result = read_source(source, "relations.MD")
    assert result["schema_version"] == 1
    assert result["source_name"] == "relations.MD"
    assert result["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert result["media_type"] == "text/markdown"
    assert result["needs_transcription"] is False
    assert [unit["text"] for unit in result["units"]] == ["# 組織", "  A は B を所有。  ", "C は D を支援。"]
    assert [unit["locator"] for unit in result["units"]] == [
        {"kind": "text", "line_start": n, "line_end": n} for n in [1, 3, 4]
    ]
    assert all(unit["grounding"] == "source_text" for unit in result["units"])
    assert result == read_source(source, "relations.MD")
    assert len({unit["id"] for unit in result["units"]}) == 3


def test_text_bom_is_explicit_and_invalid_utf8_is_rejected():
    result = read_source(b"\xef\xbb\xbfA owns B", "source.txt")
    assert result["units"][0]["text"] == "A owns B"
    assert "byte order mark" in result["warnings"][0]
    with pytest.raises(ValueError, match="UTF-8"):
        read_source(b"A\xffB", "source.txt")


@pytest.mark.parametrize("data,name", [
    (b"", "source.txt"), (b" \r\n\t", "source.md"), (b"x\x00y", "source.txt"),
    (b"x", ""), (b"x", "data.json"), (b"%PDF-1.7", "data.pdf"), ("text", "source.txt"),
])
def test_invalid_or_unsupported_inputs_reject(data, name):
    with pytest.raises(ValueError):
        read_source(data, name)


def test_limits_reject_instead_of_truncating_and_count_utf8_bytes():
    assert len(read_source(b"x" * 65536, "a.txt")["units"][0]["text"]) == 65536
    with pytest.raises(ValueError, match="64 KiB"):
        read_source(("猫" * 21846).encode(), "a.txt")
    assert len(read_source(b"x\n" * 500, "a.txt")["units"]) == 500
    with pytest.raises(ValueError, match="500 units"):
        read_source(b"x\n" * 501, "a.txt")
    with pytest.raises(ValueError, match="16 MiB"):
        read_source(b"x" * (16 * 1024 * 1024 + 1), "a.txt")


def test_csv_duplicate_headers_empty_values_and_quoted_multiline_keep_positions():
    source = 'name,name,note\r\nA,B,"line one\r\nline two"\r\nC,,"said ""yes"""\r\n'.encode()
    result = read_source(source, "relations.csv")
    first, second = result["units"]
    cells = [
        {"column": 1, "header": "name", "value": "A"},
        {"column": 2, "header": "name", "value": "B"},
        {"column": 3, "header": "note", "value": "line one\r\nline two"},
    ]
    assert json.loads(first["text"]) == {"cells": cells}
    assert first["locator"] == {"kind": "csv", "row": 2, "line_start": 2, "line_end": 3, "cells": cells}
    assert second["locator"]["row"] == 3
    assert second["locator"]["line_start"] == second["locator"]["line_end"] == 4
    assert second["locator"]["cells"][1]["value"] == ""
    assert second["locator"]["cells"][2]["value"] == 'said "yes"'


def test_tsv_multiline_header_and_cells_are_not_guessed_or_coerced():
    result = read_source('"employee\nname"\tmanager\n001\t"A\tB"\n'.encode(), "roles.tsv")
    unit = result["units"][0]
    assert unit["locator"]["line_start"] == 3
    assert unit["locator"]["cells"] == [
        {"column": 1, "header": "employee\nname", "value": "001"},
        {"column": 2, "header": "manager", "value": "A\tB"},
    ]


@pytest.mark.parametrize("source", [b"a,b\n1\n", b"a,b\n1,2,3\n", b'a,b\n1,"unterminated', b"a,b\n", b"a,b\n\n"])
def test_csv_rejects_bad_width_malformed_quoting_or_no_data(source):
    with pytest.raises(ValueError):
        read_source(source, "bad.csv")


def test_pptx_uses_presentation_order_and_combines_runs_with_paragraph_boundaries():
    data = _pptx([
        _box("2", [["second filename"], ["paragraph two"]]),
        _box("3", [["Acme ", "owns ", "Beta"], [], ["Source: register"]]),
    ], order=[1, 0])
    units = read_source(data, "relations.pptx")["units"]
    assert [unit["text"] for unit in units] == ["Acme owns Beta\n\nSource: register", "second filename\nparagraph two"]
    assert units[0]["locator"] == {"kind": "pptx", "slide": 1, "shape_id": "3", "paragraph_start": 1, "paragraph_end": 3}
    assert units[1]["locator"]["slide"] == 2


def test_pptx_text_fields_line_breaks_and_grouped_shapes():
    box = _box("4", [["A"]]).replace("</a:r></a:p>", '</a:r><a:br/><a:fld id="field"><a:t>B</a:t></a:fld></a:p>')
    group = '<p:grpSp><p:nvGrpSpPr><p:cNvPr id="2" name="group"/></p:nvGrpSpPr><p:grpSpPr/>' + box + "</p:grpSp>"
    unit = read_source(_pptx([group]), "group.pptx")["units"][0]
    assert unit["text"] == "A\nB"
    assert unit["locator"]["shape_id"] == "4"


def test_pptx_table_keeps_empty_cells_physical_positions_and_merge_flags():
    data = _pptx([_table([
        [_cell("Parent", 'gridSpan="2"'), _cell("", 'hMerge="1"')],
        [_cell("Child"), _cell("Owner")],
    ])])
    units = read_source(data, "table.pptx")["units"]
    assert [unit["text"] for unit in units] == ["Parent", "", "Child", "Owner"]
    assert [unit["locator"]["cell"] for unit in units] == [
        {"row": 1, "column": 1}, {"row": 1, "column": 2}, {"row": 2, "column": 1}, {"row": 2, "column": 2},
    ]
    assert units[0]["locator"]["column_span"] == 2
    assert units[1]["locator"]["horizontal_merge"] is True
    assert all(unit["locator"]["shape_id"] == "8" for unit in units)


@pytest.mark.parametrize("table", [
    _table([[_cell("A")]]),
    _table([[_cell("A", 'gridSpan="3"'), _cell("B")]]),
    _table([[_cell("A", 'vMerge="maybe"'), _cell("B")]]),
])
def test_pptx_invalid_table_geometry_rejects(table):
    with pytest.raises(ValueError, match="PPTX table"):
        read_source(_pptx([table]), "bad.pptx")


def test_pptx_reports_images_charts_smartart_and_speaker_notes():
    body = (
        _box("2", [["Visible source text"]])
        + '<p:pic><p:nvPicPr><p:cNvPr id="3" name="picture"/></p:nvPicPr></p:pic>'
        + '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="4" name="chart"/></p:nvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="chart"/></a:graphic></p:graphicFrame>'
    )
    parts = _parts(_pptx([body]))
    parts["ppt/slides/_rels/slide1.xml.rels"] = _rels([
        ("rId1", "notesSlide", "../notesSlides/notesSlide1.xml"),
        ("rId2", "diagramData", "../diagrams/data1.xml"),
        ("rId3", "chart", "../charts/chart1.xml"),
    ])
    parts["ppt/notesSlides/notesSlide1.xml"] = b"<notes/>"
    parts["ppt/diagrams/data1.xml"] = b"<diagram/>"
    parts["ppt/charts/chart1.xml"] = b"<chart/>"
    result = read_source(_zip(parts), "mixed.pptx")
    assert [unit["text"] for unit in result["units"]] == ["Visible source text"]
    warning = " ".join(result["warnings"])
    assert all(word in warning for word in ["images", "SmartArt", "charts", "speaker notes"])


def test_image_only_pptx_rejects_instead_of_empty_success():
    with pytest.raises(ValueError, match="no extractable text"):
        read_source(_pptx(['<p:pic><p:nvPicPr><p:cNvPr id="2" name="image"/></p:nvPicPr></p:pic>']), "image.pptx")


def test_strict_ooxml_namespaces_are_supported():
    parts = _parts(_pptx([_box("2", [["Strict source"]])]))
    replacements = {
        P: "http://purl.oclc.org/ooxml/presentationml/main",
        A: "http://purl.oclc.org/ooxml/drawingml/main",
        R: "http://purl.oclc.org/ooxml/officeDocument/relationships",
    }
    for name, data in parts.items():
        for old, new in replacements.items():
            data = data.replace(old.encode(), new.encode())
        parts[name] = data
    assert read_source(_zip(parts), "strict.pptx")["units"][0]["text"] == "Strict source"


@pytest.mark.parametrize("target,mode", [("https://example.invalid/private", "External"), ("https://example.invalid/private", "Internal"), ("../../../../escape.xml", "Internal")])
def test_pptx_external_or_escaping_relationships_reject_even_when_unused(target, mode):
    parts = _parts(_pptx([_box("2", [["Visible"]])]))
    parts["ppt/slides/_rels/slide1.xml.rels"] = (
        f'<Relationships xmlns="{PACKAGE_R}"><Relationship Id="rId1" Type="{R}/hyperlink" '
        f'Target="{target}" TargetMode="{mode}"/></Relationships>'
    ).encode()
    with pytest.raises(ValueError, match="External|escapes"):
        read_source(_zip(parts), "external.pptx")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_pptx_dtd_entity_rejection_includes_utf16(encoding):
    parts = _parts(_pptx([_box("2", [["Visible"]])]))
    original = parts["ppt/presentation.xml"].decode()
    parts["ppt/presentation.xml"] = (
        f'<?xml version="1.0" encoding="{encoding}"?><!DOCTYPE p:presentation [<!ENTITY hidden "secret">]>' + original
    ).encode(encoding)
    with pytest.raises(ValueError, match="document type"):
        read_source(_zip(parts), "entity.pptx")


def test_pptx_duplicate_entries_and_unsafe_names_reject():
    parts = _parts(_pptx([_box("2", [["Visible"]])]))
    stream = io.BytesIO(_zip(parts))
    with zipfile.ZipFile(stream, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("ppt/presentation.xml", parts["ppt/presentation.xml"])
    with pytest.raises(ValueError, match="duplicate entry"):
        read_source(stream.getvalue(), "duplicate.pptx")
    parts["../outside.xml"] = b"<x/>"
    with pytest.raises(ValueError, match="unsafe entry"):
        read_source(_zip(parts), "unsafe.pptx")


def test_pptx_zip_expansion_bound_is_checked_before_parsing():
    parts = _parts(_pptx([_box("2", [["Visible"]])]))
    parts["ppt/media/bomb.bin"] = b"0" * (32 * 1024 * 1024)
    data = _zip(parts)
    assert len(data) < 16 * 1024 * 1024
    with pytest.raises(ValueError, match="32 MiB"):
        read_source(data, "bounded.pptx")


def test_pptx_duplicate_shapes_and_missing_slide_targets_reject():
    with pytest.raises(ValueError, match="shape identifiers"):
        read_source(_pptx([_box("2", [["A"]]) + _box("2", [["B"]])]), "duplicate.pptx")
    parts = _parts(_pptx([_box("2", [["A"]])]))
    del parts["ppt/slides/slide1.xml"]
    with pytest.raises(ValueError, match="target is missing"):
        read_source(_zip(parts), "missing.pptx")


def test_pptx_crc_damage_and_invalid_zip_reject():
    parts = _parts(_pptx([_box("2", [["A"]])]))
    data = bytearray(_zip(parts, compression=zipfile.ZIP_STORED))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entry = archive.getinfo("ppt/slides/slide1.xml")
        offset = entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra)
    data[offset] ^= 1
    for source in [bytes(data), b"not a ZIP"]:
        with pytest.raises(ValueError, match="ZIP"):
            read_source(source, "invalid.pptx")


@pytest.mark.parametrize("format_name,extension", [("PNG", "png"), ("JPEG", "jpg"), ("JPEG", "jpeg"), ("WEBP", "webp")])
def test_images_require_transcription_but_do_not_perform_ocr(format_name, extension):
    if format_name == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow was built without WebP support")
    data = _image(format_name)
    result = read_source(data, "source." + extension)
    assert result["units"] == []
    assert result["needs_transcription"] is True
    assert result["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert "not been transcribed" in result["warnings"][0]


def test_transcription_remains_unverified_without_bbox_claims():
    data = _image()
    result = read_source(data, "source.png", transcription={"lines": ["A owns B", "Source: public register"]})
    assert result["needs_transcription"] is False
    assert [unit["locator"] for unit in result["units"]] == [
        {"kind": "image", "frame": 1, "transcript_line": 1},
        {"kind": "image", "frame": 1, "transcript_line": 2},
    ]
    assert all(unit["grounding"] == "model_transcription_unverified" for unit in result["units"])
    assert "not pixel bounding boxes" in " ".join(result["warnings"])
    alternate = read_source(data, "source.png", transcription={"lines": ["A does not own B"]})
    assert result["units"][0]["id"] != alternate["units"][0]["id"]


@pytest.mark.parametrize("transcription", [{}, {"lines": []}, {"lines": "text"}, {"lines": [""]}, {"lines": [42]}, {"lines": ["a\nb"]}, {"lines": ["x"], "bbox": [0, 0, 1, 1]}])
def test_invalid_transcriptions_reject(transcription):
    with pytest.raises(ValueError, match="transcription"):
        read_source(_image(), "source.png", transcription=transcription)


def test_transcription_is_rejected_for_text_sources():
    with pytest.raises(ValueError, match="only for image"):
        read_source(b"original", "source.txt", transcription={"lines": ["replacement"]})


def test_animated_images_mismatched_formats_and_truncated_images_reject():
    stream = io.BytesIO()
    Image.new("RGB", (3, 3), "red").save(stream, format="PNG", save_all=True,
                                           append_images=[Image.new("RGB", (3, 3), "blue")], duration=100, loop=0)
    with pytest.raises(ValueError, match="single-frame"):
        read_source(stream.getvalue(), "animated.png")
    for data, name in [(_image("JPEG"), "wrong.png"), (_image()[:-20], "broken.png")]:
        with pytest.raises(ValueError):
            read_source(data, name)


def test_image_pixel_bound_is_independent_of_pillow_global_configuration(monkeypatch):
    original_limit = Image.MAX_IMAGE_PIXELS
    monkeypatch.setattr(relation_sources, "_MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="dimension limit"):
        read_source(_image(), "large.png")
    assert Image.MAX_IMAGE_PIXELS == original_limit
