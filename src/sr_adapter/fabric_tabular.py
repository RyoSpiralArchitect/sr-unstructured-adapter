"""Scalar-column Fabric snapshots of selected native PowerPoint slides.

This is a structural extraction, not a semantic interpretation of a diagram.
Only selected slide XML is copied to the snapshot; the original deck stays local.
The original document hash is provenance, not proof of a remote SharePoint version.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import stat
import zipfile

from .fabric import MAX_EXPORT_BYTES, _plan, rows_digest
from .knowledge import sha256
from .knowledge_store import atomic_json
from .relation_sources import _A, _NS, _P, _R, _is_relationship, _relationship_target, _xml
from .tables import canonical_json


COMMON = "document_sha256 source_name slide page_label"
COLUMNS = {
    "pages": (COMMON + " title hidden width_emu height_emu object_count connector_count "
              "attached_count partial_count unattached_count dangling_count unsupported_count").split(),
    "objects": (COMMON + " object_id shape_id parent_object_id kind name text hidden "
                "x_emu y_emu width_emu height_emu rotation_60000deg flip_h flip_v "
                "child_x_emu child_y_emu child_width_emu child_height_emu fill_kind fill_value").split(),
    "connections": (COMMON + " connector_id start_shape_id end_shape_id start_object_id end_object_id "
                    "start_text end_text start_site_index end_site_index endpoint_status "
                    "head_arrow tail_arrow interpretation").split(),
    "cells": (COMMON + " table_object_id row_index column_index row_span column_span "
              "horizontal_merge vertical_merge text").split(),
}
INTS = set(("slide width_emu height_emu object_count connector_count attached_count partial_count "
            "unattached_count dangling_count unsupported_count x_emu y_emu rotation_60000deg "
            "child_x_emu child_y_emu child_width_emu child_height_emu start_site_index end_site_index "
            "row_index column_index row_span column_span").split())
BOOLS = {"hidden", "flip_h", "flip_v", "horizontal_merge", "vertical_merge"}
MAX_DECK_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024


def _schema(kind):
    import pyarrow as pa
    return pa.schema([pa.field(c, pa.int64() if c in INTS else pa.bool_() if c in BOOLS else pa.string())
                      for c in COLUMNS[kind]])


def _integer(value):
    if value is None:
        return None
    if not re.fullmatch(r"-?\d+", value) or not -(2**63) <= int(value) < 2**63:
        raise ValueError("Invalid OOXML integer")
    return int(value)


def _boolean(value, default=False):
    if value is None:
        return default
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("Invalid OOXML boolean")
    return value in {"1", "true"}


def _select(source, slides, page_labels):
    with Path(source).open("rb") as stream:
        data = stream.read(MAX_DECK_BYTES + 1)
    if len(data) > MAX_DECK_BYTES:
        raise ValueError("Tabular PPTX input exceeds 64 MiB")
    if (not slides or len(slides) > 20 or len(set(slides)) != len(slides)
            or any(type(n) is not int or n < 1 for n in slides)):
        raise ValueError("Select 1 to 20 distinct one-based slide numbers")
    if page_labels is not None and (len(page_labels) != len(slides)
                                  or any(not isinstance(v, str) or not v.strip() for v in page_labels)):
        raise ValueError("Page labels must match selected slides")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 4096 or sum(e.file_size for e in entries) > MAX_EXPANDED_BYTES:
            raise ValueError("PPTX archive exceeds bounded entry/expanded size limits")
        names = set()
        for entry in entries:
            name = entry.filename.rstrip("/")
            if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                    or any(p in {"", ".", ".."} for p in name.split("/")) or name in names
                    or entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
                raise ValueError("Unsafe or duplicate PPTX entry")
            names.add(name)
        presentation = _xml(archive.read("ppt/presentation.xml"))
        order = presentation.find("p:sldIdLst", _NS)
        size = presentation.find("p:sldSz", _NS)
        if order is None or size is None or any(n > len(order) for n in slides):
            raise ValueError("Selected slide is absent or presentation size is missing")
        relationships = {}
        for rel in _xml(archive.read("ppt/_rels/presentation.xml.rels")):
            identity = rel.get("Id")
            if not identity or identity in relationships:
                raise ValueError("Duplicate or absent presentation relationship ID")
            relationships[identity] = rel
        selected, files = [], {}
        for index, number in enumerate(slides):
            identity = order[number - 1].get("{" + _R + "}id")
            rel = relationships.get(identity)
            if (rel is None or rel.get("TargetMode", "Internal") != "Internal"
                    or not _is_relationship(rel.get("Type", ""), "slide")):
                raise ValueError("Selected slide must have an internal slide relationship")
            part = _relationship_target("ppt/presentation.xml", rel.get("Target", ""))
            content = archive.read(part)
            _xml(content)
            name = f"source/slide-{number}.xml"
            files[name] = content
            selected.append({"slide": number, "page_label": page_labels[index] if page_labels else None,
                             "part": part, "file": name})
    config = {"schema_version": 1, "source_name": Path(source).name, "document_sha256": sha256(data),
              "source_bytes": len(data), "slide_count": len(order), "width_emu": _integer(size.get("cx")),
              "height_emu": _integer(size.get("cy")), "selection": selected,
              "scope": "selected_native_slide_xml; notes, images, masters and visual interpretation excluded"}
    return config, files


def _text(shape):
    body = shape.find("p:txBody", _NS)
    if body is None:
        body = shape.find("a:txBody", _NS)
    if body is None:
        return ""
    paragraphs = []
    for paragraph in body.findall("a:p", _NS):
        pieces = []
        for child in paragraph:
            if child.tag in {"{" + _A + "}r", "{" + _A + "}fld"}:
                pieces.extend(n.text or "" for n in child.findall("a:t", _NS))
            elif child.tag == "{" + _A + "}br":
                pieces.append("\n")
        paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def _projection(config, files):
    if (config.get("schema_version") != 1 or not re.fullmatch(r"[a-f0-9]{64}", config["document_sha256"])
            or not 1 <= len(config["selection"]) <= 20):
        raise ValueError("Invalid tabular source selection")
    selection = config["selection"]
    numbers = [s["slide"] for s in selection]
    if len(set(numbers)) != len(numbers) or any(type(n) is not int or not 1 <= n <= config["slide_count"] for n in numbers):
        raise ValueError("Invalid selected slide numbers")
    if set(files) != {f"source/slide-{n}.xml" for n in numbers}:
        raise ValueError("Unexpected selected slide files")
    rows = {kind: [] for kind in COLUMNS}
    for selected in selection:
        number = selected["slide"]
        if selected["file"] != f"source/slide-{number}.xml":
            raise ValueError("Slide file differs from its selection")
        common = {k: config[k] for k in ("document_sha256", "source_name")}
        common.update(slide=number, page_label=selected["page_label"])
        root = _xml(files[selected["file"]])
        tree = root.find("p:cSld/p:spTree", _NS)
        if root.tag != "{" + _P + "}sld" or tree is None:
            raise ValueError("Invalid slide XML")
        objects, connectors, cells, unsupported = {}, [], [], []

        def walk(container, parent=None):
            for shape in container:
                kind = shape.tag.rsplit("}", 1)[-1]
                if kind in {"nvGrpSpPr", "grpSpPr", "extLst"}:
                    continue
                if shape.tag not in {"{" + _P + "}" + v for v in ("sp", "cxnSp", "grpSp", "graphicFrame", "pic")}:
                    unsupported.append(kind)
                    continue
                nv = shape.find(".//p:cNvPr", _NS)
                sid = nv.get("id") if nv is not None else None
                if not sid or sid in objects:
                    raise ValueError("Missing or duplicate shape ID")
                if len(objects) >= 5000:
                    raise ValueError("Selected slide exceeds 5000 objects")
                oid = f"{config['document_sha256']}:slide:{number}:shape:{sid}"
                row = {c: None for c in COLUMNS["objects"]}
                row.update(common, object_id=oid, shape_id=sid, parent_object_id=parent, kind=kind,
                           name=nv.get("name", ""), text=_text(shape), hidden=_boolean(nv.get("hidden")))
                transform = next((v for path in ("p:spPr/a:xfrm", "p:grpSpPr/a:xfrm", "p:xfrm")
                                  if (v := shape.find(path, _NS)) is not None), None)
                if transform is not None:
                    row.update(rotation_60000deg=_integer(transform.get("rot", "0")),
                               flip_h=_boolean(transform.get("flipH")), flip_v=_boolean(transform.get("flipV")))
                    for tag, attrs, columns in (("off", ("x", "y"), ("x_emu", "y_emu")),
                            ("ext", ("cx", "cy"), ("width_emu", "height_emu")),
                            ("chOff", ("x", "y"), ("child_x_emu", "child_y_emu")),
                            ("chExt", ("cx", "cy"), ("child_width_emu", "child_height_emu"))):
                        item = transform.find("a:" + tag, _NS)
                        if item is not None:
                            row.update({col: _integer(item.get(attr)) for col, attr in zip(columns, attrs)})
                fill = shape.find("p:spPr/a:solidFill", _NS)
                if fill is not None and len(fill):
                    row.update(fill_kind=fill[0].tag.rsplit("}", 1)[-1], fill_value=fill[0].get("val"))
                objects[sid] = row
                if kind == "cxnSp":
                    connectors.append(shape)
                for table in shape.findall("a:graphic/a:graphicData/a:tbl", _NS):
                    for r, tr in enumerate(table.findall("a:tr", _NS), 1):
                        for c, cell in enumerate(tr.findall("a:tc", _NS), 1):
                            rs, cs = _integer(cell.get("rowSpan", "1")), _integer(cell.get("gridSpan", "1"))
                            if rs < 1 or cs < 1:
                                raise ValueError("Invalid table span")
                            cells.append(dict(common, table_object_id=oid, row_index=r, column_index=c,
                                row_span=rs, column_span=cs, horizontal_merge=_boolean(cell.get("hMerge")),
                                vertical_merge=_boolean(cell.get("vMerge")), text=_text(cell)))
                if kind == "grpSp":
                    walk(shape, oid)

        walk(tree)
        connections = []
        for shape in connectors:
            sid = shape.find("p:nvCxnSpPr/p:cNvPr", _NS).get("id")
            row = {c: None for c in COLUMNS["connections"]}
            row.update(common, connector_id=objects[sid]["object_id"], interpretation="native_connector_only")
            for end, tag in (("start", "stCxn"), ("end", "endCxn")):
                ref = shape.find("p:nvCxnSpPr/p:cNvCxnSpPr/a:" + tag, _NS)
                if ref is not None:
                    refid = ref.get("id")
                    row[end + "_shape_id"] = refid
                    row[end + "_site_index"] = _integer(ref.get("idx"))
                    if refid in objects:
                        row[end + "_object_id"] = objects[refid]["object_id"]
                        row[end + "_text"] = objects[refid]["text"]
            refs = [row["start_shape_id"], row["end_shape_id"]]
            row["endpoint_status"] = ("dangling" if any(ref is not None and ref not in objects for ref in refs)
                else "attached" if all(refs) else "partial" if any(refs) else "unattached")
            for end in ("head", "tail"):
                arrow = shape.find("p:spPr/a:ln/a:" + end + "End", _NS)
                row[end + "_arrow"] = arrow.get("type", "none") if arrow is not None else None
            connections.append(row)
        title = ""
        for shape in tree.findall("p:sp", _NS):
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", _NS)
            if placeholder is not None and placeholder.get("type") in {"title", "ctrTitle"}:
                title = _text(shape)
                break
        rows["pages"].append(dict(common, title=title, hidden=not _boolean(root.get("show"), True),
            width_emu=config["width_emu"], height_emu=config["height_emu"], object_count=len(objects),
            connector_count=len(connections), unsupported_count=len(unsupported),
            **{status + "_count": sum(c["endpoint_status"] == status for c in connections)
               for status in ("attached", "partial", "unattached", "dangling")}))
        rows["objects"].extend(objects.values())
        rows["connections"].extend(connections)
        rows["cells"].extend(cells)
    for group in rows.values():
        group.sort(key=canonical_json)
    return rows


def _snapshot(config, source, rows):
    return sha256(canonical_json({"mode": "pptx_tabular_v1", "selection": config,
        "source": {n: sha256(v) for n, v in source.items()},
        "tables": {k: rows_digest(v) for k, v in rows.items()}}))


def _tabular_plan(snapshot, rows, target):
    # Keep empty Parquet schemas in the export, but do not submit an empty load.
    return _plan(snapshot, {kind: group for kind, group in rows.items() if group}, target)


def export_tabular(source, destination, *, target, slides, page_labels=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config, selected = _select(source, slides, page_labels)
    rows = _projection(config, selected)
    snapshot = _snapshot(config, selected, rows)
    files = {**selected, "selection.json": canonical_json(config),
             "target.json": canonical_json(target.model_dump(mode="json")),
             "deployment-plan.json": canonical_json(_tabular_plan(snapshot, rows, target))}
    for kind, group in rows.items():
        output = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(group, schema=_schema(kind)), output)
        files[f"tables/{kind}.parquet"] = output.getvalue()
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Tabular snapshot exceeds 64 MiB")
    manifest = {"schema_version": 2, "mode": "pptx_tabular", "status": "complete", "snapshot_id": snapshot,
        "tables": {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()},
        "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in files.items()}}
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    atomic_json(root / "manifest.json", manifest)
    verify_tabular(root, target=target)
    return manifest


def verify_tabular(folder, *, target):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(folder).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 2 or manifest.get("mode") != "pptx_tabular" or manifest.get("status") != "complete":
        raise ValueError("Incomplete tabular snapshot")
    files = {}
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Tabular file escapes export")
        data = path.read_bytes()
        if expected != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError("Tabular file hash/size mismatch")
        files[name] = data
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ValueError("Tabular snapshot exceeds 64 MiB")
    config = json.loads(files["selection.json"])
    selected = {n: v for n, v in files.items() if n.startswith("source/")}
    rows = _projection(config, selected)
    snapshot = _snapshot(config, selected, rows)
    expected_names = {*selected, "selection.json", "target.json", "deployment-plan.json"}
    expected_names.update(f"tables/{k}.parquet" for k in rows)
    if set(files) != expected_names or manifest["snapshot_id"] != snapshot:
        raise ValueError("Tabular snapshot identity/files mismatch")
    if json.loads(files["target.json"]) != target.model_dump(mode="json"):
        raise ValueError("Pass the explicit target matching this export")
    plan = _tabular_plan(snapshot, rows, target)
    if json.loads(files["deployment-plan.json"]) != plan:
        raise ValueError("Tabular plan differs from verified projection")
    if manifest["tables"] != {k: {"rows": len(v), "rows_sha256": rows_digest(v)} for k, v in rows.items()}:
        raise ValueError("Tabular row manifest mismatch")
    for kind, group in rows.items():
        table = pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"]))
        if table.schema != _schema(kind) or table.to_pylist() != group:
            raise ValueError("Tabular Parquet differs from selected source XML")
    files["manifest.json"] = manifest_bytes
    return manifest, files, plan
