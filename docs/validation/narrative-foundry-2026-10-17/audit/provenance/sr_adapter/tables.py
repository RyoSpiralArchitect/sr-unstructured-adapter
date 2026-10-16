"""Opt-in ruled-PDF table extraction with source coordinates and explicit spans."""

from __future__ import annotations

import hashlib
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import BBox, Block, Provenance


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identifier(kind: str, *parts) -> str:
    return f"urn:sr:{kind}:" + hashlib.sha256(canonical_json(parts)).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cell(Record):
    id: str
    row: int = Field(ge=0, strict=True)
    column: int = Field(ge=0, strict=True)
    row_span: int = Field(default=1, ge=1, strict=True)
    column_span: int = Field(default=1, ge=1, strict=True)
    text: str
    bbox: BBox


class SourceTable(Record):
    id: str
    page: int = Field(ge=1, strict=True)
    page_width: float = Field(gt=0, allow_inf_nan=False)
    page_height: float = Field(gt=0, allow_inf_nan=False)
    bbox: BBox
    rows: int = Field(ge=1, strict=True)
    columns: int = Field(ge=1, strict=True)
    cells: list[Cell] = Field(min_length=1)
    coordinate_system: Literal["pdf_points_top_left"] = "pdf_points_top_left"
    method: Literal["pdfplumber_lines"] = "pdfplumber_lines"

    @model_validator(mode="after")
    def valid_grid(self):
        if self.rows * self.columns > 100000:
            raise ValueError("Table grid exceeds supported size")
        occupied = set()
        for cell in self.cells:
            if cell.row + cell.row_span > self.rows or cell.column + cell.column_span > self.columns:
                raise ValueError("Cell span exceeds table grid")
            if (cell.bbox.x0 < self.bbox.x0 or cell.bbox.y0 < self.bbox.y0 or
                    cell.bbox.x1 > self.bbox.x1 or cell.bbox.y1 > self.bbox.y1):
                raise ValueError("Cell lies outside its table")
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    if (row, column) in occupied:
                        raise ValueError("Overlapping cell spans")
                    occupied.add((row, column))
        return self


class TableDocument(Record):
    schema_version: Literal["1"] = "1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_name: str
    extractor_version: str
    tables: list[SourceTable]
    warnings: list[str] = Field(default_factory=list)


def extract_pdf_tables(path: str | Path, *, max_pages: int = 100, max_cells: int = 100000) -> TableDocument:
    """Retain physical cells; no OCR, header guessing, or merged-cell fill-down.

    Detection uses drawn horizontal/vertical rules. A detected grid remains an
    extraction hypothesis, not a semantic or accuracy certificate.
    """
    if max_pages < 1 or max_cells < 1:
        raise ValueError("Table limits must be positive")
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF tables require sr-unstructured-adapter[tables]") from exc
    source = Path(path)
    source_bytes = source.read_bytes()
    checksum = hashlib.sha256(source_bytes).hexdigest()
    tables = []
    warnings = []
    count = 0
    with pdfplumber.open(BytesIO(source_bytes)) as pdf:
        if len(pdf.pages) > max_pages:
            raise ValueError("PDF exceeds max_pages; no partial table result returned")
        for page_number, page in enumerate(pdf.pages, 1):
            detected = page.find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
            if not detected:
                warnings.append(f"page {page_number}: no ruled table detected; no OCR performed")
            for index, table in enumerate(sorted(detected, key=lambda t: (t.bbox[1], t.bbox[0]))):
                boxes = sorted(set(table.cells), key=lambda b: (b[1], b[0]))
                count += len(boxes)
                if count > max_cells:
                    raise ValueError("PDF exceeds max_cells; no partial table result returned")
                xs = sorted({box[i] for box in boxes for i in (0, 2)})
                ys = sorted({box[i] for box in boxes for i in (1, 3)})
                grid = table.extract()
                text_by_box = {box: grid[r][c] or "" for r, row in enumerate(table.rows)
                               for c, box in enumerate(row.cells) if box is not None}
                table_id = identifier("table", checksum, page_number, index, table.bbox)
                cells = []
                for box in boxes:
                    row, column = ys.index(box[1]), xs.index(box[0])
                    cells.append(Cell(id=identifier("cell", table_id, row, column), row=row, column=column,
                                      row_span=ys.index(box[3]) - row, column_span=xs.index(box[2]) - column,
                                      text=text_by_box[box], bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])))
                box = table.bbox
                tables.append(SourceTable(id=table_id, page=page_number, page_width=page.width,
                    page_height=page.height, bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                    rows=len(ys) - 1, columns=len(xs) - 1, cells=cells))
    return TableDocument(source_sha256=checksum, source_name=source.name,
                         extractor_version=importlib.metadata.version("pdfplumber"), tables=tables, warnings=warnings)


def table_blocks(document: TableDocument) -> list[Block]:
    """Adapter blocks for callers that opt into table evidence."""
    return [Block(id=table.id, type="table", text="\n".join(cell.text for cell in table.cells),
                  attrs={"table": table.model_dump(mode="json"), "source_sha256": document.source_sha256,
                         "layout_geometry": "source", "structure_inference": "ruled_grid"},
                  prov=Provenance(uri=document.source_name, page=table.page - 1, bbox=table.bbox),
                  confidence=0.7) for table in document.tables]
