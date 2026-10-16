"""Shared real-PDF fixtures for the opt-in knowledge path."""

import pytest
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from sr_adapter.knowledge import OntologyMapping, extract_knowledge
from sr_adapter.tables import extract_pdf_tables


@pytest.fixture
def pdf_factory(tmp_path):
    def make(rows, *, spans=(), name="source.pdf"):
        path = tmp_path / name
        writer = canvas.Canvas(str(path), pagesize=(700, 600), invariant=True)
        table = Table(rows, colWidths=[150] * len(rows[0]), rowHeights=[30] * len(rows))
        table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("GRID", (0, 0), (-1, -1), 1, "black"), *[("SPAN", start, end) for start, end in spans]]))
        _, height = table.wrap(700, 600)
        table.drawOn(writer, 20, 500 - height)
        writer.save()
        return path
    return make


@pytest.fixture
def mapping():
    return OntologyMapping(namespace="https://example.org/test/", classes={"Project": "Project", "Team": "Team"},
        entities=[{"key": "project", "column": "Project", "class_id": "Project"},
                  {"key": "owner", "column": "Owner", "class_id": "Team"}],
        relations=[{"predicate": "ownedBy", "subject": "project", "object": "owner"}],
        properties=[{"predicate": "budget", "subject": "project", "column": "Budget"}])


@pytest.fixture
def rows():
    return [["Project", "Owner", "Budget"], ["Alpha", "Ops", "100"], ["Beta", "Dev", ""], ["Gamma", "UNKNOWN", "300"]]


@pytest.fixture
def package(pdf_factory, mapping, rows):
    return extract_knowledge(extract_pdf_tables(pdf_factory(rows)), mapping)
