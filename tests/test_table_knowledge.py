"""Real PDF layout controls, explicit ontology semantics, and grounded exports."""

import json

import pytest
from rdflib import Graph, URIRef
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from sr_adapter.knowledge import (KnowledgePackage, OntologyMapping, extract_knowledge,
                                  graph_jsonld, ontology_jsonld)
from sr_adapter.tables import extract_pdf_tables, table_blocks


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


def triples(package):
    labels = {entity.id: entity.label for entity in package.entities}
    return {(labels[edge.subject], edge.predicate, labels[edge.object]) for edge in package.relations}


def test_pdf_cells_preserve_real_coordinates_empty_cells_and_block_provenance(pdf_factory, rows):
    source = extract_pdf_tables(pdf_factory(rows))
    table = source.tables[0]
    assert (table.rows, table.columns, len(table.cells)) == (4, 3, 12)
    assert table.cells[0].bbox.model_dump() == {"x0": 20., "y0": 100., "x1": 170., "y1": 130.}
    assert next(c for c in table.cells if (c.row, c.column) == (2, 2)).text == ""
    block = table_blocks(source)[0]
    assert block.prov.page == 0 and table.page == 1
    assert block.prov.bbox == table.bbox
    assert block.attrs["layout_geometry"] == "source"


def test_column_permutation_preserves_semantic_bindings(pdf_factory, rows, mapping):
    first = extract_knowledge(extract_pdf_tables(pdf_factory(rows)), mapping)
    changed = [[row[i] for i in (2, 0, 1)] for row in rows]
    second = extract_knowledge(extract_pdf_tables(pdf_factory(changed, name="changed.pdf")), mapping)
    assert triples(first) == triples(second)
    assert first.id != second.id
    assert sorted((x.status, x.value) for x in first.observations) == sorted((x.status, x.value) for x in second.observations)


def test_only_swapping_headers_changes_the_relationship(pdf_factory, rows, mapping):
    changed = [row[:] for row in rows]
    changed[0][0], changed[0][1] = changed[0][1], changed[0][0]
    result = extract_knowledge(extract_pdf_tables(pdf_factory(changed)), mapping)
    assert ("Ops", mapping.namespace + "ownedBy", "Alpha") in triples(result)
    assert ("Alpha", mapping.namespace + "ownedBy", "Ops") not in triples(result)


def test_unknowns_are_not_entities_or_asserted_literals(package):
    assert len(package.entities) == 5 and len(package.relations) == 2
    assert all(entity.label != "UNKNOWN" for entity in package.entities)
    assert sum(item.status == "unknown" for item in package.observations) == 2
    graph = Graph().parse(data=json.dumps(graph_jsonld(package)), format="json-ld")
    beta = next(entity for entity in package.entities if entity.label == "Beta")
    assert not list(graph.objects(URIRef(beta.id), URIRef(package.mapping.namespace + "budget")))
    assert len(list(graph.triples((None, URIRef(package.mapping.namespace + "ownedBy"), None)))) == 2
    ontology = Graph().parse(data=json.dumps(ontology_jsonld(package.mapping)), format="json-ld")
    assert (URIRef(package.mapping.namespace + "ownedBy"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#range"), URIRef(package.mapping.namespace + "Team")) in ontology


def test_spanning_cells_are_retained_without_inventing_row_assignments(pdf_factory, rows, mapping):
    source = extract_pdf_tables(pdf_factory(rows, spans=[((1, 1), (1, 2))]))
    assert any(cell.row_span == 2 for cell in source.tables[0].cells)
    result = extract_knowledge(source, mapping)
    assert not result.relations
    assert sum(item.status == "ambiguous" for item in result.observations) >= 2


def test_duplicate_required_headers_fail_closed(pdf_factory, mapping):
    source = extract_pdf_tables(pdf_factory([["Project", "Owner", "Owner", "Budget"], ["A", "B", "C", "1"]]))
    result = extract_knowledge(source, mapping)
    assert not result.entities and not result.relations
    assert any("ambiguous headers" in warning for warning in result.warnings)


def test_same_label_in_different_rows_is_not_implicitly_one_entity(pdf_factory, mapping):
    source = extract_pdf_tables(pdf_factory([["Project", "Owner", "Budget"], ["A", "Ops", "1"], ["B", "Ops", "2"]]))
    result = extract_knowledge(source, mapping)
    owners = [entity for entity in result.entities if entity.label == "Ops"]
    assert len(owners) == 2 and owners[0].id != owners[1].id


def test_source_and_mapping_reproduce_package_id(package):
    assert package == extract_knowledge(package.source, package.mapping)
    payload = package.model_dump(mode="json")
    payload["entities"][0]["label"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        KnowledgePackage.model_validate(payload)


@pytest.mark.parametrize("change", ["missing_class", "missing_subject", "duplicate_term", "relative_namespace"])
def test_invalid_ontology_is_rejected(mapping, change):
    payload = mapping.model_dump()
    if change == "missing_class":
        payload["entities"][0]["class_id"] = "Absent"
    elif change == "missing_subject":
        payload["relations"][0]["subject"] = "absent"
    elif change == "duplicate_term":
        payload["properties"][0]["predicate"] = "ownedBy"
    else:
        payload["namespace"] = "relative/"
    with pytest.raises(ValueError):
        OntologyMapping.model_validate(payload)
