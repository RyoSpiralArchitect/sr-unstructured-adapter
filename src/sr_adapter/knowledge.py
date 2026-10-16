"""Source-scoped entity mentions, row relations, and an explicit ontology mapping.

No automatic identity resolution or ontology induction is implied. The mapping
is authored by the caller; ambiguous headers/spanning data cells are left unknown.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from .tables import Record, TableDocument, canonical_json, identifier

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
PROV = "http://www.w3.org/ns/prov#"
SR = "urn:sr:vocab:"
TERM = r"^[A-Za-z][A-Za-z0-9_]*$"


class EntityRule(Record):
    key: str = Field(pattern=TERM)
    column: str = Field(min_length=1)
    class_id: str = Field(pattern=TERM)


class RelationRule(Record):
    predicate: str = Field(pattern=TERM)
    subject: str
    object: str


class PropertyRule(Record):
    predicate: str = Field(pattern=TERM)
    subject: str
    column: str = Field(min_length=1)


class OntologyMapping(Record):
    namespace: str
    classes: dict[str, str]
    entities: list[EntityRule] = Field(min_length=1)
    relations: list[RelationRule] = Field(default_factory=list)
    properties: list[PropertyRule] = Field(default_factory=list)
    header_row: int = Field(default=0, ge=0, strict=True)
    unknown_values: list[str] = Field(default_factory=lambda: ["", "UNKNOWN", "未定", "不明", "-"])

    @field_validator("namespace")
    @classmethod
    def namespace_is_absolute(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http", "urn"} or any(c.isspace() for c in value):
            raise ValueError("Ontology namespace must be an absolute IRI")
        if parsed.scheme in {"http", "https"} and not parsed.netloc:
            raise ValueError("Ontology namespace requires a host")
        if not value.endswith(("/", "#", ":")):
            raise ValueError("Ontology namespace must end in /, #, or :")
        return value

    @model_validator(mode="after")
    def references_are_valid(self):
        keys = [rule.key for rule in self.entities]
        terms = list(self.classes) + [rule.predicate for rule in [*self.relations, *self.properties]]
        if len(set(keys)) != len(keys) or len(set(terms)) != len(terms):
            raise ValueError("Entity keys and ontology terms must be unique")
        if any(not re.fullmatch(TERM, term) for term in terms):
            raise ValueError("Ontology terms must be simple identifiers")
        for rule in self.entities:
            if rule.class_id not in self.classes:
                raise ValueError("Entity references an undefined class")
        for rule in [*self.relations, *self.properties]:
            if rule.subject not in keys or (isinstance(rule, RelationRule) and rule.object not in keys):
                raise ValueError("Relation/property references an undefined entity rule")
        return self


class Entity(Record):
    id: str
    type: str
    label: str
    evidence: list[str] = Field(min_length=1)
    identity_scope: Literal["source_row_mention"] = "source_row_mention"


class Relation(Record):
    id: str
    subject: str
    predicate: str
    object: str
    evidence: list[str] = Field(min_length=1)


class Observation(Record):
    id: str
    subject: str | None
    predicate: str
    value: str | None
    status: Literal["observed", "unknown", "ambiguous"]
    evidence: list[str]


class KnowledgePackage(Record):
    schema_version: Literal["1"] = "1"
    id: str
    source: TableDocument
    mapping: OntologyMapping
    entities: list[Entity]
    relations: list[Relation]
    observations: list[Observation]
    warnings: list[str]

    @model_validator(mode="after")
    def integrity(self):
        tables = {table.id for table in self.source.tables}
        cells = {cell.id for table in self.source.tables for cell in table.cells}
        entities = {entity.id for entity in self.entities}
        all_ids = [table.id for table in self.source.tables] + [cell.id for table in self.source.tables for cell in table.cells]
        all_ids += [record.id for group in (self.entities, self.relations, self.observations) for record in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate graph record IDs")
        for record in [*self.entities, *self.relations, *self.observations]:
            if not set(record.evidence) <= cells | tables:
                raise ValueError("Evidence references a missing source cell/table")
        for relation in self.relations:
            if relation.subject not in entities or relation.object not in entities:
                raise ValueError("Dangling relation")
        for observation in self.observations:
            if observation.subject is not None and observation.subject not in entities:
                raise ValueError("Dangling property subject")
            if observation.status == "observed" and (observation.value is None or observation.subject is None):
                raise ValueError("Observed properties require a subject and value")
        expected = identifier("package", self.model_dump(mode="json", exclude={"id"}))
        if self.id != expected:
            raise ValueError("Knowledge package content hash mismatch")
        return self


def header_key(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def extract_knowledge(source: TableDocument, mapping: OntologyMapping) -> KnowledgePackage:
    entities, relations, observations = [], [], []
    warnings = list(source.warnings)
    unknown = {value.strip() for value in mapping.unknown_values} | {""}
    required = {header_key(rule.column) for rule in [*mapping.entities, *mapping.properties]}
    for table in source.tables:
        headers = [cell for cell in table.cells if cell.row == mapping.header_row]
        by_header = {}
        for cell in headers:
            by_header.setdefault(header_key(cell.text), []).append(cell)
        if not required <= by_header.keys():
            warnings.append(f"{table.id}: mapping not applicable (required headers missing)")
            continue
        if any(len(by_header[key]) != 1 or by_header[key][0].row_span != 1 or
               by_header[key][0].column_span != 1 for key in required):
            warnings.append(f"{table.id}: ambiguous headers; no assertions emitted")
            continue
        columns = {key: by_header[key][0].column for key in required}
        lookup = {(cell.row, cell.column): cell for cell in table.cells}
        for row in range(mapping.header_row + 1, table.rows):
            row_entities = {}

            def value_for(column):
                header = by_header[header_key(column)][0]
                cell = lookup.get((row, columns[header_key(column)]))
                evidence = [header.id] + ([cell.id] if cell else [table.id])
                if cell is None or cell.row_span != 1 or cell.column_span != 1:
                    return None, "ambiguous", evidence
                if cell.text.strip() in unknown:
                    return None, "unknown", evidence
                return cell.text, "observed", evidence

            for rule in mapping.entities:
                value, status, evidence = value_for(rule.column)
                if status != "observed":
                    observations.append(Observation(id=identifier("missing", table.id, row, rule.key),
                        subject=None, predicate=mapping.namespace + rule.class_id,
                        value=None, status=status, evidence=evidence))
                    continue
                entity = Entity(id=identifier("entity", table.id, row, rule.key, value),
                    type=mapping.namespace + rule.class_id, label=value, evidence=evidence)
                entities.append(entity)
                row_entities[rule.key] = entity
            for rule in mapping.relations:
                subject, obj = row_entities.get(rule.subject), row_entities.get(rule.object)
                if subject and obj:
                    relations.append(Relation(id=identifier("relation", subject.id, rule.predicate, obj.id),
                        subject=subject.id, predicate=mapping.namespace + rule.predicate, object=obj.id,
                        evidence=list(dict.fromkeys(subject.evidence + obj.evidence))))
            for rule in mapping.properties:
                value, status, evidence = value_for(rule.column)
                subject = row_entities.get(rule.subject)
                if subject is None:
                    value, status = None, "ambiguous"
                observations.append(Observation(id=identifier("property", table.id, row, rule.predicate),
                    subject=subject.id if subject else None, predicate=mapping.namespace + rule.predicate,
                    value=value, status=status, evidence=list(dict.fromkeys((subject.evidence if subject else []) + evidence))))
    payload = dict(schema_version="1", source=source.model_dump(mode="json"), mapping=mapping.model_dump(mode="json"),
        entities=[e.model_dump(mode="json") for e in entities], relations=[r.model_dump(mode="json") for r in relations],
        observations=[o.model_dump(mode="json") for o in observations], warnings=warnings)
    return KnowledgePackage(id=identifier("package", payload), **payload)


def ontology_jsonld(mapping: OntologyMapping) -> dict:
    ns = mapping.namespace
    classes = {rule.key: ns + rule.class_id for rule in mapping.entities}
    nodes = [{"@id": ns, "@type": OWL + "Ontology"}]
    nodes += [{"@id": ns + key, "@type": OWL + "Class", RDFS + "label": label} for key, label in mapping.classes.items()]
    for rule in mapping.relations:
        nodes.append({"@id": ns + rule.predicate, "@type": OWL + "ObjectProperty",
                      RDFS + "domain": {"@id": classes[rule.subject]}, RDFS + "range": {"@id": classes[rule.object]}})
    for rule in mapping.properties:
        nodes.append({"@id": ns + rule.predicate, "@type": OWL + "DatatypeProperty",
                      RDFS + "domain": {"@id": classes[rule.subject]},
                      RDFS + "range": {"@id": "http://www.w3.org/2001/XMLSchema#string"}})
    return {"@graph": nodes}


def graph_jsonld(package: KnowledgePackage) -> dict:
    """Expanded JSON-LD, with no remote context fetches or inferred identities."""
    doc_id = "urn:sha256:" + package.source.source_sha256
    nodes = [{"@id": doc_id, "@type": PROV + "Entity", RDFS + "label": package.source.source_name}]
    for table in package.source.tables:
        nodes.append({"@id": table.id, "@type": SR + "Table", SR + "page": table.page,
                      PROV + "wasDerivedFrom": {"@id": doc_id}})
        for cell in table.cells:
            nodes.append({"@id": cell.id, "@type": SR + "Cell", RDF + "value": cell.text,
                SR + "row": cell.row, SR + "column": cell.column, SR + "rowSpan": cell.row_span,
                SR + "columnSpan": cell.column_span, SR + "coordinateSystem": table.coordinate_system,
                SR + "bbox": {"@value": cell.bbox.model_dump(), "@type": "@json"},
                PROV + "wasDerivedFrom": {"@id": table.id}})
    for entity in package.entities:
        nodes.append({"@id": entity.id, "@type": entity.type, RDFS + "label": entity.label,
            SR + "identityScope": entity.identity_scope, PROV + "wasDerivedFrom": [{"@id": e} for e in entity.evidence]})
    for relation in package.relations:
        nodes.append({"@id": relation.subject, relation.predicate: {"@id": relation.object}})
        nodes.append({"@id": relation.id, "@type": RDF + "Statement", RDF + "subject": {"@id": relation.subject},
            RDF + "predicate": {"@id": relation.predicate}, RDF + "object": {"@id": relation.object},
            PROV + "wasDerivedFrom": [{"@id": e} for e in relation.evidence]})
    for observation in package.observations:
        node = {"@id": observation.id, "@type": SR + "Observation", SR + "status": observation.status,
                RDF + "predicate": {"@id": observation.predicate},
                PROV + "wasDerivedFrom": [{"@id": e} for e in observation.evidence]}
        if observation.subject:
            node[RDF + "subject"] = {"@id": observation.subject}
        if observation.status == "observed":
            node[RDF + "value"] = observation.value
            nodes.append({"@id": observation.subject, observation.predicate: observation.value})
        nodes.append(node)
    return {"@graph": nodes}


def package_records(package: KnowledgePackage) -> list[dict[str, str]]:
    records = [("document", "urn:sha256:" + package.source.source_sha256, package.source.model_dump(mode="json", exclude={"tables"})),
               ("ontology", identifier("ontology", package.mapping.model_dump()), package.mapping.model_dump(mode="json"))]
    for table in package.source.tables:
        records.append(("table", table.id, table.model_dump(mode="json", exclude={"cells"})))
        records.extend(("cell", cell.id, {**cell.model_dump(mode="json"), "table_id": table.id}) for cell in table.cells)
    for kind, group in (("entity", package.entities), ("relation", package.relations), ("observation", package.observations)):
        records.extend((kind, record.id, record.model_dump(mode="json")) for record in group)
    return [{"package_id": package.id, "kind": kind, "record_id": key, "payload_json": canonical_json(payload).decode()}
            for kind, key, payload in records]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
