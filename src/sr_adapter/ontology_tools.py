"""Provider-neutral ontology tools bound to one local database by trusted code."""
from .local_ontology import GUIDANCE, inspect_ontology, list_ontologies, search_ontology
from .relation_agent import RelationEvidenceTools, _failure
from .tables import canonical_json


class OntologyTools(RelationEvidenceTools):
    """Reuse the host allowlist, argument validator and response-budget contract."""

    def definitions(self):
        collection = {"type": "string", "enum": list(self.collections)}
        name = {"type": "string", "maxLength": 256}
        snapshot = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
        search = {"collection": collection, "query": {"type": "string", "maxLength": 4096},
            "kind": {"type": "string", "enum": ["all", "claim", "term"]},
            "name": {**name, "type": ["string", "null"]},
            "snapshot_id": {**snapshot, "type": ["string", "null"]},
            "status": {"type": ["string", "null"], "enum": [None, "asserted", "denied", "unknown", "ambiguous"]},
            "modality": {"type": ["string", "null"], "enum": [None, "descriptive", "required", "permitted", "prohibited", "possible", "hypothetical"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": self.max_results},
            "offset": {"type": "integer", "minimum": 0}}
        inspect = {"collection": collection, "name": name, "snapshot_id": snapshot,
            "kind": {"type": "string", "enum": ["claim", "term"]}, "id": {"type": "string", "maxLength": 256},
            "vocabulary_offset": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807}}
        return [{"name": tool, "description": description, "input_schema": {"type": "object",
            "properties": fields, "required": required, "additionalProperties": False}}
            for tool, description, fields, required in [
                ("list_ontologies", "Discover the documents and semantic versions in the allowed local collection.",
                 {"collection": collection}, ["collection"]),
                ("search_ontology", "Find candidate claims or vocabulary with ONE literal substring, e.g. 'approval'. A list of keywords is NOT an OR/AND query. Use query='' to browse, then inspect returned references before reuse.",
                 search, ["collection", "query"]),
                ("inspect_ontology", "Read a same-version whole claim with conditions, dissent, exact role anchors and original context, or inspect vocabulary and its uses. vocabulary_review preserves separate declaration notes and review flags scoped by archive.",
                 inspect, [key for key in inspect if key != "vocabulary_offset"])]]

    def call(self, name, arguments):
        if name not in {definition["name"] for definition in self.definitions()}:
            return _failure("unknown_tool", "Use list_ontologies, search_ontology or inspect_ontology.")
        invalid = self._validate(name, arguments)
        if invalid is not None:
            return invalid
        try:
            if name == "list_ontologies":
                result = list_ontologies(self.database, **arguments)
            elif name == "search_ontology":
                result = search_ontology(self.database, **{"limit": min(5, self.max_results), **arguments})
                for hit in result["hits"]:
                    hit["inspect"] = {"name": "inspect_ontology", "arguments": hit["reference"]}
            else:
                result = inspect_ontology(self.database, **arguments)
            result.update(tool=name, tool_contract_version=1, guidance=GUIDANCE)
            size = len(canonical_json(result))
            if size > self.max_response_bytes:
                return _failure("response_too_large", "No partial scope was returned. Narrow the search or use the trusted local API for a large trace.",
                                response_bytes=size, limit_bytes=self.max_response_bytes)
            return result
        except Exception:
            return _failure("ontology_unavailable", "The requested local ontology reference could not be verified. Keep the reference returned by search.")
