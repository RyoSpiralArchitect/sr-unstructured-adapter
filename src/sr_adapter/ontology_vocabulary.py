"""Verified, proposal-scoped declaration context for SQLite-only model reuse."""
import json

from .knowledge import sha256
from .proposal_vocabulary import REGISTRY, STRICT, compile_vocabulary
from .tables import canonical_json

ARCHIVES_PER_PAGE = 1
MAX_REVIEW_BYTES = 32 * 1024 * 1024


def inspect_vocabulary(connection, semantics, snapshot_id, selected_iris, offset=0):
    """Keep each archive's original declarations separate, even for equal snapshots."""
    rows = connection.execute("SELECT archive_id, kind, descriptor FROM sro_archives WHERE snapshot_id=? "
        "ORDER BY archive_id LIMIT ? OFFSET ?", (snapshot_id, ARCHIVES_PER_PAGE + 1, offset)).fetchall()
    if not rows and offset == 0:
        raise ValueError("Missing ontology declaration archives")
    has_more = len(rows) > ARCHIVES_PER_PAGE
    total = 0
    archives = []
    expected_terms = sorted(semantics.data["terms"], key=canonical_json)
    for archive_id, kind, encoded in rows[:ARCHIVES_PER_PAGE]:
        total += len(encoded.encode())
        if total > MAX_REVIEW_BYTES or len(encoded.encode()) > 4 * 1024 * 1024 or sha256(encoded.encode()) != archive_id:
            raise ValueError("Ontology declaration archive differs or exceeds its budget")
        descriptor = json.loads(encoded)
        if descriptor["snapshot_id"] != snapshot_id or descriptor["kind"] != kind:
            raise ValueError("Ontology declaration archive has a different scope")

        manifest = None

        def read_json(path, limit):
            nonlocal total
            expected = descriptor["files"].get(path)
            if manifest is not None and manifest["files"].get(path) != expected:
                raise ValueError("Proposal declaration references differ")
            size = connection.execute("SELECT length(data) FROM sro_files WHERE archive_id=? AND path=?",
                                      (archive_id, path)).fetchone()
            if expected is None or size is None or not 0 <= size[0] <= limit or size[0] != expected["bytes"]:
                raise ValueError("Missing or oversized ontology declaration payload")
            total += size[0]
            if total > MAX_REVIEW_BYTES:
                raise ValueError("Ontology declaration context exceeds its budget")
            raw = connection.execute("SELECT data FROM sro_files WHERE archive_id=? AND path=?", (archive_id, path)).fetchone()[0]
            if not isinstance(raw, bytes) or sha256(raw) != expected["sha256"]:
                raise ValueError("Ontology declaration payload differs from its archive")
            return json.loads(raw)

        manifest = read_json("manifest.json", 1024 * 1024)
        if manifest["kind"] != kind:
            raise ValueError("Ontology archive kind differs from its manifest")
        if kind == "relation_semantics":
            if (manifest["snapshot_id"] != snapshot_id or manifest["input_snapshot_id"] != semantics.source.snapshot
                    or read_json("semantics.json", 4 * 1024 * 1024) != semantics.data):
                raise ValueError("Explicit vocabulary declarations belong to another ontology")
            archives.append({"archive_id": archive_id, "policy": "explicit_overlay", "declarations_recorded": False, "terms": []})
            continue
        if kind != "semantic_proposal":
            raise ValueError("Unknown ontology declaration archive kind")
        if (manifest["kind"] != kind or manifest["semantic_snapshot_id"] != snapshot_id
                or manifest["source_snapshot_id"] != semantics.source.snapshot):
            raise ValueError("Ontology proposal declarations belong to another snapshot")
        policy = manifest.get("vocabulary_policy", STRICT)
        if policy not in (STRICT, REGISTRY):
            raise ValueError("Unknown ontology vocabulary policy")
        from .narrative import _json_response
        from .semantic_proposals import ProposalConfig, _proposal_model
        config = ProposalConfig.model_validate(read_json("config.json", 65536))
        response = read_json("response.json", 8 * 1024 * 1024)
        if (config.version != manifest["version"]
                or semantics.model.author != f"semantic-proposal/{config.version} model:" + response["model"]):
            raise ValueError("Ontology declaration model provenance differs")
        original = _proposal_model(config.version).model_validate(_json_response(response))
        proposal = read_json("proposal.json", 8 * 1024 * 1024)
        if original.model_dump() != proposal or original.overlay is None:
            raise ValueError("Ontology declarations differ from the model response")
        terms = original.overlay.terms
        compiled, report = compile_vocabulary(terms)
        if (sorted(compiled, key=canonical_json) != expected_terms
                or (policy == STRICT and report["repeated_iri_count"])):
            raise ValueError("Original declarations differ from the compiled ontology")
        groups = {}
        for index, term in enumerate(terms):
            if term.iri in selected_iris:
                groups.setdefault(term.iri, []).append({"index": index, "pointer": f"/overlay/terms/{index}",
                    "kind": term.kind, "label": term.label, "description": term.description})
        selected = []
        for iri, declarations in sorted(groups.items()):
            distinct = len({d["description"] for d in declarations if d["description"]})
            selected.append({"iri": iri, "review_required": distinct > 1, "repeated": len(declarations) > 1,
                             "declarations": declarations})
        archives.append({"archive_id": archive_id, "policy": policy, "response_model": response["model"],
                         "declarations_recorded": True, "terms": selected})
    complete = offset == 0 and not has_more and all(a["declarations_recorded"] for a in archives)
    required = any(term["review_required"] for archive in archives for term in archive["terms"])
    return {"scope": "selected_snapshot_and_archive_page", "snapshot_id": snapshot_id,
        "review_required": True if required else False if complete else None,
        "complete": complete, "offset": offset, "has_more": has_more, "archives": archives}
