"""Compile a vocabulary within one model proposal. References keep their exact IRIs."""
from collections import defaultdict

STRICT = "strict"
REGISTRY = "proposal-registry/1"


class VocabularyConflict(ValueError):
    """Incompatible declarations stay available for inspection and manual review."""

    def __init__(self, conflicts):
        super().__init__("Vocabulary declarations could not be compiled")
        self.diagnostic = {"policy": REGISTRY, "conflicts": conflicts}


def compile_vocabulary(terms):
    """Accept validated Term objects. Keep original positions and all differing notes."""
    groups = defaultdict(list)
    for index, term in enumerate(terms):
        groups[term.iri].append((index, term))

    compiled, repetitions, conflicts = [], [], []
    for iri, occurrences in sorted(groups.items()):
        first = occurrences[0][1]
        declaration = first.model_dump()
        if len(occurrences) > 1:
            positions = [{"index": index, "kind": term.kind, "label": term.label, "description": term.description}
                         for index, term in occurrences]
            record = {"iri": iri, "declarations": positions}
            differences = [field for field in ("kind", "label")
                           if any(getattr(term, field) != getattr(first, field) for _, term in occurrences)]
            if differences:
                conflicts.append({**record, "reason": "incompatible_declarations", "fields": differences})
                continue
            # Descriptions are annotations. Keep their separate original strings
            # in the review and expose every distinct nonempty note to SQL readers.
            notes = sorted({term.description for _, term in occurrences if term.description})
            description = "\n\n".join(notes)
            if len(description) > 65536:  # Term.description's existing Text bound.
                conflicts.append({**record, "reason": "description_limit", "fields": ["description"]})
                continue
            declaration["description"] = description
            repetitions.append({**record, "distinct_notes": len(notes), "review_notes": len(notes) > 1})
        compiled.append(declaration)

    if conflicts:
        raise VocabularyConflict(conflicts)
    return compiled, {"policy": REGISTRY, "declaration_count": len(terms), "term_count": len(compiled),
        "repeated_iri_count": len(repetitions), "repetitions": repetitions}
