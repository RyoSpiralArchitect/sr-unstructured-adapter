# Raw-source local ontology development observation — 2026-11-02

The [machine-readable observation](validation/local-ontology-2026-11-02/observation.json)
records a synthetic DOCX and a synthetic diagram, ingested through the default
OpenAI transport with GPT-5.5. No customer materials or Fabric/Foundry resources
were used. This is a development integration smoke, not a fixed extraction-quality
benchmark or a claim of general semantic accuracy.

## Observed path

The synthetic policy contains conditional approval, a JPY 1,000,000 threshold,
an unzoned validity period, an Audit disagreement and an unspecified system
operator. Its diagram contains role/system boxes, connectors and conditional
annotations. Final active versions contain:

| Source | Scoped claims | Exact role anchors | Semantic scalar rows |
| --- | ---: | ---: | ---: |
| DOCX | 4 | 17 | 123 |
| PNG | 11 | 21 | 182 |

Counts describe saved model outputs, not correct extractions. Image claims include
headings, disclaimers and displayed items, so more claims is not better coverage.
Both were committed with their original bytes and receipts in one SQLite file.
The installed wheel, running outside the checkout with PyArrow imports blocked,
read both entries, traced all 15 claims and recovered both original sources.

## Development failures retained

There were **7 extraction requests across 5 runs**, with no automatic retries:

1. Initial DOCX: completed in 1 request using proposal v2.
2. Initial image: failed after 1 legacy direct-visual request. Three proposed
   relations had one evidence ID, while the binary relation contract requires two
   distinct endpoints. Its response was retained and no entry was activated.
3. Image with inventory-only visual v3: inventory completed, then proposal v2
   failed because an `implies` expression had three operands. Both responses were
   retained and no entry was activated.
4. DOCX with structural proposal v3: completed in 1 request, replacing its previous
   version with the expected-snapshot check.
5. Image with inventory-only v3 and structural proposal v3: completed in 2 requests.

The implementation changes preserve v1/v2 request replay. Inventory-only reading
avoids an unnecessary binary-edge stage before general semantics. Structural
proposal v3 constrains operator arity during generation while storing the existing
`relation_semantics/2` representation. The failed outputs were not edited into
successful outputs.

## Model reuse and remaining limits

A separate conversation received only the read-only tool definitions and a copy
of the SQLite file. It made six tool calls: four literal searches, a collection
listing and one exact-version claim inspection.
The six-request loop reached its bound before an answer. One explicit finalization
request, with tools disabled and the existing tool results retained, produced a
Japanese answer. Total reuse requests: **7**. No extraction was repeated for reuse.

The answer cites the document, snapshot and claim `c1`; it describes approval and
pause conditions, the amount threshold, the unresolved dispute and the unspecified
operator. It explicitly distinguishes `c2`/`c3` as uninspected candidates and reads
the remaining statements from `c1`'s full original source context. This demonstrates
DB-backed source reuse, not a complete review of every generated claim or of both
documents. The model initially used keyword lists against a literal substring
search; the tool description and result guidance now explicitly explain using one
substring or an empty query to browse. That clarification has not been separately
measured as a retrieval-quality improvement.

Offline coverage: 1,341 tests passed, 1 skipped, including 32 new tests for raw
formats, SQL-only reuse, old/new semantic profiles, scoped trace, version conflict,
rollback, source recovery, allowlists, bounded results, failures and abstention.
Ruff passed. Scanned-PDF OCR, embedded Office diagrams, automated truth evaluation,
cross-document entity resolution and local-model accuracy remain outside this
validation.
