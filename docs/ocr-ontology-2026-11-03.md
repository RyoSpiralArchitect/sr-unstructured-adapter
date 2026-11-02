# OCR ontology development observation — 2026-11-03

This is a bounded integration observation, not an OCR or semantic-accuracy
benchmark. All inputs are synthetic; no customer documents or Fabric/Foundry
resources were used. Extraction used OpenAI Responses model `gpt-6-luna`, and
local OCR used Apple Vision revision 3 on macOS 26.2, Japanese/English, language
correction disabled. No automatic retries were made.

## Fixed sources and retained failures

The PNG contains a Japanese approval table: up to 100万円 → 審査部, over 100万円
→ 部長; an incomplete antisocial check prevents disbursement; a proposed revision
is unapproved. The mixed PDF adds an embedded-English-text first page and puts
the same PNG on its scanned second page.

| Source | SHA-256 |
| --- | --- |
| `policy.png` | `a20308654a1f1b375903fe872f35a1dab1d07296bdb24c4605b221f56bbbbb3b` |
| `mixed-policy.pdf` | `83a66cf03f35c17e4df512ecced0d123c0f4063d7d4c3896666895f1172bcfb1` |

The first development attempt retained full scalar location rows in model
context. All three runs failed validation: OCR PNG had inconsistent coverage
references; OCR mixed PDF and the two-call visual-inventory baseline omitted
vocabulary declarations. Those requests/responses and failed receipts were kept.

After compacting context, retaining list bounds in the generation schema,
clarifying role declarations and selecting medium reasoning for protocol v4,
each OCR input was run once more. These are engineering follow-ups after code
changes, not repeats in a controlled comparison. The two-call baseline was not
rerun. Across both stages there were **5 runs / 6 model requests: 1 complete,
4 failed**. Failures did not activate incomplete entries.

## Final OCR attempts

| Input | Model requests | Local OCR pages | Result |
| --- | ---: | ---: | --- |
| PNG | 1 | 1 | Complete: 5 unreviewed candidate claims |
| Mixed PDF | 1 | 1 of 2 selected pages | Rejected: duplicate vocabulary IRI |

The mixed-PDF failure occurred after successful OCR, page rendering and model
response. The model declared `urn:sr:proposed:condition` twice with differing
descriptions. The adapter did not choose a description, merge declarations or
silently repair the proposal. Both page readings and the failed response remain
available. Reliable vocabulary generation remains an open model/protocol issue.

For the completed PNG, manual trace inspection found both table row pairings,
an implication from incomplete check to non-execution, and a denied approval
claim for the proposed revision. This spot check is not an exhaustive annotation
or a measured accuracy rate; no conclusion about production banking documents
or general table understanding follows from it.

## Context size and reuse

The original and compact projections of the same saved OCR datasets were compared
locally, with no model invocation. These sizes count only the serialized text
context, excluding schema, instructions and image data. Source pixels and the
full scalar database rows were preserved.

| Input | Original text context | Compact text context | Record count |
| --- | ---: | ---: | --- |
| PNG | 73,693 bytes | 5,094 bytes | 168 → 12 |
| Mixed PDF | 78,510 bytes | 5,471 bytes | 176 → 13 |

That is about a 93% reduction in text-context bytes for these fixtures. Actual
final API usage was 6,431 input / 4,910 output tokens for PNG, and 9,174 input /
5,896 output tokens for mixed PDF. Model reasoning settings and instructions also
changed between attempts; these observations do not establish latency, cost or
quality improvements. Fewer calls alone is not a cost comparison.

The successful snapshot is
`76ab55dfaabd7f6036a7817db74badb2f77b70612f56119efae4eddb4264e0ed`.
Re-ingestion with rendering, OCR and HTTP replaced by failing sentinels returned
`already_active`, with zero calls to all three. A copied SQLite database was then
read from an installed wheel outside the checkout, with imports of PyArrow,
PDFium and OCR modules blocked. All 5 claims, original PNG and saved page view
were recovered and hash-checked; the database hash remained unchanged.

## Reproducibility and scope

Local receipts are under
`/Users/ryohiga/SpiralReality/artifacts/ocr-ontology-luna-20261102/`:
the original and `compact-context/` directories each retain frozen fixture hashes,
run scripts, preparation receipts, SQLite request/response records and results.
`compact-context/context-comparison.json`, `reuse.json` and
`db-only/readback.json` record the checks above. Full local receipt directories
are not committed to the repository. Synthetic source text/images were sent to
OpenAI in the model requests reported above.

Offline tests additionally cover inline raster images hidden under a text header,
EXIF orientation, selected pages, low confidence, empty readings, malformed boxes,
failure preservation, pre-OCR replacement conflicts, compact-context bindings,
language-data identity and archive tampering. Tesseract parsing/adapter behavior
has fixture tests; native Tesseract execution has not been exercised on this Mac.
See [usage and limits](local-ontology.md) for the opt-in OCR route.
