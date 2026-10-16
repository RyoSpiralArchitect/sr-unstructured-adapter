# Narrative, CSV, PPTX and image relations in Foundry — 2026-10-17

The adapter extracted relation tables from five synthetic inputs, published all
40 assertion cards to Azure AI Search, and answered six scenarios through a
GPT-5.5 Prompt agent using native Foundry IQ retrieval. CSV row/column, PPTX
slide/shape and image-transcript locators survived this path. Image transcription
and model interpretation retain their unverified labels.

A separate controlled comparison produced strict scores of 23/24 versus 24/24
for Markdown and 11/12 versus 12/12 for PNG, comparing original sources with
relation tables plus source context. The two mismatches concern answer-category
and relevance boundaries, not missing gold facts or demonstrated OCR improvement.
Input tokens increased. This is a small development pilot with no declared winner.

## Corpus and extraction

The [hand-authored gold](../data/knowledge/narrative-mock/gold.json) has two Japanese
mock memoranda, 16 document-scoped claims and 12 questions. It distinguishes
primary/support/approval roles, denial, unknown values, unresolved alternatives,
contradictions between equal-priority sources and different entities sharing a
name. Memo A is also represented as CSV, PPTX and PNG. Five files therefore
contain **40 repeated claim instances, only 16 distinct authored claims**.
Gold answers were not sent to extraction or answer generation. Only question IDs
and question text entered the controlled QA requests.

| Input | Matched / expected tuples | Extra / missing | Source locator |
|---|---:|---:|---|
| memo-a.md | 8 / 8 | 0 / 0 | Original line |
| memo-b.md | 8 / 8 | 0 / 0 | Original line |
| memo-a.csv | 8 / 8 | 0 / 0 | Logical row, physical lines and columns |
| memo-a.pptx | 8 / 8 | 0 / 0 | Slide and shape |
| memo-a.png | 8 / 8 | 0 / 0 | Frame and unverified transcript line |

All six extraction/transcription calls completed with returned model `gpt-5.5`.
The [audit](validation/narrative-foundry-2026-10-17/audit/report.json) replays source
and quotation verification separately from semantic tuple scoring. For PNG,
matching eight authored facts does not validate all OCR text, layout or pixels.
The live PPTX example exercises text shapes; table-cell/span handling has local
tests but no separate live PPTX-table model-quality evaluation here. JPEG/WebP
and TSV have local parser tests; the live formats are the five files above.

## Original source versus relation table

The [frozen comparison](validation/narrative-foundry-2026-10-17/comparison/frozen/spec.json)
uses the same model deployment, ontology, questions, system instructions,
strict output schema, reasoning effort `low` and 12,000 output-token ceiling.
Each arm runs twice in counterbalanced order, with no retries. All eight requests
completed. Markdown uses both complete documents; PNG uses the complete memo-A
image and six memo-A questions. The relation arm contains candidate rows and the
complete source context once per document (the saved transcript for PNG).
This is not a comparison against native Foundry retrieval.

| Case / arm | Strict correct / planned | Input tokens | Output tokens | Cached input tokens |
|---|---:|---:|---:|---:|
| Markdown original | 23 / 24 | 3,644 | 3,009 | 1,280 |
| Markdown relations + source | 24 / 24 | 10,160 | 2,882 | 4,864 |
| PNG original image | 11 / 12 | 4,318 | 1,286 | 1,792 |
| PNG relations + transcript | 12 / 12 | 5,892 | 1,207 | 2,816 |

These are aggregate QA tokens for two calls per row. Input tokens increased
**178.8% for Markdown and 36.5% for PNG**. These numbers exclude preprocessing:
Markdown relation extraction used another 4,130 input / 1,467 output tokens;
PNG transcription plus extraction used 3,450 input / 1,033 output tokens.
Cached tokens are shown separately; no monetary-cost or latency advantage is claimed.

Strict scoring requires the exact document-scoped claim set and `answer_kind`:

- Markdown original, repeat 1, Q09: both opposing support claims were correct,
  and the explanation preserved the contradiction, but the category was `mixed`
  instead of `conflicting`. All 24 Markdown claim sets match gold in both arms.
- PNG original, repeat 2, Q03: the explicit support denial was correct, but the
  answer added the document's true primary-owner claim for a different team.
  This exceeded the question's requested relationship and changed the category
  to `mixed`. It was not a fabricated relation or a visual transcription error.

All other answers passed the fixed strict rule. The
[raw requests/responses and replayed summary](validation/narrative-foundry-2026-10-17/comparison/summary.json)
retain the failures without retrying or revising the rubric. Explanatory prose
and entity identifiers in Q10 are not automatically judged. Two repeats of these
authored examples do not establish generalization, significance or causality.

## Native Foundry observation

The [portable v3 handoff](validation/narrative-foundry-2026-10-17/handoff/manifest.json)
contains the source bundles and 40 assertion cards. Its snapshot ID is
`77f73ca72ce6b18270e00e6bb788d59ba92b36d827d29708cb8d66b43211ce15`.
The public handoff has no configured Azure target. An explicitly selected private
target was used to create a dedicated index, knowledge source/base, connection
and Prompt agent version 1. All 40 documents were accepted and matched on readback.
No Fabric or existing application-traffic configuration changed for this pilot.

Six fixed native-agent prompts covered Markdown, CSV, PPTX, PNG, unresolved
approval and cross-document conflicts/identity. The
[live receipt summary](validation/narrative-foundry-2026-10-17/live-verification.json)
and [offline audit](validation/narrative-foundry-2026-10-17/native-audit.json) record:

- 6/6 completed responses; 15 MCP retrieval calls; 524 returned document occurrences.
- All returned document fields match the saved cards, including structured evidence.
- 41/41 native citations identify retrieved documents in the requested source scope.
- Manual audit: 25/25 answer relation claims and their stated locators match the
  hand gold and saved evidence, with no identified error in these six answers.
- Unknown budget, unresolved approval, equal-priority contradictions and distinct
  same-name projects remain explicit. PNG answers preserve the transcription caveat.

This is a post-hoc manual audit of six prompts, covering 14 of the 16 gold claims.
It does not test all 12 gold questions; notably the absence question Q04 was not
requested in native retrieval. Other-format cards were retrieved too, even when
the final citations obeyed the requested source. The 524 occurrences are repeated
retrievals of 40 cards, not 524 independent evidence items.

The native responses consumed **255,229 input / 8,584 output tokens**, with zero
reported cached input tokens. Full source context is repeated in each assertion
card, and the index contains several versions of memo A. The high retrieval
volume is an observed efficiency problem. A follow-up should separate assertion
and document context, filter by source scope, and measure bounded retrieval while
retaining qualification, contradiction and identity evidence.

An initial validation attempt stopped before all six model calls because agent
readback represented `allowed_tools` as an object containing `tool_names` instead
of a list. The second attempt normalized only that documented-in-the-receipt
representation, verified the remaining definition and then ran all six calls.
The incomplete first receipt is retained by hash. No server-side Foundry evaluation
suite was created: evaluation used the requested synthetic pilot and local replay.

## Replay and evidence boundaries

From the repository root, with dependencies installed:

```sh
python scripts/probe_narrative_relations.py replay-audit \
  --audit docs/validation/narrative-foundry-2026-10-17/audit --allow-code-drift
python scripts/probe_narrative_relations.py replay \
  --frozen docs/validation/narrative-foundry-2026-10-17/comparison/frozen \
  --run docs/validation/narrative-foundry-2026-10-17/comparison/run --allow-code-drift
```

Audit and comparison replay need no credentials or model requests. Frozen files,
source-code hashes, dependency versions, requests, raw responses and scoring
receipts are included. Checksums establish internal consistency, not signatures
or proof of model origin. Strict replay rejects a changed implementation by default.
`--allow-code-drift` is an explicit **offline replay only** option: it uses current
code, lists every changed source hash and environment, and still checks all frozen
file hashes, reconstructed requests, source bundles and saved scores. It neither
executes the archived Python files nor permits new inference with a changed plan.
To run the original implementation itself, use repository commit `c38136d`.
See the [usage and format boundaries](narrative-relations-foundry.md) for new runs.

Review identified an escaped-cell bug: a CSV entity such as `Acme "East"` was
present in the original value but not literally in its JSON-escaped unit text.
The correction grounds labels in each cited cell's raw header/value, while still
requiring the complete unit quote and preserving positional evidence. Twelve
regression cases cover CSV/TSV quotes, backslashes, tabs, quoted headers, multiline values,
JSON-only metadata and uncited rows; eleven failed before the correction.
The original experiment artifacts remain unchanged. The
[review replay receipt](validation/narrative-foundry-2026-10-17/review-replay.json)
records the explicit implementation differences and identical recomputed results;
no additional model calls were made for this correction.

Extraction and controlled-comparison raw model bodies are public synthetic
artifacts. Native MCP raw responses, endpoint configuration and publication
receipts remain private. The
[native projection](validation/narrative-foundry-2026-10-17/native-projection.json)
retains answer text, document identifiers, usage and raw-body hashes while removing
service URLs and generated remote identifiers. The offline audit's `.git/...`
paths identify those private inputs; they are not public downloadable artifacts.
Public readers can inspect answers/citations and replay local extraction and QA
scoring, but cannot independently reconstruct the complete native wire exchange.
