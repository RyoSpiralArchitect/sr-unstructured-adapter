# Same-model document vision comparison

This pilot measures whether the current adapter helps a vision model answer
source-specific questions. Images remain present in every condition. No answer
key, expected value, evidence annotation, or case tag is sent to the model.

The [first observed result (Japanese)](vlm-pilot-2026-10-15.md) preserves a negative
result: no gain on known values in this pilot, with additional input tokens.

| Condition | Model input | Purpose |
| --- | --- | --- |
| `vision` | Page images + questions | Direct VLM baseline |
| `vision_text` | Identical images/questions + pypdf text | Separate text availability from adapter processing |
| `vision_adapter` | Identical images/questions + adapter JSON | Measure the current adapter's incremental effect |

The adapter runs `convert(..., recipe="default", profile="archival", llm_ok=False)`.
It receives only the pages shown to the model. The baseline text is extracted
from the same page subset with the same installed pypdf version. The adapter JSON
retains block types, text, attributes, confidence, spans, warnings, and provenance.
Random block IDs and absolute source paths are removed. Provenance page numbers
are explicitly converted to one-based original PDF page numbers. Heuristic
geometry stays labelled as heuristic. The production parser is unchanged.

## Frozen pilot corpus

[Specification and annotations](../data/vlm_benchmark/pilot-v1.json) contain eight
cases, nine rendered pages, and 48 questions (12 with `UNKNOWN` as the expected
answer). Four synthetic documents cover a numeric table, two versions across
pages, side-by-side tables, and an image-only PDF. Four local Digital Agency
documents cover a development timeline, roles, story development policies, and
an image-only definition diagram/table.

The real inputs are the first page of each of these existing local PDFs:

- 別添10 アプリケーション開発の進め方（想定）.pdf
- 別添12 デジタル庁の体制（想定）.pdf
- 別添6 要求事項一覧_ストーリー定義.pdf
- 別添9 要求事項一覧_定義.pdf

Original file hashes are pinned in the specification. They identify historical
source artifacts; the task does not ask for current government policy. The roles
case intentionally asks separately about the diagram and table, whose labels
differ. Approximate schedules are not scored as confirmed dates. No source PDF or
locally installed font is redistributed in Git.

Annotations were authored by the implementing agent and visually checked against
the rendered source pages before inference. They are **not independently
human-validated gold**. Synthetic cases and these public documents may be easier
than deployment inputs; public-document training exposure is unknown. Two repeat
calls per case are not two independent documents. This is a development pilot,
not a held-out test or a statistical claim of general superiority.

## Controls and scoring

The initial run fixes `gpt-4.1-mini-2025-04-14`, temperature 0, seed 17/18 for the
two repeats, image detail `high`, image long edge 1800 pixels, and 1600 output
tokens. Three arms are evaluated for every case/repeat. All six arm permutations
are rotated before a fixed shuffle of case/repeat groups; sixteen groups do not
divide evenly into six permutations. The same seed is used within each group.
Provider seeds and temperature do not guarantee deterministic execution.

The model and image message format follow the official
[snapshot documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
and [vision guide](https://developers.openai.com/api/docs/guides/images-vision).
Response model IDs and system fingerprints are recorded. A model mismatch gets
no credit and stops later calls. This older snapshot is a fixed pilot baseline,
not a recommendation of the newest or best model.

Each answer must provide a string `value`, an original `page`, and a short `quote`.

- **Value correctness:** exact match to a predeclared accepted string after NFKC,
  whitespace removal, and numeric thousands-comma normalization. Units, signs,
  dates, and precision are retained. No LLM judge or post-hoc alias expansion.
- **Source-quote correctness** (`grounded_correct` in JSON): correct value, correct
  page, and a contiguous normalized quote contained in a predeclared reference
  excerpt with a relevant anchor. This is a deliberately narrow lexical check;
  it is not an entailment certificate and may reject otherwise valid quotes or
  longer excerpts. It is reported separately from value accuracy.
- **Unknown preservation:** correct `UNKNOWN` values and `unknown_format_mismatch`
  when an unknown field returns any other string. The latter includes literal
  `未定`, so it does **not** imply fabrication. Inspect the saved actual values to
  distinguish formatting mismatches from concrete invented answers. Invalid
  responses are counted separately.
- **Completion and validity:** truncated, blocked, empty, mixed, unknown-finish,
  wrong-model, malformed, duplicate-key, missing-field, or extra-field answers
  receive no field credit. Protocol completion alone never implies correctness.
- **Denominators:** all planned fields remain in accuracy denominators. Failed or
  unattempted pairs are `unavailable`, not silently dropped or tied.

Raw responses, exact question text, supplemental JSON, request/response hashes,
image hashes, usage, and latency are retained locally. Stored request records use
image hashes instead of duplicating base64 images. The frozen corpus retains the
actual images. The receipt distinguishes 48 planned calls from attempts and
completed/valid responses. Each call is written as `running` before dispatch.
Transport/provider failures stop remaining calls; there is no retry or automatic
resume. An interrupted/ambiguous call remains visible. Output directories cannot
be reused, and credentials/errors containing provider bodies are not logged.

Input token counts include the extra text/JSON by design; this is **not an equal
input-token-budget comparison**. All arms share the output cap and images.
Provider token usage is retained, with missing usage represented as unknown.
Latency includes network and provider scheduling, and cache/order effects may
remain. Preparation time is separately recorded; do not present request-only
latency as end-to-end processing latency or token counts as a billed invoice.

## Reproduce

Install `.[evaluation]`, Poppler (`pdftoppm`), and an embeddable Japanese TrueType
font. `.[dev]` suffices for offline regression tests; rendering is not done in CI.
The font hash and library versions are recorded. Raster images are the fixed
inputs of an individual run; regenerating with another renderer/font/version
produces a new corpus and must be labelled accordingly.

Preparation also works from an unpacked source distribution or archive. Git is
optional: `source.git_head` is `null` when a revision for this source root is
unavailable, while source-file hashes and dependency versions are still recorded.
A surrounding repository's revision is never used to identify an unpacked archive.

```bash
python scripts/vlm_benchmark.py prepare \
  --sources /path/to/03_仕様書等 \
  --font /path/to/JapaneseFont.ttf \
  --output artifacts/vlm/pilot-v1/corpus
```

Inspect **every** generated PNG, including synthetic PDFs. The first local
preparation used an unembedded CID font: text extraction succeeded but the
synthetic images had no visible glyphs. It was rejected before any API call.
The generator now embeds a real font. Preparation returns `prepared`; only after
visual inspection should the operator freeze it:

```bash
python scripts/vlm_benchmark.py freeze \
  --corpus artifacts/vlm/pilot-v1/corpus \
  --review-note "Describe the actual visual and annotation checks"

python scripts/vlm_benchmark.py run \
  --corpus artifacts/vlm/pilot-v1/corpus \
  --output artifacts/vlm/pilot-v1/run-01 \
  --allow-live-api --repeats 2 --max-requests 48 --max-output-tokens 1600

python scripts/vlm_benchmark.py score artifacts/vlm/pilot-v1/run-01
pytest tests/test_vlm_benchmark.py --benchmark-disable
```

The live command reads an existing `OPENAI_API_KEY` and sends selected page images
and derived text to the fixed OpenAI endpoint. It never provisions a key. `score`
is offline: it checks the frozen manifest/response hashes and re-grades saved raw
responses instead of trusting previously computed scores. It does not overwrite
the original receipt.

The first pilot runner named the strict UNKNOWN-mismatch counter
`unsupported_fill`. That name overstated what the string comparison measures.
The current scorer renames only that key to `unknown_format_mismatch`; accepted
answers, scores, and denominators are unchanged. The original runner and original
receipt are preserved with the pilot evidence. Literal `未定` remains a strict
contract error, not evidence of an invented person or date.

Change the corpus, model, prompt, parser, or accepted answers only in a separate
version/run. Use pilot errors to propose a narrow follow-up, then freeze new
counterfactual or held-out cases before measuring that change.
