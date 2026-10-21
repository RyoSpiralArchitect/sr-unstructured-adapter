# Documents → proposed relation tables → Foundry IQ

The opt-in `knowledge narrative` path reads UTF-8 text/Markdown, CSV/TSV,
Word `.docx`, PowerPoint `.pptx`, and single-frame PNG/JPEG/WebP documents. It calls an existing
Foundry model deployment to propose statements using a caller-authored ontology.
It exports a normal CSV relation table and a verified evidence bundle, then the
existing `knowledge foundry` / `foundry-publish` path makes those statements
available to a Prompt agent through Foundry IQ's Azure AI Search knowledge base.
This path does not require Fabric or Databricks.

This is candidate extraction, not automatic truth or ontology certification.
`source_text_verified` means that a quotation exactly reproduces a source unit
extracted from the saved original bytes. By default the subject/object must appear literally
in the evidence (the raw header/value strings for CSV/TSV cells, before JSON
escaping), but this cannot detect every semantic error, including reversal,
misread negation or a wrong predicate. Assess those separately against reviewed
source material. `model_transcription_unverified` means the text came from a
model reading an image; matching that transcript does not validate the pixels.

For a PoC that should retain model labels even when they paraphrase the source,
pass `--label-policy preserve` to `knowledge narrative` (or `label_policy="preserve"`
to `extract_narrative` / `render_narrative`). The original labels and complete
quotes are preserved, with `subject_label_match` and `object_label_match` columns
indicating literal occurrence in the cited evidence. A null object has a null
match flag. False is an unverified interpretation, not certified normalization.
Quote mismatches, undeclared predicates, invalid statuses, incomplete responses
and duplicate assertions still reject. The model request remains unchanged;
this option controls acceptance, not extraction instructions. Default `literal`
bundles retain their previous bytes, identities and verification behavior.

A completed response from a failed literal attempt can be replayed offline with
`render_narrative(..., response=saved_response, label_policy="preserve")`, then
`save_narrative(new_folder, manifest, files)`. Supply the saved original bytes,
source name, ontology, model and instruction version; keep the failed receipt.
No model retry is needed. The acceptance policy changes the new bundle identity
and is checked again during verification.

Use `knowledge fabric <bundle> --mode narrative` for
[ordinary Fabric tables](fabric-lakehouse.md#word-and-other-document-relations-as-ordinary-tables).
This exports relations, entity mentions, source evidence and locator details as
scalar columns rather than JSON fields, retaining the same assertion statuses.

New extractions use instruction version 2. It asks for the applicable table
headers, unit captions, dates, scope and footnotes in each assertion's evidence,
so a separated row retains the context needed to interpret its literal value.
Numbers are not converted or combined with a separately printed unit. Empty
cells remain absent; explicit zero, unknown and unresolved alternatives remain
distinct. These are model instructions, not a guarantee of semantic correctness.
The [realistic synthetic development check](relation-context-2026-10-19.md)
scores correct values separately from correct relationships and context coverage.

## Run

Install the Foundry authentication extra and use an existing authenticated Azure
identity with access to the selected deployment:

```sh
python -m pip install -e '.[foundry]'
sr-adapt knowledge narrative data/knowledge/narrative-mock/memo-a.md \
  --ontology data/knowledge/narrative-mock/ontology.json \
  --project-endpoint https://YOUR-ACCOUNT.services.ai.azure.com/api/projects/YOUR-PROJECT \
  --model gpt-5.5 --out artifacts/narrative/memo-a --execute
```

Without `--execute`, this creates an offline request plan without authentication
or model calls. Each output directory must be new. A text/CSV/DOCX/PPTX extraction
uses one model request. An image uses two: transcription, then relation extraction
from the saved transcription. No automatic retries, implicit cloud publication,
OCR confidence scores or invented visual bounding boxes are used.

A complete run writes `bundle/relations.csv`, `assertions.json`, source bytes and
units, ontology, request/response snapshots, and a hash manifest. The parent
`receipt.json` records each attempt before submission. Failed/incomplete responses
remain in that directory; they do not become a publishable bundle. A transport
failure is an unknown outcome: inspect the receipt before a new attempt.

```sh
sr-adapt knowledge foundry artifacts/narrative/memo-a/bundle \
  --prefix sr-narrative --target local-foundry-target.json \
  --out artifacts/narrative/foundry
sr-adapt knowledge foundry-publish artifacts/narrative/foundry \
  --target local-foundry-target.json --receipt artifacts/narrative/publish.json --execute
```

Use the target shape in `data/knowledge/foundry-target.example.json` and a new,
dedicated connection name for each snapshot. Keep target credentials outside the
file; publication uses Azure identity. Inspect `deployment-plan.json` first.
Only explicitly selected existing Search/Foundry services are used; no service,
model deployment or role assignment is provisioned. Native retrieval and agent
answering must be tested separately from `configured` acknowledgements.

## Relation table

Every row has a source-scoped assertion ID, subject, ontology predicate, object,
status, subject/object types, grounding status, source filename/hash and structured
evidence locators. `object` is empty in CSV and JSON `null` when no value is
asserted. Preserve `status` when importing CSV so an empty value is not confused
with an asserted empty string.

| Status | Meaning |
|---|---|
| `asserted` | Model interprets the source as making a positive statement. |
| `denied` | Model interprets the source as explicitly denying the stated relation/value. |
| `unknown` | Source explicitly leaves the value unknown; no concrete object. |
| `ambiguous` | Unresolved choice or qualification; no definite object. |

Absence is not a negative fact. Identical names are separate source-assertion
mentions, with no global identity resolution, joins or transitive inference.
The human-authored ontology defines permitted predicate meanings and direction.
The extractor does not induce an ontology. Full source context is retained in
Foundry `evidence_json` alongside direct assertion evidence, so document-level
dates, conditions and identity qualifications remain inspectable.

## Source locators and limits

| Input | Evidence locator | Boundary |
|---|---|---|
| `.txt`, `.md` | Original one-based line | UTF-8; no Markdown semantic inference. |
| `.csv`, `.tsv` | Logical record, physical lines, column index/header/raw value | Duplicate headers remain positional; quoted multiline cells are retained; inconsistent widths reject. |
| `.docx` | Main-document XML part, block/paragraph; table number, grid row/column and explicit header context | Body paragraphs and table cells in document order, including nested tables/content controls. Merge markers and empty cells are retained; no rendered page numbers or inferred headers. |
| `.pptx` | Presentation-order slide, shape ID, table cell where applicable | OOXML text; images, diagrams, charts and notes are not visually interpreted. Unsupported content is reported. |
| `.png`, `.jpg`, `.jpeg`, `.webp` | Original image hash, frame and model transcript line | Transcript lines are not pixel coordinates; no claim of OCR/visual correctness. |

This new route does not yet accept PDF. The existing `knowledge extract` ruled-PDF
table path remains available. It also does not accept animated images, resolve
pronouns, normalize units, infer missing cells or follow external PPTX links.
Limits reject the whole input rather than silently truncating: 16 MiB input,
64 KiB extracted UTF-8 text, 500 source units, 200 candidates. Foundry cards have a
64 KiB ceiling, and publication is bounded to 1,000 cards / 12 MiB per batch.
Full evidence may reach these bounds before the raw file does.

### Word documents

The same CLI accepts a Word file without conversion or a new flag:

```sh
sr-adapt knowledge narrative requirements.docx \
  --ontology ontology.json \
  --project-endpoint https://YOUR-ACCOUNT.services.ai.azure.com/api/projects/YOUR-PROJECT \
  --model gpt-5.5 --out artifacts/narrative/requirements
```

This example prepares an offline request; add `--execute` for the existing
single-request extraction flow. The Python APIs remain
`read_source(data, "requirements.docx")` and `extract_narrative(path, ...)`.
Existing CSV exports, evidence replay and Foundry handoffs accept the resulting
bundle without a DOCX-specific publishing step.

Word locators count blocks and paragraphs within their body or cell. Tables are
numbered in document order, including nested tables; nested cells also identify
their parent table/cell. Columns follow the Word grid, so a merged cell's
`column_span` can exceed one. `horizontal_merge` and `vertical_merge` preserve
`none`, `restart` or `continue`; merged values are never filled down.
`header_context` contains overlapping earlier rows explicitly marked as repeating
headers (`w:tblHeader`), not a guess that the first row is a header. The source
units for those headers remain separately citable.

Visible hyperlink text and cached field results are included. Hyperlink targets
are never fetched; other external relationships are rejected. Deleted revisions
are omitted and inserted text is included, with a warning. Automatic list numbers,
headers/footers, footnotes, comments, drawings, text boxes and embedded objects are
not rendered or transcribed. Warnings describe these boundaries. The DOCX reader
uses the existing bounded ZIP/XML checks, including traversal, duplicate parts,
DTD/entity declarations and decompression limits. No macro-enabled Word format is
accepted.

## Reproducibility and existing bundles

Bundle verification re-extracts the saved original bytes, validates the completed
model response, and regenerates CSV/assertions/cards without another model call.
Changing derived rows and updating only their hashes is rejected. These hashes
provide internal consistency, not a digital signature or proof of model origin.

`extraction.json` records `instructions_version` for new bundles. A missing field
selects the immutable original version 1 request, preserving older bundle bytes
and their Foundry card IDs. Unsupported versions reject. `extract_narrative` and
`extraction_request` accept an explicit `instructions_version=1` for a controlled
baseline; the default is version 2. No model call is needed to replay either.

Narrative Foundry handoffs use schema v3. Existing table handoffs retain their v1/v2
byte representation and verifier. A single export accepts either table bundles
or narrative bundles; mixing these bundle types is rejected.

The synthetic fixtures are an authored development pilot, not a held-out measure
of generalization. Report extraction precision/recall, evidence location, format
coverage, model request completion, cloud readback and agent answers separately.
See [Microsoft's structured-output documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)
for the JSON-schema mechanism; it constrains output structure, not factual truth.
