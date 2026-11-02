# Local ontology memory

`sr-adapt knowledge ontology ingest` takes an original document or image through
source reading, semantic proposal and SQLite registration. One database holds the
original bytes, vocabulary, scoped claims, exact evidence, immutable versions and
model request/response receipts. No graph server, vector service, Fabric or Foundry
is needed. The model does not learn weights: it reuses the stored interpretations
through three read-only tools.

The [Luna/OCR development observation](ocr-ontology-2026-11-03.md) records the
original image and mixed-PDF runs. The [recovery example](#recover-a-rejected-vocabulary)
shows how the saved mixed-PDF response was subsequently registered.

## From a raw source

```bash
pip install 'sr-unstructured-adapter[ontology]'

# Prepare and record the request locally. No credential or model call is needed.
sr-adapt knowledge ontology ingest policy.docx --database ontology.sqlite --model gpt-6-luna

# Use the host's existing OPENAI_API_KEY; send this source for extraction.
sr-adapt knowledge ontology ingest policy.docx --database ontology.sqlite --model gpt-6-luna --execute

# All commands below use only the SQLite database, with no model/network call.
sr-adapt knowledge ontology list --database ontology.sqlite
sr-adapt knowledge ontology search 'approval' --kind claim --database ontology.sqlite
sr-adapt knowledge ontology inspect --database ontology.sqlite --collection default \
  --name policy.docx --snapshot-id <snapshot-from-search> --kind claim --id <claim-from-search>
```

Preparation and execution are separate recorded runs. Execution builds the request
again from the supplied file; it does not silently resume a saved plan. If the same
source, model, adapter and protocol are already active, ingestion verifies the SQL
snapshot and returns `already_active` with `submitted_requests_this_call: 0`.
The original receipt retains its historical `submitted_requests` count.

Use `--collection project-a` to group documents and `--name policy` for a stable
document name. Different entries can hold different source formats and ontologies.
To replace an entry, supply `--expected-snapshot <current-version>` from `list`.
Versions already used in answers remain readable by their exact search reference.
An entry with the same filename is never silently overwritten. Reusing the same
term label or IRI in two documents does not merge their identities or conclusions.

| Input | Reading before semantic proposal | Model requests |
| --- | --- | --- |
| TXT / Markdown | Local text units | 1 |
| CSV / TSV | Cells with row/column/header context | 1 |
| DOCX | Paragraphs, tables, supported document parts and locators | 1 |
| PPTX | Native text and tables with slide/shape locators | 1 |
| PDF without `--ocr` | Embedded text with physical page number | 1 |
| PNG / JPEG / WebP without `--ocr` | Visual inventory, then semantics with original pixels again | 2 |
| PDF with `--ocr auto` | Per-page embedded text or local OCR, plus every selected page's pixels | 1 |
| PDF with `--ocr always` | Local OCR for every selected page, plus pixels | 1 |
| PNG / JPEG / WebP with `--ocr auto` | Local OCR with normalized image pixels | 1 |

These are maximum planned requests for one successful run; validation can stop
earlier. No automatic retries are made. Ingestion does not interpret embedded
diagrams in DOCX/PPTX; ingest their images as separate entries. Without `--ocr`,
PDF uses the existing embedded-text route: image-only pages produce warnings and
an entirely scanned PDF is rejected. The general conversion pipeline's OCR
settings do not change this ingestion route. The CLI and Python ingestion API
default to `gpt-6-luna`; an explicit `--model` still overrides it.

## Local OCR and mixed PDFs

```bash
pip install 'sr-unstructured-adapter[ontology,ocr]'

# macOS: on-device Vision OCR via Swift command-line tools.
# Preparation performs local OCR and saves its request; no model is called.
sr-adapt knowledge ontology ingest scanned.pdf --database ontology.sqlite \
  --ocr auto --ocr-engine vision --pages 1 2

# Execute with Luna; text and selected page images are sent together.
sr-adapt knowledge ontology ingest scanned.pdf --database ontology.sqlite \
  --ocr auto --ocr-engine vision --pages 1 2 --execute

# Other platforms: install Tesseract and the required language data first.
sr-adapt knowledge ontology ingest scan.png --database ontology.sqlite \
  --ocr auto --ocr-engine tesseract --ocr-languages jpn eng --execute
```

`auto` uses embedded PDF text only when it has at least 16 non-whitespace-trimmed
characters, no replacement/NUL characters and no raster images (including Form
and inline images). Other selected pages go through OCR. This is a routing
heuristic, not a text-accuracy check. Use `always` for a suspect embedded text layer.
Both routes retain page images so the model can inspect layout and conditions.
Text in vector diagrams may still need visual interpretation. OCR word order is
not a table reconstruction; no rows, columns or cell relationships are invented
by the local reader.

Engine `auto` selects Apple Vision on macOS with Swift installed, otherwise the
[Tesseract CLI](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).
Missing engines/languages fail explicitly, without falling back to a cloud call.
Vision defaults to `ja-JP en-US`, revision 3, accurate recognition and language
correction disabled. Tesseract uses installed `jpn`/`eng` languages by default,
page segmentation mode 3 and TSV word receipts. Selected language data hashes
also participate in reuse identity. It joins words within a line
with spaces; it does not normalize Japanese spelling or repair OCR errors.
Low confidence readings are retained; these scores are not calibrated probabilities.

PDF views use [PDFium](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
at `--ocr-dpi 144` (72–300), capped to a 2,500-pixel longest edge. PDFium calls in
this module are serialized. Images are EXIF-transposed and scaled to that edge.
Line boxes use top-left **rendered pixel** coordinates; they are not PDF points
or exact boxes around selected role phrases. Each frame records dimensions/hash.
The original source, plan, rendered PNGs, per-page OCR text/word receipts and
engine/renderer versions are archived in SQLite. Replay verifies the saved
readings, hashes and source projection; it does not rerun OCR or independently
re-render the PDF to certify pixel equivalence.

Select up to 12 physical pages, in increasing 1-based order; omitting `--pages`
selects all pages only if they fit this bound. Unselected pages are explicitly
listed as outside coverage. Limits include 16 million pixels per input image,
8 MiB per rendered frame, 20 MiB across frames, 4 MiB per reading and 28 MiB per
source package, plus the existing context/archive bounds. Exceeding any bound
fails explicitly instead of truncating a successful result.

Raw ingestion defaults to source-agnostic semantic proposal protocol **v5**.
It constrains vocabulary/list sizes, distinguishes meanings with separate IRIs,
retains source-local draft/conflict scope and restricts non-null validity fields
to timezone-bearing timestamps. Native and OCR sources use medium reasoning.
`--proposal-version 3` or `4` selects an earlier generation contract; OCR requires
version 4 or 5. Reuse an older active entry with its original protocol, or supply
`--expected-snapshot` when changing its generation profile.

Protocols **v4/v5** send each narrative source unit
once with its layout attached, avoiding repeated source IDs and location rows.
Full scalar rows remain unchanged in the database and binding receipts. Page
pixels stay in the request. V4 preserves bounded list sizes in the generation
schema and uses medium reasoning effort; older v1/v2/v3 request contracts remain
unchanged. All resulting semantics still use `relation_semantics/2`.

Reuse identity includes the source hash, page selection, DPI, engine signature,
renderer versions, model and proposal protocol. An unchanged active entry
returns `already_active` **before rendering, OCR or model calls**; this call's
counts are `ocr_pages_this_call: 0`, `rendered_pages_this_call: 0` and
`submitted_requests_this_call: 0`. Historical counts remain in the original
receipt. A changed identity requires the current `--expected-snapshot` before
reading pages. Hosts can supply `ocr_reader`, a callable taking PNG bytes with
a stable JSON `signature` containing `engine` and `version`; change that signature
when recognition behavior changes. See `ocr_readers.py` for the line contract.

Images use a new inventory-only visual source contract (v3). It records visible
items, annotations and regions, then passes those readings and the original pixels
directly to semantic proposal. There is no intermediate binary relation extractor
that would force a condition or an unknown operator into a two-endpoint edge.
Existing v1/v2 visual bundles and request contracts remain unchanged.

Raw ingestion retains semantic proposal protocol v3, which constrains each logical
operator separately (`implies`: exactly two ordered operands, `not`: one, `and/or`:
two or more). It uses the supported `anyOf` and array-bound subset of
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
This addresses malformed logical shapes at generation time; it does not establish
semantic correctness. Stored ontologies remain `relation_semantics/2`; existing
proposal v1/v2 schemas and requests retain their exact replay behavior.

In v5, a syntactically valid selector that cannot resolve within its known source
unit becomes an explicit `unknown` role value. The original selector and source
reference remain in `review.json.anchor_findings`, with review-required flags.
Claim evidence remains available. Unknown aliases, wrong source-table selectors,
malformed selectors and role constraints that exclude unknown still reject the
proposal. Earlier protocols retain their strict selector validation.

For supplying several inspected claims to a model, `pack_ontology_context` and
`unpack_ontology_context` share identical fields only within the same collection,
document and snapshot. Source tables use named columns, constant fields and row
arrays. Roundtrip verification restores every input JSON value, including numeric
types and original source strings. The host supplies verified inspection results.
Bounds are 100 inspections, 32 MiB expanded and 8 MiB packed; excess data fails
explicitly. See the [cross-format observation](cross-format-ontology-2026-11-03.md).

Input is limited to 16 MiB, PDF to 100 pages, and source units to existing parser
bounds (500 units and 64 KiB of extracted UTF-8 text in total). Semantic context/request/response/archive
bounds remain those of [semantic proposals](semantic-proposals.md). Large files
fail explicitly; they are not silently split, summarized or truncated.

Ingestion temporarily uses the existing verified Parquet bundle protocol and
therefore requires the `ontology` extra (PyArrow). The temporary directory is
removed after completion or failure. **Retrieval needs neither PyArrow nor those
directories**: it reads and verifies the scalar SQL tables. The durable database
also archives portable bundle bytes for provenance, but retrieval does not open
their Parquet files. Database backups should use SQLite's normal backup facility
when writers are active; copy the closed file for offline transfer.

## Let a model reuse the DB

```python
from pathlib import Path
from sr_adapter.ontology_tools import OntologyTools

tools = OntologyTools(Path("ontology.sqlite"), collections=("project-a",))
definitions = tools.definitions()  # name, description, provider-neutral input_schema

# Pass definitions to your model's tool API, adapting only the outer envelope.
# Dispatch the selected tool with its JSON arguments:
result = tools.call("search_ontology", {
    "collection": "project-a", "query": "approval", "kind": "claim",
})
for hit in result.get("hits", []):
    action = hit["inspect"]
    evidence = tools.call(action["name"], action["arguments"])
    # Supply evidence to the model for a source-backed answer.
```

The tools are `list_ontologies`, `search_ontology` and `inspect_ontology`. Trusted
host code chooses the database and collection allowlist. The model cannot submit
SQL, select paths, ingest content, change active versions or call a provider.
Search uses a literal, case-insensitive Unicode substring: it discovers vocabulary
and claims, with filters for claim status/modality, stable paging and pinned
references. It is not an embedding search or an ontology reasoner. The collection
scan is bounded to 50 entries; select `name` for a larger collection.

Search summaries deliberately do not contain the whole logical scope. Inspect
the returned reference before answering: it returns the complete linked claim
component, including conditions, negation, time, attribution, dissent, original
source context and v2 role anchors. Inspecting a term returns its declaration,
related vocabulary/axioms and references to claims using it. These are declared
interpretations, with `semantic_validation: not_performed`; inferred subclass
closure, equivalence resolution or truth adjudication is not performed.

Inspection also returns `value.vocabulary_review`. Each archive records its
vocabulary policy and response model, then lists relevant terms with their
individual description strings, zero-based declaration positions and JSON
Pointers into the original proposal. `review_required: true` marks parallel
descriptions that need interpretation. Claim inspection includes declarations
used by its reachable expressions; unused vocabulary has its own term reference.
Search summaries set `declarations_included: false` so consumers can request the
full inspection before interpreting a combined description.

Multiple archived responses can produce the same semantic snapshot. Their
declarations remain scoped by `archive_id`. Inspection returns one archive per
page, with `has_more` and a complete `next_inspect` action for the model to call.
The CLI accepts `--vocabulary-offset` from that action. A null aggregate
`review_required` means the complete declaration history has yet to be established,
including explicit overlays that have no archived model declarations. Per-term
flags apply to the displayed archive. Follow all pages when assessing its history.

Readback verifies archive, manifest, configuration, response and proposal hashes,
re-parses the original model response and compares its compiled vocabulary with
the SQL terms. This uses SQLite and core Python dependencies. Source folders,
Arrow, OCR and PDF rendering are unnecessary for inspection. Reads are bounded
to 32 MiB of archive metadata per page, and the model tool response budget also
applies. Earlier strict proposal archives and explicit semantic overlays retain
their own provenance.

Each read checks descriptor/schema hashes, source rows and the semantic projection
inside a read-only SQLite transaction. A changed active version cannot silently
redirect a saved reference. Tool responses are capped at 256 KiB by default
(host-configurable from 1 KiB to 1 MiB). An oversized scope returns
`response_too_large`, never a clipped claim. Hosts can use `inspect_ontology`
directly when they deliberately need a larger trace. Document/model text is data,
not agent instructions.

For a trusted local UI, `read_ontology_source(database, collection=..., name=...,
snapshot_id=..., package_id=...)` returns hash-checked original bytes and filename.
The bytes API is intentionally not among the model's tools.
For OCR sources, `read_ontology_frame(database, collection=..., name=...,
snapshot_id=..., package_id=..., page=...)` also recovers a hash-checked page PNG
and its saved reading from SQLite alone, with no OCR engine, PDF renderer or
PyArrow. Use the package ID from an inspected claim and the physical page number
in its source locations. Original files may be removed after successful ingestion.

## Swap the extraction model

```python
from sr_adapter.ontology_ingest import ingest_ontology_source

receipt = ingest_ontology_source(
    "policy.docx", "ontology.sqlite", model="my-local-model",
    complete=my_model_adapter, adapter_id="local-adapter-v1", execute=True,
)
```

The host callable receives the complete Responses-shaped request, including the
source context, JSON schema and image bytes when applicable. It returns a complete
normalized response envelope (`status`, actual `model`, and assistant
`output_text` containing the JSON proposal). The host owns model invocation,
credentials and schema support; this package does not install or start a local
model. Its stable `adapter_id` participates in reuse identity and must change when
adapter behavior changes. This interface has offline callable tests; a particular
local model's extraction quality is not thereby validated.

The default CLI transport is OpenAI Responses. Credentials come only from the
explicit Python `api_key` argument or the host's `OPENAI_API_KEY`; they are not
written to requests or the database. Source parsing itself is local. The archived
narrative source reader uses an explicitly named `local-source-reader` envelope
with zero assertions to reuse the existing source package contract; that is not
an additional model invocation or a preliminary relation extractor.

## Failed, interrupted and abstained runs

```bash
sr-adapt knowledge ontology status <run-id> --database ontology.sqlite
```

The same DB stores the raw source, submitted requests, received responses and
stage receipt. `failed` and `unknown` never activate a partial ontology. A model
abstention is retained as `abstained` with its coverage report; no empty successful
ontology is invented. A transport interruption remains `unknown` because a remote
execution may have occurred. Process termination may leave a `running` stage:
inspect the saved receipt before explicitly starting another run. There is no
automatic retry or background resume.
Rendered OCR pages and received readings are saved incrementally before semantic
generation. A local OCR failure therefore retains the source, plan and completed
page records without activating a partial ontology or spending a model call.

Source/semantic SQL rows, archive, document activation and successful ingestion
receipt commit together. An entry changed during model execution makes activation
fail; its response remains available for diagnosis. Credentials, provider errors
and document text are not copied into public exception messages.

### Recover a rejected vocabulary

Ingestion compiles model vocabulary declarations locally before registering a
proposal. It groups terms by their exact IRI within that proposal. Repeated
declarations with the same kind and label produce one term. Its description
contains each distinct nonempty note verbatim, in Unicode order, separated by a
blank line. `status` and the proposal review list every original declaration,
including its position and description. Differing descriptions remain available
for review alongside the claims that use the term.

If the same IRI has different kinds or labels, the run returns `vocabulary_conflict`.
`status` shows both declarations and the conflicting fields. The existing entry
remains active. The same checks apply to classes, relations, roles and individuals
across text, image and OCR inputs. Claim IDs, evidence, argument selectors and
role constraints still pass through the semantic validator.

To reprocess an earlier run that received a complete semantic response but failed
validation, use its original run ID. Recovery verifies the archived source,
request and response hashes, rebuilds the **identical** semantic request locally,
and records a separate attempt linked to the original. It uses the `ontology`
extra to rebuild portable source tables. It makes zero model, OCR or rendering
calls. Text sources, saved visual inventories and saved OCR pages are supported.

```bash
sr-adapt knowledge ontology recover <original-run-id> --database ontology.sqlite
sr-adapt knowledge ontology status <recovery-run-id> --database ontology.sqlite
```

Recovery uses the original collection and document name. If a different version
is active, add `--expected-snapshot <current-version>` to replace it. Repeating a
completed recovery for the same active entry returns its existing run ID. The
failed receipt and original response remain available by the original run ID.
Archived source filenames follow the source reader's metadata contract, including
names containing line breaks or tabs when an explicit safe document name was used.

In the synthetic mixed-PDF case, Luna declared `urn:sr:proposed:condition` twice
with the label `条件`, once for a table and once for an exception. Replaying that
response produced 18 terms from 19 declarations. All 7 original claims, 10
expressions and 14 exact text anchors were verified. The two notes are retrievable
from SQLite. One fresh `gpt-6-luna` call with the byte-identical request also
registered 7 candidate claims, this time with 19 distinct terms. These counts
describe the two runs on this fixture.

Existing completed bundles can also be imported without a model call:

```bash
sr-adapt knowledge ontology import <semantic-or-proposal-bundle> --database ontology.sqlite
```

Both `relation_semantics/1` and `/2` work. Handwritten semantic bundles and completed
model proposals preserve their own provenance. This layer generalizes local
storage and reuse; it does not certify extraction accuracy or collapse a tentative
reading into an operational fact.
