# Local ontology memory

`sr-adapt knowledge ontology ingest` takes an original document or image through
source reading, semantic proposal and SQLite registration. One database holds the
original bytes, vocabulary, scoped claims, exact evidence, immutable versions and
model request/response receipts. No graph server, vector service, Fabric or Foundry
is needed. The model does not learn weights: it reuses the stored interpretations
through three read-only tools.

## From a raw source

```bash
pip install 'sr-unstructured-adapter[ontology]'

# Prepare and record the request locally. No credential or model call is needed.
sr-adapt knowledge ontology ingest policy.docx --database ontology.sqlite --model gpt-5.5

# Use the host's existing OPENAI_API_KEY; send this source for extraction.
sr-adapt knowledge ontology ingest policy.docx --database ontology.sqlite --model gpt-5.5 --execute

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
| PDF | Embedded text with physical page number | 1 |
| PNG / JPEG / WebP | Visual inventory, then semantics with original pixels again | 2 |

These are maximum planned requests for one successful run; validation can stop
earlier. No automatic retries are made. The first release has **no PDF page rendering
or OCR fallback**, and does not interpret embedded diagrams in DOCX/PPTX. For those
diagrams, ingest their images as separate entries. PDF layout/reading order is not
reconstructed; image-only pages produce warnings, and an entirely scanned PDF is
rejected before a model call. The general conversion pipeline's OCR settings do
not change this bounded ingestion route.

Images use a new inventory-only visual source contract (v3). It records visible
items, annotations and regions, then passes those readings and the original pixels
directly to semantic proposal. There is no intermediate binary relation extractor
that would force a condition or an unknown operator into a two-endpoint edge.
Existing v1/v2 visual bundles and request contracts remain unchanged.

Raw ingestion selects semantic proposal protocol v3, which constrains each logical
operator separately (`implies`: exactly two ordered operands, `not`: one, `and/or`:
two or more). It uses the supported `anyOf` and array-bound subset of
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
This addresses malformed logical shapes at generation time; it does not establish
semantic correctness. Stored ontologies remain `relation_semantics/2`; existing
proposal v1/v2 schemas and requests retain their exact replay behavior.

Input is limited to 16 MiB, PDF to 100 pages, and source units to existing parser
bounds (500 units and 64 KiB per unit). Semantic context/request/response/archive
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

Source/semantic SQL rows, archive, document activation and successful ingestion
receipt commit together. An entry changed during model execution makes activation
fail; its response remains available for diagnosis. Credentials, provider errors
and document text are not copied into public exception messages.

Existing completed bundles can also be imported without a model call:

```bash
sr-adapt knowledge ontology import <semantic-or-proposal-bundle> --database ontology.sqlite
```

Both `relation_semantics/1` and `/2` work. Handwritten semantic bundles and completed
model proposals preserve their own provenance. This layer generalizes local
storage and reuse; it does not certify extraction accuracy or collapse a tentative
reading into an operational fact.
