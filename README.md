<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SR Unstructured Adapter

Turn chaotic documents into structured payloads with a pipeline that speaks both native kernels and LLMs.

The [portable relation dataset layer](docs/relation-datasets.md) separates source
interpretation from destination storage. Replay-verified document and image
relations become immutable, typed scalar tables with source IDs, evidence,
uncertainty and versioned profiles. Fabric and local SQLite consume the same
tables; portable Parquet needs no cloud target or new model call. Start with
`sr-adapt knowledge dataset <saved-bundle> --profile narrative --out <new-directory>`.
The same dataset also produces [search cards and an assertion graph](docs/relation-views.md)
with shared evidence IDs: `sr-adapt knowledge dataset-view <dataset> --view search --out <new-directory>`
(or `--view graph`). Both views are local, replayable,
and readable through the existing SQLite sink.

Receive both views together with the [local receiver](docs/relation-receiver.md):
`sr-adapt knowledge dataset-receive <search-view> <graph-view> --database received.db --collection demo`.
Search and trace APIs retain the selected source version through updates.

An explicit [semantic overlay](docs/relation-semantics.md) adds typed vocabulary,
n-ary roles, nested conditions/negation, modality, validity intervals and competing
viewpoints. It preserves the source snapshot and replays into scalar Parquet/SQLite
tables. Start with `python scripts/demo_relation_semantics.py --out <new-directory>`;
the example uses handwritten synthetic declarations, with no model or cloud calls.

[Semantic proposals](docs/semantic-proposals.md) now generate that overlay from
saved document/image datasets, with original image pixels, source citations,
coverage/omission notes and a review report. Prepare with
`python -m sr_adapter.semantic_proposal_runtime propose <dataset> --model <model> --out <new-directory>`;
`--execute` makes one explicit OpenAI Responses call. Results remain unreviewed
model candidates, with the original request/response retained for replay.

The opt-in [table knowledge and lakehouse path](docs/table-knowledge-lakehouse.md)
preserves ruled PDF cells and coordinates, maps them to grounded entity mentions
and relations, and exports JSON-LD, Parquet, SQLite, ADLS Gen2, and Databricks Delta
records. Cloud connectors have offline protocol tests; deployment-specific cloud
authentication and writes require validation in the destination environment.

The [Foundry IQ bridge](docs/foundry-iq.md) projects those facts into complete row
contexts for Azure AI Search and generates a knowledge source, knowledge base,
and prompt agent using the native retrieval MCP tool. It preserves relation
direction, missing values, and cell evidence. Export locally with
`sr-adapt knowledge foundry <bundle> --out <new-directory>`; an explicit target
and `foundry-publish --execute` opt into cloud configuration.

The [document relation extraction path](docs/narrative-relations-foundry.md)
also turns Markdown/text, CSV, DOCX, PPTX and images into relation CSV tables using an
existing Foundry model. It preserves positive, negative, unknown and ambiguous
claims with their source evidence, and publishes them through the same Foundry IQ
bridge. Model interpretation and image transcription remain explicitly unverified.
Saved document relations also export as [ordinary Fabric tables](docs/fabric-lakehouse.md)
with `knowledge fabric --mode narrative`, preserving negation, source locations and
optional nonliteral model labels without another model call.
The [GPT-5.5 synthetic pilot](docs/narrative-foundry-2026-10-17.md) includes live
Foundry retrieval, frozen direct-source comparisons and offline replay, with
token overhead and evaluation limits retained.

## Why this adapter?
The [Fabric IQ blueprint](docs/fabric-iq.md) prepares a Fabric Data Agent,
Foundry IQ knowledge base, and prompt agent over one verified visual-table
snapshot. It keeps image evidence and review findings in scope and generates
configuration locally; live retrieval and user permissions require separate validation.

- **Streaming document pipeline** – Detects formats, parses into blocks, normalises text, and applies recipes with streaming output where the parser supports it. Memory use still depends on the document library and format.
- **Native acceleration** – Visual layout and text normalisation are executed by C++ kernels orchestrated through a shared runtime for deterministic telemetry and warm-up.
- **LLM escalation built-in** – Drivers share circuit breakers, exponential backoff with jitter, streaming + async APIs, and telemetry hooks while routing low-confidence spans to Azure OpenAI, OpenAI, Anthropic, Docker, or local vLLM endpoints before normalising responses.
- **Config-first ergonomics** – Recipes describe parsing behaviour, while tenant and adapter YAML plus `.env` overrides keep credentials and runtime toggles out of code.
- **Observability ready** – Kernel and LLM latency, payload sizes, and failures flow to Prometheus or Sentry with per-service labels straight from the CLI.
- **Recipe autopilot** – Feed the CLI a few labelled examples and it proposes regex-based recipes, scores them against negative samples, and emits YAML for review and use with the recipe registry.
- **Hybrid embeddings** – A lightweight `BlockEmbedder` mixes hashed text embeddings, layout, metadata, and optional semantic-field stats into deterministic vectors plus a pluggable search index for clustering or FAISS-backed recall.
- **Adaptive kernels** – The autotuner benchmarks batch sizes for the native runtime and records the best settings for future runs straight from `kernels autotune`.

## Architecture at a glance
```
  +---------------------------+
  |  Input sources            |
  |  (files, streams)         |
  +-------------+-------------+
                |
                v
       +--------+---------+
       | Type + MIME      |
       | detection        |
       +--------+---------+
                |
                v
       +--------+---------+
       | Parser registry  |
       +--------+---------+
                |
                v
       +--------+---------+
       | Parsed blocks    |
       +--------+---------+
                |
                v
       +--------+---------+
       | NativeKernelRuntime|
       | (text & layout     |
       |  kernels)          |
       +--------+---------+
                |
                v
       +--------+---------+
       | Recipe transforms |
       +---+-----------+---+
           |           |
           |           v
           |   +-------+--------+
           |   | Confidence      |
           |   | check           |
           |   +---+--------+----+
           |       |        |
           |   High|        |Low
           |       v        v
           |   +---+----+  +---------------+
           |   | Writers |  | DriverManager|
           |   | (JSONL/ |  | (Azure,      |
           |   |  API)   |  |  Docker, …)  |
           |   +---+----+  +-------+-------+
           |                        |
           |                        v
           |               +--------+--------+
           |               | LLM drivers     |
           |               +--------+--------+
           |                        |
           |                        v
           |               +--------+--------+
           |               | LLM normaliser  |
           |               +--------+--------+
           |                        |
           +------------------------+
                (enriched blocks)

       .-------------------------------------------.
       |  VisualLayoutAnalyzer (calibration store)  |
       '-------------------------+-----------------'
                                 |
                                 v
       +-------------------------+-----------------+
       |     NativeKernelRuntime (shared state)     |
       +-------------------------------------------+
```

## Key components
### Parser + recipe stack
1. `ParserRegistry` detects a best-fit parser by MIME or sniffed type and streams heavyweight formats (PDFs, images) chunk-by-chunk.
2. Each block is normalised, enriched by the active recipe, and written to structured documents or downstream sinks.

### Native kernels runtime
- `NativeKernelRuntime` coordinates the C++ text normaliser and layout analyser, capturing per-kernel telemetry and offering fast warm-up hooks.
- Layout calibration thresholds persist between runs and can be reused by matching profiles.
- Set `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` to fall back to the pure Python path or adjust batching with `SR_ADAPTER_TEXT_KERNEL_BATCH_BYTES`.

### Processing profiles
- Processing profiles bundle runtime layout preferences and LLM escalation policy so UX stays simple while the system adapts to each workload. The registry exposes built-ins (`balanced`, `realtime`, `archival`) and loads overrides from `configs/profiles/` or custom search paths.
- The `PipelineOrchestrator` resolves the active profile, warms the runtime when requested, and only escalates blocks that satisfy the profile's confidence, type, and limit criteria.
- CLI commands accept `--profile` so you can swap latency vs. fidelity trade-offs without changing recipes or code.

### LLM escalation
- `delegate.escalate_low_conf` loads the configured recipe, resolves the tenant, and invokes the appropriate driver through the shared manager cache.
- Drivers live in `src/sr_adapter/drivers/` and register themselves with a lightweight factory registry, so dropping in Azure, OpenAI, Anthropic, Docker, or vLLM backends requires no manager changes.
- Responses are normalised into a stable schema before the pipeline writes them back into documents or CLI output.
- Blocks always expose `attrs.confidence_structural` (structural confidence; layout bounded) and optionally `attrs.semantic_confidence`/`attrs.confidence_semantic` when deterministic semantic scoring is enabled.
- To enrich escalations with retrieval context, set `llm.context.top_k` (or `llm.context_top_k`) in the recipe and the adapter will append a related-context section to the prompt.
- Profiles can override recipe-level retrieval context parameters by setting `llm.context` in the selected processing profile.

### LLM completion status
Normalized responses and each choice include a `status`. HTTP success alone does not establish that text generation finished:

| Status | Meaning |
| --- | --- |
| `complete` | Nonempty text with a recognized normal stop reason |
| `truncated` | Output token limit or context limit reached |
| `blocked` | Provider refusal, content filtering, or blocked prompt |
| `tool_call` | A tool call requiring handling by the caller |
| `empty` | No choices, or a normal stop without usable text |
| `unknown` | Missing, unfamiliar, or nonfinal finish reason |
| `mixed` | Returned choices have different statuses |

`result.is_complete` is true only for `complete`; this says nothing about factual correctness or extraction accuracy. Finish reasons, partial text, and the original provider response remain available. The mappings follow the [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat), [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons), and [Gemini finish reasons](https://ai.google.dev/api/generate-content#FinishReason).

The pipeline preserves incomplete responses in `attrs.llm_escalations` and adds a document warning without changing the source text. `meta.llm_escalations` counts selected blocks receiving a new response; `llm_completed` and `llm_incomplete` split that count by completion. These counts precede output block truncation and are not provider request counts. Configuration/transport failures still use the existing failure logging path. Adaptive profile feedback uses completed responses; it does not treat refused or truncated responses as successful completion.

`llm run` prints the normalized response and exits with code 10 for any noncomplete status, including missing finish reasons. `llm replay` also fails and preserves any existing output file. With `--skip-errors`, replay retains incomplete response rows, continues processing, and still exits nonzero; only complete responses count toward `--limit`.

For bounded live integration checks using synthetic text and existing environment credentials:

```bash
python scripts/smoke_llm.py --allow-live-api --provider all --output live-results.json
```

The runner covers OpenAI (`OPENAI_API_KEY`), Mistral (`MISTRAL_API_KEY`), and Gemini (`GEMINI_API_KEY`) through sync, async, streaming, and async streaming calls. It allows at most four requests per provider, disables retries, and stops that provider after its first failure. Select a single provider with `--provider` or override its model with `--openai-model`, `--mistral-model`, or `--gemini-model`. An output path must be new and writable before any request; receipts are updated after each mode. Missing credentials, skipped modes, and interrupted requests remain visible alongside passes. A receipt left `running` after termination means the request's outcome is unknown; inspect it before deciding to run again. See the [October completion audit](docs/llm-completion-audit.md) for observed results and limitations.

### Escalation gate benchmarks
Use the built-in ablation harness to quantify the confidence gate trade-offs:

```bash
python scripts/benchmark.py --dataset data/escalation_benchmark.jsonl --update-readme
```

<!-- BENCHMARK:START -->
| use_structural_gate | use_semantic_gate | use_aif | precision | recall | F1 | mean ms | p50 ms | p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.200 | 0.333 | 0.250 | 0.01 | 0.01 | 0.01 |
| 0 | 0 | 1 | 0.667 | 0.667 | 0.667 | 0.09 | 0.09 | 0.14 |
| 0 | 1 | 0 | 0.500 | 0.667 | 0.571 | 0.07 | 0.00 | 0.19 |
| 0 | 1 | 1 | 0.750 | 1.000 | 0.857 | 0.10 | 0.08 | 0.13 |
| 1 | 0 | 0 | 1.000 | 0.667 | 0.800 | 0.00 | 0.00 | 0.01 |
| 1 | 0 | 1 | 1.000 | 0.667 | 0.800 | 0.02 | 0.01 | 0.05 |
| 1 | 1 | 0 | 1.000 | 1.000 | 1.000 | 0.02 | 0.00 | 0.06 |
| 1 | 1 | 1 | 1.000 | 1.000 | 1.000 | 0.04 | 0.05 | 0.07 |

_Generated from `data/escalation_benchmark.jsonl` via `python scripts/benchmark.py --update-readme`._
<!-- BENCHMARK:END -->

## Installation
Install the adapter in editable mode while iterating:

```bash
pip install -e .
```

Native kernels are compiled on first use when a C++17 toolchain is available. The wheel includes the C++ sources and recipe YAML. Compiled libraries live in `SR_ADAPTER_NATIVE_CACHE`, or `$XDG_CACHE_HOME/sr_adapter/native` (default `~/.cache/sr_adapter/native`), so installed package directories can remain read-only.

## Configuration
### Recipes
Recipes live under `src/sr_adapter/recipes` and control classification rules, confidence thresholds, and escalation toggles.

### LLM tenants
1. Copy `.env.llm.example` to `.env` (or export the variables manually).
2. Drop tenant YAML into `configs/tenants/` in the working directory, or set `SR_ADAPTER_TENANT_DIR`. The `driver` key selects a backend and `settings` contain endpoint-specific knobs.
3. Reference tenants from recipes via the `llm.tenant` field or override at runtime with `SR_ADAPTER_TENANT`.

OpenAI and Azure metadata are sent only when the tenant explicitly sets `store: true`; the default does not enable provider storage. Escalation metadata is retained locally regardless. OpenAI defaults to `max_completion_tokens`; set `max_tokens` explicitly for a backend that requires it. Sampling options such as `temperature` are only sent when configured. Anthropic accepts only its supported user metadata; Gemini safety settings belong in `settings.safety_settings`.

Mistral, OpenAI, Azure, Anthropic, Docker, vLLM, Gemini and compatible proxy drivers support native streaming. Gemini uses `:streamGenerateContent?alt=sse`, validates completion for every requested candidate, and retains final usage events. A custom Google proxy can set `settings.stream_endpoint`. The wire format follows the [Gemini REST reference](https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent). Bedrock and Vertex require an explicit endpoint for the supported API surface. Consume every asynchronous stream with `async for chunk in driver.async_stream_generate(prompt)`.

### Adapter settings
1. Global runtime defaults live in `configs/settings.yaml` (telemetry, driver defaults, distributed backends).
2. `sr_adapter.settings.get_settings()` merges YAML with environment overrides and `.env` values using Pydantic validation.
3. Override individual knobs via environment variables such as `SR_ADAPTER_DISTRIBUTED__DEFAULT_BACKEND=asyncio` or `SR_ADAPTER_DRIVERS__DEFAULT_TIMEOUT=45`.

For installed-wheel deployments, set `SR_ADAPTER_SETTINGS_PATH=/path/to/settings.yaml` and `SR_ADAPTER_DOTENV=/path/to/.env` explicitly. Automatic discovery of these two files is relative to the source checkout, not the service working directory. Exported environment variables need no dotenv file.

### Driver resilience defaults
- Configure exponential backoff, jitter, and retry counts globally with `drivers.retry_backoff_base`, `drivers.retry_backoff_max`, `drivers.retry_jitter`, and `drivers.max_retries`.
- Circuit breakers automatically pause failing tenants using `drivers.circuit_breaker_failures`, `drivers.circuit_breaker_window`, and `drivers.circuit_breaker_recovery`; override them per tenant when linking to fragile providers.

### Environment toggles
- `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` – force the legacy Python normalisers.
- `SR_ADAPTER_DISABLE_TEXT_KERNEL=1` – disable only native text normalization.
- `SR_ADAPTER_TEXT_KERNEL_BATCH_BYTES=<bytes>` – cap payload size per native call.
- `SR_ADAPTER_MAX_SIZE_MB=<float>` – guardrails for the classic adapter CLI.
- `SR_ADAPTER_SEMANTIC_CONFIDENCE=1` – annotate blocks with deterministic semantic confidence and include it in the escalation gate.
- `SR_ADAPTER_SEMANTIC_MAX_CONFIDENCE=<float>` – override the semantic gate threshold (defaults to `0.2` when semantic confidence is enabled via env).

## Recipe authoring guide
- Start by enumerating structural hints (`patterns`) that map regexes to block types and confidence scores; fall back to a safe default for everything else.
- Enable LLM escalation per recipe with `llm.enable` and provide prompt context/thresholds to nudge the driver toward the desired taxonomy.
- Document recipes alongside tenant configs so operators understand which driver features (streaming, retries, circuit breakers) the pipeline will exercise.
- Jump-start new regex recipes via `python -m sr_adapter.cli recipes suggest --positives examples.jsonl --name custom-title`; the helper learns a pattern from positive samples, evaluates it against negatives, and saves YAML.

## Troubleshooting
- Native build failing? Set `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` to fall back to Python while you inspect compiler logs, then re-enable once dependencies are installed.
- Validate kernel availability with `python -m sr_adapter.cli kernels status --json` and inspect `telemetry` snapshots for failure counters before escalating.
- When drivers flap, adjust the circuit-breaker window or retry backoff in `configs/settings.yaml` or the tenant override; the new metrics will show success/failure deltas immediately.

## Operations guide
- Expose Prometheus metrics (`python -m sr_adapter.cli kernels export --format prometheus`) and scrape both kernel + LLM series for SLA dashboards.
- Attach environment-specific labels (e.g. `service`, `tenant`) through `telemetry.labels` so PromQL slices align with deployments.
- Enable the Sentry exporter to capture structured runtime snapshots, including aggregated LLM metrics, for post-incident analysis.

## CLI quickstart
All orchestration commands live under `python -m sr_adapter.cli`.

### Convert documents
```bash
python -m sr_adapter.cli convert docs/*.pdf --recipe default --out output.jsonl --profile balanced --backend threadpool --concurrency 4
```
Stream parsed blocks into JSONL while optionally disabling escalation with `--no-llm`. Select another processing profile (e.g. `realtime`) to trade accuracy for latency, and choose a distributed backend (`threadpool`, `asyncio`, `dask`, `ray`) when scaling batch jobs.

## HTTP API quickstart (optional)
Install the optional API dependencies:

```bash
pip install "sr-unstructured-adapter[api]"
```

Run a local server:

```bash
sr-adapt-api --host 127.0.0.1 --port 8000
```

Convert a file upload:

```bash
curl -sS -F "file=@examples/sample.txt" "http://127.0.0.1:8000/convert?recipe=default&profile=balanced" | jq .
```

Stream parsed blocks as NDJSON (requires `llm_ok=false`):

```bash
curl -sS -F "file=@examples/sample.txt" "http://127.0.0.1:8000/convert-stream?recipe=default&profile=balanced&llm_ok=false" | jq -c .
```

Notes:

- Legacy `SR_ADAPTER_API_KEY(S)` grant unrestricted access. For tenant isolation, configure `SR_ADAPTER_API_KEY_TENANTS` as described below. Treat unrestricted keys as administrator credentials.
- Path conversion is disabled by default. Enable it with `SR_ADAPTER_API_ALLOW_PATHS=1` and then use `POST /convert-path`. This grants unrestricted callers access to server-readable regular files; scoped keys cannot use path conversion even when enabled.
- Upload guardrail: set `SR_ADAPTER_API_MAX_UPLOAD_MB=<float>` to enforce an upper bound (defaults to `SR_ADAPTER_MAX_SIZE_MB` when set, otherwise 200MB; set to `0` to disable).
- Authentication and raw body limits run before parsing uploaded files or path JSON requests, including when the app is mounted. The multipart envelope gets 1 MiB of overhead above the per-file limit. JSON booleans and integer limits are strict.
- Request IDs: responses include `X-Request-ID` (you can supply your own via the `X-Request-ID` request header).
- Optional rate limiting: set `SR_ADAPTER_API_RATE_LIMIT_RPM=<int>` (requests/minute; default disabled). When running behind a reverse proxy, set `SR_ADAPTER_API_TRUST_PROXY_HEADERS=1` to key limits by `X-Forwarded-For`; only enable this when the proxy overwrites that header. The in-memory limiter is per process.
- Telemetry endpoints: `GET /telemetry` (JSON) and `GET /metrics` (Prometheus; requires `telemetry.enable_prometheus=true`). Aggregate telemetry is unavailable to scoped keys because it includes other tenants.
- Optional API key auth: set `SR_ADAPTER_API_KEYS=key1,key2` (or `SR_ADAPTER_API_KEY=key1`) and send `X-API-Key: key1` (or `Authorization: Bearer key1`). `GET /healthz` stays public for liveness checks.
- Tenant override for LLM escalation: send `X-SR-Tenant: <tenant-name>` to force a specific tenant config.

### Tenant-scoped credentials

Set `SR_ADAPTER_API_KEY_TENANTS` to a JSON object of API keys and their exact tenant names. The following credentials are placeholders; supply your own secrets through the deployment environment:

```bash
export SR_ADAPTER_API_KEY_TENANTS='{"replace-with-alpha-key":["alpha"],"replace-with-beta-key":["beta"]}'
```

Scoped requests use `X-SR-Tenant`, or `SR_ADAPTER_TENANT` / `default` when the header is absent or blank. That tenant must appear in the key's scope and becomes the explicit conversion tenant. An invalid scope configuration prevents startup; it never silently disables authentication. A scope takes precedence if the same key also appears in the legacy unrestricted list.

Job lists are filtered before applying `limit`; status, results, and cancellation are restricted to the job's recorded tenant. Keys for the same tenant share its jobs, so key rotation retains access. Cross-tenant and historical jobs with no recorded tenant return 404 to scoped callers. Scoped keys cannot use server-path routes or aggregate telemetry. Legacy requests without a tenant header retain recipe-level tenant selection.

### Async jobs
For long-running conversions, submit a job and poll later:

```bash
# enqueue
curl -sS -F "file=@examples/sample.txt" "http://127.0.0.1:8000/jobs/convert?recipe=default&profile=balanced&llm_ok=false" | jq .

# check status
curl -sS "http://127.0.0.1:8000/jobs/<job_id>" | jq .

# fetch result (409 until ready)
curl -sS "http://127.0.0.1:8000/jobs/<job_id>/result" | jq .
```

To persist job status/results across restarts, configure the SQLite backend:

```bash
export SR_ADAPTER_API_JOBS_BACKEND=sqlite
export SR_ADAPTER_API_JOBS_DB_PATH=./sr_adapter_jobs.sqlite3
```

On startup, abandoned queued/running jobs become `interrupted`, with an unknown outcome; the service never automatically repeats conversions or paid LLM calls. Jobs owned by another live process remain untouched. Completed results remain available after restart. This is an in-process executor with persistent status, not a distributed queue: cancellation must reach the process holding the job's future.

Keep the database and its adjacent `.owners` directory together on a local filesystem. Stop old-version workers before upgrading the schema; they do not participate in owner locking. Do not delete owner sidecars while workers are running.

### Inspect LLM drivers
```bash
# List configured tenants
python -m sr_adapter.cli llm list-tenants

# Run a single prompt with inline metadata
python -m sr_adapter.cli llm run --tenant default --prompt "Summarise this" --metadata '{"source": "demo"}'

# Replay a JSONL dataset and capture responses
python -m sr_adapter.cli llm replay --input data/escalation_samples.jsonl --output replay.jsonl --skip-errors
```
The CLI validates prompts, streams normalized responses, and can skip failures while reporting a summary.

### Manage native kernels
```bash
# Show runtime status with telemetry
python -m sr_adapter.cli kernels status

# Compile and warm both kernels, emitting JSON
python -m sr_adapter.cli kernels warm --json

# Export Prometheus metrics (and optionally send to Sentry)
python -m sr_adapter.cli kernels export --format prometheus --label env=prod --sentry

# Autotune batch sizes for text/layout kernels and persist them
python -m sr_adapter.cli kernels autotune --profile balanced
```
Runtime instances are cached within a process. Compiled libraries and saved tuning parameters can be reused across CLI invocations.

### Generate recipe candidates
Create your positive and negative JSONL example files first; the paths below are placeholders.

```bash
python -m sr_adapter.cli recipes suggest \
  --positives data/title_examples.jsonl \
  --negatives data/title_noise.jsonl \
  --name auto-title --print-yaml
```
The CLI will report precision metrics, print the YAML recipe, and optionally write it to disk for review. To register it, add the YAML to `src/sr_adapter/recipes/` in a source checkout and reinstall; `configs/recipes/` is not a recipe search path.

### Embed and cluster blocks
```python
from sr_adapter.embedding import BlockEmbedder, EmbeddingIndex
from sr_adapter.schema import Block

embedder = BlockEmbedder(dimensions=64, use_sentence_transformers=False)
# Use embed_with_context(...) to inject deterministic semantic-field stats when desired.
blocks = [Block(text="An example paragraph.")]
vectors = embedder.embed_with_context(blocks)
index = EmbeddingIndex(len(vectors[0]))
for block, vector in zip(blocks, vectors):
    index.add(vector, {"id": block.id, "type": block.type})
```
Use the returned embeddings with FAISS or the built-in cosine search to cluster similar content, deduplicate documents, or pre-select the richest context window for downstream LLM calls.

## Library usage
Use the high-level helpers when embedding the adapter in another service:

```python
from sr_adapter.pipeline import batch_convert

documents = batch_convert(["examples/sample.txt"], recipe="default")
for document in documents:
    print(document.model_dump(mode="json"))
```
`batch_convert` applies detection, parsing, native normalisation, recipes, and escalation before returning structured documents.

### Async & streaming drivers
- Directly call `driver.async_generate(...)` or `driver.async_stream_generate(...)` when integrating with asyncio services; synchronous code can opt into `.stream_generate(...)` for incremental chunks.
- Provider implementations honour the same resilience, telemetry, and streaming interfaces so you can swap tenants without rewriting orchestration glue.

## Sample data
The `data/escalation_samples.jsonl` file provides quick prompts for replay testing.

## Development
Run the full test suite with:

```bash
pytest -q
```

The suite covers driver management, pipeline behaviours, native kernel orchestration, and CLI workflows.

### Continuous integration
GitHub Actions runs on Ubuntu with Python 3.10, 3.11 and 3.13, plus Windows with Python 3.11. The full suite includes native and fallback paths, API, job/profile persistence and provider protocols. Every job checks lint, then builds and exercises an installed wheel. Benchmark timings are disabled in this correctness CI; there is no performance gate.


## Verification and behavioral limits

```bash
pip install -e '.[dev,api]' ruff build
ruff check .
pytest --benchmark-disable
python -m build
```

CI exercises Python 3.10, 3.11 and 3.13, the API extra, native and Python paths, and an installed wheel away from the checkout. Provider protocol tests use simulated HTTP transports and make no billable requests.

Explicit, bounded live checks use synthetic text only (up to four requests per selected provider, retries disabled):

```bash
# Export existing credentials in your shell; never put them in tenant YAML.
python scripts/smoke_llm.py --allow-live-api --provider all --output live-results.json
```

Text normalization preserves code blocks. When prose normalization changes text with annotations, offsets are cleared and `attrs.spans_invalidated_by` explains why. Refinement preserves case and never treats an escalation probability as extraction accuracy. A failed binary parser yields a diagnostic warning instead of decoding container bytes as text; a parser that fails after emitting stream blocks raises instead of repeating the input.

Layout classification, synthesized PDF/image geometry, semantic hash scores, and adaptive profile rewards remain heuristics. They are not calibrated quality measurements. Parser-generated synthetic geometry is marked as heuristic rather than claimed as a measured source region. Actual OCR availability depends on the optional OCR installation. Adaptive profile updates reload and atomically replace the existing JSON under a stable `.lock` sidecar, so cooperating processes preserve each other's statistics. All writers must use this version and a filesystem supporting advisory locks and atomic replacement. Lock waits are bounded; failed feedback writes leave the previous state intact. SQLite preserves completed results and marks abandoned work `interrupted`; it does not automatically replay it.

The benchmark table above is historical evidence for its recorded dataset, not a performance claim for this revision. Live provider smoke tests verify integration and response handling, not general extraction quality. See [the audit notes](docs/comprehensive-audit.md) for the tested surface and remaining boundaries.

For a controlled document-quality comparison, the [paired VLM evaluation](docs/vlm-paired-evaluation.md) holds the model, images, questions, and output budget fixed while comparing images alone, images plus extracted text, and images plus adapter JSON. It includes a frozen Japanese synthetic/Digital Agency pilot, strict offline scoring, and an opt-in bounded live runner.

The [initial pilot readout](docs/vlm-pilot-2026-10-15.md) found no improvement on known values in its eight-document sample. Raw responses and replayable scoring receipts are retained; this is an exploratory observation, not a general quality claim.
