# Comprehensive adapter audit

Audit date: 2026-10-08. Starting revision: `e996558`.

This change concentrates on source fidelity, provider protocol correctness, service lifecycle, and behavior after installation. It does not establish general extraction quality or a speed improvement.

## Correctness changes

| Surface | Reproduced problem | Resulting behavior |
| --- | --- | --- |
| XLSX / DOCX / PPTX | Values-only cells treated as objects; tables moved after paragraphs; slide files sorted by filename | Cell values across sheets, document body order, and presentation relationship order are preserved |
| JSON / HTML / TSV / ICS | Repairs altered quoted URLs, code lost whitespace or duplicated, TSV delimiter and calendar folding mishandled | String-aware JSON repairs and format-aware parsing preserve source content |
| Binary detection and failures | Whole file read for a four-byte probe, PDF signature missed, failed containers decoded as text | Bounded signature read; diagnostic failure blocks and warnings; partial stream failure is visible |
| Normalization and refinement | Native/Python output differed, consumed generators lost data on fallback, code/case/spans were altered | Unicode-consistent normalization, retained generator input, exact code preservation, explicit invalidation of changed span offsets |
| PDF / image geometry | Synthetic boxes appeared as measured provenance and altered types/confidence; disabled native mode still compiled kernels | Heuristic geometry is labeled in attrs; measured provenance stays distinct; unavailable native layout preserves parsed text |
| LLM request payloads | Provider-incompatible metadata, unsupported implicit sampling, Azure streaming request attribute error | Provider-aware payloads; explicit OpenAI/Azure storage option; correctly encoded metadata |
| LLM streams | Retries replayed already-emitted chunks; malformed or truncated streams looked successful | No replay after emission; SSE completion and errors validated; consistent async iterator contract |
| Response normalization | Anthropic/Gemini shapes ignored; malformed responses escaped CLI handling | Normalized provider text/usage; retained original input on failure; ordinary CLI error status |
| Tenant configuration | Incorrect default directory, missing environment variables sent literally, shared breaker state across tenants | Installed/default tenant discovery, explicit environment expansion errors, isolated and synchronized driver cache |
| HTTP API | Event loop blocked, string false enabled LLM use, auth/limits followed multipart parsing, mounted apps bypassed early guards | Threaded blocking work, strict JSON options, early auth and bounded bodies including mounted routes |
| Jobs / CLI output | Cancelled uploads leaked, SQLite closed before draining, callback shutdown corrupted lifecycle, replay overwrote inputs/results | Cleanup hooks, safe executor shutdown, explicit rejection of worker shutdown, atomic replay output |
| Runtime / telemetry | Mutable snapshots changed retroactively, disable flags ignored tuned kernels, invalid Prometheus escaping | Detached snapshots, effective disable flags, valid text-format labels, finite geometry validation |
| Autotuning / profile state | Text samples never reached candidate batch limits; best single outlier selected; corrupt cache shapes crashed | Meaningful batch workload, median trial selection, tolerant cache reads and atomic profile writes |
| Packaging / CI | YAML/C++ resources absent from wheels; Python 3.10 incompatible datetime imports; API tests skipped in CI | Explicit package resources, supported timezone/TOML imports, API and installed-wheel validation across supported Python versions |

## Validation

The tests include real miniature Office/HTML/JSON/calendar documents, native/Python parity cases, mocked provider transports, malformed/partial SSE streams, mounted API requests, strict input validation, cleanup/cancellation, and persistent job lifecycle checks. Native libraries are compiled from the bundled C++ sources. The wheel is checked outside the source checkout with a separate installation.

The opt-in live runner uses only synthetic text. OpenAI `gpt-4.1-mini` and Mistral `mistral-small-latest` were exercised through synchronous, asynchronous, streaming, and asynchronous streaming driver interfaces. Additional checks sent synthetic uploads through a real localhost HTTP server, pipeline, tenant manager, and provider, then inspected the local escalation result. Failed live attempts were retained in the audit receipts rather than counted as passing.

Live requests exposed two gaps that mocked transport tests alone did not reveal: OpenAI rejects metadata unless storage is enabled, and Mistral rejects non-string metadata values. Storage is never enabled implicitly; typed metadata remains in the local escalation envelope.

Reproduce offline checks:

```sh
python -m pip install -e '.[dev,api]' ruff build
ruff check .
pytest --benchmark-disable
SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1 pytest --benchmark-disable
python -m build
```

Explicit live checks (credentials from the environment; four requests per selected provider, no retries):

```sh
python scripts/smoke_llm.py --allow-live-api --provider all --output live-results.json
```

## Boundaries

- Live validation covers OpenAI and Mistral only. Other providers have deterministic protocol tests; Gemini now has native SSE support, without live credential validation. Bedrock/Vertex require an explicit supported endpoint.
- Provider integration checks are small synthetic examples, not a comparative model-quality benchmark.
- Native layout classification, semantic hashing, escalation scores, and adaptive profile rewards remain heuristics. Optional OCR engines, FAISS, Dask, and Ray were not validated against production services or representative workloads.
- Legacy API keys remain unrestricted. Optional exact tenant scopes restrict conversions and persisted jobs, and disallow host-path conversion and aggregate telemetry. Scoped keys for the same tenant share its jobs; there is no user-level authorization within a tenant.
- Rate limiting remains local to one process. Profile persistence now serializes cooperating processes using an adjacent advisory lock and atomic publication. SQLite keeps completed results and marks abandoned work interrupted without replaying it. Local storage and stopped legacy workers are required for SQLite ownership migration.
- The historical README benchmark table is not regenerated into a new performance claim.


## Follow-up: operational boundaries

The follow-up preserves the earlier live receipts and adds the following changes:

| Boundary | Result |
| --- | --- |
| Tenant authorization | Exact credential scopes, authorization before body parsing, tenant ownership persisted with jobs, pre-limit list filtering, 404 for cross-tenant jobs, same-tenant key rotation, administrator-only host paths and aggregate metrics |
| Interrupted work | OS owner leases distinguish stopped workers from live siblings; abandoned jobs become `interrupted`; no automatic replay; legacy schema migration and concurrent initialization/recovery covered |
| Adaptive statistics | Existing JSON schema retained; cross-process reload/update/atomic publication under a stable sidecar lock; selection refreshes shared results; failed publication preserves the prior state |
| Gemini streaming | Native synchronous and asynchronous SSE, candidate completion validation, trailing usage retention, URL-template support, and no replay after emitted output |
| Portability | Full suite on Ubuntu Python 3.10/3.11/3.13 and Windows Python 3.11; installed-wheel conversion in each CI job |

Deterministic regression scenarios include a process terminated with `os._exit`, a simultaneously live sibling worker, four concurrent SQLite constructors, four spawned adaptive writers preserving 200 updates plus prior statistics, malformed credential scope configuration, and legacy recipe tenant routing. OpenAI/Mistral were called again through all four interfaces with synthetic data. A real localhost HTTP server verified tenant isolation, key rotation, streaming upload conversion, completed-job persistence after restart, and shutdown.

Gemini transport behavior is based on the [official generate-content REST reference](https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent), tested using mocked HTTP transport. No Gemini live credential was available. Windows native builds target GNU/MinGW-style compilers; MSVC is not supported by these compiler flags. Production OCR/FAISS/Dask/Ray workloads, calibrated extraction quality, and distributed rate limiting remain outside these results.


The initial Windows full-suite run exposed DLL loading failures from compiler runtime dependencies; its failure is retained as a separate CI receipt. Windows builds now link the compiler runtimes statically and include all compile/link flags in the cache identity, so earlier broken DLLs are not reused. Windows CI also loads fresh kernels in a child process with the compiler PATH removed.
