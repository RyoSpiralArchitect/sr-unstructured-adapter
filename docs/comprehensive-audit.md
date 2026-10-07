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

- Live validation covers OpenAI and Mistral only. Other providers have deterministic protocol tests; Gemini currently uses the non-streaming fallback. Bedrock/Vertex require an explicit supported endpoint.
- Provider integration checks are small synthetic examples, not a comparative model-quality benchmark.
- Native layout classification, semantic hashing, escalation scores, and adaptive profile rewards remain heuristics. Optional OCR engines, FAISS, Dask, and Ray were not validated against production services or representative workloads.
- API keys share configured tenants and jobs. This service does not implement per-tenant authorization. Path conversion is disabled by default and grants server file access when enabled.
- Rate limiting is local to one process. Profile state writes are atomic and thread-safe within one process, but separate processes can overwrite learned statistics. SQLite persists completed results; it does not restart interrupted work.
- The historical README benchmark table is not regenerated into a new performance claim.
