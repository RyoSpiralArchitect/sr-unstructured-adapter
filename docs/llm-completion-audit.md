# LLM completion audit

Date: 2026-10-15 JST. Starting revision: `ddea7565287d72be288cbbddf842ebce122ddeb6`.

The previous pipeline accepted every nonempty choice list as an LLM result, regardless of refusal or token-limit finish reasons. CLI commands returned success for those responses, and adaptive profile feedback counted them alongside completed text. A blocked Gemini prompt without candidates lost its response evidence.

The normalizer now assigns a status to each choice and the whole response. It requires a recognized normal stop and nonempty text before reporting `complete`. Incomplete responses retain their original payload and partial text. Pipeline metadata separates received responses from completed responses, document warnings expose incomplete outcomes, and adaptive feedback uses only completed responses. `llm run` returns code 10 for incomplete text. Replay preserves existing output on failure; `--skip-errors` keeps incomplete rows while returning a nonzero status and excludes them from the successful-prompt limit.

These are text-completion checks, not a measurement of extraction correctness. HTTP driver telemetry continues to describe transport success. Source blocks, spans and confidence are preserved by LLM delegation.

## Local validation

macOS / Python 3.13.13:

| Check | Result |
| --- | --- |
| Starting revision, native runtime enabled | 629 passed, 1 skipped |
| Updated full suite, native runtime enabled | 688 passed, 1 skipped |
| Updated full suite, native runtime disabled | 684 passed, 5 skipped |
| Ruff | Passed |
| sdist and wheel builds | Passed; sdist includes the smoke runner imported by its tests |
| Installed wheel, outside the checkout | Local synthetic HTTP provider: pipeline retains truncated evidence and counters; CLI prints `truncated` and exits 10 |

The native run skips the Windows-only DLL loader check. The fallback run also skips four native text-kernel tests. Both report a third-party Starlette/httpx deprecation warning; neither run validates Windows behavior locally.

Regression coverage includes normal/limited/filtered/unknown stop reasons, refusal metadata, tool calls, empty choices, mixed candidates, malformed shapes, preserved source evidence, CLI exit codes, replay output preservation, successful-prompt limits, old escalation histories, and adaptive feedback. Offline smoke-runner tests cover terminal chunks, candidate separation, Gemini thought filtering and trailing usage, first-failure stopping, missing credentials, output collisions and interrupted receipt persistence.

Reproduce without live calls:

```sh
python -m pip install -e '.[dev,api]' ruff build
pytest --benchmark-disable -rs
SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1 pytest --benchmark-disable -rs
ruff check .
python -m build
```

## Live observations

All generation requests used short synthetic prompts, existing environment credentials and `max_retries=0`. No source documents were sent. Each interface check allowed at most 96 output tokens. Output receipts were reserved before requests, updated after each mode, and contain no keys, provider error messages, or raw model text.

| Run | Planned | Attempted | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial OpenAI / Mistral / Gemini 2.5 Flash-Lite | 12 | 9 | 8 | 1 | 3 |
| Explicit Gemini 3.1 Flash-Lite follow-up | 4 | 4 | 4 | 0 | 0 |
| OpenAI intentional output truncation | 1 | 1 | 1 | 0 | 0 |

The initial Gemini 2.5 Flash-Lite request returned HTTP 404, despite the model appearing in that credential's model catalog; the remaining three modes were skipped. That failed receipt is retained. A separate, explicit Gemini 3.1 Flash-Lite run passed all four interfaces. The runner now defaults to the [documented stable 3.1 Flash-Lite model](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite); it does not silently switch models or retry failures.

OpenAI `gpt-4.1-mini` (response model `gpt-4.1-mini-2025-04-14`), Mistral `mistral-small-latest`, and Gemini `gemini-3.1-flash-lite` each returned the expected JSON with a normal terminal reason through synchronous, asynchronous, streaming and asynchronous streaming interfaces. Separately, an OpenAI request capped at one output token returned `finish_reason=length` and was correctly classified as `truncated`, with `is_complete=False`. That negative control passing means the adapter detected truncation; the generated answer itself was incomplete.

Receipts:

- [Initial run, including Gemini 404 and skipped modes](validation/llm-completion-2026-10-15.json)
- [Gemini 3.1 follow-up](validation/llm-completion-gemini-3.1-2026-10-15.json)
- [Intentional truncation control](validation/llm-truncation-2026-10-15.json)
- [Installed-wheel pipeline and CLI check](validation/wheel-completion-2026-10-15.json)
- [Local validation summary and source hashes](validation/completion-validation-2026-10-15.json)

Run the current positive checks with a **new** receipt filename:

```sh
python scripts/smoke_llm.py --allow-live-api --provider all --output new-live-results.json
```

## Boundaries

- Completion does not establish factual correctness, calibrated confidence, representative extraction quality, or a performance improvement.
- Live checks cover three providers and small synthetic text only. Refusal and tool-call cases use deterministic offline responses; no live safety-refusal benchmark was run.
- `llm_escalations`, `llm_completed` and `llm_incomplete` count blocks with newly attached responses before output truncation. A shared request may cover multiple blocks. Configuration/transport failures remain in the existing failure logging path and are not included in those response counts.
- The live runner retains actual usage when a provider supplies it. OpenAI streaming did not supply usage in these checks; empty usage remains unknown rather than being reported as zero tokens.
- No automatic retry or continuation was introduced for incomplete text. Caller review is required before consuming partial responses.
- Local test results, live observations and GitHub CI are separate evidence. This audit does not assert a merge or deployment.
