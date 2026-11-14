# Twelve-question paired ontology evaluation

`evaluations/ontology_questions_v1.json` defines a handwritten **development**
set over the existing six synthetic PDF, DOCX and PNG documents. It is not a
held-out benchmark or a measurement of accuracy on bank documents. Each question
names two files of different formats. Every category uses PDF–DOCX, DOCX–PNG
and PNG–PDF pairs, with the A/B interpretations retained separately.

| Category | Three questions |
| --- | --- |
| Identity | Atlas system versus team; invalid organization merge; unsupported rename history |
| Conditions | Exactly 100万円; emergency with an incomplete check; ordinary-time permission |
| Revision/authority | Current versus unapproved; no automatic winner; shared applicable period |
| Unknown | Missing approver name; amounts above the threshold; dates after the applicable period |

Gold contains original fixture paragraph text, required statements, forbidden
assertions and candidate primary claim references. The references are a manually
selected binding to the retained extraction, **not** the source of the expected
answer. In particular, absence of a name can appear as an asserted absence,
denied name-recording proposition or unknown identity; those are not collapsed
into a global status. Gold is never included in the model callback input.

## Freeze before calls

```bash
python scripts/evaluate_ontology_questions.py prepare \
  --database /path/to/ontology.sqlite \
  --suite evaluations/ontology_questions_v1.json \
  --out /path/to/new-evaluation
```

Preparation verifies active names/snapshots, original source hashes, all primary
claim references, complete vocabulary history and exact context round-trip. It
copies SQLite read-only and freezes the suite, full packed evidence, package
Python/C++ implementation hashes, model settings, budgets and execution order.
Keep the printed `protocol_id`: execution and reporting require it and reject
modified inputs or implementation. An existing destination is never reused.
The original source documents do not need to be present beside SQLite.

There are 24 planned cells, with baseline/retrieval order alternating by question.
Both arms use `gpt-6-luna`, medium reasoning, a 6,000 output-token cap per call,
the identical question and base guidance, and a 2 MiB input-context byte cap.

* `all_evidence`: all verified primary-claim inspections, one final-answer call.
* `retrieval`: existing question loop, at most six model calls, 24 tool calls and
  12 inspection pages. Discovery, comparison and inspection remain explicit.

The total ceiling is 84 model calls; there is one attempt per cell and no
automatic retry. These are **different interaction budgets**, not equal-compute
arms. This comparison measures completed answer quality and resource use for
the same questions. It does not isolate retrieval from every prompting effect.
Freezing package source does not make OS, dependency or provider execution
hermetic; record those separately when conducting a live experiment.

## Host-owned model invocation

The preparation/report CLI is offline and does not read credentials. The library
entry point `run_evaluation_cell` accepts a trusted callback; there is no hidden
HTTP client or fallback model:

```python
from sr_adapter.ontology_evaluation import run_evaluation_cell

receipt = run_evaluation_cell(
    root, protocol_id, "identity-1", "all_evidence",
    complete=approved_luna_callback,
)
```

The callback receives `{model, reasoning, max_output_tokens, input}`. `input`
contains question, guidance, tool definitions/history, packed evidence and
remaining budgets. It returns:

```json
{
  "model": "gpt-6-luna",
  "response": {"actions": [], "answer": {"text": "…", "citations": []}},
  "usage": {"input_tokens": 100, "output_tokens": 20}
}
```

This example's empty citations deliberately do not qualify as a linked answer.
Use actual five-field references for linked answers. The host must use the
frozen model/settings, retain the raw provider envelope, parse its unique final
assistant message, and report actual usage. Missing usage is unknown, not zero.
The callback must not retry internally. Raise `ProviderRefusal` on a provider
policy rejection: the failed cell is retained and further protocol cells stop.
Do not reroute, change the prompt or switch models to bypass that stop.

Call cells in `protocol.json` order. Existing cell directories reject reexecution,
including failed or interrupted calls. An interrupted cell without a receipt
needs manual audit; it is not implicitly resubmitted. Per-call requests, responses,
failure types, known usage and final receipts remain under `cells/`.

## Scoring and manual review

```bash
python scripts/evaluate_ontology_questions.py report \
  --run /path/to/new-evaluation --protocol-id PRINTED_HASH \
  --out /path/to/new-report
```

The report keeps all 24 planned cells, including `not_run`, `interrupted`, `failed`, `incomplete`
and `needs_review`. It counts linked citations, exact required-reference groups
provided in callback inputs and cited in answers, calls, elapsed time and usage.
An inspection can expose a larger connected graph or the whole document context;
therefore exact primary-reference coverage is **not semantic evidence recall**.
Full-baseline provided-reference coverage is complete by construction. Its cited
coverage and answer quality remain meaningful measurements.

When a cell directory exists without a final receipt, the report marks it
`interrupted` and recovers submitted calls, known token subtotals and unknown
usage from its retained event log. It never labels that attempt `not_run`, assumes
zero cost, resumes it or promotes a partial response to a completed answer.
For interrupted retrieval, retained loop receipts/history also recover a lower
bound on tool calls. The exact count remains unknown because interruption may
precede a log flush.

`review-template.json` contains only cells with answer text. Review against the
suite's original source units and each criterion, using `pass`, `fail` or
`uncertain`. For a forbidden criterion, `pass` means the forbidden assertion is
absent. Do not pass an otherwise empty answer merely because it avoids a bad
assertion; the required criteria must also pass. Preserve explanations in `notes`.
For less reviewer bias, read the answer/source first and reveal the arm afterward;
this implementation does not enforce blinding.

Every question includes a grounding criterion: material source-specific assertions
must be supported by the cited inspection from that source. A correct A+B answer
citing only B does not pass. Review the connected claims and original context as
well as the primary claim; exact primary-reference coverage is a separate metric.

Submit the completed file with `--reviews /path/to/reviews.json` and a new output
directory. Every judgment is bound to the protocol and exact receipt hash, with
all rubric criteria required. Stale/duplicate/incomplete judgments are rejected.
Until review, semantic results stay `unreviewed`; a valid response or a resolved
citation does not become a correct answer. Token deltas are reported only for
pairs where both answers completed, passed all reviewed criteria and have known
usage. Failed/unknown pairs remain in the 12-pair denominator and are never
silently dropped. Known token subtotals are labelled lower bounds if usage is missing.

## Previous failure and current execution boundary

The 2026-11-13 composite question reached eight inspections in its first run but
failed to select one final message from a multi-phase response. The parser fix
is covered by the existing question-loop tests. A second run was rejected with
provider `invalid_prompt` on its fifth request. That rejection does not identify
which prompt content caused it and is not evidence of retrieval accuracy.

This PR prepares the new benchmark and verifies execution/scoring with scripted
callbacks. It does not resubmit the rejected request or claim new live-Luna
results. Preparation against the retained corpus verifies six original source
hashes and 53 claim inspections. Live completion, semantic accuracy and token
savings remain unmeasured until a separately recorded, permitted provider run.
