# Relation context probe — 2026-10-16

Readable row contexts preserve the adapter's mapped facts for a prompt agent. In
this small probe, both formats answered every content question correctly. The
useful difference was attribution: placing a full assertion ID alongside each
relation/property prevented the observed confusion between search-document IDs
and relation IDs in the revised run. This is a local context-consumption result,
not an Azure Foundry IQ retrieval result or a general accuracy claim.

## Fixed comparison

Two previously recorded packages were used: a synthetic Japanese ownership table
and the Digital Agency role table. Together they contain eight rows, 19 entity
mentions, 11 relations and three property observations. The [earlier table
comparison](table-knowledge-2026-10-16.md) retains their document provenance and
the independent VLM results.

Each version contains two input formats × two packages × two repeats = eight
completed requests. Each format has 12 questions × two repeats = 24 planned
answers. Both use `gpt-4.1-mini-2025-04-14`, temperature 0, a 2,048-output-token
cap, identical system instructions and questions, no retry, and no provider-side
storage requested. The baseline supplies all normalized JSONL records; the other
format supplies complete row contexts. This is deliberately an all-context test:
there is no retrieval selection, image input, ranking, or tool execution.

Questions cover primary versus secondary ownership, another row with a repeated
organization label, a known budget, an unknown budget, reverse edges, unsupported
identity resolution, an absent project, role/task association, and an unstated
reporting line. Gold and questions were frozen before each run. Gold values and
expected evidence IDs do not appear in request instructions.

`answer_match` requires the expected value and observed/unknown/not-established
status. `strict` additionally requires the prescribed `evidence_ids` field, valid
relation/observation IDs, and the expected supporting assertion for observed and
unknown answers. A retrieval-document ID or entity ID is not an assertion ID.
For an unsupported relation, an empty evidence list is accepted; absence is not
scored as a negative fact. Missing, duplicate or extra answers fail the contract.

## Observations, including the first failure

| Run | Format | Answer content | Answer + assertion contract | Input tokens |
| --- | --- | ---: | ---: | ---: |
| Initial v1 | Normalized records | 24/24 | 15/24 | 42,698 |
| Initial v1 | Row contexts | 24/24 | 16/24 | 38,382 |
| Revised v2 | Normalized records | 24/24 | 16/24 | 42,698 |
| Revised v2 | Row contexts, inline assertion IDs | 24/24 | 24/24 | 40,006 |

In v1, the row format used a search-document hash for the three known role answers
in both repeats. Its two synthetic identity answers cited entity IDs. Baseline
failures included using `evidence` instead of `evidence_ids` and citing entity
IDs. These were attribution/schema failures; the content answers were correct.

The producer now labels citation scopes and appends `assertion_id=…` directly to
each relation/observation sentence. Missing entities also identify their source
column. Hash comparison confirms that only `synthetic/row_contexts.json` and
`roles/row_contexts.json` changed between the two frozen inputs; the model,
questions, gold, system prompt, baseline contexts and runner bytes stayed fixed.

The v2 row format used 6.30% fewer aggregate input tokens than its baseline here.
That aggregate hides a per-document tradeoff: repeating row evidence can make
long role descriptions larger. Token counts are not latency or cost measurements;
cache usage differs across calls. Baseline output also varied despite temperature
0. Two repeats and two small documents cannot establish a reliable effect size.
No answer-content improvement was observed, and v2 has no independent held-out
confirmation. The strict score partly measures output-schema adherence, not just
semantic grounding. Do not transfer this result to native MCP citation behavior.

## Reproduce and inspect

All frozen contexts, questions, gold, runner snapshots, requests, raw responses,
usage, per-question scores and offline audits are under
[`validation/foundry-context-2026-10-16/`](validation/foundry-context-2026-10-16/).
Both versions are retained. An evidence manifest binds file hashes; the handoff's
own manifest also regenerates the exported semantics from its source packages.

```bash
python scripts/probe_relation_context.py replay \
  --frozen docs/validation/foundry-context-2026-10-16/v1/frozen \
  --run docs/validation/foundry-context-2026-10-16/v1/run
python scripts/probe_relation_context.py replay \
  --frozen docs/validation/foundry-context-2026-10-16/v2/frozen \
  --run docs/validation/foundry-context-2026-10-16/v2/run
```

Replay requires no credential and verifies requests against the frozen inputs,
provider completion/model, usage, hashes and recomputed scores. A new paid run is
explicit: `probe_relation_context.py run --frozen <folder> --out <new-folder>
--execute` with an existing `OPENAI_API_KEY` in the process environment. Without
the flag/key, all eight planned slots remain visibly unexecuted. Execution stops
after a provider/protocol failure and retains the remaining skipped denominator.

[`handoff/`](validation/foundry-context-2026-10-16/handoff/) is the final offline
Foundry export. It was verified to retain every package entity, relation and
observation, with no cloud destination configured. The [connection guide](foundry-iq.md)
describes the next live checks: visibility of indexed rows, managed-identity
access through MCP, returned evidence, and native citations in the agent answer.
