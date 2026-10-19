# Relation context preservation — 2026-10-19

The extractor now asks the model to retain the table headers, units, applicable
dates and scope alongside each relation. In a fixed synthetic development check,
complete context rose from **9/17 to 17/17** relation rows. Literal values already
matched in both runs. The improvement observed here concerns interpretation and
context preservation, not OCR or general real-world extraction accuracy.

The [fixtures](../data/knowledge/realistic-relations/README.md) imitate a municipal
project review memo and procurement ledger. All institutions, projects, decisions
and amounts are invented. They exercise current versus historical responsibilities,
unapproved future transfers, primary/support/approval roles, budget versus estimate,
thousand-yen versus yen headers, explicit zero, absent cells and unknown values.
No private user documents or real public-agency decisions were used.

## Fixed comparison

The 17 tuples and required context lines were hand-authored before inference.
Gold was never sent to the model. Each source was sent once to GPT-5.5 with
instruction version 1 and once with version 2, with the same source units,
ontology, structured schema, reasoning effort `low` and 12,000 output-token
ceiling. All four calls completed, returned model `gpt-5.5`, and used no retries.
Baseline calls preceded contextual calls. This is not randomized, held-out,
statistically powered or a Foundry managed evaluation. No documents were published
and no native agent or direct VLM comparison was run in this check.

| Measurement | Original instructions | Context instructions v2 |
|---|---:|---:|
| Exact non-null literal values | 13/13 | 13/13 |
| Exact gold relation tuples found | 17/17 | 17/17 |
| Additional relation tuples | 1 | 0 |
| Rows retaining all required context | 9/17 | 17/17 |
| Markdown context coverage | 2/10 | 10/10 |
| CSV context coverage | 7/7 | 7/7 |
| Input tokens | 4,731 | 5,393 |
| Output tokens (including reasoning) | 3,997 | 5,501 |
| Total tokens | 8,728 | 10,894 |

The baseline emitted both an ambiguous approval choice and a redundant unknown
approval row for the same source-scoped fact. It also omitted headers, monetary
units or document dates/scope from eight otherwise matching Markdown relations.
Both errors remain in the saved baseline. Version 2 produced one ambiguous row
for that unresolved choice and retained all required context. Quotation matching
still does not prove that the interpretation is correct.

The context version consumed 24.8% more tokens in this observation. A smaller or
cheaper extraction is not claimed; monetary cost is **unknown** because account
billing rates were not verified. There was no cached input in these four calls.
PPTX and image extraction quality were not remeasured by this two-format check.

## Replay and compatibility

Requests, raw responses, source snapshots, model usage, provenance and score
reports are under [validation/relation-context-2026-10-19](validation/relation-context-2026-10-19/).
Execution receipts omit the selected project endpoint and retain its SHA-256;
authentication tokens were never saved. Bundle verification regenerates the
saved requests and rows without model calls. The comparison manifest records
every published artifact hash.

For either `baseline` or `contextual`, run the offline scorer with a fresh output:

```sh
python scripts/probe_relation_evidence.py \
  --gold data/knowledge/realistic-relations/gold.json \
  --bundle operations-review.md=docs/validation/relation-context-2026-10-19/contextual/operations-review.md/bundle \
  --bundle procurement-ledger.csv=docs/validation/relation-context-2026-10-19/contextual/procurement-ledger.csv/bundle \
  --out replay-contextual.json
```

Version 1 remains immutable, and old bundles without an instruction-version field
replay byte-for-byte. Version 2 adds `instructions_version: 2` to extraction
metadata; silently relabelling a new request as legacy is rejected. This does not
publish or mutate existing Foundry indexes, agents, connections or stored bundles.
