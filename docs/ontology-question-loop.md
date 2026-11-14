# Question-driven SQLite evidence retrieval

`answer_ontology_question` in `sr_adapter.ontology_question` accepts a trusted
`complete(request)` model callback, a SQLite path, allowed collections, a question
and an exclusive output directory. The callback receives tool definitions,
history and losslessly packed inspected evidence. It returns either
`{"actions":[{"name":"search_ontology","arguments":{"collection":"default","query":"approval"}}],"answer":null}`
or `{"actions":[],"answer":{"text":"…","citations":["ref_<issued handle>"]}}`.

A private SQLite backup pins the session. Discovery references gate inspection;
only actually inspected references are linked to the answer. Readable answers
with missing or invalid citations are retained with `needs_review` and
`evidence_status=unlinked`. Linking proves reference consistency, not factual
correctness or that every contrary source was found. Search paging and archive
paging remain explicit model actions. The budgets bound rounds, calls,
inspections and request bytes; exhaustion returns `incomplete`, never silent
truncation or automatic retries. Request/response files and successful tool
results survive later failures. Host callbacks own credentials and network use.

The opt-in `scripts/probe_ontology_question.py` fixes `gpt-6-luna`; without
`--execute` it makes no calls. Provide `--database`, repeatable `--collection`,
`--question`, `--out` and optionally `--max-rounds` (default six). It exits nonzero
on failed/incomplete/unlinked runs and retains raw provider responses.

## 2026-11-13 observations

The previous six-source synthetic SQLite was reused without extraction calls.
The frozen question asks whether Atlas denotes the same target across documents,
who reviews an application below the amount threshold, and which source wins.
An initial six-call run selected eight inspections but stopped because a response
contained commentary plus a final assistant message. Total input was 145,591
tokens; the earlier all-evidence run used 161,296, but had seven questions, so
these runs do not establish an equivalent-task efficiency improvement.

The response adapter now selects one explicit `final_answer` message, rejects
ambiguous final messages, and preserves the original envelope. A separately
recorded follow-up stopped at request five with provider `invalid_prompt` after
15 search/discovery calls. It was not retried. Live final-answer completion and
counterevidence recall remain **unverified**; local deterministic tests cover
successful discovery/inspection/citation flow, backup isolation, budgets and
failure retention. Artifacts are under
`/Users/ryohiga/SpiralReality/artifacts/ontology-four-pr-20261113/`.

## Shared limits and citation handles

Each callback now receives `response_contract`, with the per-turn action ceiling
(eight), `allowed_actions_this_turn` (the smaller of that ceiling and the remaining
tool budget), a `citation_catalog`, and a provider-neutral `response_schema`.
The receiver uses the same ceiling; a nine-action response is retained and fails
with `reason=action_limit`, `requested_actions`, and `allowed_actions`, before
executing any action in that batch. There are still no automatic retries.
Hosts may use the schema for generation; supplying it in the request does not
itself enable a provider's structured-output enforcement. Receiver validation
also enforces that final answers have no actions.

The preferred final response is:

```json
{"actions":[],"answer":{"text":"Source-specific answer","citations":["ref_<copy the complete issued handle>"]}}
```

Copy handles from `response_contract.citation_catalog`. Each maps to an exact
collection, document name, snapshot, kind, and ID. Handles are deterministic
SHA-256 values of canonical references, independent of catalog ordering, and
only resolve against evidence provided in the current session. Search results
alone do not issue citation handles. Identical names/IDs in other documents or
snapshots do not authorize retargeting. Models should select the exact documents
named in the question, even when another format contains similar text.

The saved `answer.citations` and `linked_citations` contain expanded references
for existing consumers. `raw_answer` retains the returned handles. Unknown
handles remain unlinked with `citation_issues` and a readable `needs_review`
answer. Exact legacy reference objects remain supported for old callbacks, even
though the preferred generation schema asks for strings. Neither a valid handle
nor a matching reference establishes semantic support: selecting a valid handle
from the wrong source can still occur and requires review.

The paired evaluator shares this contract and resolver in both arms. The
all-evidence arm issues handles for all provided inspections and advertises zero
available actions. Each revised implementation requires a newly frozen run;
prior inputs, receipts and scores must not be rewritten.
