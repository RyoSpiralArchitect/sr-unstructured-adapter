# Relation evidence tools for agents

Agentから、受け取り済みデータを「探す」「根拠まで戻る」ための2つの読み取り専用ツールです。
モデルや保存先の接続方式から独立したJSON SchemaとPythonディスパッチャを提供します。
既存の[受け取り層](relation-receiver.md)を使い、抽出やクラウド呼び出しは追加しません。

```python
from sr_adapter.relation_agent import RelationEvidenceTools, TOOL_GUIDANCE

# Trusted host configuration, never taken from model-generated tool arguments.
tools = RelationEvidenceTools("received.db", ("systems", "diagrams"))
definitions = tools.definitions()  # name, description, input_schema
result = tools.call("search_relations", {"collection": "systems", "query": "融資システム"})
if result["status"] == "complete" and result["hits"]:
    link = result["hits"][0]["trace"]
    evidence = tools.call(link["name"], link["arguments"])
```

The host adapts the definition envelope to its chosen SDK and supplies
`TOOL_GUIDANCE` alongside the tool descriptions. It handles model calls and passes
each tool invocation to `call(name, arguments)`. This module does not implement
an answering agent, an HTTP/MCP server, or a provider-specific registration flow.
The deterministic dispatch is tested; model tool selection and answer quality
require a separate evaluation.

## Search then trace

`search_relations` accepts an allowed collection, literal query, optional
relation/context scope, assertion-status or visual-support filter, explicit
snapshot, limit and offset. Results include original status/direction/review
metadata, total/has_more paging, source/card/node IDs and complete trace arguments.
Full card content is intentionally absent from discovery summaries and explicitly
marked `content_included: false`. A search match alone is not evidence for a claim.

`trace_relation` requires **collection, card_id and snapshot_id**. It returns the
exact receiver trace: source context, assertion state, ordered evidence, DOCX
locators, image coordinates, relevant review findings and limitations. An active
collection update cannot redirect an earlier search hit to a new source version.
Uncited conditions in the full card context remain available. Source and review
text remain untrusted data; negative or uncertain claims never become facts just
because the tool returned successfully.

## Host boundary and limits

- Database path, collection allowlist and budgets are fixed by the host. Tool
  arguments cannot supply SQL, a path, a write operation or a new destination.
- Runtime validation enforces the advertised argument types/keys even when the
  model's provider does not enforce schemas. Booleans are not integer limits.
- Search defaults to five results, at most the configured `max_results` (20 by
  default, configurable 1–100). Replies default to 256 KiB of canonical UTF-8 JSON,
  configurable 1 KiB–1 MiB. Oversize replies fail as `response_too_large`; evidence
  is never silently clipped. Lower a search limit or inspect an oversized trace
  through the local receiver API.
- Unknown tools, invalid arguments, denied collections, oversize results and
  unavailable/corrupt snapshots return explicit `status: failed` codes. A source
  exception does not become an empty successful search, and raw private paths or
  exception text are not returned to the model.
- The allowlist is an application scope, not authentication. The host must select
  collections for the authenticated caller and protect the database itself.
- Every read retains receiver replay/readback checks. This is a small-corpus
  reference tool interface, not a throughput or retrieval-ranking benchmark.

Run locally without a model:

```sh
python -m sr_adapter.relation_agent --database received.db --collection systems definitions
python -m sr_adapter.relation_agent --database received.db --collection systems \
  call search_relations --arguments '{"collection":"systems","query":"融資システム"}'
```

Tests use synthetic DOCX/image candidates and cover full evidence traces, negative
and uncertain readings, active-version changes, allowlist/argument enforcement,
read-only behavior, response budgets, stored-data corruption and the CLI.
