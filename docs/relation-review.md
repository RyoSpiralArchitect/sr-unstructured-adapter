# Document review from received relations

受け取り済みの資料から、確認したい読み取りを理由付きで一覧にします。業務上の正誤や
要件の網羅性を採点する機能ではなく、元資料へ戻ってレビューするための入口です。
Agent用ツールとは独立して、[共通の受け取り層](relation-receiver.md)だけを利用します。

```python
from sr_adapter.relation_review import review_received_relations
from sr_adapter.relation_receiver import trace_received_relation

queue = review_received_relations("received.db", collection="systems")
for item in queue["items"]:
    print(item["reasons"], item["record"])
    for reference in item["trace_refs"]:
        evidence = trace_received_relation("received.db", **reference)
```

`relation_review/1` is a deterministic, versioned ruleset over replay-verified
source data. Every item retains its original source record, package, source
metadata and snapshot-bound trace references. The function is read-only: it does
not change extraction results, mark findings resolved, or write approval status.

| Signal | What the reviewer should check |
| --- | --- |
| `unknown_object`, `ambiguous_object` | The source wording and why the reading has no definite object |
| `nonliteral_label` | A model label differs from the literal quoted text; check the source terminology |
| `uncertain_visual_relation`, `unspecified_direction` | The drawn line, labels and direction in the original image |
| `recorded_review` | Existing reviewer comments and the exact image/item/relation they reference |
| `source_limitation` | Saved extraction limitations, once per limitation record |
| `no_extracted_relations` | The source context; empty extraction does not establish a missing requirement |
| `denied_assertion` (opt-in) | Negative statements as information, not automatically errors or contradictions |

A literal denial is not a default review defect. It can still appear for an
independent reason, such as a nonliteral entity label. `--include-denials` adds an
informational reason without changing the original assertion status. Recorded
review text remains a separate record and is never applied as a correction. A
review of an unconnected image item survives with its original label and box even
when there is no relation card to trace. Source-wide findings and limitations may
link to multiple cards; this does not create new business relationships.

```sh
python -m sr_adapter.relation_review --database received.db --collection systems
python -m sr_adapter.relation_review --database received.db --collection systems \
  --reason unknown_object
python -m sr_adapter.relation_review --database received.db --collection systems \
  --include-denials --reason denied_assertion
```

Results contain `total`, `reason_counts`, `offset`, `limit` (1–100) and `has_more`.
Items are sorted deterministically by package, kind and item ID; they are not
ranked by business severity. One item can have multiple reasons, so reason counts
need not sum to the number of items. A reason filter selects items and retains all
their reasons. IDs bind to the source snapshot and source record, never just the
display label. Pass `--snapshot` to inspect an earlier version after an update.

An empty queue means no selected rule matched, not that the document is complete
or correct. There is no entity resolution, requirements inference, hidden LLM
judge, risk score or automatic review closure. Receiver corruption produces a
failure instead of an apparently clean review. Tests use synthetic DOCX/image
sources and mock responses; real review utility still needs domain feedback.
