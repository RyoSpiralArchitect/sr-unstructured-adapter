# Investigating changes between received sources

利用者が選んだ旧版・新版の資料を比べ、関係の読み取り、根拠、資料内の位置、レビュー
記録の差を確認します。同じ名前から実体の同一性や業務上の変更を推定する機能ではありません。
[受け取り層](relation-receiver.md)だけに依存し、Agentツールやレビューキューとは独立しています。

## Select the pair explicitly

Both source snapshot IDs and both package IDs are required, even if the receiver
has only one active document. Select them from receiver receipts/search results.
Both snapshots must already belong to the named collection, and each package must
belong to its selected snapshot. Filenames and equal labels are never used to guess
which documents or entities correspond. An active-version update cannot redirect
either side of a comparison.

```python
from sr_adapter.relation_changes import compare_received_sources
from sr_adapter.relation_receiver import trace_received_relation

report = compare_received_sources(
    "received.db", collection="systems",
    before_snapshot=old_snapshot_id, after_snapshot=new_snapshot_id,
    before_package=old_package_id, after_package=new_package_id,
)
for group in report["groups"]:
    print(group["classification"], group["content"])
    for reference in group["before_references"] + group["after_references"]:
        evidence = trace_received_relation("received.db", **reference["trace_ref"])
```

For example, changing the saved owner reading from B to C produces a B group
present only before and a C group present only after. It does not automatically
join those observations as one confirmed ownership transfer.

## What the comparison means

`explicit_source_pair_exact_values/1` compares multisets of normalized values.
Equal content has one group containing **all** before/after occurrence references.
It does not select a particular matching occurrence or merge their IDs.

| Classification | Meaning |
| --- | --- |
| `before_only` | These values appear only in the selected older reading |
| `after_only` | These values appear only in the selected newer reading |
| `count_changed` | Identical values occur a different number of times; references are retained without assigning a particular removed/added identity |
| `unchanged_values` | Equal values and occurrence count, not proven identity or unchanged business reality; hidden unless requested |

The relation scope retains predicates, assertion status, null objects, literal
match flags, entity types, grounding and interpretation; visual values include
support, direction, quantities, normalized boxes and reading/review status. It
also compares ordered cited evidence, narrative locator details, image observations
and relevant review records. A change in a source location or evidence ordering
therefore counts as a reading/evidence difference even if the displayed relation
labels are unchanged. Denial, unknown/ambiguous objects and visual uncertainty are
never converted to positive edges.

Snapshot-dependent package, relation, mention, item, evidence-unit and finding
IDs are omitted from comparison values and retained in original references.
Common source metadata (filename, file hash, page label, extraction strategy)
is shown in `before_source`/`after_source` rather than repeated in every candidate
fingerprint. Entity roles, scalar types, nulls, false, zero, positions and original
evidence order remain distinct. A new deployment/package ID alone does not imply
different candidate values. This compares projected readings, not every byte of
the original extraction request or a certified ontology.

Relation-review targets include the normalized endpoint observations and ordered
evidence, so moving a finding between same-label connectors in different regions
remains visible. Image-review targets retain the pixel dimensions and coordinate
system; deployment, response-model and endpoint metadata remain in the source
records and do not re-enter candidate fingerprints through review targets.

## Check conditions outside the cited relation

The response always distinguishes:

- `source_bytes_changed`: original source file hashes differ.
- `source_metadata_changed`: source records differ after excluding package ID,
  including deployment metadata and extraction counts.
- `source_context_changed`: the full saved source-unit/observation, limitation or
  review multiset differs, including units never cited by a relation.

Thus "A is managed by B" can remain an unchanged reading while an uncited
"approval required" condition changes. Inspect `scope="context"` to see the
actual context differences, rather than concluding that the document is unchanged.
Context groups return exact normalized records/locators and original table keys;
relation groups also return snapshot-bound card trace references. Empty extraction
does not suppress the context comparison. For images, original pixel changes are
reported by the source hash even if saved model observations remain identical;
this is not a pixel-difference algorithm.

```sh
python -m sr_adapter.relation_changes --database received.db --collection systems \
  --before-snapshot <old-snapshot> --after-snapshot <new-snapshot> \
  --before-package <old-package> --after-package <new-package>

# Use the same explicit pair to inspect uncited conditions and source notices.
python -m sr_adapter.relation_changes --database received.db --collection systems \
  --before-snapshot <old-snapshot> --after-snapshot <new-snapshot> \
  --before-package <old-package> --after-package <new-package> --scope context
```

Results include counts of before/after/shared/before-only/after-only occurrences,
`total_groups` before filtering, `total` after excluding unchanged groups, and
`offset`, `limit` (1–100), `has_more`. Use `--include-unchanged` for a complete
comparison. Fingerprint order makes paging deterministic; it is not importance
ranking. Counts describe readings or context records, not distinct business
entities, document defects or confirmed impact. Identical visual labels with
different regions remain different values; identical duplicate readings retain
their multiplicity.

Both sides replay and verify receiver tables before comparison. The operation
is read-only and makes no model/cloud call. Corruption returns an error instead
of an empty difference. Existing database access remains the host's responsibility.
Synthetic tests cover changed owners, status/null preservation, DOCX locators,
ordered evidence, uncited conditions, stable values across package revisions,
visual direction/boxes/reviews, duplicate multiplicity, empty extraction, explicit
selection, paging and CLI failures. Domain identity mapping and automatic impact
reasoning require a separate explicit contract and validation.
