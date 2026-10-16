# Fabric lakehouse live validation — 2026-10-16

The existing synthetic and Digital Agency roles knowledge examples were published
to the explicitly selected existing Fabric lakehouse. All six asynchronous table
loads completed successfully. Reading their Delta version-0 transaction logs and
active Parquet files reproduced the projected rows exactly, including their
order-independent content hashes. All 21 uploaded source/configuration files also
matched their local bytes.

| Table | Expected rows | Read back |
| --- | ---: | ---: |
| records | 67 | 67 |
| entities | 19 | 19 |
| relations | 11 | 11 |
| observations | 3 | 3 |
| evidence | 30 | 30 |
| evidence_links | 94 | 94 |

Snapshot:
`c47481c7ddad99e9d1896458d9688a190d8ae659ecae84ca78d9e913e8a148aa`.

The source inputs are the already documented
[`synthetic`](validation/table-correspondence-2026-10-16/knowledge-examples/synthetic/manifest.json)
and [`roles`](validation/table-correspondence-2026-10-16/knowledge-examples/roles/manifest.json)
bundles. The [sanitized verification receipt](validation/fabric-lakehouse-2026-10-16/live-verification.json)
contains the exact counts, row hashes, and Delta versions. Tenant/resource IDs and
credentials are omitted; full operation receipts remain with the deployment operator.

This demonstrates the adapter-to-Fabric persistence path with source-scoped entity
mentions, directed relations, and evidence joins intact. The SQL analytics endpoint
was provisioned, but SQL query execution was not tested in this validation. Foundry
retrieval directly from these Fabric tables was not tested. The original PDFs were
not uploaded; source names, hashes, cells, coordinates, mappings, and generated
knowledge representations were retained in the verified bundles.

See [the Fabric workflow](fabric-lakehouse.md) for the explicit target, opt-in
publication, immutable snapshot rules, beta Load API, and read-only status command.
