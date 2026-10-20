# Grounded knowledge in an existing Fabric lakehouse

## Native PowerPoint as ordinary tables

For a diagram whose labels and connectors are native PowerPoint objects, use
`--mode tabular`. This mode needs no model call or ontology mapping. It produces
ordinary string, integer and boolean columns in Parquet, then loads Delta tables
through the same explicit publisher and readback checks described below. There
are no `payload_json`, `evidence_json`, array or nested columns to unpack in SQL.

```bash
sr-adapt knowledge fabric architecture.pptx --mode tabular \
  --slides 57 58 --page-labels 56 57 \
  --target fabric-target.json --out artifacts/architecture-tables
sr-adapt knowledge fabric-publish artifacts/architecture-tables \
  --target fabric-target.json --receipt artifacts/architecture-attempt-01.json --execute
sr-adapt knowledge fabric-status artifacts/architecture-tables \
  --target fabric-target.json --receipt artifacts/architecture-attempt-01.json
```

`--slides` always uses one-based positions in the presentation's slide order.
`--page-labels` records optional printed page labels supplied by the caller; it
does not change slide selection or claim OCR verification of those labels. This
distinction matters when a cover offsets the printed numbering. Explicitly
selected hidden slides are included and marked `hidden=true`.

| Table | Contents |
| --- | --- |
| `pages` | Document hash/name, physical slide number, printed label, native title when available, hidden flag, page size, object/connection counts and unsupported-element count |
| `objects` | Page-scoped object ID, original shape ID, parent group ID, kind, name, literal text, explicit geometry and raw fill token |
| `connections` | Connector ID, start/end shape IDs, joined object IDs/text when available, attachment status, explicit arrow tokens and interpretation scope |
| `cells` | Native table ID, one-based physical row/column, merge spans/flags and literal text, including blanks and zero |

All four schemas are retained in the local export. Only nonempty tables receive
a cloud Load request. Snapshots use the existing `sr_<snapshot>_<kind>` names;
new exports never replace an existing table. Join objects by `object_id`, not by
label: repeated labels and shape IDs on different slides remain distinct.

`connections.endpoint_status` is `attached` when both referenced objects exist,
`partial` when just one endpoint is declared, `unattached` when neither is
declared, and `dangling` when a declared ID is absent. Missing endpoint IDs/text
remain null. Attached endpoints can themselves be unlabelled shapes or groups.
Start/end are drawing endpoints, **not** inferred data-flow direction. The
`interpretation` column is `native_connector_only`; a connector does not prove a
business dependency, system ownership, failover relationship or ontology edge.

Geometry is in the immediate parent's EMU coordinates, with group transforms
preserved in separate child-coordinate columns. Rotation uses OOXML units of
1/60000 degree. Missing inherited geometry/style remains null. Fill values are
raw direct OOXML tokens, not rendered colors (theme transforms and inheritance
are not resolved). Text preserves paragraph boundaries; no category, missing
label or connection is inferred from proximity. Unsupported elements are counted.
Picture text, diagrams embedded as images, slide masters/layouts, animations and
speaker notes are not extracted by this first native mode.

The exporter accepts a deck up to 64 MiB, with bounded ZIP expansion and XML
parsing, and 1–20 explicit slides. It copies only the selected slide XML for
reconstruction, not the original deck or unselected slides. JSON files inside
the snapshot are internal manifests and delivery metadata, not the queryable
table payload. The original deck hash is provenance recorded at local extraction;
it does not certify that a remote SharePoint version has identical bytes.

For example, after the SQL endpoint has discovered the tables:

```sql
SELECT page_label, start_text, end_text, endpoint_status, head_arrow, tail_arrow
FROM dbo.sr_<snapshot>_connections
ORDER BY slide, connector_id;
```

Parquet/Delta readback verifies the structural projection. It does not establish
semantic accuracy or SQL endpoint synchronization. Use separate visual/model
evaluation when interpreting the meaning of diagram colors, arrows or containers.

## Ontology knowledge packages

The Fabric path loads the same verified knowledge packages used by the Foundry IQ
bridge into an existing **schema-enabled** lakehouse. It retains directed relations,
source-scoped entity identity, explicit missingness, ontology records, and source
cell evidence. No model is called and no Fabric capacity, workspace, lakehouse,
connection, role, or SQL endpoint is provisioned.

Install the optional dependencies and use an existing Microsoft Entra login:

```bash
pip install '.[fabric]'
cp data/knowledge/fabric-target.example.json fabric-target.json
# Fill in the existing workspace/item GUIDs and exact lakehouse display name.
sr-adapt knowledge fabric path/to/bundle-a path/to/bundle-b \
  --target fabric-target.json --out artifacts/fabric-export
```

This step is local. Inspect `deployment-plan.json`, `manifest.json`, and the Parquet
files before opting into publication:

```bash
sr-adapt knowledge fabric-publish artifacts/fabric-export \
  --target fabric-target.json --receipt artifacts/fabric-attempt-01.json --execute
sr-adapt knowledge fabric-status artifacts/fabric-export \
  --target fabric-target.json --receipt artifacts/fabric-attempt-01.json
```

`DefaultAzureCredential` obtains separate Fabric (`https://api.fabric.microsoft.com`)
and Storage (`https://storage.azure.com`) tokens in memory. Neither the target nor
the receipt stores credentials. A regional OneLake DFS origin can be selected to
match the workspace region. The target is checked against the live item ID,
workspace ID, exact display name, OneLake path, and schema-enabled properties.

## Files and queryable tables

Files are copied to `Files/sr-adapter/<full snapshot SHA-256>/`. This includes the
original verified bundles, six projected Parquet files, target/plan, and completion
manifest. Only descendants of the existing managed `Files` folder are created;
the uploader never tries to create the lakehouse or its managed root folders.
Files are created conditionally and read back byte-for-byte. Existing identical
files are reused; conflicting or partially uploaded files stop the attempt. The
snapshot manifest is uploaded last among the exported source files. Separate
publication control records live below
`Files/sr-adapter/_submissions/<schema>/<table>/`; they are not source bundle files.

Six Delta tables are loaded into the configured schema. Their names are
`sr_<full snapshot SHA-256>_<kind>`:

| Kind | Columns and purpose |
| --- | --- |
| `records` | Original `package_id`, `kind`, `record_id`, `payload_json`, including documents, ontology, tables, cells, entities, relations and observations |
| `entities` | Package ID, entity ID, type IRI, label, identity scope |
| `relations` | Package ID, relation ID, subject ID, predicate IRI, object ID |
| `observations` | Package ID, observation ID, nullable subject/value, predicate IRI, observed/unknown/ambiguous status |
| `evidence` | Package ID, table/cell ID, document SHA-256, page, physical row/column/span, raw text, bounding box and coordinate system |
| `evidence_links` | Package ID, assertion kind/ID, evidence ID and its order in the original assertion |

Join on **both package ID and record ID**. An entity label is not a global identity.
The original mapping and complete source representations remain in `records` and
the copied bundles. No missing value is filled or converted to a fabricated fact.

For example, replace `<snapshot>` with the exported full hash and use the SQL
analytics endpoint after it has discovered the tables:

```sql
SELECT s.label AS subject, r.predicate_iri, o.label AS object,
       e.document_sha256, e.page, e.row_index, e.column_index, e.text
FROM dbo.sr_<snapshot>_relations AS r
JOIN dbo.sr_<snapshot>_entities AS s
  ON s.package_id = r.package_id AND s.entity_id = r.subject_id
JOIN dbo.sr_<snapshot>_entities AS o
  ON o.package_id = r.package_id AND o.entity_id = r.object_id
JOIN dbo.sr_<snapshot>_evidence_links AS l
  ON l.package_id = r.package_id AND l.assertion_id = r.relation_id
 AND l.assertion_kind = 'relation'
JOIN dbo.sr_<snapshot>_evidence AS e
  ON e.package_id = l.package_id AND e.evidence_id = l.evidence_id;
```

## Delivery and validation contract

The publisher supports snapshots up to 64 MiB. All table projections are checked
against the original package reconstruction before network access. A receipt path
is reserved exclusively. Before each Load request, the publisher also conditionally
creates a durable OneLake reservation directory for the destination schema/table.
Only the process that creates that directory may submit the Load request. The
reservation survives a new local receipt, process interruption, and lost responses.
The acknowledged operation ID is saved locally first and then in the reservation's
`operation.json` so a later attempt can reconcile the same operation.

Schema-scoped Fabric Load Table is currently a **beta API**. It loads Parquet in
Append mode only after checking that the destination table is absent. An existing
table is reused only after exact row verification; it is never overwritten.
An existing pending or unknown reservation triggers read-only reconciliation,
even when the caller chooses another receipt path. No load is resent and no
remaining table is submitted while that earlier outcome is unresolved. Failed
operations also retain their reservations. The API does not expose an atomic
create-if-absent table operation: other writers that bypass these reservations
can still append duplicate rows, which exact readback detects. There is no
transactional commit across all six tables or distributed exactly-once guarantee.

HTTP 202 means `pending`, not committed. The receipt records each operation ID.
The separate status command issues only reads and never resubmits a load. HTTP 4xx,
server/transport ambiguity, failed operations, partial delivery, and visibility
delays remain distinct from a verified commit. There is no automatic transport retry
or reservation expiry. Interruption after reserving but before submission can leave
an unused slot blocked. Losing the operation ID after submission can leave an
unknown slot until exact table readback succeeds. These cases require operator
reconciliation using the original receipt and service history; this tool never
deletes or reclaims the reservation. The protection requires the reservations to
remain intact and applies to attempts made by this publisher version; it cannot
identify earlier in-flight loads that were submitted without a reservation.

Status reads the Delta transaction log and its active Parquet files, compares exact
row counts and order-independent content hashes, and records the Delta version.
`committed` requires all six tables and every source file to match. This bounded
reader requires complete JSON logs from version zero and rejects checkpoint-only,
partitioned, or deletion-vector tables. It is intended for these small immutable
snapshots. It does **not** prove SQL endpoint synchronization, SQL permissions,
Foundry retrieval, or agent answer quality; verify those separately.

Official protocol references: [OneLake access and authentication](https://learn.microsoft.com/en-us/fabric/onelake/onelake-access-api),
[managed-folder restrictions](https://learn.microsoft.com/en-us/fabric/onelake/onelake-api-parity),
[schema-scoped table loading](https://learn.microsoft.com/en-us/rest/api/fabric/lakehouse/tables/load-schema-table%28beta%29),
and [Delta metadata API](https://learn.microsoft.com/en-us/fabric/onelake/table-apis/delta-table-apis-get-started).
