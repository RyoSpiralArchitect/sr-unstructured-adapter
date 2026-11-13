# Reopen a source region and retain a new reading

`prepare_ontology_reread(database, reference=<pinned claim>, node_id=<evidence node>)`
returns a verified descriptor and PNG bytes using only SQLite. The source unit
must belong to the inspected claim; page/region coordinates come from its saved
locator. The descriptor retains original text, source/frame hashes, page, pixel
coordinates and the selected snapshot. Native DOCX sources without stored page
views are explicitly unsupported by this visual API.

`reread_ontology_source` takes the same binding, a trusted `complete(packet)`
callback, `model` identifier and exclusive `destination`. The host adapter can
send `packet['image']` as image input. It returns one JSON observation:
`{"text":"visible text","status":"read","notes":"..."}`. Status may also be
`uncertain` or `unreadable` (the latter requires empty text). Credentials and
provider calls belong to the host callback; there is one call and no retry.

Requests, image regions and responses survive validation failures in the output
directory. Accepted observations are stored alongside the ontology in the
append-only `sro_reading_reviews` table, with content-derived identities and exact
image bytes. `read_ontology_reread(database, review_id)` verifies both observation
and source. Identical observations are idempotent. New readings are unverified;
claims, original readings, selectors and active snapshots are not automatically
changed. The API is host-owned; applications must apply their collection
allowlist before invoking it.

On the existing synthetic image, the saved OCR string contained `審査担` while
the recovered page region visibly contains `審査担当`. The region was recovered
from the copied SQLite, without reopening source files. This is a visual
inspection, not a new Luna result or a measured OCR accuracy improvement.
Deterministic tests exercise DB-only pixels, foreign-unit rejection, append-only
readings, idempotence, tamper rejection and retained failed responses.
