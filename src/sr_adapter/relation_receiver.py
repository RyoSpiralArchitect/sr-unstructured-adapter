"""A replayable search/graph delivery contract and local transactional receiver.

Publication makes a snapshot available, not semantically correct. Query results
always carry a source snapshot so later updates cannot change their evidence.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
from typing import Protocol

from .knowledge import sha256
from .relation_dataset import RelationDataset
from .relation_dataset_store import (
    MAX_BYTES, _load_dataset, _verify_payloads, _verify_sqlite_dataset, _write_sqlite_dataset,
)
from .relation_views import project_relation_view
from .tables import canonical_json


class ReceiverConflict(ValueError):
    """The collection changed, or its source profile differs; no activation occurred."""


@dataclass(frozen=True)
class RelationDelivery:
    """Receiver contract v1: replay inputs and both exact, immutable projections.

    Constructing from saved base-bundle bytes verifies and regenerates all three
    datasets. It neither trusts a caller-supplied flag nor calls a model/service.
    """

    files: tuple[tuple[str, bytes], ...]
    source: RelationDataset = field(init=False)
    search: RelationDataset = field(init=False)
    graph: RelationDataset = field(init=False)

    def __post_init__(self):
        files = tuple(tuple(entry) for entry in self.files)
        if (any(len(entry) != 2 or type(entry[0]) is not str or type(entry[1]) is not bytes for entry in files)
                or len({entry[0] for entry in files}) != len(files)):
            raise ValueError("Delivery requires distinct filenames and immutable bytes")
        payloads = dict(files)
        raw = payloads.pop("manifest.json")
        if len(raw) > 1024 * 1024 or sum(map(len, payloads.values())) > MAX_BYTES:
            raise ValueError("Delivery source exceeds the portable bundle size limit")
        source = _verify_payloads(json.loads(raw), payloads, allow_view=False)
        object.__setattr__(self, "files", tuple(sorted(files)))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "search", project_relation_view(source, view="search"))
        object.__setattr__(self, "graph", project_relation_view(source, view="graph"))

    def describe(self):
        return {"receiver_contract_version": 1, "source_snapshot_id": self.source.snapshot_id,
                "source_profile": self.source.profile,
                "datasets": {kind: {"snapshot_id": dataset.snapshot_id, **dataset.describe()}
                             for kind, dataset in self.datasets().items()}}

    def datasets(self):
        return {"source": self.source, "search": self.search, "graph": self.graph}

    @property
    def delivery_id(self):
        return sha256(canonical_json(self.describe()))


def load_relation_delivery(search_bundle, graph_bundle):
    """Reject incomplete, altered or mismatched views before contacting a receiver."""
    search, search_files = _load_dataset(search_bundle, allow_view=True)
    graph, graph_files = _load_dataset(graph_bundle, allow_view=True)
    sm, gm = (json.loads(files["manifest.json"]) for files in (search_files, graph_files))
    if (sm.get("view") != "search" or gm.get("view") != "graph"
            or sm.get("input_snapshot_id") != gm.get("input_snapshot_id")
            or sm.get("input_profile") != gm.get("input_profile")):
        raise ValueError("Receive requires search and graph views of the same source snapshot")
    delivery = RelationDelivery(tuple((name.removeprefix("input/"), data)
        for name, data in search_files.items() if name.startswith("input/")))
    if delivery.search != search or delivery.graph != graph:
        raise ValueError("Delivery views differ from source replay")
    return delivery


class RelationReceiver(Protocol):
    """A sink accepts a complete delivery with compare-and-set activation.

    Implementations must verify readback before reporting committed; the interface
    alone does not certify a sink. See the SQLite implementation and conformance tests.
    """

    def receive(self, delivery: RelationDelivery, *, collection: str,
                expected_snapshot: str | None = None) -> dict: ...


_SCHEMA = {
    "srr_snapshots": "CREATE TABLE srr_snapshots (snapshot_id TEXT PRIMARY KEY NOT NULL, descriptor TEXT NOT NULL)",
    "srr_files": "CREATE TABLE srr_files (snapshot_id TEXT NOT NULL, path TEXT NOT NULL, data BLOB NOT NULL, "
                 "PRIMARY KEY (snapshot_id, path))",
    "srr_collections": "CREATE TABLE srr_collections (collection TEXT PRIMARY KEY NOT NULL, "
                       "active_snapshot_id TEXT NOT NULL, source_profile TEXT NOT NULL)",
    "srr_members": "CREATE TABLE srr_members (collection TEXT NOT NULL, snapshot_id TEXT NOT NULL, "
                   "PRIMARY KEY (collection, snapshot_id))",
}


def _schema(connection, *, create=False):
    existing = {name: connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
                for name in _SCHEMA}
    if any(existing.values()) and not all(existing.values()):
        raise ValueError("Receiver catalog is incomplete; no implicit repair")
    for name, sql in _SCHEMA.items():
        saved = existing[name]
        if saved is None and create:
            connection.execute(sql)
        elif saved != (sql,):
            raise ValueError("Receiver catalog schema is missing or incompatible")


def _collection(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Collection must be a lowercase name of at most 64 characters")


def _snapshot(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Snapshot must be a SHA-256 identity")


def _read_delivery(connection, snapshot):
    _snapshot(snapshot)
    saved = connection.execute("SELECT descriptor FROM srr_snapshots WHERE snapshot_id=?", (snapshot,)).fetchone()
    if saved is None:
        raise ValueError("Receiver snapshot is incomplete")
    size = connection.execute("SELECT SUM(length(data)) FROM srr_files WHERE snapshot_id=?", (snapshot,)).fetchone()[0]
    if size is None or size > MAX_BYTES + 1024 * 1024:
        raise ValueError("Receiver replay archive is missing or exceeds its limit")
    delivery = RelationDelivery(tuple(connection.execute(
        "SELECT path, data FROM srr_files WHERE snapshot_id=?", (snapshot,))))
    if delivery.source.snapshot_id != snapshot or saved != (canonical_json(delivery.describe()).decode(),):
        raise ValueError("Receiver descriptor differs from source replay")
    for dataset in delivery.datasets().values():
        _verify_sqlite_dataset(connection, dataset)
    return delivery


class SQLiteRelationReceiver:
    """Reference receiver: archive, three datasets and collection activation are atomic."""

    def __init__(self, database):
        self.database = database

    def receive(self, delivery: RelationDelivery, *, collection: str, expected_snapshot=None):
        if not isinstance(delivery, RelationDelivery):
            raise TypeError("Receiver requires a replay-verified RelationDelivery")
        _collection(collection)
        if expected_snapshot is not None:
            _snapshot(expected_snapshot)
        snapshot = delivery.source.snapshot_id
        with closing(sqlite3.connect(self.database, timeout=30)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _schema(connection, create=True)
            current = connection.execute("SELECT active_snapshot_id, source_profile FROM srr_collections "
                                         "WHERE collection=?", (collection,)).fetchone()
            active = current[0] if current else None
            if current and current[1] != delivery.source.profile:
                raise ReceiverConflict("Use a separate collection for a different source profile")
            # An already-active delivery is safe to retry even after a lost response.
            # A stale retry of an older delivery cannot reactivate it without an
            # explicit expectation matching the currently active snapshot.
            if active != snapshot and active != expected_snapshot:
                raise ReceiverConflict("Collection changed; inspect it and supply its expected snapshot")
            if active is not None:
                if connection.execute("SELECT 1 FROM srr_members WHERE collection=? AND snapshot_id=?",
                                      (collection, active)).fetchone() is None:
                    raise ValueError("Active receiver snapshot has no collection membership")
                _read_delivery(connection, active)
            exists = connection.execute("SELECT 1 FROM srr_snapshots WHERE snapshot_id=?", (snapshot,)).fetchone()
            if exists:
                existing = _read_delivery(connection, snapshot)
                if existing.describe() != delivery.describe():
                    raise ValueError("Existing delivery differs from the incoming snapshot")
            else:
                # Orphan files without a snapshot marker are a conflict, not a repair.
                if connection.execute("SELECT 1 FROM srr_files WHERE snapshot_id=? LIMIT 1", (snapshot,)).fetchone():
                    raise ValueError("Receiver archive exists without its complete snapshot")
                connection.executemany("INSERT INTO srr_files VALUES (?, ?, ?)",
                                       ((snapshot, name, data) for name, data in delivery.files))
            receipts = {kind: _write_sqlite_dataset(connection, dataset)
                        for kind, dataset in delivery.datasets().items()}
            if not exists:
                connection.execute("INSERT INTO srr_snapshots VALUES (?, ?)",
                                   (snapshot, canonical_json(delivery.describe()).decode()))
            _read_delivery(connection, snapshot)  # Archive and every table, before activation.
            if connection.execute("SELECT 1 FROM srr_members WHERE collection=? AND snapshot_id=?",
                                  (collection, snapshot)).fetchone() is None:
                connection.execute("INSERT INTO srr_members VALUES (?, ?)", (collection, snapshot))
            if current is None:
                connection.execute("INSERT INTO srr_collections VALUES (?, ?, ?)",
                                   (collection, snapshot, delivery.source.profile))
            elif active != snapshot:
                connection.execute("UPDATE srr_collections SET active_snapshot_id=? WHERE collection=?",
                                   (snapshot, collection))
            observed = connection.execute("SELECT active_snapshot_id FROM srr_collections WHERE collection=?",
                                          (collection,)).fetchone()
            if observed != (snapshot,):
                raise ValueError("Receiver activation readback differs")
        # Returning from outside the transaction means commit succeeded as well.
        return {"status": "committed", "receiver_contract_version": 1, "collection": collection,
                "snapshot_id": snapshot, "delivery_id": delivery.delivery_id,
                "previous_snapshot_id": active, "outcome": "already_active" if active == snapshot else "activated",
                "readback_verified": True, "inserted_rows": sum(r["inserted_rows"] for r in receipts.values()),
                "datasets": receipts}


def receive_relation_views(search_bundle, graph_bundle, database, *, collection, expected_snapshot=None):
    delivery = load_relation_delivery(search_bundle, graph_bundle)
    return SQLiteRelationReceiver(database).receive(delivery, collection=collection, expected_snapshot=expected_snapshot)


def _load_received(database, collection, snapshot_id=None):
    _collection(collection)
    if snapshot_id is not None:
        _snapshot(snapshot_id)
    # Read-only URI avoids creating a database on a miss. BEGIN pins activation,
    # replay inputs and tables to the same consistent read transaction.
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection, connection:
        connection.execute("BEGIN")
        _schema(connection)
        current = connection.execute("SELECT active_snapshot_id, source_profile FROM srr_collections "
                                     "WHERE collection=?", (collection,)).fetchone()
        if current is None:
            raise ValueError("Receiver collection does not exist")
        selected = current[0] if snapshot_id is None else snapshot_id
        if connection.execute("SELECT 1 FROM srr_members WHERE collection=? AND snapshot_id=?",
                              (collection, selected)).fetchone() is None:
            raise ValueError("Snapshot does not belong to this collection")
        delivery = _read_delivery(connection, selected)
        if current[1] != delivery.source.profile:
            raise ValueError("Receiver collection profile differs from source replay")
    return delivery, current[0]


def _query_receipt(delivery, collection, active):
    return {"status": "complete", "receiver_contract_version": 1, "collection": collection,
            "snapshot_id": delivery.source.snapshot_id, "active_snapshot_id": active,
            "delivery_id": delivery.delivery_id, "readback_verified": True,
            "semantic_validation": "not_performed"}


def search_received_relations(database, *, collection, query="", scope="relation",
                              assertion_status=None, support=None, limit=20, offset=0, snapshot_id=None):
    """Literal, Unicode-casefolded search; no embeddings, ranking, or ontology reasoning.

    The default searches relation labels/predicate/title. Context search is explicit:
    shared full-document context may match every card from the same source.
    """
    if not isinstance(query, str) or len(query) > 4096 or scope not in {"relation", "context"}:
        raise ValueError("Select relation/context scope and a query of at most 4096 characters")
    if assertion_status not in {None, "asserted", "denied", "unknown", "ambiguous"}:
        raise ValueError("Unsupported assertion status filter")
    if support not in {None, "explicit", "uncertain"}:
        raise ValueError("Unsupported visual support filter")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Use limit 1..100 and a nonnegative offset")
    delivery, active = _load_received(database, collection, snapshot_id)
    columns = ("content",) if scope == "context" else ("title", "subject_label", "predicate", "object_label")
    term = query.casefold()
    hits = [card for card in delivery.search.rows()["cards"]
            if (assertion_status is None or card["assertion_status"] == assertion_status)
            and (support is None or card["support"] == support)
            and any(term in (card[column] or "").casefold() for column in columns)]
    hits.sort(key=lambda card: (card["package_id"], card["card_id"]))
    return {**_query_receipt(delivery, collection, active), "query": query, "scope": scope,
            "filters": {"assertion_status": assertion_status, "support": support},
            "match_mode": "literal_casefold", "total": len(hits), "offset": offset, "limit": limit,
            "has_more": offset + limit < len(hits), "hits": hits[offset:offset + limit]}


def trace_received_relation(database, *, collection, card_id, snapshot_id=None):
    """Resolve one received card to its exact assertion, cited evidence and locators."""
    if not isinstance(card_id, str) or not re.fullmatch(r"c_[0-9a-f]{64}", card_id):
        raise ValueError("Use a card ID returned by the receiver search")
    delivery, active = _load_received(database, collection, snapshot_id)
    card = next((r for r in delivery.search.rows()["cards"] if r["card_id"] == card_id), None)
    if card is None:
        raise ValueError("Card does not belong to the selected snapshot; use the search result's snapshot ID")
    rows = delivery.graph.rows()
    nodes = {r["node_id"]: r for r in rows["nodes"] if r["package_id"] == card["package_id"]}
    properties = {node: {} for node in nodes}
    for row in rows["properties"]:
        if row["node_id"] in nodes:
            column = {"string": "text_value", "integer": "integer_value", "boolean": "boolean_value"}.get(row["value_kind"])
            properties[row["node_id"]][row["property_name"]] = row[column] if column else None
    links = {}
    for edge in rows["edges"]:
        if edge["package_id"] == card["package_id"]:
            links.setdefault((edge["source_id"], edge["role"]), []).append(edge)

    def record(node):
        return {"node": nodes[node], "properties": properties[node]}

    def linked(node, role):
        return sorted(links.get((node, role), []), key=lambda e: (e["position"] or 0, e["edge_id"]))

    root = card["graph_node_id"]
    source = next(n for n, row in nodes.items() if row["node_kind"] == "source")
    relevant = {root, source}
    mentions = {}
    for role in ("subject", "object"):
        for edge in linked(root, role):
            mentions[role] = record(edge["target_id"])
            relevant.add(edge["target_id"])
    evidence = []
    for edge in linked(root, "cites"):
        node = edge["target_id"]
        observed = [e["target_id"] for e in linked(node, "observes")]
        relevant.update([node, *observed])
        evidence.append({"position": edge["position"], **record(node),
            "locations": [record(e["target_id"]) for e in linked(node, "location")],
            "observations": [record(n) for n in observed]})
    reviews = [record(e["source_id"]) for e in rows["edges"]
               if e["package_id"] == card["package_id"] and e["role"] == "reviews" and e["target_id"] in relevant]
    return {**_query_receipt(delivery, collection, active), "card": card, "source": record(source),
            "assertion": record(root) if card["card_kind"] == "relation" else None,
            "mentions": mentions, "evidence": evidence, "reviews": reviews,
            "limitations": [record(n) for n, row in nodes.items() if row["node_kind"] == "limitation"]}
