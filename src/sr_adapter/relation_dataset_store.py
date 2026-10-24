"""Portable relation snapshots and a local SQL sink, independent of cloud targets.

Saved inputs allow replay without a model call. Parquet is only a serialization;
source verification and the versioned profile remain authoritative for the rows.
"""
from __future__ import annotations

from contextlib import closing
import io
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3

from .knowledge import sha256
from .knowledge_store import atomic_json
from .relation_dataset import RelationTable
from .relation_profiles import narrative_dataset, visual_dataset
from .tables import canonical_json

MAX_BYTES = 64 * 1024 * 1024
PROFILES = {"narrative": "narrative_relations/1", "visual": "visual_relations/1"}


def _key(package_id):
    key = package_id.rsplit(":", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Source package ID must end in a SHA-256 hash")
    return key


def _project(profile, bundles, review):
    if profile == PROFILES["narrative"]:
        from .narrative import verify_narrative_payloads
        if review is not None:
            raise ValueError("Visual review findings require the visual profile")
        return narrative_dataset([(verify_narrative_payloads(files), files) for files in bundles])
    if profile != PROFILES["visual"]:
        raise ValueError("Unsupported dataset profile/version")
    from .visual_relations import project_visual
    packages = []
    for bundle in bundles:
        files = dict(bundle)
        saved = json.loads(files.pop("manifest.json"))
        package = project_visual(files)
        if saved != {"version": package["config"]["version"], "status": "complete",
                     "package_id": package["package_id"], "files": _hashes(files)}:
            raise ValueError("Visual source receipt mismatch")
        packages.append(package)
    return visual_dataset(packages, review)


def _hashes(files):
    return {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in files.items()}


def load_relation_dataset(bundles, *, profile, review=None):
    """Verify saved source bundles and return an immutable dataset plus source files."""
    if profile not in PROFILES or not 1 <= len(bundles) <= 20:
        raise ValueError("Select 1 to 20 bundles and an explicit narrative or visual profile")
    if profile == "narrative":
        from .narrative import verify_narrative as verify
    else:
        from .visual_relations import verify_visual as verify
    payloads, sources, total = [], {}, 0
    for bundle in bundles:
        package, files = verify(bundle)
        key = _key(package["package_id"])
        if key in sources:
            raise ValueError("Duplicate source package")
        total += sum(map(len, files.values()))
        if total > MAX_BYTES:
            raise ValueError("Source selection exceeds 64 MiB")
        sources[key] = files
        payloads.append(files)
    dataset = _project(PROFILES[profile], payloads, review)
    return dataset, sources


def _manifest(dataset, files):
    description = dataset.describe()
    return {"schema_version": 1, "kind": "relation_dataset", "status": "complete",
            "snapshot_id": sha256(canonical_json(description)), **description, "files": _hashes(files)}


def export_relation_dataset(bundles, destination, *, profile, review=None):
    """Save all scalar tables and replay inputs; no endpoint, identity or network."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    dataset, sources = load_relation_dataset(bundles, profile=profile, review=review)
    files = {"dataset.json": canonical_json(dataset.describe())}
    files.update({f"sources/{key}/{name}": data for key, entries in sources.items() for name, data in entries.items()})
    if review is not None:
        files["review.json"] = canonical_json(review)
    for table in dataset.tables:
        stream = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(table.rows(), schema=table.arrow_schema()), stream)
        files[f"tables/{table.name}.parquet"] = stream.getvalue()
    if sum(map(len, files.values())) > MAX_BYTES:
        raise ValueError("Relation dataset exceeds 64 MiB")
    manifest = _manifest(dataset, files)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def _read_files(root, manifest):
    files, total = {}, 0
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if (not name or name != str(relative) or relative.is_absolute() or "\\" in name
                or ".." in relative.parts or name == "manifest.json"):
            raise ValueError("Invalid dataset file path")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Dataset file escapes its root")
        with path.open("rb") as stream:
            data = stream.read(MAX_BYTES - total + 1)
        total += len(data)
        if total > MAX_BYTES or expected != {"bytes": len(data), "sha256": sha256(data)}:
            raise ValueError("Dataset size/hash mismatch")
        files[name] = data
    return files


def verify_relation_dataset(folder):
    """Rebuild every table from original sources; rehashed edits cannot bypass replay."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(folder).resolve()
    with (root / "manifest.json").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Dataset manifest exceeds 1 MiB")
    manifest = json.loads(raw)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("kind") != "relation_dataset" or manifest.get("status") != "complete"
            or type(manifest.get("contract_version")) is not int or manifest["contract_version"] != 1
            or manifest.get("profile") not in PROFILES.values()):
        raise ValueError("Incomplete or unsupported relation dataset")
    packages = manifest["package_ids"]
    if not 1 <= len(packages) <= 20 or len(set(packages)) != len(packages):
        raise ValueError("Invalid dataset package selection")
    files = _read_files(root, manifest)
    payloads, expected = [], {"dataset.json"}
    for package_id in packages:
        prefix = "sources/" + _key(package_id) + "/"
        payload = {name[len(prefix):]: data for name, data in files.items() if name.startswith(prefix)}
        if json.loads(payload["manifest.json"])["package_id"] != package_id:
            raise ValueError("Source package ID differs from its directory")
        payloads.append(payload)
        expected.update(prefix + name for name in payload)
    review = None
    if "review.json" in files:
        review = json.loads(files["review.json"])
        if review is None:
            raise ValueError("A saved review cannot be null")
        expected.add("review.json")
    dataset = _project(manifest["profile"], payloads, review)
    for table in dataset.tables:
        filename = f"tables/{table.name}.parquet"
        expected.add(filename)
        saved = pq.read_table(pa.BufferReader(files[filename]))
        if saved.schema != table.arrow_schema() or saved.to_pylist() != table.rows():
            raise ValueError("Dataset table differs from replayed source/candidates")
    if (set(files) != expected or files["dataset.json"] != canonical_json(dataset.describe())
            or manifest != _manifest(dataset, files)):
        raise ValueError("Dataset contract differs from replayed source/candidates")
    return dataset


def _create_sql(name, table):
    fields = []
    for column in table.columns:
        sql = f'"{column.name}" ' + ("TEXT" if column.kind == "string" else "INTEGER")
        if not column.nullable:
            sql += " NOT NULL"
        if column.kind == "boolean":
            sql += f' CHECK ("{column.name}" IN (0, 1))'
        fields.append(sql)
    fields.append("PRIMARY KEY (" + ", ".join(f'"{c}"' for c in table.primary_key) + ")")
    return f'CREATE TABLE "{name}" (' + ", ".join(fields) + ")"


def _read_sql_table(connection, name, table):
    fields = ", ".join(f'"{c.name}"' for c in table.columns)
    rows = []
    for values in connection.execute(f'SELECT {fields} FROM "{name}"'):
        row = {}
        for column, value in zip(table.columns, values):
            if column.kind == "boolean" and value is not None:
                if type(value) is not int or value not in (0, 1):
                    raise ValueError("SQLite boolean differs from the scalar contract")
                value = bool(value)
            row[column.name] = value
        rows.append(row)
    return RelationTable.from_rows(table.name, table.columns, table.primary_key, rows)


def write_dataset_sqlite(bundle, database):
    """Commit a complete snapshot transactionally; repeated writes require exact readback.

    Each snapshot gets separate tables, so equal labels or multiple source versions
    are never implicitly merged. Existing databases and other snapshots are kept.
    """
    dataset = verify_relation_dataset(bundle)
    snapshot = dataset.snapshot_id
    descriptor = canonical_json(dataset.describe()).decode()
    names = {t.name: f"srd_{snapshot}_{t.name}" for t in dataset.tables}
    inserted = 0
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS sr_relation_datasets (snapshot_id TEXT PRIMARY KEY NOT NULL, "
                           "profile TEXT NOT NULL, descriptor TEXT NOT NULL)")
        old = connection.execute("SELECT profile, descriptor FROM sr_relation_datasets WHERE snapshot_id=?",
                                 (snapshot,)).fetchone()
        if old is not None and old != (dataset.profile, descriptor):
            raise ValueError("Existing SQLite snapshot has different metadata")
        for table in dataset.tables:
            name = names[table.name]
            sql = _create_sql(name, table)
            if old is None:
                connection.execute(sql)  # A table without its snapshot marker is a conflict.
                placeholders = ", ".join("?" for _ in table.columns)
                connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', table.values)
                inserted += len(table.values)
            saved = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
            if saved != (sql,) or _read_sql_table(connection, name, table) != table:
                raise ValueError("Existing SQLite table differs from the complete dataset")
        if old is None:
            connection.execute("INSERT INTO sr_relation_datasets VALUES (?, ?, ?)",
                               (snapshot, dataset.profile, descriptor))
    return {"status": "committed", "snapshot_id": snapshot, "profile": dataset.profile,
            "inserted_rows": inserted, "tables": {t.name: {"name": names[t.name], **t.describe()} for t in dataset.tables}}
