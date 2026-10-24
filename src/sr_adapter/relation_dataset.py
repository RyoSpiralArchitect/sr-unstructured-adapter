"""Destination-independent, immutable scalar tables for relation handoffs.

Profiles retain source-family semantics; this contract does not merge entities,
infer edges, or certify model readings. No cloud or Arrow dependency is required.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .knowledge import sha256
from .tables import canonical_json


def rows_digest(rows):
    """Order-independent digest; duplicate rows remain part of the content."""
    return sha256(canonical_json(sorted(rows, key=canonical_json)))


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("Use a simple lowercase table/column name")


@dataclass(frozen=True)
class Column:
    name: str
    kind: str = "string"
    nullable: bool = False

    def __post_init__(self):
        _name(self.name)
        if self.kind not in {"string", "integer", "boolean"} or type(self.nullable) is not bool:
            raise ValueError("Unsupported scalar column contract")

    def describe(self):
        return {"name": self.name, "kind": self.kind, "nullable": self.nullable}


@dataclass(frozen=True)
class RelationTable:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    values: tuple[tuple, ...]

    def __post_init__(self):
        _name(self.name)
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "primary_key", tuple(self.primary_key))
        object.__setattr__(self, "values", tuple(tuple(row) for row in self.values))
        names = [c.name for c in self.columns]
        if not names or len(set(names)) != len(names):
            raise ValueError("Columns must be nonempty and distinct")
        if (not self.primary_key or len(set(self.primary_key)) != len(self.primary_key)
                or not set(self.primary_key) <= set(names)):
            raise ValueError("A table requires a distinct, declared primary key")
        key_indexes = [names.index(c) for c in self.primary_key]
        if any(self.columns[i].nullable for i in key_indexes):
            raise ValueError("Primary key columns must be non-nullable")
        seen = set()
        for row in self.values:
            if len(row) != len(self.columns):
                raise ValueError("Row width differs from column contract")
            for value, column in zip(row, self.columns):
                if value is None and column.nullable:
                    continue
                expected = {"string": str, "integer": int, "boolean": bool}[column.kind]
                if type(value) is not expected:
                    raise ValueError(f"Invalid scalar type for {self.name}.{column.name}")
                if column.kind == "integer" and not -(2**63) <= value < 2**63:
                    raise ValueError("Integer exceeds the signed 64-bit handoff contract")
            key = tuple(row[i] for i in key_indexes)
            if None in key or key in seen:
                raise ValueError("Null or duplicate table primary key")
            seen.add(key)
        object.__setattr__(self, "values", tuple(sorted(self.values,
            key=lambda row: canonical_json(dict(zip(names, row))))))

    @classmethod
    def from_rows(cls, name, columns, primary_key, rows):
        rows = list(rows)
        columns = tuple(columns)
        names = [c.name for c in columns]
        if any(set(row) != set(names) for row in rows):
            raise ValueError("Row fields differ from column contract; fields must not be dropped")
        return cls(name, columns, tuple(primary_key), tuple(tuple(row[c] for c in names) for row in rows))

    def rows(self):
        """Return fresh dictionaries; consumers cannot mutate the shared snapshot."""
        return [dict(zip((c.name for c in self.columns), row)) for row in self.values]

    def describe(self):
        return {"columns": [c.describe() for c in self.columns], "primary_key": list(self.primary_key),
                "rows": len(self.values), "rows_sha256": rows_digest(self.rows())}

    def arrow_schema(self):
        import pyarrow as pa
        types = {"string": pa.string(), "integer": pa.int64(), "boolean": pa.bool_()}
        return pa.schema([pa.field(c.name, types[c.kind], nullable=c.nullable) for c in self.columns])


@dataclass(frozen=True)
class RelationDataset:
    """Contract v1: one versioned semantic profile and all of its scalar tables."""

    profile: str
    package_ids: tuple[str, ...]
    tables: tuple[RelationTable, ...]

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]+/\d+", self.profile):
            raise ValueError("Dataset profile must include its version")
        packages = tuple(self.package_ids)
        if not packages or any(not isinstance(p, str) or not p for p in packages) or len(set(packages)) != len(packages):
            raise ValueError("Dataset packages must be nonempty and distinct")
        object.__setattr__(self, "package_ids", tuple(sorted(packages)))
        object.__setattr__(self, "tables", tuple(sorted(self.tables, key=lambda table: table.name)))
        if not self.tables or len({t.name for t in self.tables}) != len(self.tables):
            raise ValueError("Dataset tables must be nonempty and distinct")
        for table in self.tables:
            columns = {c.name: c for c in table.columns}
            if columns.get("package_id") != Column("package_id"):
                raise ValueError("Every table requires a non-null string package_id")
            if any(row["package_id"] not in self.package_ids for row in table.rows()):
                raise ValueError("Row references an unselected package")

    def describe(self):
        return {"contract_version": 1, "profile": self.profile, "package_ids": list(self.package_ids),
                "tables": {t.name: t.describe() for t in self.tables}}

    @property
    def snapshot_id(self):
        # Neither destination endpoint nor transport state is part of this identity.
        return sha256(canonical_json(self.describe()))

    def rows(self):
        return {table.name: table.rows() for table in self.tables}
