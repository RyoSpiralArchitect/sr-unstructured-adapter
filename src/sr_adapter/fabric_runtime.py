"""Bounded Fabric publication with immutable files, async receipts, and Delta readback.

This is a single-writer snapshot publisher, not a general Delta reader or MERGE
engine. Unknown submissions are never retried. Status performs only GET/HEAD.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from time import time
from urllib.parse import quote, unquote
import uuid

import httpx

from .fabric import FabricTarget, rows_digest, verify_fabric
from .knowledge import sha256
from .knowledge_store import _reserve_receipt, atomic_json
from .tables import canonical_json

FABRIC_API = "https://api.fabric.microsoft.com/v1"
MAX_READ_BYTES = 64 * 1024 * 1024


class SnapshotConflict(ValueError):
    pass


class _Session:
    def __init__(self, target, credential, transport):
        self.target = target
        self.owned = credential is None
        if self.owned:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        self.credential = credential
        self.tokens = {}
        self.client = httpx.Client(transport=transport, timeout=60, follow_redirects=False)

    def close(self):
        self.client.close()
        if self.owned:
            self.credential.close()

    def request(self, method, url, *, storage=False, **kwargs):
        scope = "https://storage.azure.com/.default" if storage else "https://api.fabric.microsoft.com/.default"
        token = self.tokens.get(scope)
        if token is None or getattr(token, "expires_on", 0) <= time() + 60:
            token = self.credential.get_token(scope)
            self.tokens[scope] = token
        headers = {"Authorization": "Bearer " + token.token,
                   "x-ms-client-request-id": str(uuid.uuid4())}
        if storage:
            headers["x-ms-version"] = "2021-06-08"
        headers.update(kwargs.pop("headers", {}))
        response = self.client.request(method, url, headers=headers, **kwargs)
        if 300 <= response.status_code < 400:
            raise ValueError("Unexpected redirect")
        return response

    def read(self, url, **kwargs):
        response = self.request("GET", url, storage=True, **kwargs)
        response.raise_for_status()
        if len(response.content) > MAX_READ_BYTES:
            raise ValueError("Readback exceeds bounded snapshot size")
        return response

    def path(self, relative):
        return f"{self.target.onelake_endpoint}/{self.target.workspace_id}/{self.target.lakehouse_id}/{quote(relative, safe='/')}"

    def check_target(self):
        t = self.target
        response = self.request("GET", f"{FABRIC_API}/workspaces/{t.workspace_id}/lakehouses/{t.lakehouse_id}")
        response.raise_for_status()
        item = response.json()
        properties = item.get("properties", {})
        if (item.get("id") != t.lakehouse_id or item.get("workspaceId") != t.workspace_id
                or item.get("displayName") != t.lakehouse_name or item.get("type") != "Lakehouse"
                or properties.get("defaultSchema") is None):
            raise ValueError("Target must match the existing schema-enabled lakehouse")
        expected_path = f"https://onelake.dfs.fabric.microsoft.com/{t.workspace_id}/{t.lakehouse_id}/Files"
        if properties.get("oneLakeFilesPath") != expected_path:
            raise ValueError("Unexpected OneLake target path")
        return {"id": item["id"], "display_name": item["displayName"],
                "default_schema": properties["defaultSchema"],
                "sql_endpoint_id": properties.get("sqlEndpointProperties", {}).get("id")}

    def directory(self, relative):
        # Callers only supply children of Files; never create item/Files/Tables.
        if relative != "Files/sr-adapter" and not relative.startswith("Files/sr-adapter/"):
            raise ValueError("Directory must be below the managed Files folder")
        response = self.request("PUT", self.path(relative), storage=True, params={"resource": "directory"},
                                headers={"If-None-Match": "*"}, content=b"")
        if response.status_code in (409, 412):
            existing = self.request("HEAD", self.path(relative), storage=True)
            existing.raise_for_status()
            if existing.headers.get("x-ms-resource-type") != "directory":
                raise SnapshotConflict("Existing path is not a directory")
            return False
        else:
            response.raise_for_status()
            if response.status_code != 201:
                raise ValueError("Unexpected directory acknowledgement")
            return True

    def upload(self, relative, data):
        url = self.path(relative)
        old = self.request("GET", url, storage=True)
        if old.status_code != 404:
            old.raise_for_status()
            if old.content != data:
                raise SnapshotConflict("Existing OneLake object differs")
            return "already_present_verified"
        created = self.request("PUT", url, storage=True, params={"resource": "file"},
                               headers={"If-None-Match": "*"}, content=b"")
        if created.status_code in (409, 412):
            if self.read(url).content != data:
                raise SnapshotConflict("Concurrent OneLake object differs")
            return "already_present_verified"
        created.raise_for_status()
        if created.status_code != 201:
            raise ValueError("Unexpected file acknowledgement")
        response = self.request("PATCH", url, storage=True, params={"action": "append", "position": 0},
                                content=data, headers={"Content-Type": "application/octet-stream"})
        response.raise_for_status()
        response = self.request("PATCH", url, storage=True,
                                params={"action": "flush", "position": len(data), "close": "true"}, content=b"")
        response.raise_for_status()
        if self.read(url).content != data:
            raise SnapshotConflict("OneLake upload readback differs")
        return "uploaded_and_verified"

    def submission_path(self, table):
        # One durable reservation per destination/schema/table, independent of
        # local receipts. Table names include the complete snapshot hash.
        return f"Files/sr-adapter/_submissions/{self.target.schema_name}/{table['table_name']}"

    def submission(self, table):
        relative = self.submission_path(table)
        response = self.request("HEAD", self.path(relative), storage=True)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if response.headers.get("x-ms-resource-type") != "directory":
            raise SnapshotConflict("Submission reservation is not a directory")
        response = self.request("GET", self.path(relative + "/operation.json"), storage=True)
        if response.status_code == 404:
            return {}  # Reservation exists; submission may have happened.
        response.raise_for_status()
        try:
            record = response.json()
        except ValueError:
            return {}  # Interrupted operation-record upload; never resubmit.
        if not isinstance(record, dict) or not {"schema_version", "request_sha256", "operation_id"} <= record.keys():
            return {}
        operation_id = str(uuid.UUID(record["operation_id"]))
        if record != self.operation_record(table, operation_id):
            raise SnapshotConflict("Submission record differs from verified load request")
        return record

    def reserve_submission(self, table):
        self.directory("Files/sr-adapter/_submissions")
        self.directory(f"Files/sr-adapter/_submissions/{self.target.schema_name}")
        # Only this successful conditional creation authorizes one POST. Never
        # remove or reuse the reservation, including after an ambiguous failure.
        return self.directory(self.submission_path(table))

    @staticmethod
    def operation_record(table, operation_id):
        return {"schema_version": 1, "request_sha256": sha256(canonical_json(table["load_body"])),
                "operation_id": operation_id}

    def record_submission(self, table):
        self.upload(self.submission_path(table) + "/operation.json",
                    canonical_json(self.operation_record(table, table["operation_id"])))

    def reconcile_table(self, table, submission=None):
        if submission is not None:
            table["submission_reserved"] = True
            if submission.get("operation_id"):
                table["operation_id"] = submission["operation_id"]
        if table.get("operation_id"):
            operation_id = str(uuid.UUID(table["operation_id"]))
            response = self.request("GET", f"{FABRIC_API}/operations/{operation_id}")
            response.raise_for_status()
            state = response.json().get("status")
            table["operation_status"] = state
            if state in {"NotStarted", "Running"}:
                table["status"] = "pending"
                return
            if state in {"Failed", "Canceled", "Cancelled"}:
                table["status"] = "failed"
                return
            if state != "Succeeded":
                table["status"] = "unknown"
                return
        verification = self.verify_table(table)
        if verification is None:
            table["status"] = "awaiting_visibility" if table.get("operation_status") == "Succeeded" else "unknown"
        else:
            table.update(status="verified", verification=verification)

    def catalog(self, name):
        t = self.target
        base = f"https://onelake.table.fabric.microsoft.com/delta/{t.workspace_id}/{t.lakehouse_id}/api/2.1/unity-catalog"
        # The catalog selector accepts the immutable item GUID, avoiding name escaping.
        return self.request("GET", f"{base}/tables/{t.lakehouse_id}.{t.schema_name}.{name}", storage=True)

    def list_paths(self, relative):
        t = self.target
        response = self.read(f"{t.onelake_endpoint}/{t.workspace_id}", params={"resource": "filesystem",
            "directory": f"{t.lakehouse_id}/{relative}", "recursive": "false", "maxResults": 100})
        body = response.json()
        if response.headers.get("x-ms-continuation") or body.get("ContinuationToken"):
            raise ValueError("Delta log exceeds bounded reader limit")
        return body.get("paths", [])

    def verify_table(self, table):
        """Read a small immutable Delta snapshot, retaining duplicate detection.

        Checkpoint-only logs/deletion vectors are rejected rather than read
        incompletely. These are unnecessary for the publisher's new snapshots.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        response = self.catalog(table["table_name"])
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if response.json().get("data_source_format", "").upper() != "DELTA":
            raise SnapshotConflict("Existing table is not Delta")
        prefix = f"Tables/{self.target.schema_name}/{table['table_name']}"
        logs = self.list_paths(prefix + "/_delta_log")
        names = sorted(row["name"].rsplit("/", 1)[-1] for row in logs
                       if re.fullmatch(r"[0-9]{20}\.json", row["name"].rsplit("/", 1)[-1]))
        if not names or [int(name[:-5]) for name in names] != list(range(len(names))):
            raise ValueError("Readback requires a complete Delta JSON log from version zero")
        active, read_bytes = {}, 0
        for name in names:
            log = self.read(self.path(prefix + "/_delta_log/" + name)).content
            read_bytes += len(log)
            for line in log.splitlines():
                action = json.loads(line)
                if "remove" in action:
                    active.pop(action["remove"]["path"], None)
                if "add" in action:
                    add = action["add"]
                    if add.get("deletionVector") or add.get("partitionValues"):
                        raise ValueError("Partitioned/deletion-vector snapshots need a full Delta reader")
                    active[add["path"]] = add
        records = []
        for name in sorted(active):
            decoded = unquote(name)
            if (decoded.startswith("/") or ":" in decoded or "\\" in decoded
                    or any(part in ("", ".", "..") for part in decoded.split("/"))):
                raise ValueError("Delta file path escapes table")
            data = self.read(self.path(prefix + "/" + decoded)).content
            read_bytes += len(data)
            if read_bytes > MAX_READ_BYTES:
                raise ValueError("Delta readback exceeds bounded publisher limit")
            records.extend(pq.read_table(pa.BufferReader(data)).to_pylist())
        checksum = rows_digest(records)
        if len(records) != table["row_count"] or checksum != table["rows_sha256"]:
            raise SnapshotConflict("Delta rows differ from the verified package projection")
        return {"row_count": len(records), "rows_sha256": checksum, "delta_version": len(names) - 1,
                "active_parquet_files": len(active)}


def _failure(report, exc):
    if isinstance(exc, SnapshotConflict):
        report.update(status="conflict", error_type=type(exc).__name__)
    elif isinstance(exc, httpx.HTTPStatusError):
        report.update(status="failed" if 400 <= exc.response.status_code < 500 else "unknown",
                      error_type=type(exc).__name__, http_status=exc.response.status_code)
    else:
        report.update(status="unknown", error_type=type(exc).__name__)


def _status(tables):
    statuses = {table["status"] for table in tables}
    return ("committed" if statuses <= {"verified", "already_present_verified"}
            else "partial" if "failed" in statuses
            else "unknown" if "unknown" in statuses else "pending")


def publish_fabric(folder, *, target: FabricTarget, receipt, credential=None, transport=None):
    manifest, files, plan = verify_fabric(folder, target=target)
    report = {"destination": "fabric", "snapshot_id": manifest["snapshot_id"], "status": "running",
              "target": target.model_dump(mode="json"), "manifest_sha256": sha256(files["manifest.json"]),
              "files": [], "tables": [{**table, "status": "not_submitted"} for table in plan["tables"]]}
    receipt_path = _reserve_receipt(receipt, report)
    session = None
    try:
        session = _Session(target, credential, transport)
        report["observed_target"] = session.check_target()
        # Preflight every table before any upload/load. Never overwrite or append
        # to an existing table whose rows differ, even in a disposable lakehouse.
        for table in report["tables"]:
            verification = session.verify_table(table)
            if verification is not None:
                table.update(status="already_present_verified", verification=verification)
            submission = session.submission(table)
            if submission is not None:
                session.reconcile_table(table, submission)
        atomic_json(receipt_path, report)
        prefix = plan["files_prefix"]
        if any(table["status"] in {"pending", "unknown", "failed", "awaiting_visibility"}
               for table in report["tables"]):
            # An earlier attempt may still write this snapshot. Reconcile only;
            # a new receipt must not authorize another load or partial restart.
            for name, data in files.items():
                if session.read(session.path(prefix + "/" + name)).content != data:
                    raise SnapshotConflict("Uploaded snapshot differs on readback")
                report["files"].append({"name": name, "sha256": sha256(data),
                                        "status": "already_present_verified"})
            report["status"] = _status(report["tables"])
            atomic_json(receipt_path, report)
            return report
        directories = {"Files/sr-adapter", prefix}
        for name in files:
            parts = name.split("/")[:-1]
            directories.update(prefix + "/" + "/".join(parts[:i]) for i in range(1, len(parts) + 1))
        for directory in sorted(directories, key=lambda x: (x.count("/"), x)):
            session.directory(directory)
        markers = sorted(name for name in files if name.endswith("/manifest.json"))
        ordinary = sorted(set(files) - set(markers) - {"manifest.json"})
        for name in [*ordinary, *markers, "manifest.json"]:
            row = {"name": name, "sha256": sha256(files[name]), "status": "running"}
            report["files"].append(row)
            atomic_json(receipt_path, report)
            row["status"] = session.upload(prefix + "/" + name, files[name])
            atomic_json(receipt_path, report)
        for table in report["tables"]:
            if table["status"] in {"already_present_verified", "verified"}:
                continue
            # Re-check before claiming the remote, durable submission slot.
            verification = session.verify_table(table)
            if verification is not None:
                table.update(status="already_present_verified", verification=verification)
                continue
            if not session.reserve_submission(table):
                session.reconcile_table(table, session.submission(table))
                break
            table["submission_reserved"] = True
            table.update(status="submitting", request_sha256=sha256(canonical_json(table["load_body"])))
            atomic_json(receipt_path, report)
            response = session.request("POST", table["load_url"], json=table["load_body"])
            table["http_status"] = response.status_code
            table["response_sha256"] = sha256(response.content)
            response.raise_for_status()
            if response.status_code != 202:
                raise ValueError("Unexpected load acknowledgement; reconcile before retrying")
            operation_id = str(uuid.UUID(response.headers["x-ms-operation-id"]))
            table.update(status="pending", operation_id=operation_id,
                         retry_after=response.headers.get("Retry-After"))
            atomic_json(receipt_path, report)
            session.record_submission(table)
        report["status"] = _status(report["tables"])
    except Exception as exc:
        _failure(report, exc)
        for table in report["tables"]:
            if table["status"] == "submitting":
                table["status"] = report["status"]
    finally:
        if session:
            session.close()
    atomic_json(receipt_path, report)
    return report


def fabric_status(folder, *, target: FabricTarget, receipt, credential=None, transport=None):
    """Reconcile existing operations/files/tables, without submitting or retrying writes."""
    manifest, files, plan = verify_fabric(folder, target=target)
    path = Path(receipt)
    report = json.loads(path.read_bytes())
    if (report.get("destination") != "fabric" or report.get("target") != target.model_dump(mode="json")
            or report.get("snapshot_id") != manifest["snapshot_id"]
            or report.get("manifest_sha256") != sha256(files["manifest.json"])):
        raise ValueError("Receipt does not match verified export and target")
    if len(report["tables"]) != len(plan["tables"]) or any(
            any(row.get(key) != value for key, value in expected.items())
            for row, expected in zip(report["tables"], plan["tables"])):
        raise ValueError("Receipt table plan differs")
    session = None
    try:
        session = _Session(target, credential, transport)
        session.check_target()
        for name, data in files.items():
            if session.read(session.path(plan["files_prefix"] + "/" + name)).content != data:
                raise SnapshotConflict("Uploaded snapshot differs on readback")
        for table in report["tables"]:
            submission = session.submission(table) if not table.get("operation_id") else None
            session.reconcile_table(table, submission)
            atomic_json(path, report)
        report["status"] = _status(report["tables"])
        report.pop("error_type", None)
        report.pop("http_status", None)
    except Exception as exc:
        _failure(report, exc)
    finally:
        if session:
            session.close()
    atomic_json(path, report)
    return report
