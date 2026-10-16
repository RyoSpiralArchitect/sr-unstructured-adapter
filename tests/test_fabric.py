"""Fabric projections, managed-folder boundaries, async receipts, and real Delta bytes."""

import json
from types import SimpleNamespace
from urllib.parse import unquote
import uuid

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.cli import main
from sr_adapter.fabric import FabricTarget, export_fabric, verify_fabric
from sr_adapter.fabric_runtime import fabric_status, publish_fabric
from sr_adapter.knowledge import sha256
from sr_adapter.knowledge_store import export_bundle
from sr_adapter.tables import canonical_json


@pytest.fixture
def target():
    return FabricTarget(workspace_id="11111111-1111-1111-1111-111111111111",
                        lakehouse_id="22222222-2222-2222-2222-222222222222", lakehouse_name="test")


@pytest.fixture
def fabric_export(package, target, tmp_path):
    source = tmp_path / "source"
    export_bundle(package, source, parquet=True)
    output = tmp_path / "fabric"
    export_fabric([source], output, target=target)
    return output


def test_fabric_projection_retains_relations_unknowns_and_source_joins(fabric_export, target, package):
    manifest, files, plan = verify_fabric(fabric_export, target=target)
    def read(kind):
        return pq.read_table(pa.BufferReader(files[f"tables/{kind}.parquet"])).to_pylist()
    entities = {row["entity_id"]: row for row in read("entities")}
    evidence = {row["evidence_id"]: row for row in read("evidence")}
    for relation in read("relations"):
        assert relation["subject_id"] in entities and relation["object_id"] in entities
    for link in read("evidence_links"):
        assert link["evidence_id"] in evidence
    assert any(row["status"] == "unknown" and row["value"] is None for row in read("observations"))
    assert all(row["document_sha256"] == package.source.source_sha256 for row in evidence.values())
    assert all(row["coordinate_system"] == "pdf_points_top_left" for row in evidence.values())
    assert manifest["tables"]["entities"]["rows"] == len(package.entities)
    assert all(table["load_body"]["mode"] == "Append" for table in plan["tables"])
    assert all("/schemas/dbo/" in table["load_url"] and table["load_url"].endswith("?beta=true") for table in plan["tables"])


def test_rehashed_projection_tamper_still_fails(fabric_export, target):
    file = fabric_export / "tables/relations.parquet"
    table = pq.read_table(file)
    rows = table.to_pylist()
    rows[0]["subject_id"], rows[0]["object_id"] = rows[0]["object_id"], rows[0]["subject_id"]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    path = fabric_export / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"]["tables/relations.parquet"] = {"sha256": sha256(file.read_bytes()), "bytes": file.stat().st_size}
    path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="Parquet differs"):
        verify_fabric(fabric_export, target=target)


class FabricFixture:
    def __init__(self, target):
        self.target = target
        self.files = {}
        self.directories = set()
        self.tables = set()
        self.loads = {}
        self.requests = []
        self.state = "Running"
        self.poll_error = False
        self.load_status = 202
        self.fail_upload = False

    def handle(self, request):
        self.requests.append(request)
        url = request.url
        path = unquote(url.path)
        method = request.method
        if url.host == "api.fabric.microsoft.com":
            if path.endswith("/lakehouses/" + self.target.lakehouse_id):
                return httpx.Response(200, json={"id": self.target.lakehouse_id, "workspaceId": self.target.workspace_id,
                    "displayName": self.target.lakehouse_name, "type": "Lakehouse", "properties": {"defaultSchema": "dbo",
                    "oneLakeFilesPath": f"https://onelake.dfs.fabric.microsoft.com/{self.target.workspace_id}/{self.target.lakehouse_id}/Files"}})
            if path.endswith("/load"):
                assert method == "POST" and url.params["beta"] == "true"
                if self.load_status != 202:
                    return httpx.Response(self.load_status, json={"error": "secret must not enter receipts"})
                name = path.split("/")[-2]
                operation = str(uuid.UUID(int=len(self.loads) + 1))
                self.loads[operation] = (name, json.loads(request.content))
                return httpx.Response(202, headers={"x-ms-operation-id": operation, "Retry-After": "30",
                    "Location": "https://evil.invalid/not-followed"})
            if "/operations/" in path:
                assert method == "GET"
                if self.poll_error:
                    return httpx.Response(503, json={"error": "secret provider details"})
                name, body = self.loads[path.rsplit("/", 1)[-1]]
                if self.state == "Succeeded":
                    self.tables.add(name)
                    prefix = f"Tables/dbo/{name}"
                    self.files[prefix + "/part-0000.parquet"] = self.files[body["relativePath"]]
                    self.files[prefix + "/_delta_log/00000000000000000000.json"] = canonical_json({
                        "add": {"path": "part-0000.parquet", "partitionValues": {}}}) + b"\n"
                return httpx.Response(200, json={"status": self.state})
        if url.host == "onelake.table.fabric.microsoft.com":
            name = path.rsplit(".", 1)[-1]
            return httpx.Response(200, json={"data_source_format": "DELTA"}) if name in self.tables else httpx.Response(404)
        if method == "GET" and url.params.get("resource") == "filesystem":
            directory = url.params["directory"].split("/", 1)[1]
            return httpx.Response(200, json={"paths": [{"name": self.target.lakehouse_id + "/" + name}
                for name in self.files if name.startswith(directory + "/")]})
        relative = path.split("/", 3)[-1]
        if method == "GET":
            return httpx.Response(200, content=self.files[relative]) if relative in self.files else httpx.Response(404)
        if method == "HEAD":
            return httpx.Response(200, headers={"x-ms-resource-type": "directory"}) if relative in self.directories else httpx.Response(404)
        if method == "PUT":
            assert request.headers["If-None-Match"] == "*"
            assert relative.startswith("Files/sr-adapter")  # No managed-root creation.
            if url.params["resource"] == "directory":
                if relative in self.directories:
                    return httpx.Response(409)
                self.directories.add(relative)
            else:
                if relative in self.files:
                    return httpx.Response(412)
                self.files[relative] = b""
            return httpx.Response(201)
        if method == "PATCH":
            if self.fail_upload:
                raise httpx.ReadTimeout("secret transport detail", request=request)
            if url.params["action"] == "append":
                self.files[relative] += request.content
            return httpx.Response(200)
        raise AssertionError((method, str(url)))

    @property
    def kwargs(self):
        return {"credential": SimpleNamespace(get_token=lambda _: SimpleNamespace(token="fixture-secret")),
                "transport": httpx.MockTransport(self.handle)}


def test_async_load_poll_exact_readback_and_identical_repeat(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    receipt = tmp_path / "receipt.json"
    pending = publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert pending["status"] == "pending" and len(service.loads) == 6
    assert all(row["status"] == "pending" for row in pending["tables"])
    puts = [request for request in service.requests if request.method == "PUT"
            and request.url.params.get("resource") == "file" and "/_submissions/" not in request.url.path]
    assert puts[-1].url.path.endswith("/manifest.json")
    for index, request in enumerate(puts):
        if "/bundles/" in request.url.path and request.url.path.endswith("/manifest.json"):
            source_prefix = request.url.path.rsplit("/", 1)[0]
            assert not any(later.url.path.startswith(source_prefix + "/") for later in puts[index + 1:])
    with pytest.raises(FileExistsError):
        publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    service.requests.clear()
    assert fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)["status"] == "pending"
    assert {r.method for r in service.requests} == {"GET"}
    service.state = "Succeeded"
    result = fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert result["status"] == "committed"
    assert all(table["verification"]["row_count"] == table["row_count"] for table in result["tables"])
    service.requests.clear()
    result = publish_fabric(fabric_export, target=target, receipt=tmp_path / "repeat.json", **service.kwargs)
    assert result["status"] == "committed" and not any(r.method == "POST" for r in service.requests)


def test_pending_snapshot_with_new_receipt_only_reconciles_existing_loads(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    first = publish_fabric(fabric_export, target=target, receipt=tmp_path / "first.json", **service.kwargs)
    assert first["status"] == "pending" and len(service.loads) == 6
    service.requests.clear()
    second_path = tmp_path / "second.json"
    second = publish_fabric(fabric_export, target=target, receipt=second_path, **service.kwargs)
    assert second["status"] == "pending" and len(service.loads) == 6
    assert [row["operation_id"] for row in second["tables"]] == [row["operation_id"] for row in first["tables"]]
    assert {r.method for r in service.requests} <= {"GET", "HEAD"}
    service.state = "Succeeded"
    assert fabric_status(fabric_export, target=target, receipt=second_path, **service.kwargs)["status"] == "committed"


def test_accepted_but_unacknowledged_load_is_not_repeated_with_new_receipt(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    original = service.handle

    def lose_acknowledgement(request):
        response = original(request)
        if request.method == "POST":
            raise httpx.ReadTimeout("lost load acknowledgement", request=request)
        return response

    kwargs = {**service.kwargs, "transport": httpx.MockTransport(lose_acknowledgement)}
    first = publish_fabric(fabric_export, target=target, receipt=tmp_path / "first.json", **kwargs)
    assert first["status"] == "unknown" and len(service.loads) == 1
    service.requests.clear()
    second = publish_fabric(fabric_export, target=target, receipt=tmp_path / "second.json", **service.kwargs)
    assert second["status"] == "unknown" and len(service.loads) == 1
    assert second["tables"][0]["submission_reserved"] is True
    assert {r.method for r in service.requests} <= {"GET", "HEAD"}


def test_interruption_after_reservation_never_reuses_the_submission_slot(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    original = service.handle

    def interrupt_before_post(request):
        if request.method == "POST":
            raise KeyboardInterrupt()
        return original(request)

    kwargs = {**service.kwargs, "transport": httpx.MockTransport(interrupt_before_post)}
    with pytest.raises(KeyboardInterrupt):
        publish_fabric(fabric_export, target=target, receipt=tmp_path / "interrupted.json", **kwargs)
    assert not service.loads
    service.requests.clear()
    result = publish_fabric(fabric_export, target=target, receipt=tmp_path / "second.json", **service.kwargs)
    assert result["status"] == "unknown" and not service.loads
    assert {r.method for r in service.requests} <= {"GET", "HEAD"}


def test_losing_conditional_reservation_race_never_submits_load(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    original = service.handle

    def claim_before_create(request):
        relative = unquote(request.url.path).split("/", 3)[-1]
        if request.method == "PUT" and relative.startswith("Files/sr-adapter/_submissions/dbo/sr_"):
            service.directories.add(relative)  # Another publisher won after preflight.
        return original(request)

    kwargs = {**service.kwargs, "transport": httpx.MockTransport(claim_before_create)}
    result = publish_fabric(fabric_export, target=target, receipt=tmp_path / "race.json", **kwargs)
    assert result["status"] == "unknown" and not service.loads
    assert not any(request.method == "POST" for request in service.requests)
    assert result["tables"][0]["submission_reserved"] is True
    assert result["tables"][1]["status"] == "not_submitted"


def test_interrupted_operation_record_keeps_local_id_and_blocks_resubmission(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    original = service.handle

    def fail_operation_record(request):
        if request.method == "PATCH" and request.url.path.endswith("/operation.json"):
            raise httpx.ReadTimeout("operation record unavailable", request=request)
        return original(request)

    receipt = tmp_path / "first.json"
    kwargs = {**service.kwargs, "transport": httpx.MockTransport(fail_operation_record)}
    first = publish_fabric(fabric_export, target=target, receipt=receipt, **kwargs)
    assert first["status"] == "unknown" and first["tables"][0]["operation_id"]
    assert len(service.loads) == 1
    service.requests.clear()
    second = publish_fabric(fabric_export, target=target, receipt=tmp_path / "second.json", **service.kwargs)
    assert second["status"] == "unknown" and len(service.loads) == 1
    assert {r.method for r in service.requests} <= {"GET", "HEAD"}
    service.state = "Succeeded"
    reconciled = fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert reconciled["tables"][0]["status"] == "verified" and len(service.loads) == 1


@pytest.mark.parametrize("status, expected", [(400, "failed"), (500, "unknown"), (207, "unknown")])
def test_load_errors_are_not_retried_or_claimed_committed(fabric_export, target, tmp_path, status, expected):
    service = FabricFixture(target)
    service.load_status = status
    receipt = tmp_path / "receipt.json"
    result = publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert result["status"] == expected
    assert len([r for r in service.requests if r.method == "POST"]) == 1
    assert "secret" not in receipt.read_text()
    assert result["tables"][1]["status"] == "not_submitted"


def test_upload_failure_does_not_submit_any_load(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    service.fail_upload = True
    receipt = tmp_path / "receipt.json"
    result = publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert result["status"] == "unknown" and not service.loads
    assert "secret" not in receipt.read_text()
    assert not any(name.endswith("/manifest.json") for name in service.files)


def test_partial_submission_keeps_operation_for_read_only_reconciliation(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    original = service.handle

    def fail_second(request):
        if request.method == "POST" and len(service.loads) == 1:
            return httpx.Response(503, json={"error": "ambiguous"})
        return original(request)

    receipt = tmp_path / "receipt.json"
    kwargs = {**service.kwargs, "transport": httpx.MockTransport(fail_second)}
    result = publish_fabric(fabric_export, target=target, receipt=receipt, **kwargs)
    assert result["status"] == "unknown"
    assert result["tables"][0]["operation_id"]
    assert result["tables"][1]["status"] == "unknown"
    assert result["tables"][2]["status"] == "not_submitted"
    service.state = "Succeeded"
    service.requests.clear()
    result = fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert result["status"] == "unknown" and result["tables"][0]["status"] == "verified"
    assert {request.method for request in service.requests} <= {"GET", "HEAD"}


def test_failed_operation_and_poll_transport_never_resubmit(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    receipt = tmp_path / "receipt.json"
    publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    service.requests.clear()
    service.poll_error = True
    assert fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)["status"] == "unknown"
    assert {r.method for r in service.requests} == {"GET"}
    service.poll_error = False
    service.state = "Failed"
    assert fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)["status"] == "partial"
    assert len(service.loads) == 6


def test_existing_mismatched_table_stops_before_any_write(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    plan = json.loads((fabric_export / "deployment-plan.json").read_bytes())
    name = plan["tables"][0]["table_name"]
    service.tables.add(name)
    service.files[f"Tables/dbo/{name}/_delta_log/00000000000000000000.json"] = b'{"commitInfo":{}}\n'
    result = publish_fabric(fabric_export, target=target, receipt=tmp_path / "receipt.json", **service.kwargs)
    assert result["status"] == "conflict"
    assert {r.method for r in service.requests} == {"GET"}


def test_tampered_receipt_cannot_redirect_status_or_resubmit(fabric_export, target, tmp_path):
    service = FabricFixture(target)
    receipt = tmp_path / "receipt.json"
    report = publish_fabric(fabric_export, target=target, receipt=receipt, **service.kwargs)
    report["tables"][0]["load_url"] = "https://evil.invalid/"
    receipt.write_bytes(canonical_json(report))
    service.requests.clear()
    with pytest.raises(ValueError, match="table plan differs"):
        fabric_status(fabric_export, target=target, receipt=receipt, **service.kwargs)
    assert not service.requests


def test_cli_requires_execute_before_fabric_publication(fabric_export, target, tmp_path, capsys):
    path = tmp_path / "target.json"
    path.write_text(target.model_dump_json())
    receipt = tmp_path / "receipt.json"
    assert main(["knowledge", "fabric-publish", str(fabric_export), "--target", str(path), "--receipt", str(receipt)]) == 1
    assert not receipt.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("endpoint", ["https://onelake.dfs.fabric.microsoft.com.evil.test", "http://onelake.dfs.fabric.microsoft.com",
    "https://user@onelake.dfs.fabric.microsoft.com", "https://onelake.dfs.fabric.microsoft.com/elsewhere"])
def test_target_rejects_credential_destinations(target, endpoint):
    with pytest.raises(ValueError):
        FabricTarget(**{**target.model_dump(), "onelake_endpoint": endpoint})
