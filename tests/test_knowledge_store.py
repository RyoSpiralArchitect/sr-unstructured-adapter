"""Storage round trips, failure receipts, and cloud protocol contracts (offline)."""

import json
import sqlite3
from types import SimpleNamespace

from azure.core.exceptions import ResourceExistsError
import httpx
import pytest

from sr_adapter.cli import main
from sr_adapter.knowledge import extract_knowledge, package_records
from sr_adapter.knowledge_store import (databricks_plan, databricks_status, export_bundle, upload_adls,
                                        verify_bundle, write_databricks, write_sqlite)


@pytest.fixture
def bundle(package, tmp_path):
    path = tmp_path / "bundle"
    export_bundle(package, path, parquet=True)
    return path


def test_jsonld_parquet_jsonl_are_one_verified_package(bundle, package):
    manifest, files, records = verify_bundle(bundle)
    assert manifest["package_id"] == package.id
    assert "records.parquet" in files and "ontology.jsonld" in files
    assert records == package_records(package)
    with pytest.raises(FileExistsError):
        export_bundle(package, bundle)


def test_mapping_class_order_survives_canonical_json_storage(package, tmp_path):
    mapping = package.mapping.model_copy(update={"classes": dict(reversed(list(package.mapping.classes.items())))})
    reordered = extract_knowledge(package.source, mapping)
    assert reordered.id == package.id
    bundle = tmp_path / "reordered-classes"
    export_bundle(reordered, bundle, parquet=True)
    manifest, _, records = verify_bundle(bundle)
    assert manifest["package_id"] == package.id
    assert write_sqlite(bundle, tmp_path / "roundtrip.sqlite3")["inserted"] == len(records)


def test_bundle_tamper_is_rejected_before_any_sqlite_write(bundle, tmp_path):
    (bundle / "records.jsonl").write_text('{}\n')
    destination = tmp_path / "test.sqlite3"
    with pytest.raises(ValueError, match="hash/size mismatch"):
        write_sqlite(bundle, destination)
    assert not destination.exists()


def test_sqlite_round_trip_and_exact_repeat(bundle, package, tmp_path):
    database = tmp_path / "test.sqlite3"
    first = write_sqlite(bundle, database)
    second = write_sqlite(bundle, database)
    assert first["inserted"] == first["records"] and second["inserted"] == 0
    with sqlite3.connect(database) as connection:
        records = connection.execute('SELECT package_id, kind, record_id, payload_json FROM sr_knowledge_records').fetchall()
    expected = package_records(package)
    assert sorted(records) == sorted(tuple(row[k] for k in ("package_id", "kind", "record_id", "payload_json")) for row in expected)


def test_sqlite_conflict_rolls_back_earlier_rows(bundle, tmp_path):
    database = tmp_path / "test.sqlite3"
    write_sqlite(bundle, database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM sr_knowledge_records WHERE kind='document'")
        connection.execute("UPDATE sr_knowledge_records SET payload_json='corrupt' WHERE kind='ontology'")
    with pytest.raises(ValueError, match="different content"):
        write_sqlite(bundle, database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM sr_knowledge_records WHERE kind='document'").fetchone()[0] == 0


class FakeADLS:
    def __init__(self):
        self.objects = {}
        self.directories = set()
        self.uploads = []
        self.fail_on = None

    def get_file_system_client(self, name):
        assert name == "data"
        return self

    def get_directory_client(self, path):
        def create_directory(**kwargs):
            assert kwargs["etag"] == "*"
            if path in self.directories:
                raise ResourceExistsError()
            self.directories.add(path)
        return SimpleNamespace(create_directory=create_directory, get_directory_properties=lambda: {})

    def get_file_client(self, path):
        def upload_data(data, *, overwrite):
            assert overwrite is False
            self.uploads.append(path)
            if path.endswith(self.fail_on or "no-such-file"):
                raise TimeoutError("secret credential must never enter the receipt")
            if path in self.objects:
                raise ResourceExistsError()
            self.objects[path] = data
        return SimpleNamespace(upload_data=upload_data,
            download_file=lambda: SimpleNamespace(readall=lambda: self.objects[path]))


def test_adls_manifest_is_last_and_identical_repeat_is_verified(bundle, tmp_path):
    client = FakeADLS()
    args = dict(account_url="https://demo.dfs.core.windows.net", filesystem="data", prefix="knowledge/bronze", service_client=client)
    first = upload_adls(bundle, receipt=tmp_path / 'first.json', **args)
    assert first["status"] == "committed" and client.uploads[-1].endswith('/manifest.json')
    second = upload_adls(bundle, receipt=tmp_path / 'second.json', **args)
    assert all(row["status"] == "already_present_verified" for row in second["files"])
    with pytest.raises(FileExistsError):
        upload_adls(bundle, receipt=tmp_path / 'second.json', **args)


def test_adls_failure_has_no_completion_marker_and_no_error_secret(bundle, tmp_path):
    client = FakeADLS()
    client.fail_on = "package.json"
    receipt = tmp_path / 'failure.json'
    with pytest.raises(RuntimeError, match="incomplete"):
        upload_adls(bundle, account_url="https://demo.dfs.core.windows.net", filesystem="data", prefix="knowledge",
                    service_client=client, receipt=receipt)
    assert not any(path.endswith('/manifest.json') for path in client.objects)
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "incomplete" and saved["files"][-1]["status"] == "running"
    assert "secret credential" not in receipt.read_text()


def test_adls_conflict_never_overwrites(bundle, tmp_path):
    client = FakeADLS()
    args = dict(account_url="https://demo.dfs.core.windows.net", filesystem="data", prefix="knowledge", service_client=client)
    upload_adls(bundle, receipt=tmp_path / 'first.json', **args)
    key = next(path for path in client.objects if path.endswith('/graph.jsonld'))
    client.objects[key] = b'changed elsewhere'
    with pytest.raises(RuntimeError):
        upload_adls(bundle, receipt=tmp_path / 'second.json', **args)
    assert client.objects[key] == b'changed elsewhere'


def test_databricks_values_are_parameters_and_identifiers_are_checked(bundle):
    plan = databricks_plan(bundle, table="catalog.schema.knowledge", warehouse_id="abc123")
    assert "Alpha" not in plan['request']['statement']
    assert "Alpha" in plan['request']['parameters'][1]['value']
    assert "MERGE INTO" in plan['request']['statement'] and "raise_error" in plan['request']['statement']
    assert "USING DELTA" in plan['required_table_ddl']
    with pytest.raises(ValueError):
        databricks_plan(bundle, table="catalog.schema.name;DROP", warehouse_id="abc123")


def test_databricks_pending_is_not_success_and_status_never_resends(bundle, tmp_path):
    calls = []
    def handler(request):
        calls.append(request.method)
        assert request.headers['authorization'] == 'Bearer fixture-token'
        state = 'PENDING' if request.method == 'POST' else 'SUCCEEDED'
        return httpx.Response(200, json={'statement_id':'statement-123', 'status':{'state':state}})
    transport = httpx.MockTransport(handler)
    receipt = tmp_path / 'statement.json'
    first = write_databricks(bundle, host='https://workspace.azuredatabricks.net', token='fixture-token',
        table='catalog.schema.knowledge', warehouse_id='abc123', receipt=receipt, transport=transport)
    assert first['status'] == 'pending' and 'fixture-token' not in receipt.read_text()
    second = databricks_status(receipt, host='https://workspace.azuredatabricks.net', token='fixture-token', transport=transport)
    assert second['status'] == 'committed' and calls == ['POST', 'GET']
    with pytest.raises(FileExistsError):
        write_databricks(bundle, host='https://workspace.azuredatabricks.net', token='fixture-token',
            table='catalog.schema.knowledge', warehouse_id='abc123', receipt=receipt, transport=transport)
    assert calls == ['POST', 'GET']


def test_databricks_ambiguous_transport_is_recorded_without_retry(bundle, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('secret provider data')
    receipt = tmp_path / 'unknown.json'
    report = write_databricks(bundle, host='https://workspace.azuredatabricks.net', token='fixture-token',
        table='catalog.schema.knowledge', warehouse_id='abc123', receipt=receipt, transport=httpx.MockTransport(handler))
    assert report['status'] == 'unknown' and len(calls) == 1
    assert 'secret provider' not in receipt.read_text()
    with pytest.raises(ValueError, match='No statement ID'):
        databricks_status(receipt, host='https://workspace.azuredatabricks.net', token='fixture-token')


def test_status_does_not_send_credentials_to_a_changed_receipt_host(bundle, tmp_path):
    receipt = tmp_path / 'statement.json'
    receipt.write_text(json.dumps({'destination':'databricks', 'host':'https://unexpected.example', 'statement_id':'valid-id'}))
    with pytest.raises(ValueError, match='does not match'):
        databricks_status(receipt, host='https://workspace.azuredatabricks.net', token='fixture-token')


def test_cli_extract_to_sqlite_and_default_databricks_plan(pdf_factory, mapping, rows, tmp_path, capsys):
    pdf = pdf_factory(rows)
    spec = tmp_path / 'mapping.json'
    spec.write_text(mapping.model_dump_json())
    bundle = tmp_path / 'cli-bundle'
    assert main(['knowledge','extract',str(pdf),'--mapping',str(spec),'--out',str(bundle),'--parquet']) == 0
    assert main(['knowledge','sqlite',str(bundle),'--database',str(tmp_path / 'demo.sqlite3')]) == 0
    assert main(['knowledge','databricks',str(bundle),'--table','cat.schema.records','--warehouse-id','abc']) == 0
    assert 'required_table_ddl' in capsys.readouterr().out
