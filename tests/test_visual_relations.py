import copy
import json
from types import SimpleNamespace

import httpx
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.fabric import FabricTarget, rows_digest, verify_fabric
from sr_adapter.fabric_visual import COLUMNS, export_visual
from sr_adapter.knowledge import sha256
from sr_adapter.tables import canonical_json
from sr_adapter.visual_relations import (Box, Inventory, Relations, image_info,
                                        validate_relations, verify_visual, visual_request)
from sr_adapter.visual_runtime import extract_visual

ENDPOINT = "https://example.services.ai.azure.com/api/projects/test"
TARGET = FabricTarget(workspace_id="11111111-1111-1111-1111-111111111111",
                      lakehouse_id="22222222-2222-2222-2222-222222222222", lakehouse_name="test")
BOX = {"x0": 0, "y0": 0, "x1": 500, "y1": 500}
INVENTORY = {"observations": [
    {"id": "n1", "label": "共通システム", "kind": "system", "box": BOX},
    {"id": "n2", "label": "共通システム", "kind": "system", "box": dict(BOX, x0=500, x1=1000)},
    {"id": "n3", "label": "HD 0%出資", "kind": "annotation", "box": dict(BOX, y0=500, y1=1000)}],
    "limitations": ["方向は不明"]}
RELATIONS = {"relations": [{"id": "r1", "subject_id": "n1", "predicate": "connected_to", "object_id": "n2",
    "value_text": "0%", "direction": "unspecified", "basis": "connector", "support": "uncertain",
    "evidence_ids": ["n1", "n2", "n3"], "evidence_description": "線と注記", "box": BOX}], "limitations": []}


def response(payload, **extra):
    return {"status": "completed", "model": "test-model-version", "usage": {"total_tokens": 10},
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(payload)}]}], **extra}


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (100, 100), "white").save(path)
    return path


def run(source, out, *, strategy="grounded", responses=None, exception=None, execute=True):
    values = responses if responses is not None else [response(INVENTORY), response(RELATIONS)]
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        if exception:
            raise exception
        return httpx.Response(200, json=values[len(calls) - 1])

    credential = SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="test"))
    report = extract_visual(source, out, endpoint=ENDPOINT, model="gpt-test", page_label="7",
                            strategy=strategy, execute=execute, credential=credential,
                            transport=httpx.MockTransport(handle))
    return report, calls


def test_grounded_preserves_image_each_stage_and_distinct_occurrences(source, tmp_path):
    report, calls = run(source, tmp_path / "run")
    assert report["status"] == "complete"
    assert len(calls) == 2
    assert all(c["store"] is False for c in calls)
    assert calls[0]["input"][1]["content"][0] == calls[1]["input"][1]["content"][0]
    assert json.loads(calls[1]["input"][1]["content"][1]["text"]) == INVENTORY
    package, _ = verify_visual(tmp_path / "run")
    assert len(package["inventory"]["observations"]) == 3
    assert package["semantic_validation"] == "not_performed"
    assert package["relations"]["relations"][0]["direction"] == "unspecified"


def test_direct_comparator_same_image_model_and_schema(source, tmp_path):
    data = {**INVENTORY, "relations": RELATIONS["relations"]}
    report, calls = run(source, tmp_path / "direct", strategy="direct", responses=[response(data)])
    assert report["status"] == "complete" and len(calls) == 1
    assert calls[0]["model"] == "gpt-test"
    package, _ = verify_visual(tmp_path / "direct")
    assert package["config"]["strategy"] == "direct"


@pytest.mark.parametrize("raw", [response(INVENTORY, status="incomplete"), response(INVENTORY, error={"message": "failed"}),
    {"status": "completed", "model": "test", "output": [{"type": "message", "role": "assistant",
     "content": [{"type": "refusal", "refusal": "no"}]}]}])
def test_incomplete_or_refused_never_publishes(source, tmp_path, raw):
    report, calls = run(source, tmp_path / "failed", responses=[raw])
    assert report["status"] == "failed" and len(calls) == 1
    assert not (tmp_path / "failed/manifest.json").exists()
    assert (tmp_path / "failed/inventory-response.json").exists()


def test_transport_unknown_no_retry(source, tmp_path):
    report, calls = run(source, tmp_path / "unknown", exception=httpx.ReadTimeout("unknown"))
    assert report["status"] == "unknown" and len(calls) == 1
    assert report["automatic_retries"] == 0
    assert not (tmp_path / "unknown/manifest.json").exists()


def test_dry_run_no_network_and_repeated_run_no_overwrite(source, tmp_path):
    report, calls = run(source, tmp_path / "prepared", execute=False)
    assert report["status"] == "prepared" and calls == []
    with pytest.raises(FileExistsError):
        run(source, tmp_path / "prepared")


@pytest.mark.parametrize("change", [
    {"subject_id": "absent"}, {"evidence_ids": ["n1", "n3"]}, {"evidence_ids": ["n1", "n2", "n2"]},
    {"value_text": "100%"}, {"basis": "legend"}, {"basis": "ambiguous", "support": "explicit"},
    {"object_id": "n1"},
])
def test_reference_and_literal_constraints(change):
    values = copy.deepcopy(RELATIONS)
    values["relations"][0].update(change)
    with pytest.raises(ValueError):
        validate_relations(Inventory.model_validate(INVENTORY), Relations.model_validate(values))


def test_bounds_and_nonimages_rejected():
    with pytest.raises(ValueError):
        Box.model_validate(dict(BOX, x1=0))
    with pytest.raises(ValueError):
        Box.model_validate(dict(BOX, y1=1001))
    with pytest.raises(ValueError):
        image_info(b"text", "source.txt")
    data = copy.deepcopy(INVENTORY)
    data["observations"][1]["id"] = "n1"
    with pytest.raises(ValueError):
        Inventory.model_validate(data)


def test_scalar_fabric_roundtrip_and_tamper(source, tmp_path):
    run(source, tmp_path / "run")
    folder = tmp_path / "fabric"
    result = export_visual([tmp_path / "run"], folder, target=TARGET)
    manifest, _, plan = verify_fabric(folder, target=TARGET)
    assert result == manifest and len(plan["tables"]) == 5
    for kind in COLUMNS:
        table = pq.read_table(folder / f"tables/{kind}.parquet")
        assert all(pa.types.is_string(f.type) or pa.types.is_int64(f.type) for f in table.schema)
        assert not any(c.endswith("_json") for c in table.column_names)
    rows = pq.read_table(folder / "tables/relations.parquet").to_pylist()
    assert rows[0]["subject_id"] != rows[0]["object_id"]
    assert rows[0]["subject_label"] == rows[0]["object_label"]
    assert rows[0]["review_status"] == "unreviewed_model_candidate"
    assert rows[0]["value_text"] == "0%"
    path = folder / "tables/relations.parquet"
    schema = pq.read_table(path).schema
    rows[0]["subject_label"] = "tampered"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    data = path.read_bytes()
    manifest["files"]["tables/relations.parquet"] = {"bytes": len(data), "sha256": sha256(data)}
    manifest["tables"]["relations"]["rows_sha256"] = rows_digest(rows)
    (folder / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError):
        verify_fabric(folder, target=TARGET)


def test_target_duplicate_and_raw_request_tamper(source, tmp_path):
    run(source, tmp_path / "run")
    with pytest.raises(ValueError, match="Duplicate"):
        export_visual([tmp_path / "run", tmp_path / "run"], tmp_path / "duplicate", target=TARGET)
    export_visual([tmp_path / "run"], tmp_path / "fabric", target=TARGET)
    with pytest.raises(ValueError):
        verify_fabric(tmp_path / "fabric", target=TARGET.model_copy(update={"lakehouse_name": "other"}))
    path = tmp_path / "run/inventory-request.json"
    request = json.loads(path.read_bytes())
    request["store"] = True
    path.write_bytes(canonical_json(request))
    manifest_path = tmp_path / "run/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="versioned contract"):
        verify_visual(tmp_path / "run")


def test_empty_extraction_creates_only_image_table(source, tmp_path):
    run(source, tmp_path / "empty", responses=[response({"observations": [], "limitations": []}),
                                               response({"relations": [], "limitations": []})])
    export_visual([tmp_path / "empty"], tmp_path / "fabric", target=TARGET)
    _, _, plan = verify_fabric(tmp_path / "fabric", target=TARGET)
    assert [t["kind"] for t in plan["tables"]] == ["images"]


def test_cli_visual_registered_and_invalid_options(tmp_path):
    from sr_adapter.cli import main
    target = tmp_path / "target.json"
    target.write_text(TARGET.model_dump_json())
    assert main(["knowledge", "fabric", "nonexistent", "--mode", "visual", "--slides", "7",
                 "--target", str(target), "--out", str(tmp_path / "unused")]) == 1


def test_request_schema_bounded_locally_and_stages(source):
    request = visual_request(source.read_bytes(), source.name, "gpt-test", "direct")
    assert request["text"]["format"]["strict"] is True
    schema_text = json.dumps(request["text"]["format"]["schema"])
    assert '"minimum"' not in schema_text and '"maximum"' not in schema_text
    with pytest.raises(ValueError):
        visual_request(source.read_bytes(), source.name, "gpt-test", "relations")
    with pytest.raises(ValueError):
        visual_request(source.read_bytes(), source.name, "bad/path", "inventory")


def test_execution_receipt_is_sealed_and_exported(source, tmp_path):
    run(source, tmp_path / "run")
    _, files = verify_visual(tmp_path / "run")
    execution = json.loads(files["execution.json"])
    assert execution["project_endpoint"] == ENDPOINT
    assert all(r["http_status"] == 200 and r["status"] == "completed" for r in execution["requests"])
    export_visual([tmp_path / "run"], tmp_path / "fabric", target=TARGET)
    _, exported, _ = verify_fabric(tmp_path / "fabric", target=TARGET)
    assert any(n.endswith("/execution.json") and data == files["execution.json"] for n, data in exported.items())
    rows = pq.read_table(tmp_path / "fabric/tables/images.parquet").to_pylist()
    assert rows[0]["project_endpoint"] == ENDPOINT and rows[0]["contract_version"] == 2


def test_response_stream_is_bounded_before_disk_write(source, tmp_path, monkeypatch):
    import sr_adapter.visual_runtime as runtime
    monkeypatch.setattr(runtime, "MAX_BYTES", 1024)
    consumed = []

    class LargeBody(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(1000):
                consumed.append(i)
                yield b"x" * 1024

    credential = SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="test"))
    report = extract_visual(source, tmp_path / "bounded", endpoint=ENDPOINT, model="gpt-test",
        page_label="7", execute=True, credential=credential,
        transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=LargeBody())))
    assert report["status"] == "failed"
    assert len(consumed) <= 64
    assert (tmp_path / "bounded/inventory-response.partial").stat().st_size == 1024
    assert not (tmp_path / "bounded/manifest.json").exists()
    assert report["requests"][0]["response_truncated_locally"] is True


def test_version_one_receipts_remain_replayable(source, tmp_path):
    from sr_adapter.visual_relations import project_visual
    config = dict(version=1, source_name=source.name, page_label="7", model="gpt-test", strategy="grounded")
    files = {"config.json": canonical_json(config), "source.bin": source.read_bytes()}
    inventory = None
    for stage, payload in [("inventory", INVENTORY), ("relations", RELATIONS)]:
        files[stage + "-request.json"] = canonical_json(visual_request(source.read_bytes(), source.name,
            "gpt-test", stage, inventory, version=1))
        files[stage + "-response.json"] = canonical_json(response(payload))
        inventory = Inventory.model_validate(INVENTORY)
    package = project_visual(files)
    manifest = {"version": 1, "status": "complete", "package_id": package["package_id"],
                "files": {n: {"bytes": len(v), "sha256": sha256(v)} for n, v in files.items()}}
    folder = tmp_path / "legacy"
    folder.mkdir()
    for name, content in files.items():
        (folder / name).write_bytes(content)
    (folder / "manifest.json").write_bytes(canonical_json(manifest))
    assert verify_visual(folder)[0]["config"]["version"] == 1
    export_visual([folder], tmp_path / "fabric", target=TARGET)
    assert verify_fabric(tmp_path / "fabric", target=TARGET)[0]["status"] == "complete"


def test_review_preserves_readings_and_flags_linked_relations(source, tmp_path):
    run(source, tmp_path / "run")
    package, _ = verify_visual(tmp_path / "run")
    review = {"version": 1, "reviewer": "assistant_visual_spot_check", "findings": [
        {"package_id": package["package_id"], "reference_kind": "item", "reference_id": "n1",
         "finding": "text_mismatch", "detail": "Visible text needs another reading."}]}
    export_visual([tmp_path / "run"], tmp_path / "fabric", target=TARGET, review=review)
    _, files, plan = verify_fabric(tmp_path / "fabric", target=TARGET)
    assert len(plan["tables"]) == 6 and json.loads(files["review.json"]) == review
    relations = pq.read_table(tmp_path / "fabric/tables/relations.parquet").to_pylist()
    assert relations[0]["subject_label"] == "共通システム"
    assert relations[0]["review_status"] == "model_candidate_needs_review"
    assert pq.read_table(tmp_path / "fabric/tables/reviews.parquet").num_rows == 1
    review["findings"][0]["reference_id"] = "n999"
    with pytest.raises(ValueError, match="reference"):
        export_visual([tmp_path / "run"], tmp_path / "bad", target=TARGET, review=review)


def test_review_rehash_cannot_retarget_a_finding(source, tmp_path):
    run(source, tmp_path / "run")
    package, _ = verify_visual(tmp_path / "run")
    review = {"version": 1, "reviewer": "reviewer", "findings": [
        {"package_id": package["package_id"], "reference_kind": "relation", "reference_id": "r1",
         "finding": "ambiguous_relation", "detail": "The line is ambiguous."}]}
    folder = tmp_path / "fabric"
    export_visual([tmp_path / "run"], folder, target=TARGET, review=review)
    review["findings"][0]["reference_kind"] = "image"
    review["findings"][0]["reference_id"] = None
    payload = canonical_json(review)
    (folder / "review.json").write_bytes(payload)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    manifest["files"]["review.json"] = {"bytes": len(payload), "sha256": sha256(payload)}
    (folder / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError):
        verify_fabric(folder, target=TARGET)
