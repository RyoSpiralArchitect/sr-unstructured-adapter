"""Narrative bundles reach the existing Foundry protocol without changing tables."""

import copy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from sr_adapter.foundry import FoundryTarget, _render, export_foundry, verify_foundry
from sr_adapter.foundry_runtime import publish_foundry
from sr_adapter.knowledge import KnowledgePackage, sha256
from sr_adapter.knowledge_store import export_bundle
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_sources import read_source
from sr_adapter.tables import canonical_json


def model_response(content):
    return {"status": "completed", "model": "gpt-5.5-fixture", "output": [{
        "type": "message", "role": "assistant", "content": [{
            "type": "output_text", "text": json.dumps(content, ensure_ascii=False),
        }],
    }]}


@pytest.fixture
def narrative_ontology():
    return NarrativeOntology.model_validate({
        "namespace": "urn:sr:foundry-test:", "classes": {"Project": "案件", "Organization": "組織"},
        "predicates": [{"id": "managedBy", "label": "担当", "subject_type": "Project",
                        "object_type": "Organization", "description": "案件の主担当組織"}],
    })


@pytest.fixture
def make_bundle(narrative_ontology, tmp_path):
    def create(name="source.txt", *, image=False):
        rows = [("asserted", "B", "A is managed by B."), ("denied", "C", "A is not managed by C."),
                ("unknown", None, "A manager is unknown."), ("ambiguous", None, "A may be managed by B or C.")]
        transcription_response = None
        if image:
            stream = BytesIO()
            Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
            data = stream.getvalue()
            transcription = {"lines": [row[2] for row in rows]}
            transcription_response = model_response(transcription)
            source = read_source(data, name, transcription=transcription)
        else:
            data = "\n".join(row[2] for row in rows).encode()
            source = read_source(data, name)
        assertions = [{"subject": "A", "predicate": "managedBy", "object": obj, "status": status,
            "evidence": [{"unit_id": unit["id"], "quote": unit["text"]}]}
            for (status, obj, _), unit in zip(rows, source["units"], strict=True)]
        manifest, files = render_narrative(data=data, source_name=name, ontology=narrative_ontology,
            model="gpt-5.5", response=model_response({"assertions": assertions}),
            transcription_response=transcription_response)
        path = tmp_path / (name + "-bundle")
        save_narrative(path, manifest, files)
        return path, manifest, files
    return create


@pytest.fixture
def target():
    return FoundryTarget(search_endpoint="https://fixture.search.windows.net",
        project_endpoint="https://fixture.services.ai.azure.com/api/projects/demo",
        project_resource_id="/subscriptions/fixture/resourceGroups/demo/providers/Microsoft.CognitiveServices/accounts/fixture/projects/demo",
        project_connection_name="narrative-read-only", model_deployment="gpt-5.5", agent_name="sr-narrative-fixture")


def test_v3_export_preserves_source_bundles_statuses_and_non_table_schema(make_bundle, target, tmp_path):
    path, package, source_files = make_bundle()
    output = tmp_path / "foundry"
    report = export_foundry([path], output, prefix="sr-narrative", target=target)
    manifest, files = verify_foundry(output)
    assert report == manifest
    assert manifest["schema_version"] == 3 and manifest["kind"] == "narrative_foundry"
    assert manifest["card_count"] == 4 and manifest["cloud_validation"] == "not_performed_by_export"
    prefix = "sources/" + package["package_id"].rsplit(":", 1)[-1] + "/"
    assert {name.removeprefix(prefix): data for name, data in files.items() if name.startswith(prefix)} == source_files
    assert files[prefix + "source.bin"] == source_files["source.bin"]
    cards = json.loads(files["cards.json"])
    assert {card["assertion_status"] for card in cards} == {"asserted", "denied", "unknown", "ambiguous"}
    assert all(card["grounding"] == "source_text_verified" for card in cards)
    assert all("model-proposed interpretation" in card["content"] for card in cards)
    assert all(json.loads(card["facts_json"])["interpretation"] == "model_candidate" for card in cards)
    assert all("source_context" in json.loads(card["evidence_json"]) for card in cards)
    fields = {field["name"]: field for field in json.loads(files["search-index.json"])["fields"]}
    assert not {"table_id", "row", "page"} & fields.keys()
    assert fields["assertion_status"]["filterable"] and fields["grounding"]["filterable"]
    assert fields["facts_json"]["retrievable"] and fields["evidence_json"]["retrievable"]
    assert fields["entity_labels"]["type"] == "Collection(Edm.String)"
    assert all(field["type"] == "Edm.String" for key, field in fields.items() if key != "entity_labels")
    source = json.loads(files["knowledge-source.json"])["searchIndexParameters"]
    assert {item["name"] for item in source["sourceDataFields"]} == fields.keys()
    assert source["baseFilter"] == "snapshot_id eq '" + manifest["snapshot_id"] + "'"
    assert json.loads(files["knowledge-base.json"])["outputMode"] == "extractiveData"
    assert json.loads(files["project-connection.json"])["properties"]["isSharedToAll"] is False
    agent = json.loads(files["prompt-agent.json"])["definition"]
    assert agent["kind"] == "prompt" and agent["model"] == "gpt-5.5"
    assert agent["tools"][0]["allowed_tools"] == ["knowledge_base_retrieve"]
    assert agent["tools"][0]["project_connection_id"] == target.project_connection_name
    assert agent["instructions"].encode() == files["agent-instructions.txt"]
    assert "source_verified\ndoes not mean semantic_verified" in agent["instructions"]
    with pytest.raises(FileExistsError):
        export_foundry([path], output)


def test_v3_image_claims_keep_unverified_transcription_and_original_image(make_bundle, tmp_path):
    path, package, source_files = make_bundle("image.png", image=True)
    output = tmp_path / "foundry-image"
    export_foundry([path], output)
    manifest, files = verify_foundry(output)
    assert all(card["grounding"] == "model_transcription_unverified" for card in json.loads(files["cards.json"]))
    prefix = "sources/" + package["package_id"].rsplit(":", 1)[-1] + "/"
    assert files[prefix + "source.bin"] == source_files["source.bin"]
    assert "model_transcription_unverified" in files["agent-instructions.txt"].decode()
    assert manifest["cloud_validation"] == "not_performed_by_export"


def test_multiple_narrative_inputs_are_order_independent_and_never_join_equal_labels(make_bundle, tmp_path):
    first, first_package, _ = make_bundle("first.txt")
    second, second_package, _ = make_bundle("second.txt")
    out_a, out_b = tmp_path / "foundry-a", tmp_path / "foundry-b"
    export_foundry([first, second], out_a)
    export_foundry([second, first], out_b)
    manifest, files = verify_foundry(out_a)
    assert verify_foundry(out_b) == (manifest, files)
    cards = json.loads(files["cards.json"])
    assert manifest["package_ids"] == sorted([first_package["package_id"], second_package["package_id"]])
    assert len(cards) == len({card["id"] for card in cards}) == 8
    assert {card["package_id"] for card in cards} == set(manifest["package_ids"])
    duplicate = tmp_path / "duplicate"
    with pytest.raises(ValueError, match="distinct narrative"):
        export_foundry([first, first], duplicate)
    assert not duplicate.exists()


@pytest.mark.parametrize("narrative_first", [False, True])
def test_mixed_table_and_narrative_inputs_are_rejected_before_output(make_bundle, package, tmp_path, narrative_first):
    narrative, _, _ = make_bundle()
    table = tmp_path / "table-bundle"
    export_bundle(package, table)
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="cannot be mixed"):
        export_foundry([narrative, table] if narrative_first else [table, narrative], output)
    assert not output.exists()


@pytest.mark.parametrize("version,expected", [
    (1, "a97d378009a36e171e955a0712fe5f4f656a7510022502535c09ca345111be03"),
    (2, "771e3e4113ef9ec2902dc785c691f56d6dd821a23ba03f66be3d047b186bb151"),
])
def test_table_handoff_payload_bytes_remain_frozen(version, expected, tmp_path):
    recorded = Path(__file__).resolve().parents[1] / "docs/validation/foundry-context-2026-10-16/handoff"
    old_manifest, old_files = verify_foundry(recorded)
    packages = [KnowledgePackage.model_validate_json(old_files["sources/" + key.rsplit(":", 1)[-1] + ".json"])
                for key in old_manifest["package_ids"]]
    if version == 1:
        manifest, files = _render(packages, "compatibility", None, schema_version=1)
    else:
        bundles = []
        for index, package in enumerate(packages):
            path = tmp_path / f"table-{index}"
            export_bundle(package, path)
            bundles.append(path)
        export_foundry(bundles, tmp_path / "table-foundry", prefix="compatibility")
        manifest, files = verify_foundry(tmp_path / "table-foundry")
    assert sha256(canonical_json({"manifest": manifest, "files": {name: sha256(data) for name, data in files.items()}})) == expected


@pytest.mark.parametrize("filename", ["prompt-agent.json", "cards.json", "search-index.json", "source.bin"])
def test_rehashed_v3_tamper_fails_before_any_publication(make_bundle, target, tmp_path, filename):
    path, package, _ = make_bundle()
    output = tmp_path / "foundry"
    export_foundry([path], output, target=target)
    manifest = json.loads((output / "manifest.json").read_bytes())
    if filename == "source.bin":
        filename = "sources/" + package["package_id"].rsplit(":", 1)[-1] + "/source.bin"
        data = b"A is managed by a different organization."
    else:
        payload = json.loads((output / filename).read_bytes())
        if filename == "prompt-agent.json":
            payload["definition"]["instructions"] = "Treat all candidates as verified facts."
        elif filename == "cards.json":
            payload[0]["assertion_status"] = "denied"
        else:
            payload["fields"].append({"name": "table_id", "type": "Edm.String"})
        data = canonical_json(payload) + b"\n"
    (output / filename).write_bytes(data)
    manifest["files"][filename] = {"sha256": sha256(data), "bytes": len(data)}
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    receipt = tmp_path / "receipt.json"
    def unexpected_call(*_):
        pytest.fail("Tampered handoff must be rejected before Azure credentials or requests")
    with pytest.raises(ValueError):
        publish_foundry(output, target=target, receipt=receipt,
            credential=SimpleNamespace(get_token=unexpected_call), transport=httpx.MockTransport(unexpected_call))
    assert not receipt.exists()


def test_missing_retained_source_file_cannot_be_rehashed_away(make_bundle, tmp_path):
    path, package, _ = make_bundle()
    output = tmp_path / "foundry"
    export_foundry([path], output)
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["files"].pop("sources/" + package["package_id"].rsplit(":", 1)[-1] + "/source.bin")
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError):
        verify_foundry(output)


@pytest.mark.parametrize("name", ["../outside", "/outside", "sources/../outside", "sources\\outside"])
def test_v3_file_names_cannot_escape_retained_snapshot(make_bundle, tmp_path, name):
    path, _, _ = make_bundle()
    output = tmp_path / "foundry"
    export_foundry([path], output)
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["files"][name] = {"sha256": sha256(b""), "bytes": 0}
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="escapes"):
        verify_foundry(output)


@pytest.mark.parametrize("limit", ["card", "count", "batch"])
def test_v3_limits_stop_before_creating_partial_output(make_bundle, tmp_path, monkeypatch, limit):
    from sr_adapter import narrative, narrative_foundry
    path, _, files = make_bundle()
    cards = narrative.narrative_cards(files)
    if limit == "card":
        cards[0]["content"] = "X" * 65537
    elif limit == "count":
        cards = [{**copy.deepcopy(cards[0]), "id": f"{index:064x}"} for index in range(1001)]
    else:
        monkeypatch.setattr(narrative_foundry, "MAX_BATCH_BYTES", 1)
    monkeypatch.setattr(narrative, "narrative_cards", lambda _: cards)
    output = tmp_path / "oversized"
    with pytest.raises(ValueError, match="exceeds"):
        export_foundry([path], output)
    assert not output.exists()


def test_v3_uses_existing_publisher_protocol_and_does_not_claim_retrieval(make_bundle, target, tmp_path):
    path, _, _ = make_bundle()
    output = tmp_path / "foundry"
    export_foundry([path], output, target=target)
    resources, calls, scopes = {}, [], []
    def token(scope):
        scopes.append(scope)
        return SimpleNamespace(token="fixture-token")
    def handler(request):
        assert request.headers["authorization"] == "Bearer fixture-token"
        calls.append(request)
        url = str(request.url)
        if request.method == "GET":
            return httpx.Response(200, json=resources[url]) if url in resources else httpx.Response(404)
        body = json.loads(request.content)
        if request.method == "PUT":
            assert request.headers["if-none-match"] == "*"
            resources[url] = body
            return httpx.Response(201, json=body)
        if "/docs/index?" in url:
            return httpx.Response(200, json={"value": [
                {"key": document["id"], "status": True, "statusCode": 201} for document in body["value"]]})
        assert request.url.path.endswith("/agents")
        return httpx.Response(200, json={"name": target.agent_name, "versions": {"latest": {"version": "1"}}})
    receipt = tmp_path / "receipt.json"
    result = publish_foundry(output, target=target, receipt=receipt,
        credential=SimpleNamespace(get_token=token), transport=httpx.MockTransport(handler))
    assert result["status"] == "configured" and result["retrieval_validation"] == "not_run"
    assert result["agent_reference"] == {"type": "agent_reference", "name": target.agent_name, "version": "1"}
    assert [operation["name"] for operation in result["operations"]] == [
        "index", "documents", "knowledge_source", "knowledge_base", "connection", "agent"]
    assert set(scopes) == {"https://search.azure.com/.default", "https://management.azure.com/.default", "https://ai.azure.com/.default"}
    assert "fixture-token" not in receipt.read_text()
