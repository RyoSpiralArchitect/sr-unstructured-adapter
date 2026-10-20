"""Synthetic source replay and identity/authorization boundaries for IQ blueprints."""
import base64
import json

from PIL import Image
import pytest

from sr_adapter.fabric_iq import FabricIQRetrievalModel, export_fabric_iq, verify_fabric_iq
from sr_adapter.fabric_visual import export_visual
from sr_adapter.foundry import FoundryTarget
from sr_adapter.knowledge import sha256
from sr_adapter.tables import canonical_json
from sr_adapter.visual_relations import verify_visual
from test_visual_relations import TARGET, response, run

FOUNDRY = FoundryTarget(search_endpoint="https://fixture.search.windows.net",
    project_endpoint="https://fixture.services.ai.azure.com/api/projects/demo",
    project_resource_id="/subscriptions/fixture/resourceGroups/demo/providers/Microsoft.CognitiveServices/accounts/fixture/projects/demo",
    project_connection_name="fabric-read-only", model_deployment="answer-deployment", agent_name="fabric-demo")
MODEL = FabricIQRetrievalModel(resource_uri="https://fixture.openai.azure.com",
                               deployment_id="planning-deployment", model_name="gpt-5.4-mini")
AGENT_ID = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def visual_export(tmp_path):
    source = tmp_path / "private-source.png"
    Image.new("RGB", (100, 100), "white").save(source)
    run(source, tmp_path / "candidates")
    package, _ = verify_visual(tmp_path / "candidates")
    review = {"version": 1, "reviewer": "synthetic-reviewer", "findings": [{
        "package_id": package["package_id"], "reference_kind": "item", "reference_id": "n1",
        "finding": "text_mismatch", "detail": "PRIVATE REVIEW TEXT MUST NOT BE COPIED INTO THE BLUEPRINT"}]}
    folder = tmp_path / "fabric"
    export_visual([tmp_path / "candidates"], folder, target=TARGET, review=review)
    return folder


def kwargs(visual_export, **extra):
    return {"fabric_export": visual_export, "fabric_target": TARGET, "foundry_target": FOUNDRY,
            "data_agent_id": AGENT_ID, "retrieval_model": MODEL, **extra}


def test_exact_snapshot_allowlist_and_candidate_semantics(visual_export, tmp_path):
    folder = tmp_path / "iq"
    manifest = export_fabric_iq(destination=folder, **kwargs(visual_export))
    replay, files = verify_fabric_iq(folder, **kwargs(visual_export))
    assert replay == manifest and replay["status"] == "prepared_only" and replay["configured"] is True
    binding = json.loads(files["source-binding.json"])
    expected_kinds = {"images", "items", "relations", "evidence", "limitations", "reviews"}
    assert {t["kind"] for t in binding["tables"]} == expected_kinds
    assert all(t["table_name"] == f"sr_{manifest['snapshot_id']}_{t['kind']}" for t in binding["tables"])
    parts = {p["path"]: json.loads(base64.b64decode(p["payload"])) for p in json.loads(files["fabric-data-agent-definition.json"])["parts"]}
    assert not any("/published/" in name or "publish_info" in name for name in parts)
    datasource = next(v for k, v in parts.items() if k.endswith("/datasource.json"))
    assert all(v["$schema"].startswith("https://developer.microsoft.com/json-schemas/fabric/item/dataAgent/definition/") for v in parts.values())
    tables = datasource["elements"][0]["children"]
    assert {t["display_name"] for t in tables} == {t["table_name"] for t in binding["tables"]}
    assert len(tables) == 6 and all(t["is_selected"] for t in tables)
    assert all("id" not in table and all("id" not in column for column in table["children"]) for table in tables)
    assert "Never join on labels" in datasource["dataSourceInstructions"]
    assert "model_candidate_needs_review" in datasource["dataSourceInstructions"]
    assert "unreviewed_model_candidate remains unverified" in datasource["dataSourceInstructions"]
    assert "共通システム" not in str(parts) and "PRIVATE REVIEW TEXT" not in str(parts)
    assert "private-source.png" not in b"".join(files.values()).decode()
    second = tmp_path / "same"
    assert export_fabric_iq(destination=second, **kwargs(visual_export)) == manifest
    assert all((folder / name).read_bytes() == (second / name).read_bytes() for name in files)
    with pytest.raises(FileExistsError):
        export_fabric_iq(destination=folder, **kwargs(visual_export))


def test_fabric_knowledge_source_and_obo_not_service_identity_substitution(visual_export, tmp_path):
    folder = tmp_path / "iq"
    export_fabric_iq(destination=folder, **kwargs(visual_export))
    _, files = verify_fabric_iq(folder, **kwargs(visual_export))
    source = json.loads(files["knowledge-source.json"])
    assert source["kind"] == "fabricDataAgent"
    assert source["fabricDataAgentParameters"] == {"workspaceId": TARGET.workspace_id, "dataAgentId": AGENT_ID}
    kb = json.loads(files["knowledge-base.json"])
    assert kb["retrievalReasoningEffort"] == {"kind": "low"}
    assert kb["models"][0]["azureOpenAIParameters"]["modelName"] == MODEL.model_name
    agent = json.loads(files["prompt-agent.json"])["definition"]
    assert agent["model"] == "answer-deployment"
    assert agent["structured_inputs"]["search_auth_token"]["required"] is True
    assert agent["tools"][0]["headers"] == {"x-ms-query-source-authorization": "{{search_auth_token}}"}
    assert agent["tools"][0]["allowed_tools"] == ["knowledge_base_retrieve"]
    retrieval = json.loads(files["retrieval-request.json"])
    assert retrieval["knowledgeSourceParams"][0]["includeReferenceSourceData"] is True
    plan = json.loads(files["deployment-plan.json"])
    assert plan["authentication"]["obo_required"] is True
    assert plan["authentication"]["service_authentication_also_required"] is True
    assert plan["authentication"]["user_token_scope"] == "https://search.azure.com/.default"
    assert plan["execution_supported"] is False and plan["cloud_validation"] == "not_performed"
    assert {op["name"] for op in plan["operations"]} == {"knowledge_source", "knowledge_base", "connection", "agent"}


@pytest.mark.parametrize("missing", [{"data_agent_id": None}, {"retrieval_model": None},
                                      {"data_agent_id": None, "retrieval_model": None}])
def test_missing_cloud_inputs_are_unconfigured_not_fabricated(visual_export, tmp_path, missing):
    folder = tmp_path / "iq"
    manifest = export_fabric_iq(destination=folder, **kwargs(visual_export, **missing))
    _, files = verify_fabric_iq(folder, **kwargs(visual_export, **missing))
    assert manifest["configured"] is False
    plan = json.loads(files["deployment-plan.json"])
    assert plan["operations"] == [] and plan["missing_configuration"]
    if "data_agent_id" in missing:
        assert json.loads(files["knowledge-source.json"])["fabricDataAgentParameters"]["dataAgentId"] is None


@pytest.mark.parametrize("invalid", ["", "not-a-guid", "../../agent", "33333333-3333-3333-3333-333333333333?key=secret",
                                     "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", "00000000-0000-0000-0000-000000000000", 1])
def test_malformed_data_agent_ids_rejected_before_output(visual_export, tmp_path, invalid):
    with pytest.raises(ValueError, match="canonical"):
        export_fabric_iq(destination=tmp_path / "bad", **kwargs(visual_export, data_agent_id=invalid))
    assert not (tmp_path / "bad").exists()


def test_target_and_snapshot_binding_cannot_be_changed(visual_export, tmp_path):
    folder = tmp_path / "iq"
    export_fabric_iq(destination=folder, **kwargs(visual_export))
    with pytest.raises(ValueError):
        verify_fabric_iq(folder, **kwargs(visual_export, fabric_target=TARGET.model_copy(update={"lakehouse_name": "other"})))
    with pytest.raises(ValueError):
        verify_fabric_iq(folder, **kwargs(visual_export, foundry_target=FOUNDRY.model_copy(update={"agent_name": "other-agent"})))
    with pytest.raises(ValueError):
        verify_fabric_iq(folder, **kwargs(visual_export, data_agent_id="44444444-4444-4444-4444-444444444444"))
    source_manifest = json.loads((visual_export / "manifest.json").read_bytes())
    source_manifest["snapshot_id"] = "a" * 64
    (visual_export / "manifest.json").write_bytes(canonical_json(source_manifest))
    with pytest.raises(ValueError):
        verify_fabric_iq(folder, **kwargs(visual_export))


@pytest.mark.parametrize("filename,mutate", [
    ("prompt-agent.json", lambda x: x["definition"]["structured_inputs"]["search_auth_token"].update(required=False)),
    ("knowledge-source.json", lambda x: x.update(kind="searchIndex")),
    ("knowledge-base.json", lambda x: x.update(retrievalReasoningEffort={"kind": "minimal"})),
    ("source-binding.json", lambda x: x["tables"][0].update(table_name="other_snapshot_table")),
])
def test_rehashed_tampering_cannot_weaken_contract(visual_export, tmp_path, filename, mutate):
    folder = tmp_path / "iq"
    manifest = export_fabric_iq(destination=folder, **kwargs(visual_export))
    value = json.loads((folder / filename).read_bytes())
    mutate(value)
    data = canonical_json(value)
    (folder / filename).write_bytes(data)
    manifest["files"][filename] = {"bytes": len(data), "sha256": sha256(data)}
    (folder / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="differs"):
        verify_fabric_iq(folder, **kwargs(visual_export))


def test_no_secret_inputs_or_environment_capture(visual_export, tmp_path, monkeypatch, capsys):
    secret = "fixture-token-never-export"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)
    folder = tmp_path / "iq"
    export_fabric_iq(destination=folder, **kwargs(visual_export))
    assert secret not in b"".join(p.read_bytes() for p in folder.iterdir()).decode()
    assert capsys.readouterr().out == "" and capsys.readouterr().err == ""
    with pytest.raises(ValueError):
        FabricIQRetrievalModel(**MODEL.model_dump(), api_key=secret)
    with pytest.raises(ValueError):
        FabricIQRetrievalModel(resource_uri="https://fixture.openai.azure.com/?key=" + secret,
                               deployment_id="model", model_name="model")
    with pytest.raises(TypeError):
        export_fabric_iq(destination=tmp_path / "other", **kwargs(visual_export), search_auth_token=secret)


def test_empty_tables_are_not_added_to_fabric_data_agent(tmp_path):
    source = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), "white").save(source)
    run(source, tmp_path / "empty", responses=[response({"observations": [], "limitations": []}),
                                               response({"relations": [], "limitations": []})])
    fabric = tmp_path / "fabric"
    export_visual([tmp_path / "empty"], fabric, target=TARGET)
    export_fabric_iq(destination=tmp_path / "iq", **kwargs(fabric))
    _, files = verify_fabric_iq(tmp_path / "iq", **kwargs(fabric))
    assert [t["kind"] for t in json.loads(files["source-binding.json"])["tables"]] == ["images"]
