from __future__ import annotations

import json

from sr_adapter.cli import main
from sr_adapter.drivers.base import LLMDriver
from sr_adapter.recipe import RecipeConfig


def test_cli_llm_list_tenants(monkeypatch, capsys):
    class _DummyTenantManager:
        def list_tenants(self):
            return ["alpha", "beta"]

    class _DummyManager:
        def __init__(self):
            self.tenant_manager = _DummyTenantManager()

    monkeypatch.setattr("sr_adapter.cli.DriverManager", lambda: _DummyManager())

    exit_code = main(["llm", "list-tenants"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines() == ["alpha", "beta"]


class _PromptDriver(LLMDriver):
    def __init__(self):
        super().__init__("dummy", {})

    def generate(self, prompt: str, *, metadata=None):
        assert prompt == "Hello there"
        assert metadata == {"trace": "id-123"}
        return {
            "model": "dummy-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"Echo: {prompt}"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }


def test_cli_llm_run_executes_driver(monkeypatch, capsys):
    driver = _PromptDriver()

    class _DummyTenantManager:
        def get_default_tenant(self):
            return "default"

        def list_tenants(self):
            return ["default"]

    class _DummyManager:
        def __init__(self):
            self.tenant_manager = _DummyTenantManager()

        def get_driver(self, tenant, llm_config):
            assert tenant == "default"
            assert llm_config["driver"] == "dummy"
            return driver

    recipe = RecipeConfig(
        name="demo",
        patterns=[],
        fallback_type=None,
        fallback_confidence=None,
        fallback_attrs={},
        llm={"enable": True, "driver": "dummy"},
    )

    monkeypatch.setattr("sr_adapter.cli.DriverManager", lambda: _DummyManager())
    monkeypatch.setattr("sr_adapter.cli.load_recipe", lambda _: recipe)

    exit_code = main(
        [
            "llm",
            "run",
            "--prompt",
            "Hello there",
            "--metadata",
            json.dumps({"trace": "id-123"}),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["provider"] == "dummy"
    assert payload["choices"][0]["text"].startswith("Echo: Hello there")
    assert payload["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}


def test_cli_llm_replay_executes_batch(monkeypatch, tmp_path):
    calls: list[tuple[str, dict | None]] = []

    class _BatchDriver(LLMDriver):
        def __init__(self):
            super().__init__("dummy", {})

        def generate(self, prompt: str, *, metadata=None):
            calls.append((prompt, metadata))
            return {
                "model": "dummy-model",
                "choices": [
                    {
                        "index": len(calls) - 1,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"Echo: {prompt}"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

    driver = _BatchDriver()

    class _DummyTenantManager:
        def get_default_tenant(self):
            return "default"

        def list_tenants(self):
            return ["default"]

    class _DummyManager:
        def __init__(self):
            self.tenant_manager = _DummyTenantManager()

        def get_driver(self, tenant, llm_config):
            assert tenant == "default"
            assert llm_config["driver"] == "dummy"
            return driver

    recipe = RecipeConfig(
        name="demo",
        patterns=[],
        fallback_type=None,
        fallback_confidence=None,
        fallback_attrs={},
        llm={"enable": True, "driver": "dummy"},
    )

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "text": "First prompt", "metadata": {"trace": "a"}}),
                json.dumps({"id": 2, "prompt": "Second prompt"}),
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output.jsonl"

    monkeypatch.setattr("sr_adapter.cli.DriverManager", lambda: _DummyManager())
    monkeypatch.setattr("sr_adapter.cli.load_recipe", lambda _: recipe)

    exit_code = main(
        [
            "llm",
            "replay",
            "--input",
            str(dataset),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls == [("First prompt", {"trace": "a"}), ("Second prompt", None)]

    contents = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2
    first = json.loads(contents[0])
    second = json.loads(contents[1])

    assert first["record"]["id"] == 1
    assert first["response"]["choices"][0]["text"].startswith("Echo: First prompt")
    assert first["driver"] == "dummy"
    assert second["record"]["id"] == 2
    assert second["response"]["choices"][0]["metadata"]["index"] == 1
    assert second["response"]["provider"] == "dummy"
