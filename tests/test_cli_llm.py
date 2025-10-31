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
