from __future__ import annotations

from sr_adapter.delegate import escalate_low_conf
from sr_adapter.drivers.base import LLMDriver
from sr_adapter.recipe import RecipeConfig
from sr_adapter.schema import Block


class _DummyDriver(LLMDriver):
    def __init__(self) -> None:
        super().__init__("dummy", {})

    def generate(self, prompt: str, *, metadata=None):
        return {
            "model": "dummy-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"Reviewed: {prompt}"},
                }
            ],
            "usage": {"total_tokens": 32},
        }


class _DummyTenantManager:
    def get_default_tenant(self) -> str:
        return "default"


class _DummyDriverManager:
    def __init__(self, driver: LLMDriver) -> None:
        self._driver = driver
        self.tenant_manager = _DummyTenantManager()

    def get_driver(self, tenant: str, llm_config):
        assert tenant == "default"
        return self._driver


def _recipe(enable: bool = True) -> RecipeConfig:
    return RecipeConfig(
        name="test",
        patterns=[],
        fallback_type=None,
        fallback_confidence=None,
        fallback_attrs={},
        llm={"enable": enable, "driver": "dummy"},
    )


def test_escalate_low_conf_attaches_llm_payload(monkeypatch):
    dummy_driver = _DummyDriver()
    dummy_manager = _DummyDriverManager(dummy_driver)
    monkeypatch.setattr("sr_adapter.delegate._driver_manager", dummy_manager)
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(True))
    blocks = [Block(text="First"), Block(text="Second")]

    escalated = escalate_low_conf(blocks, "test")

    assert escalated != blocks
    payload = escalated[0].attrs["llm_escalations"][0]
    assert payload["provider"] == "dummy"
    assert payload["choices"][0]["text"].startswith("Reviewed: First\n\nSecond")
    assert payload["tenant"] == "default"


def test_escalate_low_conf_returns_original_when_disabled(monkeypatch):
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(False))
    blocks = [Block(text="First"), Block(text="Second")]
    assert escalate_low_conf(blocks, "test") == blocks
