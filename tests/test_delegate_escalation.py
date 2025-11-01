from __future__ import annotations

from sr_adapter.delegate import escalate_low_conf, select_escalation_indices
from sr_adapter.drivers.base import LLMDriver
from sr_adapter.recipe import RecipeConfig
from sr_adapter.schema import Block


class _DummyDriver(LLMDriver):
    def __init__(self) -> None:
        super().__init__("dummy", {})
        self.last_metadata = None

    def generate(self, prompt: str, *, metadata=None):
        self.last_metadata = metadata or {}
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
    assert payload["target_index"] == 0
    assert dummy_driver.last_metadata["indices"] == [0, 1]


def test_escalate_low_conf_returns_original_when_disabled(monkeypatch):
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(False))
    blocks = [Block(text="First"), Block(text="Second")]
    assert escalate_low_conf(blocks, "test") == blocks


def test_select_escalation_indices_filters_by_confidence():
    blocks = [
        Block(text="high", confidence=0.9),
        Block(text="mid", confidence=0.5),
        Block(text="low", confidence=0.2),
    ]

    indices = select_escalation_indices(blocks, max_confidence=0.5)
    assert indices == [1, 2]


def test_escalate_low_conf_respects_filters(monkeypatch):
    dummy_driver = _DummyDriver()
    dummy_manager = _DummyDriverManager(dummy_driver)
    monkeypatch.setattr("sr_adapter.delegate._driver_manager", dummy_manager)
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(True))
    blocks = [
        Block(text="keep", type="paragraph", confidence=0.9),
        Block(text="fix", type="paragraph", confidence=0.2),
        Block(text="ignore", type="table", confidence=0.1),
    ]

    escalated = escalate_low_conf(
        blocks,
        "test",
        max_confidence=0.4,
        allow_types=("paragraph",),
        limit=1,
    )

    assert "llm_escalations" not in escalated[0].attrs
    assert escalated[1].attrs["llm_escalations"][0]["target_index"] == 1
    assert "llm_escalations" not in escalated[2].attrs
    assert dummy_driver.last_metadata["indices"] == [1]
