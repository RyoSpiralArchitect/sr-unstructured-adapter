from __future__ import annotations

import json

from sr_adapter.delegate import escalate_low_conf, select_escalation_indices
from sr_adapter.drivers.base import LLMDriver
from sr_adapter.recipe import RecipeConfig
from sr_adapter.schema import Block
from sr_adapter.escalation import (
    SelectionCandidate,
    SelectionResult,
    reset_escalation_policy,
)
from sr_adapter import settings as adapter_settings


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
    reset_escalation_policy()
    dummy_driver = _DummyDriver()
    dummy_manager = _DummyDriverManager(dummy_driver)
    monkeypatch.setattr("sr_adapter.delegate._driver_manager", dummy_manager)
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(True))
    blocks = [Block(text="First"), Block(text="Second")]

    escalated = escalate_low_conf(blocks, "test")

    assert "llm_escalations" in escalated[0].attrs
    payload = escalated[0].attrs["llm_escalations"][0]
    assert payload["provider"] == "dummy"
    choice_text = payload["choices"][0]["text"]
    assert choice_text.startswith("Reviewed:")
    assert "First" in choice_text and "Second" in choice_text
    assert payload["tenant"] == "default"
    assert payload["target_index"] == 0
    meta = escalated[0].attrs["llm_meta"]
    assert meta["escalation_rank"] in {1, 2}
    assert meta["escalation_score"] >= 0.5
    assert sorted(dummy_driver.last_metadata["indices"]) == [0, 1]


def test_escalate_low_conf_returns_original_when_disabled(monkeypatch):
    reset_escalation_policy()
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(False))
    blocks = [Block(text="First"), Block(text="Second")]
    assert escalate_low_conf(blocks, "test") == blocks


def test_select_escalation_indices_prioritises_low_confidence():
    reset_escalation_policy()
    blocks = [
        Block(text="high", confidence=0.9),
        Block(text="mid", confidence=0.5),
        Block(text="low", confidence=0.2),
    ]

    indices = select_escalation_indices(blocks, max_confidence=0.95)
    assert indices == [2, 1]


def test_escalate_low_conf_respects_filters(monkeypatch):
    reset_escalation_policy()
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


def test_select_escalation_indices_supports_custom_model(tmp_path, monkeypatch):
    reset_escalation_policy()
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "weights": {"confidence": 4.0},
                "bias": -1.5,
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SR_ADAPTER_ESCALATION__MODEL_PATH", str(model_path))
    adapter_settings.get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_escalation_policy()

    blocks = [
        Block(text="high", confidence=0.9),
        Block(text="low", confidence=0.1),
    ]

    indices = select_escalation_indices(blocks)
    assert indices == [0]

    adapter_settings.get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_escalation_policy()


def test_escalate_low_conf_reuses_preselection(monkeypatch):
    reset_escalation_policy()
    dummy_driver = _DummyDriver()
    dummy_manager = _DummyDriverManager(dummy_driver)
    monkeypatch.setattr("sr_adapter.delegate._driver_manager", dummy_manager)
    monkeypatch.setattr("sr_adapter.delegate.load_recipe", lambda _: _recipe(True))

    selection = SelectionResult(
        indices=[1],
        candidates=[
            SelectionCandidate(index=1, score=0.9, features={}, selected=True, rank=1),
            SelectionCandidate(index=0, score=0.2, features={}, selected=False, rank=None),
        ],
        threshold=0.5,
        limit=1,
    )

    def _fail(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("select_escalation_indices should not be invoked when selection is provided")

    monkeypatch.setattr("sr_adapter.delegate.select_escalation_indices", _fail)

    blocks = [
        Block(text="keep", confidence=0.95),
        Block(text="fix", confidence=0.1),
    ]

    escalated = escalate_low_conf(blocks, "test", selection=selection)

    assert escalated[0].attrs.get("llm_escalations") is None
    payload = escalated[1].attrs["llm_escalations"][0]
    assert payload["target_index"] == 1
    assert dummy_driver.last_metadata["indices"] == [1]
