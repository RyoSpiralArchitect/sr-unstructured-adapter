from __future__ import annotations

import pytest

from sr_adapter.normalizer import LLMNormalizer


def test_anthropic_response_includes_text_and_usage():
    raw = {
        "id": "msg_123", "type": "message", "model": "claude", "role": "assistant",
        "content": [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "Hello "}, {"type": "text", "text": "猫"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 5, "output_tokens": 2},
    }
    result = LLMNormalizer().normalize("my-claude-alias", raw, prompt="Hi")
    assert result.choices[0].text == "Hello 猫"
    assert result.choices[0].finish_reason == "end_turn"
    assert result.usage["prompt_tokens"] == 8
    assert result.usage["total_tokens"] == 10
    assert result.model == "claude"
    assert result.raw is raw


def test_gemini_response_includes_text_and_usage():
    raw = {
        "modelVersion": "gemini-model",
        "candidates": [{"content": {"role": "model", "parts": [{"thought": True, "text": "thinking"}, {"text": "answer"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": 12, "thoughtsTokenCount": 3},
    }
    result = LLMNormalizer().normalize("google", raw, prompt="Hi")
    assert result.choices[0].text == "answer"
    assert result.choices[0].finish_reason == "STOP"
    assert result.model == "gemini-model"
    assert result.usage["total_tokens"] == 12
    assert result.usage["thoughtsTokenCount"] == 3


def test_openai_parts_and_refusal_are_preserved_with_null_usage():
    result = LLMNormalizer().normalize("openai", {
        "id": "completion-is-not-model",
        "choices": [{"message": {"content": [{"type": "text", "text": "answer"}], "refusal": "reason", "tool_calls": [{"id": "call"}]}, "finish_reason": "stop"}],
        "usage": None,
    }, prompt="Hi")
    assert result.model is None
    assert result.usage == {}
    assert result.choices[0].text == "answer"
    assert result.choices[0].metadata["refusal"] == "reason"
    assert result.choices[0].metadata["tool_calls"] == [{"id": "call"}]


@pytest.mark.parametrize("raw", [[], {"unknown": "response"}, {"error": {"message": "bad"}}, {"choices": ["invalid"]}])
def test_invalid_response_raises(raw):
    with pytest.raises(ValueError):
        LLMNormalizer().normalize("test", raw, prompt="Hi")


def test_blocked_gemini_does_not_invent_choices():
    result = LLMNormalizer().normalize("gemini", {"promptFeedback": {"blockReason": "SAFETY"}}, prompt="Hi")
    assert result.choices == []
