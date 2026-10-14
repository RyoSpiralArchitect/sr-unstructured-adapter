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
    assert result.status == "blocked"
    assert not result.is_complete


@pytest.mark.parametrize("reason,text,extra,status", [
    ("stop", "answer", {}, "complete"),
    ("length", '{"looks":"valid"}', {}, "truncated"),
    ("content_filter", "partial", {}, "blocked"),
    ("stop", "partial", {"refusal": "Declined"}, "blocked"),
    ("stop", "", {}, "empty"),
    ("stop", " \n\t", {}, "empty"),
    (None, "answer", {}, "unknown"),
    ("new-provider-reason", "answer", {}, "unknown"),
    ("tool_calls", "", {}, "tool_call"),
    ("stop", "preface", {"tool_calls": [{"id": "call"}]}, "tool_call"),
])
def test_openai_completion_status(reason, text, extra, status):
    raw = {"choices": [{"message": {"content": text, **extra}, "finish_reason": reason}]}
    result = LLMNormalizer().normalize("compatible-alias", raw, prompt="test")
    assert result.status == result.choices[0].status == status
    assert result.is_complete is (status == "complete")
    assert result.raw is raw
    assert result.choices[0].text == text


@pytest.mark.parametrize("reason,status", [
    ("end_turn", "complete"), ("stop_sequence", "complete"),
    ("max_tokens", "truncated"), ("model_context_window_exceeded", "truncated"),
    ("refusal", "blocked"), ("tool_use", "tool_call"), ("pause_turn", "unknown"),
])
def test_anthropic_completion_status(reason, status):
    result = LLMNormalizer().normalize("claude", {
        "type": "message", "content": [{"type": "text", "text": "answer"}], "stop_reason": reason,
    }, prompt="test")
    assert result.status == status


@pytest.mark.parametrize("reason,status", [
    ("STOP", "complete"), ("MAX_TOKENS", "truncated"), ("SAFETY", "blocked"),
    ("RECITATION", "blocked"), ("SPII", "blocked"), ("OTHER", "unknown"),
    ("MALFORMED_FUNCTION_CALL", "unknown"), (None, "unknown"),
])
def test_gemini_completion_status(reason, status):
    result = LLMNormalizer().normalize("google", {
        "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": reason}],
    }, prompt="test")
    assert result.status == status


def test_gemini_function_call_is_not_complete_text():
    result = LLMNormalizer().normalize("google", {
        "candidates": [{"content": {"parts": [{"text": "preface"}, {"functionCall": {"name": "lookup", "args": {}}}]}, "finishReason": "STOP"}],
    }, prompt="test")
    assert result.status == "tool_call"
    assert result.choices[0].metadata["tool_calls"][0]["name"] == "lookup"


def test_one_complete_choice_does_not_mask_an_incomplete_choice():
    result = LLMNormalizer().normalize("openai", {"choices": [
        {"message": {"content": "first"}, "finish_reason": "stop"},
        {"message": {"content": "second"}, "finish_reason": "length"},
    ]}, prompt="test")
    assert result.status == "mixed"
    assert not result.is_complete


@pytest.mark.parametrize("raw", [
    {"choices": {}}, {"choices": ""}, {"candidates": {}},
    {"choices": [{"message": {"content": "answer"}, "finish_reason": []}]},
    {"promptFeedback": []},
])
def test_malformed_completion_metadata_is_not_silently_accepted(raw):
    with pytest.raises(ValueError):
        LLMNormalizer().normalize("test", raw, prompt="test")
