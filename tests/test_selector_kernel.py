import os
import random

import pytest

from sr_adapter.native.selector import find_text_occurrence, SelectorKernelError
from sr_adapter.relation_semantics import resolve_text_selector


def test_native_byte_budget_checked_before_source_sized_allocations():
    from sr_adapter.native.selector import _bounded_utf8

    class NoFullEncoding(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("Must reject before full encoding")

    for text in (NoFullEncoding("a" * 65), NoFullEncoding("あ" * 30)):
        with pytest.raises(SelectorKernelError, match="byte budget"):
            _bounded_utf8(text, 64)
    assert _bounded_utf8("あ" * 21, 64) == ("あ" * 21).encode("utf-8")
    assert _bounded_utf8("a" * 64, 64) == b"a" * 64


def test_fallback_keeps_overlaps_codepoints_nuls_and_non_normalization():
    assert find_text_occurrence("あ\x00あ\x00あ", "あ\x00あ", 1, backend="python") == 2
    assert find_text_occurrence("e\u0301", "é", 0, backend="python") == -1
    assert find_text_occurrence("\ud800\ud800", "\ud800", 1, backend="python") == 1
    with pytest.raises(ValueError):
        find_text_occurrence("x", "", 0)
    with pytest.raises(ValueError):
        find_text_occurrence("x", "x", True)


def test_auto_falls_back_when_native_unavailable(monkeypatch):
    import sr_adapter.native.selector as selector

    def unavailable():
        raise SelectorKernelError("unavailable")

    monkeypatch.setattr(selector, "_native", unavailable)
    text = "あ" * 20000
    assert find_text_occurrence(text, "あ" * 2048, 1000) == 1000
    assert (
        resolve_text_selector(
            text, {"kind": "text", "exact": "あ" * 2048, "occurrence": 1000}
        )["start"]
        == 1000
    )
    monkeypatch.setenv("SR_ADAPTER_DISABLE_NATIVE_RUNTIME", "1")
    assert find_text_occurrence(text, "あ" * 2048, 1000) == 1000
    with pytest.raises(SelectorKernelError):
        find_text_occurrence(text, "あ", 0, backend="native")


@pytest.mark.skipif(
    os.environ.get("SR_ADAPTER_DISABLE_NATIVE_RUNTIME") == "1",
    reason="Native disabled by host",
)
def test_native_matches_python_overlapping_unicode_and_random_inputs():
    rng = random.Random(72153)
    cases = [
        ("あ" * 200000, "あ" * 2048, 50000),
        ("🙂a\x00" * 20000, "🙂a\x00" * 32, 16000),
        ("aaa", "aa", 2),
        ("abc", "bc", 0),
        ("", "a", 0),
    ]
    alphabet = ["a", "b", "\x00", "あ", "🙂", "\u0301", "\u3000", "é"]
    for _ in range(1500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(100)))
        start = rng.randrange(len(text) + 1)
        needle = text[start : start + rng.randrange(1, 10)] or "a"
        cases.append((text, needle, rng.randrange(4)))
    for text, needle, occurrence in cases:
        assert find_text_occurrence(
            text, needle, occurrence, backend="native"
        ) == find_text_occurrence(text, needle, occurrence, backend="python")
