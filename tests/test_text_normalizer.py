import pytest

from sr_adapter.normalize import (
    NativeTextNormalizer,
    _normalize_block_py,
    _get_native_normalizer,
    normalize_block,
    normalize_blocks,
)
from sr_adapter.schema import Block


@pytest.fixture(autouse=True)
def enable_text_kernel(monkeypatch):
    monkeypatch.delenv("SR_ADAPTER_DISABLE_TEXT_KERNEL", raising=False)


def require_native() -> NativeTextNormalizer:
    normalizer = _get_native_normalizer()
    if normalizer is None:
        pytest.skip("native text kernel unavailable")
    assert isinstance(normalizer, NativeTextNormalizer)
    return normalizer


def test_native_normalizer_matches_python_semantics():
    require_native()
    block = Block(type="paragraph", text="\u2022  Foo\r\n\n\nBar", attrs={"text": "  Hello\r"}, confidence=0.8)
    native = normalize_block(block)
    python = _normalize_block_py(block)
    assert native.text == python.text
    assert native.type == python.type
    assert native.attrs == python.attrs
    assert native.confidence == pytest.approx(python.confidence)


def test_native_normalizer_handles_large_payload():
    require_native()
    blocks = [
        Block(type="paragraph", text=f"Item {i}: VALUE", confidence=0.6)
        for i in range(200)
    ]
    normalized = normalize_blocks(blocks)
    assert len(normalized) == len(blocks)
    kv_count = sum(1 for block in normalized if block.type == "kv")
    assert kv_count == len(blocks)
    assert all(block.confidence >= 0.6 for block in normalized)


def test_native_normalizer_normalizes_attrs():
    require_native()
    block = Block(
        type="paragraph",
        text="Heading",
        attrs={"text": "  Some Value\r\n", "other": "unchanged"},
        confidence=0.9,
    )
    normalized = normalize_block(block)
    assert normalized.attrs["text"] == "Some Value"
    assert normalized.attrs["other"] == "unchanged"
