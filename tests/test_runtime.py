import json

from sr_adapter.runtime import (
    NativeKernelRuntime,
    get_native_runtime,
    reset_native_runtime,
    runtime_status_json,
)
from sr_adapter.schema import Block, Provenance
from sr_adapter.visual import LayoutCandidate


def test_runtime_normalize_and_stats() -> None:
    runtime = NativeKernelRuntime()
    blocks = [
        Block(
            type="paragraph",
            text="Hello\n\nWorld",
            prov=Provenance(uri="test"),
            confidence=0.4,
        ),
        Block(
            type="heading",
            text="TITLE",
            prov=Provenance(uri="test"),
            confidence=0.9,
        ),
    ]
    assert {block.source for block in blocks} == {"test"}

    normalized = runtime.normalize(blocks)
    assert "\n\n" not in normalized[0].text
    snapshot = runtime.snapshot()
    assert snapshot.text_stats.calls >= 1
    assert snapshot.text_stats.total_units >= len("Hello\n\nWorld")


def test_runtime_layout_analysis_records_stats() -> None:
    runtime = NativeKernelRuntime()
    if not runtime.layout_enabled:
        return
    block = Block(
        type="paragraph",
        text="body",
        prov=Provenance(uri="test"),
        confidence=0.5,
    )
    assert block.source == "test"
    candidate = LayoutCandidate(block=block, bbox=(0, 0, 10, 10), page=0, score=0.5, order_hint=0)
    segments = list(runtime.analyze([candidate]))
    snapshot = runtime.snapshot()
    assert snapshot.layout_stats.calls >= 1
    assert snapshot.layout_stats.total_units >= len(segments)


def test_runtime_status_json_round_trip() -> None:
    reset_native_runtime()
    runtime = get_native_runtime()
    assert runtime is not None
    data = json.loads(runtime_status_json())
    assert set(data.keys()) == {"text", "layout"}
