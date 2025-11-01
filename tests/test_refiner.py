from __future__ import annotations

from sr_adapter.refiner import HybridRefiner
from sr_adapter.schema import Block


def test_hybrid_refiner_enriches_low_confidence_text():
    refiner = HybridRefiner()
    block = Block(text="   HELLO   WORLD   ", confidence=0.2)

    refined = refiner.refine([block])
    assert refined[0].confidence >= block.confidence
    assert refined[0].text == "Hello World"
    assert refined[0].attrs["ml_refine"]["original_confidence"] == block.confidence
