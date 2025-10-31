import math
from pathlib import Path

from sr_adapter.native import LayoutBox, ensure_layout_kernel
from sr_adapter.schema import Block
from sr_adapter.visual import LayoutCandidate, VisualLayoutAnalyzer


def test_layout_kernel_orders_segments(tmp_path: Path) -> None:
    kernel = ensure_layout_kernel()
    boxes = [
        LayoutBox(x0=0.0, y0=120.0, x1=180.0, y1=180.0, score=0.4, page=0, order_hint=2),
        LayoutBox(x0=0.0, y0=12.0, x1=140.0, y1=44.0, score=0.8, page=0, order_hint=1),
    ]
    results = kernel.analyze(boxes, 0.35)
    assert [result.order for result in results] == [0, 1]
    assert results[0].index == 1
    assert results[0].label in {"heading", "paragraph", "table", "figure"}
    assert math.isclose(results[0].confidence, results[0].confidence, rel_tol=1e-6)


def test_visual_analyzer_calibrates_threshold() -> None:
    analyzer = VisualLayoutAnalyzer(initial_threshold=0.2, low_conf_cutoff=0.55, batch_size=2)
    base = Block(type="paragraph", text="First segment", confidence=0.3)
    cand1 = LayoutCandidate(
        block=base,
        bbox=(0.0, 40.0, 120.0, 60.0),
        page=0,
        score=0.25,
        order_hint=1,
    )
    cand2 = LayoutCandidate(
        block=Block(type="paragraph", text="Second", confidence=0.4),
        bbox=(0.0, 0.0, 120.0, 20.0),
        page=0,
        score=0.3,
        order_hint=0,
    )
    segments = list(analyzer.process([cand1, cand2]))
    assert len(segments) == 2
    assert analyzer.threshold > 0.2
    assert segments[0].block.attrs["layout_label"] in {"heading", "paragraph", "table", "figure"}
