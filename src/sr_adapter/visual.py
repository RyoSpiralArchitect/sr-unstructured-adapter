"""High-level visual layout analysis utilities backed by the native kernel."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence

from .native import LayoutBox, LayoutKernel, LayoutResult, ensure_layout_kernel
from .schema import BBox, Block, clone_model


@dataclass
class LayoutCandidate:
    """Input candidate sent to the native kernel."""

    block: Block
    bbox: Sequence[float] | None
    page: int
    score: float
    order_hint: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class LayoutSegment:
    """Post-processed block enhanced with layout metadata."""

    block: Block
    label: str
    confidence: float
    page: int
    order: int
    threshold: float


class VisualLayoutAnalyzer:
    """Batching helper that streams candidates through the native kernel."""

    def __init__(
        self,
        *,
        kernel: Optional[LayoutKernel] = None,
        initial_threshold: float = 0.35,
        low_conf_cutoff: float = 0.6,
        batch_size: int = 32,
    ) -> None:
        self.kernel = kernel or ensure_layout_kernel()
        self.threshold = float(initial_threshold)
        self.low_conf_cutoff = float(low_conf_cutoff)
        self.batch_size = max(1, int(batch_size))
        self._history: Deque[float] = deque(maxlen=128)

    def process(self, candidates: Iterable[LayoutCandidate]) -> Iterator[LayoutSegment]:
        buffer: List[LayoutCandidate] = []
        for candidate in candidates:
            buffer.append(candidate)
            if len(buffer) >= self.batch_size:
                yield from self._flush(buffer)
                buffer.clear()
        if buffer:
            yield from self._flush(buffer)

    def record_feedback(self, confidences: Iterable[float]) -> None:
        updated = self.kernel.calibrate(confidences, self.threshold)
        if updated > 0 and abs(updated - self.threshold) > 1e-3:
            self.threshold = updated

    def _flush(self, buffer: List[LayoutCandidate]) -> Iterator[LayoutSegment]:
        if not buffer:
            return iter(())
        boxes = [
            LayoutBox(
                x0=float(candidate.bbox[0]) if candidate.bbox else 0.0,
                y0=float(candidate.bbox[1]) if candidate.bbox else float(candidate.order_hint) * 10.0,
                x1=float(candidate.bbox[2]) if candidate.bbox else 100.0,
                y1=float(candidate.bbox[3]) if candidate.bbox else float(candidate.order_hint) * 10.0 + 10.0,
                score=float(candidate.score),
                page=int(candidate.page),
                order_hint=int(candidate.order_hint),
            )
            for candidate in buffer
        ]
        results = self.kernel.analyze(boxes, self.threshold)
        if not results:
            return iter(())
        ordered = sorted(results, key=lambda result: result.order)
        low_conf = [res for res in ordered if res.confidence < self.low_conf_cutoff]
        if low_conf:
            updated = self.kernel.calibrate([cand.score for cand in buffer], self.threshold)
            if updated > self.threshold + 1e-3:
                self.threshold = updated
                return self._flush(buffer)
        segments: List[LayoutSegment] = []
        for res in ordered:
            candidate = buffer[res.index]
            block = self._merge(candidate, res)
            segments.append(
                LayoutSegment(
                    block=block,
                    label=res.label,
                    confidence=block.confidence,
                    page=res.page,
                    order=res.order,
                    threshold=self.threshold,
                )
            )
            self._history.append(block.confidence)
        return iter(segments)

    def _merge(self, candidate: LayoutCandidate, result: LayoutResult) -> Block:
        attrs = dict(candidate.block.attrs)
        attrs["layout_label"] = result.label
        attrs["layout_confidence"] = round(result.confidence, 4)
        attrs["layout_order"] = int(result.order)
        attrs["layout_calibrated_threshold"] = round(self.threshold, 4)
        attrs.setdefault("layout_kernel", "native-cpp-v1")
        if candidate.metadata:
            for key, value in candidate.metadata.items():
                attrs.setdefault(key, value)
        bbox = None
        if candidate.bbox:
            try:
                bbox = BBox(x0=float(candidate.bbox[0]), y0=float(candidate.bbox[1]), x1=float(candidate.bbox[2]), y1=float(candidate.bbox[3]))
            except Exception:  # pragma: no cover - defensive
                bbox = None
        prov_data = {
            "page": int(result.page),
            "order": int(result.order),
        }
        if bbox:
            prov_data["bbox"] = bbox
        prov = clone_model(candidate.block.prov, **prov_data)
        new_type = candidate.block.type
        if result.label != "paragraph":
            new_type = result.label
        confidence = max(candidate.block.confidence, result.confidence)
        return clone_model(candidate.block, attrs=attrs, prov=prov, type=new_type, confidence=confidence)

    @property
    def history(self) -> Sequence[float]:
        return tuple(self._history)


__all__ = [
    "LayoutCandidate",
    "LayoutSegment",
    "VisualLayoutAnalyzer",
]
