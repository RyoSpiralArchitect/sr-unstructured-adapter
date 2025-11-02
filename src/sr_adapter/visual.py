"""High-level visual layout analysis utilities backed by the native kernel."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutSegment:
    """Post-processed block enhanced with layout metadata."""

    block: Block
    label: str
    confidence: float
    page: int
    order: int
    threshold: float


class LayoutCalibrationStore:
    """Persist calibration thresholds between analyzer runs."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        env_disable = os.getenv("SR_ADAPTER_DISABLE_LAYOUT_CACHE", "").strip().lower()
        disable = env_disable in {"1", "true", "yes"}
        self.enabled = bool(enabled if enabled is not None else not disable)
        if path is not None:
            self.path = Path(path)
        else:
            override = os.getenv("SR_ADAPTER_LAYOUT_CACHE")
            if override:
                self.path = Path(override)
            else:
                home = Path(os.getenv("HOME") or Path.home())
                self.path = home / ".cache" / "sr_adapter" / "layout_calibration.json"
        self._lock = Lock()
        self._loaded = False
        self._data: Dict[str, float] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.enabled:
            self._loaded = True
            return
        data: Dict[str, float] = {}
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            pass
        except Exception:
            raw = {}
        else:
            if isinstance(raw, dict):
                for key, value in raw.items():
                    try:
                        data[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        self._data = data
        self._loaded = True

    def get(self, profile: str, default: float) -> float:
        key = profile or "default"
        with self._lock:
            self._ensure_loaded()
            value = self._data.get(key)
        return float(value) if value is not None else float(default)

    def update(self, profile: str, threshold: float) -> None:
        if not self.enabled:
            return
        key = profile or "default"
        value = float(threshold)
        if not math.isfinite(value):
            return
        with self._lock:
            self._ensure_loaded()
            current = self._data.get(key)
            if current is not None and abs(current - value) <= 1e-6:
                return
            self._data[key] = value
            self._write_locked()

    def _write_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, ensure_ascii=False, sort_keys=True)
            tmp.replace(self.path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


class VisualLayoutAnalyzer:
    """Batching helper that streams candidates through the native kernel."""

    def __init__(
        self,
        *,
        kernel: Optional[LayoutKernel] = None,
        initial_threshold: float = 0.35,
        low_conf_cutoff: float = 0.6,
        batch_size: int = 32,
        profile: str = "default",
        store: Optional[LayoutCalibrationStore] = None,
    ) -> None:
        self.kernel = kernel or ensure_layout_kernel()
        self.profile = profile or "default"
        self.store = store or LayoutCalibrationStore()
        self.threshold = float(self.store.get(self.profile, float(initial_threshold)))
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
        self._update_threshold(updated)

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
                self._update_threshold(updated)
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

    def _update_threshold(self, value: float) -> None:
        if value > 0 and abs(value - self.threshold) > 1e-3:
            self.threshold = float(value)
            if self.store:
                self.store.update(self.profile, self.threshold)


__all__ = [
    "LayoutCalibrationStore",
    "LayoutCandidate",
    "LayoutSegment",
    "VisualLayoutAnalyzer",
]
