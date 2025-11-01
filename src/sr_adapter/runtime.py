"""High-level runtime that coordinates native layout and text kernels."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .normalize import NativeTextNormalizer, normalize_blocks as _fallback_normalize
from .schema import Block
from .visual import LayoutCandidate, LayoutSegment, VisualLayoutAnalyzer


@dataclass
class KernelStats:
    """Simple accumulator for kernel invocation telemetry."""

    name: str
    calls: int = 0
    total_ms: float = 0.0
    total_units: int = 0
    failures: int = 0

    def record(self, duration: float, units: int) -> None:
        self.calls += 1
        self.total_ms += max(0.0, duration * 1000.0)
        self.total_units += max(0, units)

    def record_failure(self) -> None:
        self.failures += 1

    @property
    def avg_ms(self) -> float:
        if not self.calls:
            return 0.0
        return self.total_ms / self.calls

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "total_ms": round(self.total_ms, 4),
            "avg_ms": round(self.avg_ms, 4),
            "total_units": self.total_units,
        }


@dataclass
class RuntimeSnapshot:
    """Serializable view of the runtime status."""

    text_enabled: bool
    layout_enabled: bool
    text_stats: KernelStats = field(default_factory=lambda: KernelStats("text"))
    layout_stats: KernelStats = field(default_factory=lambda: KernelStats("layout"))

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": {
                "enabled": self.text_enabled,
                **self.text_stats.to_dict(),
            },
            "layout": {
                "enabled": self.layout_enabled,
                **self.layout_stats.to_dict(),
            },
        }


class NativeKernelRuntime:
    """Co-ordinate native layout + text kernels with shared telemetry."""

    def __init__(
        self,
        *,
        text_normalizer: Optional[NativeTextNormalizer] = None,
        layout_analyzer: Optional[VisualLayoutAnalyzer] = None,
        layout_profile: str = "default",
        layout_batch_size: int = 32,
    ) -> None:
        self._text_normalizer: Optional[NativeTextNormalizer]
        try:
            self._text_normalizer = text_normalizer or NativeTextNormalizer()
        except Exception:
            self._text_normalizer = None

        self._layout_analyzer: Optional[VisualLayoutAnalyzer]
        if layout_analyzer is not None:
            self._layout_analyzer = layout_analyzer
        else:
            try:
                self._layout_analyzer = VisualLayoutAnalyzer(
                    profile=layout_profile,
                    batch_size=layout_batch_size,
                )
            except Exception:
                self._layout_analyzer = None

        self._text_stats = KernelStats("text")
        self._layout_stats = KernelStats("layout")

    # ------------------------------------------------------------------ helpers
    @property
    def text_enabled(self) -> bool:
        return self._text_normalizer is not None

    @property
    def layout_enabled(self) -> bool:
        return self._layout_analyzer is not None

    # ---------------------------------------------------------------- text flow
    def normalize(self, blocks: Sequence[Block]) -> List[Block]:
        if not blocks:
            return []
        start = time.perf_counter()
        payload_units = sum(len(block.text or "") for block in blocks)
        if self._text_normalizer is None:
            normalized = list(_fallback_normalize(blocks))
            duration = time.perf_counter() - start
            self._text_stats.record(duration, payload_units)
            return normalized
        try:
            normalized = self._text_normalizer.normalize_blocks(blocks)
        except Exception:
            self._text_stats.record_failure()
            raise
        duration = time.perf_counter() - start
        self._text_stats.record(duration, payload_units)
        return normalized

    def normalize_stream(
        self,
        blocks: Iterable[Block],
        *,
        batch_size: int = 32,
    ) -> Iterator[Block]:
        batch: List[Block] = []
        for block in blocks:
            batch.append(block)
            if len(batch) >= max(1, batch_size):
                yield from self.normalize(batch)
                batch.clear()
        if batch:
            yield from self.normalize(batch)

    # --------------------------------------------------------------- layout flow
    def analyze(
        self,
        candidates: Iterable[LayoutCandidate],
    ) -> Iterator[LayoutSegment]:
        if self._layout_analyzer is None:
            return

        start = time.perf_counter()
        emitted = 0

        for segment in self._layout_analyzer.process(candidates):
            emitted += 1
            yield segment

        duration = time.perf_counter() - start
        self._layout_stats.record(duration, emitted)

    # ---------------------------------------------------------------- telemetry
    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            text_enabled=self.text_enabled,
            layout_enabled=self.layout_enabled,
            text_stats=self._text_stats,
            layout_stats=self._layout_stats,
        )

    # ------------------------------------------------------------------- warming
    def warm(self) -> RuntimeSnapshot:
        """Compile kernels and emit a snapshot after a lightweight invocation."""

        if self._text_normalizer is not None:
            dummy = Block(type="paragraph", text="warmup", source="runtime", confidence=0.5)
            try:
                _ = self.normalize([dummy])
            except Exception:
                pass

        if self._layout_analyzer is not None:
            dummy_block = Block(type="paragraph", text="warmup", source="runtime", confidence=0.5)
            candidate = LayoutCandidate(
                block=dummy_block,
                bbox=(0.0, 0.0, 10.0, 10.0),
                page=0,
                score=0.5,
                order_hint=0,
            )
            try:
                list(self.analyze([candidate]))
            except Exception:
                pass

        return self.snapshot()


_RUNTIME_CACHE: Dict[Tuple[str, int], NativeKernelRuntime | bool] = {}
_RUNTIME_LOCK = Lock()


def get_native_runtime(
    layout_profile: str = "default",
    layout_batch_size: int = 32,
) -> Optional[NativeKernelRuntime]:
    if os.getenv("SR_ADAPTER_DISABLE_NATIVE_RUNTIME"):
        return None

    key = (layout_profile, max(1, layout_batch_size))
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(key)
        if isinstance(cached, NativeKernelRuntime):
            return cached
        if cached is False:
            return None
        try:
            runtime = NativeKernelRuntime(
                layout_profile=layout_profile,
                layout_batch_size=max(1, layout_batch_size),
            )
        except Exception:
            _RUNTIME_CACHE[key] = False
            return None
        _RUNTIME_CACHE[key] = runtime
        return runtime


def reset_native_runtime() -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE.clear()


def runtime_status_json() -> str:
    runtime = get_native_runtime()
    if runtime is None:
        snapshot = RuntimeSnapshot(
            text_enabled=False,
            layout_enabled=False,
        )
    else:
        snapshot = runtime.snapshot()
    return json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)


__all__ = [
    "KernelStats",
    "NativeKernelRuntime",
    "RuntimeSnapshot",
    "get_native_runtime",
    "reset_native_runtime",
    "runtime_status_json",
]

