# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native kernel autotuning helpers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import Block
from .settings import get_settings


@dataclass
class KernelAutoTuneResult:
    layout_profile: str
    layout_batch_size: Optional[int]
    text_batch_bytes: Optional[int]
    layout_trials: List[Dict[str, float]] = field(default_factory=list)
    text_trials: List[Dict[str, float]] = field(default_factory=list)


class KernelAutoTuneStore:
    """Persist kernel tuning results across runs."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except Exception:
            return
        if isinstance(payload, dict):
            for profile, config in payload.get("layout", {}).items():
                try:
                    value = int(config)
                except (TypeError, ValueError):
                    continue
                self._layout_map()[str(profile)] = value
            text_bytes = payload.get("text_batch_bytes")
            try:
                if text_bytes is not None:
                    self._data.setdefault("text", {})["batch_bytes"] = int(text_bytes)
            except (TypeError, ValueError):
                pass

    def _layout_map(self) -> Dict[str, int]:
        return self._data.setdefault("layout", {})  # type: ignore[return-value]

    def layout_batch_size(self, profile: str) -> Optional[int]:
        with self._lock:
            self._ensure_loaded()
            return self._layout_map().get(profile or "default")

    def text_batch_bytes(self) -> Optional[int]:
        with self._lock:
            self._ensure_loaded()
            data = self._data.get("text", {})
            value = data.get("batch_bytes")
            return int(value) if value is not None else None

    def update(
        self,
        *,
        profile: str,
        layout_batch_size: Optional[int],
        text_batch_bytes: Optional[int],
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._ensure_loaded()
            if layout_batch_size is not None:
                self._layout_map()[profile or "default"] = int(layout_batch_size)
            if text_batch_bytes is not None:
                self._data.setdefault("text", {})["batch_bytes"] = int(text_batch_bytes)
            self._flush_locked()

    def _flush_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = {
            "layout": dict(self._layout_map()),
            "text_batch_bytes": self._data.get("text", {}).get("batch_bytes"),
        }
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


_STORE: Optional[KernelAutoTuneStore] = None


def get_autotune_store() -> KernelAutoTuneStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    settings = get_settings()
    kernel_settings = getattr(settings, "kernel_autotune", None)
    enabled = True
    path = Path(os.getenv("SR_ADAPTER_KERNEL_AUTOTUNE", "")) if os.getenv("SR_ADAPTER_KERNEL_AUTOTUNE") else None
    if kernel_settings is not None:
        enabled = bool(kernel_settings.enabled)
        resolved = getattr(kernel_settings, "resolved_state_path", None)
        if resolved is not None:
            path = resolved
    if path is None:
        base = Path(os.getenv("HOME") or str(Path.home()))
        path = base / ".cache" / "sr_adapter" / "kernel_autotune.json"
    _STORE = KernelAutoTuneStore(Path(path), enabled=enabled)
    return _STORE


class KernelAutoTuner:
    """Measure kernel throughput and persist improved parameters."""

    def __init__(self, *, layout_profile: str = "default") -> None:
        self.layout_profile = layout_profile or "default"
        self.settings = getattr(get_settings(), "kernel_autotune", None)
        self.store = get_autotune_store()

    # ---------------------------------------------------------------- sampling
    def _sample_blocks(self, count: int, *, text: str) -> List[Block]:
        content = text or "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        return [
            Block(type="paragraph", text=f"{content} #{idx}", confidence=0.5)
            for idx in range(count)
        ]

    def _layout_candidates(self, count: int) -> Iterable:
        from .visual import LayoutCandidate

        blocks = self._sample_blocks(count, text="Layout sample")
        for idx, block in enumerate(blocks):
            yield LayoutCandidate(
                block=block,
                bbox=(0.0, float(idx) * 10.0, 100.0, float(idx) * 10.0 + 10.0),
                page=idx // 20,
                score=0.5,
                order_hint=idx,
            )

    # ----------------------------------------------------------------- tuning
    def _benchmark_layout(self, batch_size: int) -> Optional[Dict[str, float]]:
        from .runtime import NativeKernelRuntime

        try:
            runtime = NativeKernelRuntime(layout_profile=self.layout_profile, layout_batch_size=batch_size)
        except Exception:
            return None
        if not runtime.layout_enabled:
            return None
        candidates = list(self._layout_candidates(batch_size * 2))
        start = time.perf_counter()
        try:
            _ = list(runtime.analyze(candidates))
        except Exception:
            return None
        elapsed = (time.perf_counter() - start) * 1000.0
        throughput = len(candidates) / elapsed if elapsed > 0 else 0.0
        return {
            "batch_size": float(batch_size),
            "duration_ms": round(elapsed, 3),
            "throughput": round(throughput, 6),
        }

    def _benchmark_text(self, batch_bytes: int) -> Optional[Dict[str, float]]:
        from .normalize import NativeTextNormalizer, TextKernelError

        try:
            normalizer = NativeTextNormalizer(max_batch_bytes=batch_bytes)
        except Exception:
            return None
        samples = self._sample_blocks(12, text="Text normalisation sample")
        start = time.perf_counter()
        try:
            _ = normalizer.normalize_blocks(samples)
        except TextKernelError:
            return None
        except Exception:
            return None
        elapsed = (time.perf_counter() - start) * 1000.0
        payload = sum(len(block.text.encode("utf-8")) for block in samples)
        throughput = payload / elapsed if elapsed > 0 else 0.0
        return {
            "batch_bytes": float(batch_bytes),
            "duration_ms": round(elapsed, 3),
            "throughput": round(throughput, 6),
        }

    def tune(self) -> KernelAutoTuneResult:
        layout_candidates: Sequence[int] = (16, 32, 48, 64)
        text_candidates: Sequence[int] = (256_000, 384_000, 512_000, 768_000)
        warmup = 1
        measurements = 2
        if self.settings is not None:
            layout_candidates = tuple(int(v) for v in getattr(self.settings, "layout_batch_sizes", layout_candidates)) or layout_candidates
            text_candidates = tuple(int(v) for v in getattr(self.settings, "text_batch_bytes", text_candidates)) or text_candidates
            warmup = max(0, int(getattr(self.settings, "warmup_trials", warmup)))
            measurements = max(1, int(getattr(self.settings, "measure_trials", measurements)))

        layout_trials: List[Dict[str, float]] = []
        for candidate in layout_candidates:
            for _ in range(warmup):
                self._benchmark_layout(candidate)
            for _ in range(measurements):
                result = self._benchmark_layout(candidate)
                if result:
                    layout_trials.append(result)

        text_trials: List[Dict[str, float]] = []
        for candidate in text_candidates:
            for _ in range(warmup):
                self._benchmark_text(candidate)
            for _ in range(measurements):
                result = self._benchmark_text(candidate)
                if result:
                    text_trials.append(result)

        best_layout: Optional[int] = None
        if layout_trials:
            top_layout = max(layout_trials, key=lambda entry: entry.get("throughput", 0.0))
            value = top_layout.get("batch_size")
            if value is not None:
                best_layout = int(value)
        else:
            stored_layout = self.store.layout_batch_size(self.layout_profile)
            if stored_layout is not None:
                best_layout = int(stored_layout)

        best_text: Optional[int] = None
        if text_trials:
            top_text = max(text_trials, key=lambda entry: entry.get("throughput", 0.0))
            value = top_text.get("batch_bytes")
            if value is not None:
                best_text = int(value)
        else:
            stored_text = self.store.text_batch_bytes()
            if stored_text is not None:
                best_text = int(stored_text)

        self.store.update(
            profile=self.layout_profile,
            layout_batch_size=best_layout,
            text_batch_bytes=best_text,
        )

        return KernelAutoTuneResult(
            layout_profile=self.layout_profile,
            layout_batch_size=best_layout,
            text_batch_bytes=best_text,
            layout_trials=layout_trials,
            text_trials=text_trials,
        )


__all__ = [
    "KernelAutoTuneResult",
    "KernelAutoTuneStore",
    "KernelAutoTuner",
    "get_autotune_store",
]
