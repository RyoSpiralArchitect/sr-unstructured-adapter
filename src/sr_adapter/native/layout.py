"""Bindings for the native layout kernel."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from .build import ensure_library
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "LayoutKernel",
    "LayoutKernelError",
    "LayoutBox",
    "LayoutResult",
    "ensure_layout_kernel",
]


class LayoutKernelError(RuntimeError):
    """Raised when the native kernel cannot be built or invoked."""


@dataclass(frozen=True)
class LayoutBox:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    page: int
    order_hint: int = 0


@dataclass(frozen=True)
class LayoutResult:
    index: int
    order: int
    page: int
    label: str
    confidence: float
    center: tuple[float, float]


_LABEL_MAP = {
    0: "paragraph",
    1: "heading",
    2: "table",
    3: "figure",
}

def _ensure_library() -> Path:
    return ensure_library(Path(__file__).with_name("_layout_kernel.cpp"), LayoutKernelError)

class LayoutKernel:
    """Thin ctypes wrapper around the native layout kernel."""

    class _Box(ctypes.Structure):
        _fields_ = [
            ("x0", ctypes.c_double),
            ("y0", ctypes.c_double),
            ("x1", ctypes.c_double),
            ("y1", ctypes.c_double),
            ("score", ctypes.c_double),
            ("page", ctypes.c_int32),
            ("order_hint", ctypes.c_int32),
        ]

    class _Result(ctypes.Structure):
        _fields_ = [
            ("original_index", ctypes.c_int32),
            ("order", ctypes.c_int32),
            ("page", ctypes.c_int32),
            ("label", ctypes.c_int32),
            ("confidence", ctypes.c_double),
            ("x_center", ctypes.c_double),
            ("y_center", ctypes.c_double),
        ]

    def __init__(self, library_path: Optional[Path] = None) -> None:
        self.path = Path(library_path or _ensure_library())
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:  # pragma: no cover - surfaced in tests
            raise LayoutKernelError(f"failed to load kernel {self.path}: {exc}") from exc
        self._lib.analyze_layout.argtypes = [
            ctypes.POINTER(self._Box),
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.POINTER(self._Result),
        ]
        self._lib.analyze_layout.restype = ctypes.c_int32
        self._lib.calibrate_threshold.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_double,
        ]
        self._lib.calibrate_threshold.restype = ctypes.c_double
        self._lock = Lock()

    def analyze(self, boxes: Sequence[LayoutBox], threshold: float) -> List[LayoutResult]:
        if not boxes:
            return []
        if not math.isfinite(threshold):
            raise ValueError("layout threshold must be finite")
        c_boxes = (self._Box * len(boxes))()
        for idx, box in enumerate(boxes):
            if not all(math.isfinite(v) for v in (box.x0, box.y0, box.x1, box.y1, box.score)):
                raise ValueError("layout coordinates and scores must be finite")
            if box.x1 < box.x0 or box.y1 < box.y0:
                raise ValueError("layout boxes must have ordered coordinates")
            c_boxes[idx] = self._Box(
                float(box.x0),
                float(box.y0),
                float(box.x1),
                float(box.y1),
                float(box.score),
                int(box.page),
                int(box.order_hint),
            )
        results = (self._Result * len(boxes))()
        with self._lock:
            written = self._lib.analyze_layout(
                c_boxes,
                ctypes.c_int32(len(boxes)),
                ctypes.c_double(float(threshold)),
                results,
            )
        if not 0 <= written <= len(boxes):
            raise LayoutKernelError("kernel returned an invalid result count")
        output: List[LayoutResult] = []
        for i in range(int(written)):
            entry = results[i]
            label = _LABEL_MAP.get(int(entry.label), "paragraph")
            output.append(
                LayoutResult(
                    index=int(entry.original_index),
                    order=int(entry.order),
                    page=int(entry.page),
                    label=label,
                    confidence=float(entry.confidence),
                    center=(float(entry.x_center), float(entry.y_center)),
                )
            )
        return output

    def calibrate(self, scores: Iterable[float], current: float) -> float:
        values = [float(v) for v in scores if not isinstance(v, bool) and math.isfinite(float(v))]
        if not values:
            return float(current)
        arr = (ctypes.c_double * len(values))(*values)
        with self._lock:
            updated = self._lib.calibrate_threshold(
                arr,
                ctypes.c_int32(len(values)),
                ctypes.c_double(float(current)),
            )
        return float(updated)


_kernel: Optional[LayoutKernel] = None
_KERNEL_LOCK = Lock()


def ensure_layout_kernel() -> LayoutKernel:
    global _kernel
    with _KERNEL_LOCK:
        if _kernel is None:
            _kernel = LayoutKernel()
        return _kernel


__all__ = ["LayoutKernel", "LayoutKernelError", "LayoutBox", "LayoutResult", "ensure_layout_kernel"]
