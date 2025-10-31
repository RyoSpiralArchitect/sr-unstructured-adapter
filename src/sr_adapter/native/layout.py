"""Bindings for the native layout kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
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

_SUFFIX = {
    "linux": ".so",
    "darwin": ".dylib",
    "win32": ".dll",
}


def _library_suffix() -> str:
    for key, suffix in _SUFFIX.items():
        if sys.platform.startswith(key):
            return suffix
    return ".so"


def _library_path() -> Path:
    return Path(__file__).with_name("_layout_kernel" + _library_suffix())


def _source_path() -> Path:
    return Path(__file__).with_name("_layout_kernel.cpp")


def _compile_library(target: Path) -> None:
    source = _source_path()
    if not source.exists():
        raise LayoutKernelError(f"missing kernel source: {source}")

    compiler = os.environ.get("CXX", "c++")
    cmd = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        str(source),
        "-o",
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced in tests
        raise LayoutKernelError(exc.stderr.decode("utf-8", "ignore") or str(exc)) from exc


def _ensure_library() -> Path:
    target = _library_path()
    source = _source_path()
    needs_build = not target.exists()
    if not needs_build and source.exists():
        needs_build = target.stat().st_mtime < source.stat().st_mtime
    if needs_build:
        target.parent.mkdir(parents=True, exist_ok=True)
        _compile_library(target)
    legacy = Path(__file__).with_suffix(_library_suffix())
    if legacy.exists() and legacy != target:
        try:
            legacy.unlink()
        except OSError:
            pass
    return target


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
        c_boxes = (self._Box * len(boxes))()
        for idx, box in enumerate(boxes):
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
        values = [float(v) for v in scores if not isinstance(v, bool)]
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


def ensure_layout_kernel() -> LayoutKernel:
    global _kernel
    if _kernel is None:
        _kernel = LayoutKernel()
    return _kernel


__all__ = ["LayoutKernel", "LayoutKernelError", "LayoutBox", "LayoutResult", "ensure_layout_kernel"]
