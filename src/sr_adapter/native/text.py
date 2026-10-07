"""Bindings for the native text normalization kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from .build import ensure_library
from pathlib import Path
from threading import Lock
from typing import List, Optional, Sequence

__all__ = [
    "TextKernel",
    "TextKernelError",
    "TextKernelResult",
    "ensure_text_kernel",
]


class TextKernelError(RuntimeError):
    """Raised when the native text kernel cannot be built or invoked."""


@dataclass(frozen=True)
class TextKernelResult:
    text: str
    type_code: int
    confidence: float


def _ensure_library() -> Path:
    return ensure_library(Path(__file__).with_name("_text_kernel.cpp"), TextKernelError)

class TextKernel:
    """ctypes wrapper around the native text normalization kernel."""

    class _Input(ctypes.Structure):
        _fields_ = [
            ("text", ctypes.c_void_p),
            ("length", ctypes.c_size_t),
            ("type_code", ctypes.c_int32),
            ("infer", ctypes.c_int32),
            ("confidence", ctypes.c_double),
        ]

    class _Output(ctypes.Structure):
        _fields_ = [
            ("offset", ctypes.c_size_t),
            ("length", ctypes.c_size_t),
            ("type_code", ctypes.c_int32),
            ("confidence", ctypes.c_double),
        ]

    def __init__(self, library_path: Optional[Path] = None) -> None:
        self.path = Path(library_path or _ensure_library())
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:  # pragma: no cover - surfaced in tests
            raise TextKernelError(f"failed to load text kernel {self.path}: {exc}") from exc
        self._lib.normalize_text_blocks.argtypes = [
            ctypes.POINTER(self._Input),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(self._Output),
        ]
        self._lib.normalize_text_blocks.restype = ctypes.c_size_t
        self._lock = Lock()

    def normalize(
        self,
        payloads: Sequence[tuple[bytes, int, bool, float]],
    ) -> List[TextKernelResult]:
        if not payloads:
            return []

        inputs = (self._Input * len(payloads))()
        holders = []
        for idx, (data, type_code, infer, confidence) in enumerate(payloads):
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("kernel input must be utf-8 bytes")
            data = bytes(data)
            data.decode("utf-8")  # Reject malformed UTF-8 before entering C++.
            buf = ctypes.create_string_buffer(data)
            holders.append(buf)
            inputs[idx] = self._Input(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(data)),
                ctypes.c_int32(int(type_code)),
                ctypes.c_int32(1 if infer else 0),
                ctypes.c_double(float(confidence)),
            )

        outputs = (self._Output * len(payloads))()
        with self._lock:
            required = self._lib.normalize_text_blocks(
                inputs,
                ctypes.c_size_t(len(payloads)),
                ctypes.c_void_p(),
                ctypes.c_size_t(),
                outputs,
            )
            buffer = ctypes.create_string_buffer(required or 1)
            written = self._lib.normalize_text_blocks(
                inputs,
                ctypes.c_size_t(len(payloads)),
                ctypes.cast(buffer, ctypes.c_void_p),
                ctypes.c_size_t(len(buffer)),
                outputs,
            )

        if written > len(buffer):
            raise TextKernelError("kernel returned an invalid output size")
        data = buffer.raw[:written]
        results: List[TextKernelResult] = []
        for entry in outputs:
            start = int(entry.offset)
            end = start + int(entry.length)
            if end > len(data):
                raise TextKernelError("kernel returned an invalid output range")
            chunk = data[start:end]
            results.append(
                TextKernelResult(
                    text=chunk.decode("utf-8", "replace"),
                    type_code=int(entry.type_code),
                    confidence=float(entry.confidence),
                )
            )
        return results


_KERNEL: TextKernel | None | bool = None
_LOCK = Lock()


def ensure_text_kernel() -> TextKernel:
    global _KERNEL
    with _LOCK:
        if isinstance(_KERNEL, TextKernel):
            return _KERNEL
        if _KERNEL is False:
            raise TextKernelError("text kernel disabled")
        _KERNEL = TextKernel()
        return _KERNEL
