"""Optional linear-time native literal selectors with exact Python fallback."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from threading import Lock

from .build import ensure_library


class SelectorKernelError(RuntimeError):
    pass


def _bounded_utf8(text, limit):
    # Check before making a source-sized allocation. A Unicode code point uses
    # at least one UTF-8 byte; non-ASCII near the limit needs a bounded preflight.
    if len(text) > limit:
        raise SelectorKernelError("Native selector byte budget exceeded")
    if len(text) > limit // 4 and not text.isascii():
        size = 0
        for offset in range(0, len(text), 16384):
            size += len(text[offset : offset + 16384].encode("utf-8"))
            if size > limit:
                raise SelectorKernelError("Native selector byte budget exceeded")
    raw = text.encode("utf-8")
    if len(raw) > limit:
        raise SelectorKernelError("Native selector byte budget exceeded")
    return raw


class SelectorKernel:
    def __init__(self):
        path = ensure_library(
            Path(__file__).with_name("_selector_kernel.cpp"), SelectorKernelError
        )
        self._library = ctypes.CDLL(str(path))
        self._find = self._library.sr_find_occurrence
        self._find.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        self._find.restype = ctypes.c_int64

    def find(self, text, needle, occurrence):
        target = _bounded_utf8(needle, 1024 * 1024)
        raw = _bounded_utf8(text, 32 * 1024 * 1024)
        text_buffer, needle_buffer = (
            ctypes.create_string_buffer(raw),
            ctypes.create_string_buffer(target),
        )
        offset = self._find(
            text_buffer, len(raw), needle_buffer, len(target), occurrence
        )
        if offset < -1 or offset >= 0 and text[offset : offset + len(needle)] != needle:
            raise SelectorKernelError("Native selector returned an invalid result")
        return offset


_kernel = None
_lock = Lock()


def _native():
    global _kernel
    with _lock:
        if _kernel is False:
            raise SelectorKernelError("Native selector unavailable")
        if _kernel is None:
            try:
                _kernel = SelectorKernel()
            except (RuntimeError, OSError, AttributeError):
                _kernel = False
                raise SelectorKernelError("Native selector unavailable") from None
        return _kernel


def find_text_occurrence(text, needle, occurrence, *, backend="auto"):
    """Find one overlapping literal occurrence in code points, without normalization."""
    if (
        not isinstance(text, str)
        or not isinstance(needle, str)
        or not needle
        or type(occurrence) is not int
        or not 0 <= occurrence <= 1_000_000
        or backend not in ("auto", "python", "native")
    ):
        raise ValueError("Invalid literal selector input or backend")
    disabled = os.environ.get("SR_ADAPTER_DISABLE_NATIVE_RUNTIME") == "1"
    use_native = backend == "native" or (
        backend == "auto"
        and not disabled
        and len(text) >= 16384
        and len(needle) >= 32
        and (occurrence + 1) * len(needle) >= 1_000_000
    )
    if use_native:
        try:
            if disabled:
                raise SelectorKernelError("Native runtime disabled")
            return _native().find(text, needle, occurrence)
        except (SelectorKernelError, UnicodeEncodeError):
            if backend == "native":
                raise
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(needle, start + 1)
        if start < 0:
            return -1
    return start
