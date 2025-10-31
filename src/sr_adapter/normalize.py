# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared normalization routines used by all parsers."""

from __future__ import annotations

import os
import re
import unicodedata
from threading import Lock
from typing import Dict, Iterable, List, Optional

from .schema import Block, clone_model

_BULLET_NORMALISER = re.compile(r"^[\u2022\u30fb]\s*")
_HEADER_PREFIX = re.compile(r"^(?:\d+[.)]|[（(][^)]+[)）])\s+")

try:  # pragma: no cover - import failure handled at runtime
    from .native import TextKernelError, TextKernelResult, ensure_text_kernel
except Exception:  # pragma: no cover - native module optional
    TextKernelError = TextKernelResult = ensure_text_kernel = None  # type: ignore


_TYPE_TO_CODE = {
    "paragraph": 0,
    "heading": 1,
    "list": 2,
    "kv": 3,
    "other": 4,
}

_CODE_TO_TYPE = {value: key for key, value in _TYPE_TO_CODE.items()}


def _normalise_text_py(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _BULLET_NORMALISER.sub("- ", text)
    return text.strip()


def _infer_type_py(block: Block) -> str:
    if block.type != "paragraph":
        return block.type
    text = block.text.strip()
    if not text:
        return "other"
    if text.count("\n") >= 1 and all(line.startswith("- ") for line in text.splitlines()):
        return "list"
    if _HEADER_PREFIX.match(text) and len(text) < 120:
        return "header"
    if len(text.split()) <= 6 and text.isupper():
        return "header"
    if ":" in text and text.count(":") == 1 and len(text) < 80:
        return "kv"
    return "paragraph"


def normalize_block(block: Block) -> Block:
    """Normalise a single block."""

    normalizer = _get_native_normalizer()
    if normalizer:
        try:
            return normalizer.normalize_block(block)
        except TextKernelError:  # pragma: no cover - surfaced in tests
            _disable_native()
        except Exception:  # pragma: no cover - defensive fallback
            _disable_native()
    return _normalize_block_py(block)


def normalize_blocks(blocks: Iterable[Block]) -> List[Block]:
    """Apply text normalisation and lightweight type inference."""
    normalizer = _get_native_normalizer()
    if normalizer:
        try:
            return normalizer.normalize_blocks(blocks)
        except TextKernelError:  # pragma: no cover - surfaced in tests
            _disable_native()
        except Exception:  # pragma: no cover - defensive fallback
            _disable_native()
    return [_normalize_block_py(block) for block in blocks]


def _normalize_block_py(block: Block) -> Block:
    text = _normalise_text_py(block.text)
    attrs = dict(block.attrs)
    if "text" in attrs:
        attrs["text"] = _normalise_text_py(attrs["text"])
    candidate = clone_model(block, text=text)
    inferred_type = _infer_type_py(candidate)
    updated = clone_model(candidate, attrs=attrs, type=inferred_type)
    if not updated.text:
        updated = clone_model(updated, confidence=min(updated.confidence, 0.2))
    return updated


class NativeTextNormalizer:
    """Wrapper that batches normalisation through the native text kernel."""

    DEFAULT_BATCH_BYTES = 512 * 1024

    def __init__(self, max_batch_bytes: Optional[int] = None) -> None:
        if ensure_text_kernel is None:
            raise TextKernelError("text kernel unavailable")
        if max_batch_bytes is None:
            env_value = os.getenv("SR_ADAPTER_TEXT_KERNEL_BATCH_BYTES")
            if env_value:
                try:
                    max_batch_bytes = int(env_value)
                except ValueError:
                    max_batch_bytes = None
        if max_batch_bytes is None:
            max_batch_bytes = self.DEFAULT_BATCH_BYTES
        self._max_batch_bytes = max_batch_bytes if max_batch_bytes > 0 else None
        self._kernel = ensure_text_kernel()

    def normalize_block(self, block: Block) -> Block:
        return self.normalize_blocks([block])[0]

    def normalize_blocks(self, blocks: Iterable[Block]) -> List[Block]:
        block_list = list(blocks)
        if not block_list:
            return []
        payloads: List[tuple[bytes, int, bool, float]] = []
        payload_sizes: List[int] = []
        block_positions: List[int] = []
        attr_positions: Dict[int, List[tuple[str, int]]] = {}
        for idx, block in enumerate(block_list):
            nfkc_text = unicodedata.normalize("NFKC", block.text or "")
            text_bytes = nfkc_text.encode("utf-8")
            allow_infer = block.type == "paragraph"
            type_code = _TYPE_TO_CODE.get(block.type, _TYPE_TO_CODE["other"])
            block_positions.append(len(payloads))
            payloads.append((text_bytes, type_code, allow_infer, block.confidence))
            payload_sizes.append(len(text_bytes))
            for key, value in list(block.attrs.items()):
                if isinstance(value, str):
                    attr_nfkc = unicodedata.normalize("NFKC", value)
                    attr_bytes = attr_nfkc.encode("utf-8")
                    payloads.append((attr_bytes, _TYPE_TO_CODE["other"], False, block.confidence))
                    payload_sizes.append(len(attr_bytes))
                    attr_positions.setdefault(idx, []).append((key, len(payloads) - 1))

        results: List[TextKernelResult] = []
        if self._max_batch_bytes is None:
            results = self._kernel.normalize(payloads)
        else:
            start = 0
            total = len(payloads)
            while start < total:
                batch_size = 0
                end = start
                while end < total:
                    payload_size = payload_sizes[end]
                    if batch_size and batch_size + payload_size > self._max_batch_bytes:
                        break
                    if batch_size == 0 and payload_size > self._max_batch_bytes:
                        end += 1
                        break
                    batch_size += payload_size
                    end += 1
                if end == start:
                    end += 1
                results.extend(self._kernel.normalize(payloads[start:end]))
                start = end
        updated_blocks: List[Block] = []
        for idx, block in enumerate(block_list):
            result = results[block_positions[idx]]
            text = result.text
            attrs = dict(block.attrs)
            for key, attr_index in attr_positions.get(idx, []):
                attr_result = results[attr_index]
                attrs[key] = attr_result.text
            new_type = block.type
            if block.type == "paragraph":
                new_type = _CODE_TO_TYPE.get(result.type_code, "paragraph")
            new_conf = max(0.0, min(1.0, result.confidence))
            updated_blocks.append(
                clone_model(
                    block,
                    text=text,
                    attrs=attrs,
                    type=new_type,
                    confidence=new_conf,
                )
            )
        return updated_blocks


_NATIVE_NORMALIZER: Optional[NativeTextNormalizer | bool] = None
_NATIVE_LOCK = Lock()


def _disable_native() -> None:
    global _NATIVE_NORMALIZER
    with _NATIVE_LOCK:
        _NATIVE_NORMALIZER = False


def _get_native_normalizer() -> Optional[NativeTextNormalizer]:
    if os.getenv("SR_ADAPTER_DISABLE_TEXT_KERNEL"):
        return None
    if ensure_text_kernel is None:
        return None
    global _NATIVE_NORMALIZER
    with _NATIVE_LOCK:
        if isinstance(_NATIVE_NORMALIZER, NativeTextNormalizer):
            return _NATIVE_NORMALIZER
        if _NATIVE_NORMALIZER is False:
            return None
        try:
            normalizer = NativeTextNormalizer()
        except Exception:
            _NATIVE_NORMALIZER = False
            return None
        _NATIVE_NORMALIZER = normalizer
        return normalizer

