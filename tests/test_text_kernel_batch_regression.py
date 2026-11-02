"""One-write text batches preserve the native ABI and exact normalization output."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import random
import unicodedata

import pytest

from sr_adapter import Block
from sr_adapter.native import TextKernel
from sr_adapter.normalize import NativeTextNormalizer, _normalise_text_py, _normalize_block_py
from sr_adapter.schema import Span


def legacy_two_pass(kernel, payloads):
    """Exercise the existing size-query/write ABI directly."""
    inputs = (kernel._Input * len(payloads))()
    holders = []
    for i, (raw, kind, infer, confidence) in enumerate(payloads):
        buffer = ctypes.create_string_buffer(raw)
        holders.append(buffer)
        inputs[i] = kernel._Input(ctypes.cast(buffer, ctypes.c_void_p), len(raw), kind, infer, confidence)
    outputs = (kernel._Output * len(payloads))()
    required = kernel._lib.normalize_text_blocks(inputs, len(payloads), None, 0, outputs)
    buffer = ctypes.create_string_buffer(required or 1)
    written = kernel._lib.normalize_text_blocks(inputs, len(payloads), buffer, len(buffer), outputs)
    return [(buffer.raw[o.offset:o.offset + o.length].decode(), o.type_code, o.confidence) for o in outputs], written


def test_one_native_write_matches_legacy_query_abi_for_mixed_unicode_and_nuls():
    whitespace = "\t\n\r\x1c\x85\u00a0\u1680\u2000\u2009\u2028\u2029\u202f\u205f\u3000"
    texts = ["", "\x00", "a\x00b", whitespace, "•", "・\t", "  • untouched after leading spaces",
        "•  審査部\r\n\n承認\u3000\n", "a\n\n\nb", "a\r\r\nb", "🔎 条件: 未確認", "e\u0301", "𠮷野"]
    rng = random.Random(479)
    alphabet = "ab\x00条件•・🔎" + whitespace
    texts.extend("".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 100))) for _ in range(120))
    texts = [unicodedata.normalize("NFKC", text) for text in texts]
    payloads = [(text.encode(), i % 5, i % 2 == 0, .75) for i, text in enumerate(texts)]
    kernel = TextKernel()
    legacy, size = legacy_two_pass(kernel, payloads)
    results = kernel.normalize(payloads)
    assert [(r.text, r.type_code, r.confidence) for r in results] == legacy
    assert [r.text for r in results] == [_normalise_text_py(text) for text in texts]
    assert size <= sum(len(p[0]) for p in payloads)


def test_batch_transforms_once_instead_of_repeating_size_probe():
    kernel = TextKernel()
    native = kernel._lib.normalize_text_blocks
    calls = []
    def observe(*args):
        calls.append(args)
        return native(*args)
    kernel._lib.normalize_text_blocks = observe
    result = kernel.normalize([(b"  approve\x00now \r\n", 0, True, .9), (b"", 0, True, .8)])
    assert len(calls) == 1
    assert result[0].text == "approve\x00now" and result[1].text == ""
    assert result[1].type_code == 4 and result[1].confidence == .2


def test_single_write_keeps_public_block_attrs_offsets_and_code_behavior():
    blocks = [Block(text="  Ａ\r\n条件  ", attrs={"name": "・  承認\n\n\n", "binary": 3},
                    spans=[Span(start=2, end=3)]),
              Block(type="code", text="  Ａ\r\n", spans=[Span(start=2, end=3)]),
              Block(text="\x00申請\x00", attrs={"name": " a\x00b "})]
    native = NativeTextNormalizer(max_batch_bytes=8).normalize_blocks(blocks)
    assert [b.model_dump() for b in native] == [_normalize_block_py(b).model_dump() for b in blocks]


def test_external_expanding_library_keeps_the_size_query_protocol():
    kernel = TextKernel()
    calls = []
    def expand(inputs, count, buffer, capacity, outputs):
        calls.append((buffer.value, capacity.value))
        outputs[0] = kernel._Output(0, 6, 0, .8)
        if buffer.value and capacity.value >= 6:
            ctypes.memmove(buffer, b"expand", 6)
        return 6
    kernel._lib.normalize_text_blocks = expand
    result = kernel.normalize([(b"x", 0, False, .8)])
    assert result[0].text == "expand"
    assert len(calls) == 3 and calls[1] == (None, 0)


def test_concurrent_calls_keep_separate_batches():
    kernel = TextKernel()
    payloads = [[(("• 条件\x00" + str(i) + "\r\n").encode(), 0, False, .7)] for i in range(24)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(kernel.normalize, payloads))
    assert [r[0].text for r in results] == ["- 条件\x00" + str(i) for i in range(24)]


def test_malformed_utf8_is_rejected_before_native_write():
    with pytest.raises(UnicodeDecodeError):
        TextKernel().normalize([(b"\xc0\x80", 0, False, .9)])
