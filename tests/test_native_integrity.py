"""Cross-backend and persistence regressions found by the comprehensive audit."""

import json

import pytest

from sr_adapter import Block
from sr_adapter import normalize as normalization
from sr_adapter.kernel_autotune import KernelAutoTuneStore
from sr_adapter.llm_metrics import LLMMetricsRegistry
from sr_adapter.native import LayoutBox, TextKernel, ensure_layout_kernel
from sr_adapter.normalize import NativeTextNormalizer, _normalize_block_py
from sr_adapter.runtime import NativeKernelRuntime
from sr_adapter.schema import Block as SchemaBlock, Span
from sr_adapter.visual import LayoutCalibrationStore, LayoutCandidate, VisualLayoutAnalyzer


@pytest.mark.parametrize("text", ["TITLE", "1.title", "é X", "ÉTÉ", "\x85hello\x85", "\x00hello\x00", "a\u2028\nb", "(a) name", "•  Foo\r\n\nBar"])
def test_native_python_unicode_parity(text):
    block = Block(text=text, attrs={"text": None, "value": "  Ａ\r\n", "count": 1})
    native = NativeTextNormalizer().normalize_block(block)
    assert native.model_dump() == _normalize_block_py(block).model_dump()


def test_checkout_public_types_have_one_identity():
    assert Block is SchemaBlock


def test_normalization_preserves_code_and_invalidates_changed_offsets():
    code = Block(type="code", text="  Ａ = 1\r\n\n", spans=[Span(start=2, end=3)])
    prose = Block(text="  Ａ ", spans=[Span(start=2, end=3)])
    for normalize in (_normalize_block_py, NativeTextNormalizer().normalize_block):
        assert normalize(code).model_dump() == code.model_dump()
        changed = normalize(prose)
        assert changed.text == "A"
        assert changed.spans == []
        assert changed.attrs["spans_invalidated_by"] == "text_normalization"
        Block.model_validate(changed.model_dump())


def test_native_failure_preserves_generator_inputs(monkeypatch):
    class BrokenNormalizer:
        def normalize_blocks(self, blocks):
            list(blocks)
            raise RuntimeError("broken kernel")
    monkeypatch.setattr(normalization, "_get_native_normalizer", lambda: BrokenNormalizer())
    monkeypatch.setattr(normalization, "_NATIVE_NORMALIZER", None)
    result = normalization.normalize_blocks(Block(text=f"item {i}") for i in range(3))
    assert [block.text for block in result] == ["item 0", "item 1", "item 2"]


def test_runtime_disable_also_disables_text_kernel(monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_DISABLE_NATIVE_RUNTIME", "1")
    assert normalization._get_native_normalizer() is None


def test_native_accepts_bytearray_and_rejects_invalid_utf8():
    kernel = TextKernel()
    assert kernel.normalize([(bytearray(b" hi "), 0, False, 0.5)])[0].text == "hi"
    with pytest.raises(UnicodeDecodeError):
        kernel.normalize([(b"\xff", 0, False, 0.5)])


def test_snapshots_are_detached():
    registry = LLMMetricsRegistry()
    registry.record_failure("driver", latency_ms=1, error="one")
    snapshot = registry.snapshot()
    registry.record_failure("driver", latency_ms=1, error="two")
    assert snapshot.stats[0].calls == 1
    snapshot.stats[0].calls = 99
    assert registry.snapshot().stats[0].calls == 2
    runtime = NativeKernelRuntime()
    before = runtime.snapshot()
    runtime.normalize([Block(text="example")])
    assert before.text_stats.calls == 0
    assert runtime.snapshot().text_stats.calls == 1


def test_layout_order_continues_across_batches(tmp_path):
    analyzer = VisualLayoutAnalyzer(batch_size=1, store=LayoutCalibrationStore(tmp_path / "state.json"))
    candidates = [LayoutCandidate(Block(text="body"), (0, i * 20, 10, i * 20 + 10), 0, .9, i) for i in range(4)]
    segments = list(analyzer.process(candidates))
    assert [segment.order for segment in segments] == [0, 1, 2, 3]
    assert [segment.block.prov.order for segment in segments] == [0, 1, 2, 3]
    assert [segment.order for segment in analyzer.process(candidates)] == [0, 1, 2, 3]


def test_layout_rejects_nonfinite_coordinates():
    kernel = ensure_layout_kernel()
    with pytest.raises(ValueError, match="finite"):
        kernel.analyze([LayoutBox(0, float("nan"), 1, 1, .5, 0)], .35)


def test_corrupt_cache_shapes_do_not_break_runtime(tmp_path):
    path = tmp_path / "tune.json"
    path.write_text(json.dumps({"layout": None, "text_batch_bytes": -4}))
    store = KernelAutoTuneStore(path)
    assert store.layout_batch_size("default") is None
    assert store.text_batch_bytes() is None
    path.write_text('{"default": NaN}')
    assert LayoutCalibrationStore(path).get("default", .35) == .35


def test_text_disable_with_persisted_tuning_never_constructs_kernel(monkeypatch):
    from sr_adapter import runtime as runtime_module
    from types import SimpleNamespace

    calls = []
    monkeypatch.setenv("SR_ADAPTER_DISABLE_TEXT_KERNEL", "1")
    monkeypatch.delenv("SR_ADAPTER_DISABLE_NATIVE_RUNTIME", raising=False)
    monkeypatch.setattr(runtime_module, "get_autotune_store", lambda: SimpleNamespace(
        layout_batch_size=lambda _: None, text_batch_bytes=lambda: 64,
    ))
    monkeypatch.setattr(runtime_module, "NativeTextNormalizer", lambda **kw: calls.append(kw))
    monkeypatch.setattr(runtime_module, "_RUNTIME_CACHE", {})
    runtime = runtime_module.get_native_runtime()
    assert runtime is not None and not runtime.text_enabled
    assert calls == []


def test_invalid_calibration_does_not_recurse(tmp_path):
    from sr_adapter.native import LayoutResult

    class InvalidCalibration:
        def analyze(self, boxes, threshold):
            return [LayoutResult(0, 0, 0, "paragraph", .1, (0, 0))]

        def calibrate(self, scores, threshold):
            return 1.5

    analyzer = VisualLayoutAnalyzer(kernel=InvalidCalibration(), store=LayoutCalibrationStore(tmp_path / "cache"))
    candidate = LayoutCandidate(Block(text="body"), (0, 0, 10, 10), 0, .1, 0)
    assert len(list(analyzer.process([candidate]))) == 1
    assert analyzer.threshold == .35


def test_prometheus_preserves_unicode_and_escapes_only_text_format_characters():
    from sr_adapter.settings import TelemetrySettings
    from sr_adapter.telemetry import TelemetryExporter

    exporter = TelemetryExporter(settings=TelemetrySettings(enable_prometheus=True), runtime=NativeKernelRuntime())
    metrics = exporter.render_prometheus(extra_labels={"service": '猫\t"\\\n'})
    assert 'service="猫\t\\"\\\\\\n"' in metrics
    assert '\\u732b' not in metrics


def test_autotuner_uses_median_instead_of_single_outlier(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from sr_adapter.kernel_autotune import KernelAutoTuner

    tuner = KernelAutoTuner()
    tuner.store = KernelAutoTuneStore(tmp_path / "tuning.json")
    tuner.settings = SimpleNamespace(layout_batch_sizes=(16, 32), text_batch_bytes=(256, 512), warmup_trials=0, measure_trials=3)
    measurements = {16: iter([1, 1, 100]), 32: iter([2, 2, 2])}
    monkeypatch.setattr(tuner, "_benchmark_layout", lambda size: {"batch_size": size, "throughput": next(measurements[size])})
    monkeypatch.setattr(tuner, "_benchmark_text", lambda size: None)
    assert tuner.tune().layout_batch_size == 32
