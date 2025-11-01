# SPDX-License-Identifier: AGPL-3.0-or-later

from sr_adapter.kernel_autotune import KernelAutoTuneStore, KernelAutoTuner


def test_kernel_autotune_store_roundtrip(tmp_path):
    path = tmp_path / "autotune.json"
    store = KernelAutoTuneStore(path, enabled=True)
    store.update(profile="default", layout_batch_size=64, text_batch_bytes=512_000)
    assert path.exists()
    reloaded = KernelAutoTuneStore(path, enabled=True)
    assert reloaded.layout_batch_size("default") == 64
    assert reloaded.text_batch_bytes() == 512_000


def test_kernel_autotuner_selects_best(monkeypatch, tmp_path):
    tuner = KernelAutoTuner(layout_profile="default")
    tuner.store = KernelAutoTuneStore(tmp_path / "state.json", enabled=True)

    def fake_layout(self, batch_size: int):
        return {"batch_size": float(batch_size), "duration_ms": 1.0, "throughput": float(batch_size)}

    def fake_text(self, batch_bytes: int):
        return {
            "batch_bytes": float(batch_bytes),
            "duration_ms": 1.0,
            "throughput": float(batch_bytes),
        }

    monkeypatch.setattr(KernelAutoTuner, "_benchmark_layout", fake_layout)
    monkeypatch.setattr(KernelAutoTuner, "_benchmark_text", fake_text)

    report = tuner.tune()
    assert report.layout_batch_size is not None
    assert report.text_batch_bytes is not None
    assert tuner.store.layout_batch_size("default") == report.layout_batch_size


def test_kernel_autotuner_uses_existing_state(monkeypatch, tmp_path):
    tuner = KernelAutoTuner(layout_profile="balanced")
    store = KernelAutoTuneStore(tmp_path / "cache.json", enabled=True)
    store.update(profile="balanced", layout_batch_size=48, text_batch_bytes=400_000)
    tuner.store = store

    monkeypatch.setattr(KernelAutoTuner, "_benchmark_layout", lambda self, size: None)
    monkeypatch.setattr(KernelAutoTuner, "_benchmark_text", lambda self, size: None)

    report = tuner.tune()
    assert report.layout_trials == []
    assert report.text_trials == []
    assert report.layout_batch_size == 48
    assert report.text_batch_bytes == 400_000
