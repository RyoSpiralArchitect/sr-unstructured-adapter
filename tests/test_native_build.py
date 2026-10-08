"""Native cache publication and Windows compiler-runtime regressions."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_adapter.native import build


def test_link_flags_invalidate_cached_library(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cpp"
    source.write_text("extern \"C\" int answer() { return 42; }", encoding="utf-8")
    monkeypatch.setenv("SR_ADAPTER_NATIVE_CACHE", str(tmp_path / "cache"))
    commands = []

    def compile_library(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(str(len(commands)).encode())

    monkeypatch.setattr(build.subprocess, "run", compile_library)
    original_flags = build._compiler_flags
    first = build.ensure_library(source, RuntimeError)
    assert build.ensure_library(source, RuntimeError) == first
    assert len(commands) == 1

    monkeypatch.setattr(build, "_compiler_flags", lambda target: (*original_flags(target), "-DNEW_LINKAGE=1"))
    changed = build.ensure_library(source, RuntimeError)
    assert changed != first
    assert changed.read_bytes() == b"2"
    assert first.read_bytes() == b"1"


def test_failed_compile_does_not_publish_partial_library(tmp_path, monkeypatch):
    source = tmp_path / "kernel.cpp"
    source.write_text("invalid source", encoding="utf-8")
    cache = tmp_path / "cache"
    monkeypatch.setenv("SR_ADAPTER_NATIVE_CACHE", str(cache))

    def fail_compile(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial library")
        raise subprocess.CalledProcessError(1, command, stderr=b"link failed")

    monkeypatch.setattr(build.subprocess, "run", fail_compile)
    with pytest.raises(RuntimeError, match="link failed"):
        build.ensure_library(source, RuntimeError)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("other_builder_published", [False, True])
def test_windows_loaded_dll_publication_race(tmp_path, monkeypatch, other_builder_published):
    source = tmp_path / "kernel.cpp"
    source.write_text("extern \"C\" int answer() { return 42; }", encoding="utf-8")
    cache = tmp_path / "cache"
    monkeypatch.setenv("SR_ADAPTER_NATIVE_CACHE", str(cache))
    monkeypatch.setattr(build, "sys", SimpleNamespace(platform="win32"))

    def compile_library(command, **kwargs):
        Path(command[-1]).write_bytes(b"compiled library")

    def loaded_destination(output, target):
        if other_builder_published:
            target.write_bytes(b"other complete library")
        raise PermissionError("destination is loaded or unwritable")

    monkeypatch.setattr(build.subprocess, "run", compile_library)
    monkeypatch.setattr(Path, "replace", loaded_destination)
    if other_builder_published:
        target = build.ensure_library(source, RuntimeError)
        assert target.read_bytes() == b"other complete library"
        assert list(cache.iterdir()) == [target]
    else:
        with pytest.raises(RuntimeError, match="PermissionError"):
            build.ensure_library(source, RuntimeError)
        assert list(cache.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader regression")
def test_windows_kernels_load_without_compiler_runtime_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_ADAPTER_NATIVE_CACHE", str(tmp_path / "cache"))
    sources = Path(build.__file__).parent
    libraries = [
        build.ensure_library(sources / "_text_kernel.cpp", RuntimeError),
        build.ensure_library(sources / "_layout_kernel.cpp", RuntimeError),
    ]
    # Load in a fresh process, outside the compiler directory, with only
    # Windows system DLLs on PATH. Do not register compiler DLL directories.
    environment = dict(os.environ)
    environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    probe = """import ctypes, sys
text = ctypes.CDLL(sys.argv[1])
layout = ctypes.CDLL(sys.argv[2])
assert text.normalize_text_blocks
assert layout.analyze_layout
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, *(str(path) for path in libraries)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
