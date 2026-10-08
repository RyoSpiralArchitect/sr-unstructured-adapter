"""Compile bundled kernels atomically outside the installed package."""

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _compiler_flags(target_platform: str) -> tuple[str, ...]:
    if target_platform == "win32":
        # GNU/MinGW DLLs must carry their compiler runtimes. Python's safe
        # DLL search does not inherit the compiler's PATH; linking only
        # libstdc++/libgcc statically can still leave a winpthread dependency.
        return ("-std=c++17", "-O3", "-shared", "-static", "-static-libgcc", "-static-libstdc++")
    return ("-std=c++17", "-O3", "-fPIC", "-shared")


def ensure_library(source: Path, error_type: type[RuntimeError]) -> Path:
    try:
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler:
            raise ValueError("CXX must specify a compiler")
        flags = _compiler_flags(sys.platform)
        identity = source.read_bytes() + repr((compiler, flags, sys.platform, platform.machine())).encode()
        digest = hashlib.sha256(identity).hexdigest()[:24]
        suffix = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
        cache = Path(os.environ.get("SR_ADAPTER_NATIVE_CACHE") or (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "sr_adapter" / "native"
        ))
        target = cache / f"{source.stem}-{digest}{suffix}"
        if target.is_file():
            return target
        cache.mkdir(parents=True, exist_ok=True)
        # Concurrent builders may compile the same source, but readers only ever
        # see a complete library. Read-only site-packages also remain supported.
        with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as temporary:
            output = Path(temporary) / target.name
            subprocess.run(
                [*compiler, *flags, str(source), "-o", str(output)],
                check=True, capture_output=True, timeout=120,
            )
            try:
                output.replace(target)
            except PermissionError:
                # Another builder may have published this same cache identity
                # while we compiled. Windows cannot replace a DLL already
                # loaded by that process; its complete published file is usable.
                if sys.platform != "win32" or not target.is_file():
                    raise
        return target
    except subprocess.CalledProcessError as exc:
        raise error_type(exc.stderr.decode("utf-8", "replace")) from exc
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise error_type(f"native kernel build failed: {type(exc).__name__}") from exc
