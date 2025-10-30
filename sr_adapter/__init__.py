"""Shim package to expose ``src.sr_adapter`` without installation.

This keeps ``python -m sr_adapter.cli`` working in editable checkouts by
forwarding imports to the actual implementation living under ``src/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PKG_ROOT = Path(__file__).resolve().parent
_SRC_IMPL = _PKG_ROOT.parent / "src" / "sr_adapter"

__path__ = [str(_SRC_IMPL)] if _SRC_IMPL.exists() else list(__path__)  # type: ignore[name-defined]

if _SRC_IMPL.exists():
    spec = importlib.util.spec_from_file_location("_sr_adapter_impl", _SRC_IMPL / "__init__.py")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name, value in vars(module).items():
            if name.startswith("__") and name not in {"__all__", "__doc__"}:
                continue
            globals()[name] = value
        __all__ = getattr(module, "__all__", [name for name in globals() if not name.startswith("_")])
    else:  # pragma: no cover - defensive fallback
        __all__: list[str] = []
else:  # pragma: no cover - when installed via pip the shim is redundant
    __all__: list[str] = []
