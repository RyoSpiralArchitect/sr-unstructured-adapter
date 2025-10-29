"""Compatibility shim to expose the src-based package without installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_src_package = _pkg_root.parent / "src" / "sr_adapter"
_src_init = _src_package / "__init__.py"

if str(_src_package.parent) not in sys.path:
    sys.path.insert(0, str(_src_package.parent))

_spec = importlib.util.spec_from_file_location(__name__, _src_init)
if _spec is None or _spec.loader is None:  # pragma: no cover - environment issue guard
    raise ImportError("Cannot locate sr_adapter package in src directory")

_module = importlib.util.module_from_spec(_spec)
_module.__path__ = [str(_src_package)]  # type: ignore[attr-defined]
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

globals().update(_module.__dict__)
