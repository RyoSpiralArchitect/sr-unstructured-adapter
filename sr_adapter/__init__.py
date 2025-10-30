"""Compatibility shim ensuring the package works without installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parent.parent / "src" / "sr_adapter"

if not _SRC_ROOT.exists():  # pragma: no cover - defensive
    raise ModuleNotFoundError("sr_adapter sources are missing; expected src/sr_adapter")

_SPEC = importlib.util.spec_from_file_location(
    "sr_adapter",
    _SRC_ROOT / "__init__.py",
    submodule_search_locations=[str(_SRC_ROOT)],
)

if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive
    raise ModuleNotFoundError("Unable to create module spec for sr_adapter")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["sr_adapter"] = _MODULE
_SPEC.loader.exec_module(_MODULE)

globals().update(_MODULE.__dict__)
