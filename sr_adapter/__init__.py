# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expose the source package in an uninstalled checkout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC_IMPL = Path(__file__).resolve().parent.parent / "src" / "sr_adapter"
_spec = importlib.util.spec_from_file_location(
    __name__, _SRC_IMPL / "__init__.py", submodule_search_locations=[str(_SRC_IMPL)]
)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError("sr_adapter source package is unavailable; install the distribution")
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
