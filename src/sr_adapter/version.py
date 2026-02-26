# SPDX-License-Identifier: AGPL-3.0-or-later
"""Package version helpers."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_adapter_version() -> str:
    """Return the installed package version (best-effort)."""

    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - extremely old Python
        return "0.0.0"
    try:
        return version("sr-unstructured-adapter")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_adapter_version"]

