# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processing profile registry coordinating runtime and LLM policies."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class LLMPolicy:
    """Fine-grained controls for LLM escalation."""

    enabled: bool = True
    max_confidence: Optional[float] = None
    limit_block_types: Tuple[str, ...] = ()
    max_blocks: Optional[int] = None
    deadline_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LLMPolicy":
        if not data:
            return cls()
        enabled = bool(data.get("enabled", True))
        max_confidence = data.get("max_confidence")
        if max_confidence is not None:
            max_confidence = float(max_confidence)
        limit_block_types: Sequence[str] = tuple(data.get("limit_block_types", ()))  # type: ignore[assignment]
        limit_block_types = tuple(str(t) for t in limit_block_types)
        max_blocks = data.get("max_blocks")
        if max_blocks is not None:
            max_blocks = int(max_blocks)
            if max_blocks < 0:
                max_blocks = None
        deadline = data.get("deadline_ms")
        if deadline is not None:
            deadline = int(deadline)
            if deadline <= 0:
                deadline = None
        metadata = {key: value for key, value in (data.get("metadata") or {}).items()}
        return cls(
            enabled=enabled,
            max_confidence=max_confidence,
            limit_block_types=tuple(limit_block_types),
            max_blocks=max_blocks,
            deadline_ms=deadline,
            metadata=metadata,
        )

    def to_meta(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_confidence": self.max_confidence,
            "limit_block_types": list(self.limit_block_types),
            "max_blocks": self.max_blocks,
            "deadline_ms": self.deadline_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProcessingProfile:
    """Bundle runtime and LLM orchestration knobs under a friendly name."""

    name: str
    layout_profile: str = "default"
    layout_batch_size: int = 32
    text_batch_size: int = 32
    stream_normalize: bool = True
    warm_runtime: bool = False
    default_deadline_ms: Optional[int] = None
    max_blocks: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_policy: LLMPolicy = field(default_factory=LLMPolicy)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, name: Optional[str] = None) -> "ProcessingProfile":
        profile_name = name or str(data.get("name") or "profile")
        layout_profile = str(data.get("layout_profile") or "default")
        layout_batch_size = int(data.get("layout_batch_size") or 32)
        text_batch_size = int(data.get("text_batch_size") or 32)
        stream_normalize = bool(data.get("stream_normalize", True))
        warm_runtime = bool(data.get("warm_runtime", False))
        default_deadline = data.get("default_deadline_ms")
        if default_deadline is not None:
            default_deadline = int(default_deadline)
            if default_deadline <= 0:
                default_deadline = None
        max_blocks = data.get("max_blocks")
        if max_blocks is not None:
            max_blocks = int(max_blocks)
            if max_blocks <= 0:
                max_blocks = None
        metadata = {key: value for key, value in (data.get("metadata") or {}).items()}
        llm_policy = LLMPolicy.from_dict(data.get("llm"))
        return cls(
            name=profile_name,
            layout_profile=layout_profile,
            layout_batch_size=layout_batch_size,
            text_batch_size=text_batch_size,
            stream_normalize=stream_normalize,
            warm_runtime=warm_runtime,
            default_deadline_ms=default_deadline,
            max_blocks=max_blocks,
            metadata=metadata,
            llm_policy=llm_policy,
        )

    def to_meta(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layout_profile": self.layout_profile,
            "layout_batch_size": self.layout_batch_size,
            "text_batch_size": self.text_batch_size,
            "stream_normalize": self.stream_normalize,
            "warm_runtime": self.warm_runtime,
            "default_deadline_ms": self.default_deadline_ms,
            "max_blocks": self.max_blocks,
            "metadata": dict(self.metadata),
            "llm": self.llm_policy.to_meta(),
        }


_DEFAULT_PROFILE_NAME = "balanced"
_DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "layout_profile": "balanced",
        "layout_batch_size": 48,
        "text_batch_size": 64,
        "stream_normalize": True,
        "warm_runtime": True,
        "default_deadline_ms": 2800,
        "metadata": {
            "description": "Balanced profile for day-to-day batch conversions.",
        },
        "llm": {
            "enabled": True,
            "max_confidence": 0.6,
            "limit_block_types": ["paragraph", "heading", "title"],
            "max_blocks": 12,
            "deadline_ms": 3800,
            "metadata": {"cadence": "batch"},
        },
    },
    "realtime": {
        "layout_profile": "realtime",
        "layout_batch_size": 24,
        "text_batch_size": 32,
        "stream_normalize": True,
        "warm_runtime": False,
        "default_deadline_ms": 900,
        "max_blocks": 200,
        "metadata": {
            "description": "Latency-optimised profile for live ingestion.",
        },
        "llm": {
            "enabled": True,
            "max_confidence": 0.45,
            "limit_block_types": ["paragraph", "heading"],
            "max_blocks": 4,
            "deadline_ms": 1200,
            "metadata": {"cadence": "low-latency"},
        },
    },
    "archival": {
        "layout_profile": "archival",
        "layout_batch_size": 16,
        "text_batch_size": 128,
        "stream_normalize": True,
        "warm_runtime": True,
        "default_deadline_ms": 6000,
        "metadata": {
            "description": "High fidelity profile tuned for large archival dumps.",
        },
        "llm": {
            "enabled": True,
            "max_confidence": 0.7,
            "limit_block_types": ["paragraph", "heading", "table"],
            "max_blocks": 48,
            "metadata": {"cadence": "deep-analysis"},
        },
    },
}


class ProfileStore:
    """Resolve profiles from built-ins, repo configs, and custom paths."""

    def __init__(
        self,
        *,
        search_paths: Optional[Sequence[Path]] = None,
        builtins: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        repo_profiles = base_dir / "configs" / "profiles"
        paths: List[Path] = []
        if repo_profiles.exists():
            paths.append(repo_profiles)
        env_paths = os.getenv("SR_ADAPTER_PROFILE_PATH", "")
        for entry in env_paths.split(os.pathsep):
            if not entry:
                continue
            candidate = Path(entry).expanduser()
            if candidate.exists():
                paths.append(candidate)
        for path in search_paths or ():
            if path not in paths and path.exists():
                paths.append(path)
        self._search_paths = paths
        self._builtins = builtins or dict(_DEFAULT_PROFILES)
        self._cache: Dict[str, ProcessingProfile] = {}

    def list_available(self) -> List[str]:
        names = set(self._builtins.keys())
        for directory in self._search_paths:
            for path in directory.glob("*.y*ml"):
                names.add(path.stem)
        return sorted(names)

    def load(self, name: Optional[str]) -> ProcessingProfile:
        target = name or _DEFAULT_PROFILE_NAME
        if target in self._cache:
            return self._cache[target]
        if target in self._builtins:
            profile = ProcessingProfile.from_dict(self._builtins[target], name=target)
            self._cache[target] = profile
            return profile
        for directory in self._search_paths:
            candidate_yaml = directory / f"{target}.yaml"
            candidate_yml = directory / f"{target}.yml"
            for candidate in (candidate_yaml, candidate_yml):
                if candidate.exists():
                    with candidate.open("r", encoding="utf-8") as handle:
                        data = yaml.safe_load(handle) or {}
                    profile = ProcessingProfile.from_dict(data, name=target)
                    self._cache[target] = profile
                    return profile
        raise KeyError(f"Processing profile '{target}' could not be resolved")


_PROFILE_STORE: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _PROFILE_STORE
    if _PROFILE_STORE is None:
        _PROFILE_STORE = ProfileStore()
    return _PROFILE_STORE


def load_processing_profile(name: Optional[str] = None) -> ProcessingProfile:
    """Load *name* from the shared profile store (defaults to the balanced profile)."""

    return get_profile_store().load(name)


def _maybe_auto_selector():
    try:
        from .profile_auto import get_auto_selector
    except Exception:  # pragma: no cover - defensive guard
        return None
    try:
        return get_auto_selector()
    except Exception:  # pragma: no cover - selector construction failure
        return None


def resolve_profile(
    profile: Optional[str | ProcessingProfile],
    *,
        context: Optional[Mapping[str, Any]] = None,
) -> ProcessingProfile:
    if isinstance(profile, ProcessingProfile):
        return profile

    if isinstance(profile, str):
        candidate = profile.strip()
        if candidate and candidate.lower() != "auto":
            return load_processing_profile(candidate)
    selector = _maybe_auto_selector()
    if selector and selector.enabled:
        return selector.select(context=context)

    if isinstance(profile, str) and profile.strip():
        return load_processing_profile(profile.strip())
    return load_processing_profile()


__all__ = [
    "LLMPolicy",
    "ProcessingProfile",
    "ProfileStore",
    "get_profile_store",
    "load_processing_profile",
    "resolve_profile",
]
