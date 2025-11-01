"""Adaptive profile selection and feedback loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

from .profiles import ProcessingProfile, get_profile_store, load_processing_profile
from .settings import AutoProfileSettings, get_settings
from .telemetry import TelemetryExporter


@dataclass
class ProfileStats:
    trials: int = 0
    reward_sum: float = 0.0
    last_updated: float = 0.0

    @property
    def average_reward(self) -> float:
        if self.trials <= 0:
            return 0.0
        return self.reward_sum / self.trials


class AdaptiveProfileSelector:
    """Epsilon-greedy selector that also considers runtime heuristics."""

    def __init__(
        self,
        *,
        settings: Optional[AutoProfileSettings] = None,
        telemetry: Optional[TelemetryExporter] = None,
    ) -> None:
        self.settings = settings or get_settings().profile_automation
        self.telemetry = telemetry or TelemetryExporter()
        self.store = get_profile_store()
        self._state_path = self.settings.resolved_state_path
        self._stats: MutableMapping[str, ProfileStats] = {}
        self._load_state()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    @property
    def candidates(self) -> Tuple[str, ...]:
        return tuple(self.settings.candidate_profiles)

    def _load_state(self) -> None:
        path = self._state_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        payload = data.get("profiles", {}) if isinstance(data, dict) else {}
        if not isinstance(payload, dict):
            return
        for name, stats in payload.items():
            if not isinstance(stats, Mapping):
                continue
            self._stats[name] = ProfileStats(
                trials=int(stats.get("trials", 0)),
                reward_sum=float(stats.get("reward_sum", 0.0)),
                last_updated=float(stats.get("last_updated", 0.0)),
            )

    def _save_state(self) -> None:
        path = self._state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiles": {
                name: asdict(stats)
                for name, stats in self._stats.items()
            }
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_stats(self, name: str) -> ProfileStats:
        if name not in self._stats:
            self._stats[name] = ProfileStats()
        return self._stats[name]

    def _candidate_profiles(self) -> Dict[str, ProcessingProfile]:
        resolved: Dict[str, ProcessingProfile] = {}
        for name in self.candidates:
            try:
                resolved[name] = self.store.load(name)
            except KeyError:
                continue
        if not resolved:
            default = load_processing_profile()
            resolved[default.name] = default
        return resolved

    def _llm_failure_rate(self) -> float:
        try:
            snapshot = self.telemetry.llm_snapshot()
        except Exception:
            return 0.0
        drivers = snapshot.get("drivers", []) if isinstance(snapshot, dict) else []
        total_calls = 0
        total_failures = 0
        for record in drivers:
            if not isinstance(record, Mapping):
                continue
            total_calls += int(record.get("calls", 0))
            total_failures += int(record.get("failures", 0))
        if total_calls <= 0:
            return 0.0
        return max(0.0, min(1.0, total_failures / total_calls))

    def _kernel_latency(self) -> float:
        try:
            snapshot = self.telemetry.snapshot()
        except Exception:
            return 0.0
        return float(snapshot.text_stats.avg_ms if snapshot else 0.0)

    def _rule_based_choice(
        self,
        context: Optional[Mapping[str, object]],
        available: Mapping[str, ProcessingProfile],
    ) -> Optional[str]:
        size_hint = int(context.get("size_bytes", 0)) if context else 0
        deadline = context.get("deadline_ms") if context else None
        try:
            deadline_val = int(deadline) if deadline is not None else None
        except (TypeError, ValueError):
            deadline_val = None

        if size_hint and size_hint >= self.settings.large_document_bytes:
            return next((name for name in available if name == "archival"), None)

        if deadline_val is not None and deadline_val <= self.settings.tight_deadline_ms:
            return next((name for name in available if name == "realtime"), None)

        kernel_latency = self._kernel_latency()
        if kernel_latency >= self.settings.high_kernel_latency_ms:
            return next((name for name in available if name == "realtime"), None)

        failure_rate = self._llm_failure_rate()
        if failure_rate >= self.settings.max_llm_failure_rate:
            return next((name for name in available if name == "balanced"), None)

        return None

    def select(
        self,
        *,
        context: Optional[Mapping[str, object]] = None,
    ) -> ProcessingProfile:
        available = self._candidate_profiles()
        if not self.enabled:
            return next(iter(available.values()))

        heuristic_choice = self._rule_based_choice(context, available)
        if heuristic_choice and heuristic_choice in available:
            return available[heuristic_choice]

        stats_pairs = [(name, self._ensure_stats(name)) for name in available]
        unexplored = [name for name, stats in stats_pairs if stats.trials <= 0]
        if unexplored:
            chosen_name = unexplored[0]
            return available[chosen_name]

        if random.random() < self.settings.epsilon:
            chosen_name = random.choice(list(available.keys()))
            return available[chosen_name]

        chosen_name = max(stats_pairs, key=lambda item: item[1].average_reward)[0]
        return available[chosen_name]

    def record_outcome(
        self,
        profile: ProcessingProfile,
        meta: Mapping[str, object],
    ) -> None:
        if not self.enabled:
            return
        name = profile.name
        stats = self._ensure_stats(name)
        latency = float(meta.get("metrics_total_ms", 0.0) or 0.0)
        block_count = int(meta.get("block_count", 0) or 0)
        escalations = int(meta.get("llm_escalations", 0) or 0)
        truncated = int(meta.get("truncated_blocks", 0) or 0)

        target = max(1.0, float(self.settings.latency_target_ms))
        latency_score = max(0.0, 1.0 - (latency / target))
        quality_score = 0.0
        penalty = 0.0
        if block_count > 0:
            quality_score = escalations / max(block_count, 1)
            penalty = truncated / max(block_count, 1)
        reward = (0.6 * latency_score) + (0.4 * quality_score) - (0.25 * penalty)
        reward = max(self.settings.min_reward, min(self.settings.max_reward, reward))

        stats.trials += 1
        stats.reward_sum += reward
        stats.last_updated = time.time()
        self._stats[name] = stats
        self._save_state()


_SELECTOR: AdaptiveProfileSelector | None = None


def get_auto_selector() -> AdaptiveProfileSelector:
    global _SELECTOR
    if _SELECTOR is None:
        _SELECTOR = AdaptiveProfileSelector()
    return _SELECTOR


def resolve_auto_profile(context: Optional[Mapping[str, object]] = None) -> ProcessingProfile:
    selector = get_auto_selector()
    return selector.select(context=context)


def record_profile_outcome(profile: ProcessingProfile, meta: Mapping[str, object]) -> None:
    selector = get_auto_selector()
    selector.record_outcome(profile, meta)


__all__ = [
    "AdaptiveProfileSelector",
    "ProfileStats",
    "get_auto_selector",
    "record_profile_outcome",
    "resolve_auto_profile",
]
