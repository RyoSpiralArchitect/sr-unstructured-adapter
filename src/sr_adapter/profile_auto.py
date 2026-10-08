"""Adaptive profile selection and feedback loop."""

from __future__ import annotations

import errno
import json
import math
import os
import random
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from .llm_metrics import LLMMetricsSnapshot
from .profiles import ProcessingProfile, get_profile_store, load_processing_profile
from .settings import AutoProfileSettings, get_settings
from .telemetry import TelemetryExporter


_STATE_LOCK_TIMEOUT = 10.0


@contextmanager
def _state_file_lock(path: Path) -> Iterator[None]:
    """Lock a stable sidecar, never the JSON inode replaced by each update.

    All writers must use this protocol on a filesystem supporting advisory
    locks and atomic replacement. The sidecar is deliberately never unlinked:
    removing it would let queued and new writers lock different inodes.
    """
    with path.with_name(path.name + ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()

            def acquire() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        deadline = time.monotonic() + _STATE_LOCK_TIMEOUT
        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring profile state lock: {path}") from exc
                time.sleep(0.01)
        try:
            yield
        finally:
            release()


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
        # Canonicalise aliases before deriving the sidecar lock's name.
        self._state_path = self.settings.resolved_state_path.resolve()
        self._stats: Dict[str, ProfileStats] = {}
        self._lock = RLock()
        self._load_state()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    @property
    def candidates(self) -> Tuple[str, ...]:
        return tuple(self.settings.candidate_profiles)

    def _load_state(self) -> None:
        with self._lock:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._stats = self._read_state()
            except OSError:
                # Selection remains usable when this optional cache cannot be
                # read. Recording uses a strict read so it cannot replace state
                # whose contents are unknown to this worker.
                pass

    def _read_state(self) -> Dict[str, ProfileStats]:
        path = self._state_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
            return {}
        payload = data.get("profiles", {}) if isinstance(data, dict) else {}
        if not isinstance(payload, dict):
            return {}
        result: Dict[str, ProfileStats] = {}
        for name, stats in payload.items():
            if not isinstance(stats, Mapping):
                continue
            try:
                candidate = ProfileStats(
                    trials=int(stats.get("trials", 0)),
                    reward_sum=float(stats.get("reward_sum", 0.0)),
                    last_updated=float(stats.get("last_updated", 0.0)),
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if candidate.trials < 0 or not all(math.isfinite(value) for value in (candidate.reward_sum, candidate.last_updated)):
                continue
            result[name] = candidate
        return result

    def _write_state(self, stats: Mapping[str, ProfileStats]) -> None:
        """Atomically publish a snapshot while the caller holds the file lock."""
        path = self._state_path
        payload = {
            "profiles": {
                name: asdict(value)
                for name, value in stats.items()
            }
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", encoding="utf-8", delete=False) as handle:
                temporary = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            deadline = time.monotonic() + _STATE_LOCK_TIMEOUT
            while True:
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    # Windows readers may briefly deny delete sharing. Keep
                    # the writer lock and retry publishing the same snapshot.
                    if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

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
        drivers: Iterable[Mapping[str, Any]]
        if isinstance(snapshot, LLMMetricsSnapshot):
            drivers = (stat.to_dict() for stat in snapshot.stats)
        elif isinstance(snapshot, Mapping):
            raw = snapshot.get("drivers", [])
            drivers = [record for record in raw if isinstance(record, Mapping)]
        else:
            return 0.0
        total_calls = 0
        total_failures = 0
        for record in drivers:
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
        context: Optional[Mapping[str, Any]],
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
        context: Optional[Mapping[str, Any]] = None,
    ) -> ProcessingProfile:
        available = self._candidate_profiles()
        if not self.enabled:
            return next(iter(available.values()))

        # Another worker may have recorded outcomes since this selector started.
        self._load_state()
        heuristic_choice = self._rule_based_choice(context, available)
        if heuristic_choice and heuristic_choice in available:
            return available[heuristic_choice]

        with self._lock:
            stats_pairs = [(name, replace(self._stats.get(name, ProfileStats()))) for name in available]
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
        meta: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        name = profile.name
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

        # The read-modify-write is one cross-process operation. Saving a cached
        # snapshot, even atomically, would discard outcomes recorded elsewhere.
        with self._lock, _state_file_lock(self._state_path):
            current = self._read_state()
            stats = current.setdefault(name, ProfileStats())
            stats.trials += 1
            stats.reward_sum += reward
            stats.last_updated = time.time()
            self._write_state(current)
            self._stats = current


_SELECTOR: AdaptiveProfileSelector | None = None


def get_auto_selector() -> AdaptiveProfileSelector:
    global _SELECTOR
    if _SELECTOR is None:
        _SELECTOR = AdaptiveProfileSelector()
    return _SELECTOR


def resolve_auto_profile(context: Optional[Mapping[str, Any]] = None) -> ProcessingProfile:
    selector = get_auto_selector()
    return selector.select(context=context)


def record_profile_outcome(profile: ProcessingProfile, meta: Mapping[str, Any]) -> None:
    selector = get_auto_selector()
    selector.record_outcome(profile, meta)


__all__ = [
    "AdaptiveProfileSelector",
    "ProfileStats",
    "get_auto_selector",
    "record_profile_outcome",
    "resolve_auto_profile",
]
