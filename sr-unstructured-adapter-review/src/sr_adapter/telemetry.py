"""Telemetry exporters for kernel runtime metrics."""

from __future__ import annotations

import json
import time
from typing import Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency
    import sentry_sdk
except ImportError:  # pragma: no cover - handled gracefully in exporter
    sentry_sdk = None  # type: ignore[assignment]

from .llm_metrics import LLMMetricsRegistry, get_llm_registry
from .runtime import NativeKernelRuntime, RuntimeSnapshot, get_native_runtime
from .settings import TelemetrySettings, get_settings


def _snapshot(runtime: Optional[NativeKernelRuntime]) -> RuntimeSnapshot:
    if runtime is None:
        runtime = get_native_runtime()
    if runtime is None:
        return RuntimeSnapshot(text_enabled=False, layout_enabled=False)
    return runtime.snapshot()


class TelemetryExporter:
    """Produce telemetry payloads and push them to external sinks."""

    def __init__(
        self,
        *,
        settings: Optional[TelemetrySettings] = None,
        runtime: Optional[NativeKernelRuntime] = None,
        llm_registry: Optional[LLMMetricsRegistry] = None,
    ) -> None:
        self.settings = settings or get_settings().telemetry
        self._runtime = runtime
        self._llm_registry = llm_registry or get_llm_registry()
        self._sentry_inited = False

    # ----------------------------------------------------------------- snapshots
    def snapshot(self) -> RuntimeSnapshot:
        return _snapshot(self._runtime)

    def llm_snapshot(self) -> Dict[str, object]:
        return self._llm_registry.snapshot().to_dict()

    def snapshot_dict(self) -> Dict[str, object]:
        payload = self.snapshot().to_dict()
        payload["llm"] = self.llm_snapshot()
        return payload

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot_dict(), ensure_ascii=False, indent=2)

    # -------------------------------------------------------------- prometheus
    def render_prometheus(
        self,
        *,
        extra_labels: Optional[Mapping[str, str]] = None,
    ) -> str:
        """Render the telemetry snapshot using the Prometheus exposition format."""

        if not self.settings.enable_prometheus:
            raise RuntimeError("Prometheus export is disabled in settings")
        snapshot = self.snapshot()

        def _format_labels(additional: Mapping[str, str]) -> str:
            if not additional:
                return ""
            payload = ",".join(
                f"{key}={json.dumps(value)}" for key, value in sorted(additional.items())
            )
            return f"{{{payload}}}"

        base_labels = dict(self.settings.labels)
        base_labels.update(extra_labels or {})
        lines = [
            "# HELP sr_adapter_kernel_calls_total Total kernel invocations.",
            "# TYPE sr_adapter_kernel_calls_total counter",
        ]
        for name, stats in (
            ("text", snapshot.text_stats),
            ("layout", snapshot.layout_stats),
        ):
            labels = dict(base_labels)
            labels["kernel"] = name
            lines.append(
                f"sr_adapter_kernel_calls_total{_format_labels(labels)} {stats.calls}"
            )
        lines.append(
            "# HELP sr_adapter_kernel_failures_total Kernel invocation failures."  # noqa: PIE790
        )
        lines.append("# TYPE sr_adapter_kernel_failures_total counter")
        for name, stats in (
            ("text", snapshot.text_stats),
            ("layout", snapshot.layout_stats),
        ):
            labels = dict(base_labels)
            labels["kernel"] = name
            lines.append(
                f"sr_adapter_kernel_failures_total{_format_labels(labels)} {stats.failures}"
            )
        lines.append(
            "# HELP sr_adapter_kernel_duration_ms_total Total runtime in milliseconds."
        )
        lines.append("# TYPE sr_adapter_kernel_duration_ms_total counter")
        for name, stats in (
            ("text", snapshot.text_stats),
            ("layout", snapshot.layout_stats),
        ):
            labels = dict(base_labels)
            labels["kernel"] = name
            lines.append(
                "sr_adapter_kernel_duration_ms_total"
                f"{_format_labels(labels)} {round(stats.total_ms, 4)}"
            )
        lines.append(
            "# HELP sr_adapter_kernel_units_total Processed units (characters or segments)."
        )
        lines.append("# TYPE sr_adapter_kernel_units_total counter")
        for name, stats in (
            ("text", snapshot.text_stats),
            ("layout", snapshot.layout_stats),
        ):
            labels = dict(base_labels)
            labels["kernel"] = name
            lines.append(
                f"sr_adapter_kernel_units_total{_format_labels(labels)} {stats.total_units}"
            )
        llm_snapshot = self._llm_registry.snapshot()
        lines.append("# HELP sr_adapter_llm_calls_total Total LLM invocations grouped by status.")
        lines.append("# TYPE sr_adapter_llm_calls_total counter")
        for stats in llm_snapshot.stats:
            labels = dict(base_labels)
            labels["driver"] = stats.driver
            success = dict(labels)
            success["status"] = "success"
            failure = dict(labels)
            failure["status"] = "failure"
            lines.append(
                f"sr_adapter_llm_calls_total{_format_labels(success)} {stats.successes}"
            )
            lines.append(
                f"sr_adapter_llm_calls_total{_format_labels(failure)} {stats.failures}"
            )
        lines.append("# HELP sr_adapter_llm_latency_ms_total Cumulative latency spent waiting for LLMs.")
        lines.append("# TYPE sr_adapter_llm_latency_ms_total counter")
        for stats in llm_snapshot.stats:
            labels = dict(base_labels)
            labels["driver"] = stats.driver
            lines.append(
                "sr_adapter_llm_latency_ms_total"
                f"{_format_labels(labels)} {round(stats.total_latency_ms, 4)}"
            )
        lines.append("# HELP sr_adapter_llm_request_bytes_total Total request payload bytes sent to LLMs.")
        lines.append("# TYPE sr_adapter_llm_request_bytes_total counter")
        for stats in llm_snapshot.stats:
            labels = dict(base_labels)
            labels["driver"] = stats.driver
            lines.append(
                "sr_adapter_llm_request_bytes_total"
                f"{_format_labels(labels)} {stats.total_request_bytes}"
            )
        lines.append("# HELP sr_adapter_llm_response_bytes_total Total response payload bytes received from LLMs.")
        lines.append("# TYPE sr_adapter_llm_response_bytes_total counter")
        for stats in llm_snapshot.stats:
            labels = dict(base_labels)
            labels["driver"] = stats.driver
            lines.append(
                "sr_adapter_llm_response_bytes_total"
                f"{_format_labels(labels)} {stats.total_response_bytes}"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ sentry
    def _ensure_sentry(self) -> None:
        if self._sentry_inited:
            return
        if sentry_sdk is None:
            raise RuntimeError("sentry-sdk is not installed")
        if not self.settings.sentry_dsn:
            raise RuntimeError("Sentry DSN is not configured")
        hub_cls = getattr(sentry_sdk, "Hub", None)
        if hub_cls is not None:
            try:
                current_hub = hub_cls.current  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                current_hub = None
            existing = getattr(current_hub, "client", None)
            if existing is not None:
                self._sentry_inited = True
                return
        sentry_sdk.init(  # type: ignore[call-arg]
            dsn=self.settings.sentry_dsn,
            environment=self.settings.environment,
            release=self.settings.release,
            traces_sample_rate=0.0,
            default_integrations=False,
        )
        self._sentry_inited = True

    def push_sentry(self) -> str:
        """Send the telemetry snapshot as a Sentry event."""

        self._ensure_sentry()
        snapshot = self.snapshot()
        event = {
            "level": "info",
            "timestamp": time.time(),
            "message": "sr-adapter kernel telemetry",
            "extra": {
                "telemetry": snapshot.to_dict(),
            },
            "tags": dict(self.settings.labels),
        }
        sentry_sdk.capture_event(event)  # type: ignore[union-attr]
        return "ok"


def export_prometheus(
    *,
    runtime: Optional[NativeKernelRuntime] = None,
    extra_labels: Optional[Mapping[str, str]] = None,
) -> str:
    """Convenience helper returning Prometheus metrics for the runtime."""

    exporter = TelemetryExporter(runtime=runtime)
    return exporter.render_prometheus(extra_labels=extra_labels)


__all__ = [
    "TelemetryExporter",
    "export_prometheus",
]

