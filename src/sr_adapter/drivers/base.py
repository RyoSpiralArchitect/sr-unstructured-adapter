"""Base classes, registry helpers, and shared exceptions for LLM drivers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Dict, Mapping, MutableMapping, Protocol, TypeVar

from .resilience import CircuitBreaker


class DriverError(RuntimeError):
    """Raised when a driver cannot complete a request."""


class LLMDriver(ABC):
    """Abstract base class for LLM drivers."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        self.name = name
        self.config = dict(config)
        threshold = int(self.config.get("circuit_breaker_failures", 3))
        recovery = float(self.config.get("circuit_breaker_recovery", 30.0))
        window = float(self.config.get("circuit_breaker_window", 10.0))
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=max(1, threshold),
            recovery_time=max(0.1, recovery),
            window=max(0.1, window),
        )

    @abstractmethod
    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Invoke the underlying LLM and return the raw response."""

    # ----------------------------------------------------------------- circuit
    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Expose the driver's circuit breaker."""

        return self._circuit_breaker

    def _ensure_circuit_closed(self) -> None:
        if not self._circuit_breaker.allow_request():
            raise DriverError(
                f"Driver '{self.name}' circuit breaker is open; temporarily refusing calls"
            )

    # --------------------------------------------------------------- streaming
    def stream_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Yield streaming responses.

        Drivers that do not implement native streaming fall back to yielding the
        result of :meth:`generate` once.
        """

        yield self.generate(prompt, metadata=metadata)

    def supports_streaming(self) -> bool:
        """Return ``True`` if :meth:`stream_generate` yields incremental updates."""

        return type(self).stream_generate is not LLMDriver.stream_generate

    # -------------------------------------------------------------- async flows
    async def async_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Async wrapper delegating to the synchronous ``generate`` implementation."""

        return await asyncio.to_thread(self.generate, prompt, metadata=metadata)

    async def async_stream_generate(
        self,
        prompt: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Async wrapper over :meth:`stream_generate`."""

        def _sync_iter() -> Iterator[Mapping[str, Any]]:
            return self.stream_generate(prompt, metadata=metadata)

        iterator = await asyncio.to_thread(lambda: list(_sync_iter()))

        async def _aiter(payload: list[Mapping[str, Any]]) -> AsyncIterator[Mapping[str, Any]]:
            for item in payload:
                yield item

        return _aiter(iterator)


class DriverFactory(Protocol):
    """Factory protocol responsible for instantiating drivers."""

    def __call__(self, name: str, settings: Mapping[str, Any]) -> LLMDriver:
        ...


_FactoryType = Callable[[str, Mapping[str, Any]], LLMDriver]
_Registry: MutableMapping[str, _FactoryType] = {}
_FactoryMetadata: MutableMapping[str, Dict[str, Any]] = {}
_DriverT = TypeVar("_DriverT", bound=LLMDriver)


def _coerce_factory(factory: DriverFactory | type[_DriverT]) -> _FactoryType:
    if isinstance(factory, type) and issubclass(factory, LLMDriver):
        driver_cls = factory

        def _create(name: str, settings: Mapping[str, Any]) -> LLMDriver:
            return driver_cls(name, settings)

        _create.__sr_driver_cls__ = driver_cls  # type: ignore[attr-defined]
        return _create
    if callable(factory):
        return factory  # type: ignore[return-value]
    raise TypeError("Factory must be callable or an LLMDriver subclass")


def register_driver(
    *names: str,
    factory: DriverFactory | type[_DriverT],
    overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Register a driver factory under one or more names.

    Parameters
    ----------
    *names:
        Names or aliases the driver should be discoverable under.
    factory:
        Callable or :class:`~LLMDriver` subclass used to build instances.
    overwrite:
        Whether to replace existing registrations for the same name.
    metadata:
        Optional descriptive metadata exposed through :func:`driver_metadata`.
    """

    if not names:
        raise ValueError("At least one name must be provided when registering a driver")

    coerced = _coerce_factory(factory)
    meta_payload = dict(metadata or {})
    if isinstance(factory, type) and issubclass(factory, LLMDriver):
        meta_payload.setdefault("class", f"{factory.__module__}.{factory.__name__}")

    for alias in names:
        key = alias.lower()
        existing = _Registry.get(key)
        if existing is not None and not overwrite:
            # Skip if the exact same factory is already registered; otherwise error.
            if existing is coerced:
                continue
            existing_cls = getattr(existing, "__sr_driver_cls__", None)
            coerced_cls = getattr(coerced, "__sr_driver_cls__", None)
            if existing_cls is not None and existing_cls is coerced_cls:
                continue
            raise DriverError(f"Driver '{alias}' is already registered")
        _Registry[key] = coerced
        _FactoryMetadata[key] = dict(meta_payload)


def unregister_driver(name: str) -> None:
    """Remove a driver registration if present."""

    key = name.lower()
    _Registry.pop(key, None)
    _FactoryMetadata.pop(key, None)


def available_drivers() -> tuple[str, ...]:
    """Return the list of registered driver names."""

    return tuple(sorted(_Registry))


def driver_metadata(name: str) -> Mapping[str, Any]:
    """Return metadata associated with a registered driver."""

    key = name.lower()
    if key not in _Registry:
        raise DriverError(f"Driver '{name}' is not registered")
    return dict(_FactoryMetadata.get(key, {}))


def create_registered_driver(driver_name: str, settings: Mapping[str, Any]) -> LLMDriver:
    """Instantiate a registered driver by name."""

    key = driver_name.lower()
    factory = _Registry.get(key)
    if factory is None:
        raise DriverError(f"Unknown driver '{driver_name}'")
    return factory(driver_name, settings)
