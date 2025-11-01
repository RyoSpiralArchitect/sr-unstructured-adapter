"""Base classes and shared exceptions for LLM drivers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class DriverError(RuntimeError):
    """Raised when a driver cannot complete a request."""


class LLMDriver(ABC):
    """Abstract base class for LLM drivers."""

    def __init__(self, name: str, config: Mapping[str, Any]):
        self.name = name
        self.config = dict(config)

    @abstractmethod
    def generate(self, prompt: str, *, metadata: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Invoke the underlying LLM and return the raw response."""
