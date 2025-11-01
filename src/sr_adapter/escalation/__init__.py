"""Helpers powering learned escalation decisions."""

from .features import build_features
from .logging import EscalationLogger, get_escalation_logger
from .model import EscalationModel, LinearEscalationModel, load_escalation_model
from .policy import (
    EscalationPolicyEngine,
    SelectionCandidate,
    SelectionResult,
    get_escalation_policy,
    reset_escalation_policy,
)

__all__ = [
    "build_features",
    "EscalationLogger",
    "EscalationModel",
    "EscalationPolicyEngine",
    "LinearEscalationModel",
    "SelectionCandidate",
    "SelectionResult",
    "get_escalation_logger",
    "get_escalation_policy",
    "load_escalation_model",
    "reset_escalation_policy",
]
