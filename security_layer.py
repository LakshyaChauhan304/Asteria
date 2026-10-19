"""Deterministic, dry-run security boundary primitives for Student-2 work.

This module performs in-memory validation and policy evaluation only. It never
executes a tool, invokes infrastructure, changes permissions, or remediates a
system.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}")
_EVIDENCE_TEXT_LIMIT = 256
_EVIDENCE_INPUT_FIELDS = frozenset(
    {
        "event_id",
        "model_prediction",
        "model_score",
        "anomaly_prediction",
        "anomaly_score",
        "protocol",
        "service",
        "state",
        "features",
    }
)


class CapabilityAction(StrEnum):
    """Actions that a future agent capability may request."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEND = "send"
    CREATE = "create"


class DecisionState(StrEnum):
    """Possible results from deterministic policy evaluation."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class DetectionEvidenceV1:
    """Immutable Student-2 representation of the DetectionEvidence contract.

    Construct this value through from_dict with the plain built-in dictionary
    returned by Student 1's DetectionEvidence.to_dict method.
    """

    event_id: str
    model_prediction: int
    model_score: float
    anomaly_prediction: int
    anomaly_score: float
    protocol: str
    service: str
    state: str
    features: tuple[tuple[str, str | int | float | bool], ...]

    CONTRACT_VERSION: ClassVar[str] = "asteria.student2.detection-evidence.v1"

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as factory construction."""

        _validate_evidence_text("event_id", self.event_id)
        _validate_binary_signal("model_prediction", self.model_prediction)
        _validate_finite_number("model_score", self.model_score, probability=True)
        _validate_binary_signal("anomaly_prediction", self.anomaly_prediction)
        _validate_finite_number("anomaly_score", self.anomaly_score, probability=False)
        _validate_evidence_text("protocol", self.protocol)
        _validate_evidence_text("service", self.service)
        _validate_evidence_text("state", self.state)
        _validate_normalized_features(self.features)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DetectionEvidenceV1:
        """Validate and copy a plain Student-1 evidence dictionary.

        Only exact built-in dictionaries are accepted. This intentionally
        rejects custom mappings and provider objects before any of their
        methods can run.
        """

        if type(payload) is not dict:
            raise TypeError("evidence payload must be a built-in dict")

        if any(type(field_name) is not str for field_name in payload):
            raise TypeError("evidence field names must be built-in strings")

        actual_fields = frozenset(payload)
        missing_fields = _EVIDENCE_INPUT_FIELDS - actual_fields
        unexpected_fields = actual_fields - _EVIDENCE_INPUT_FIELDS
        if missing_fields:
            raise ValueError(f"evidence is missing fields: {sorted(missing_fields)}")
        if unexpected_fields:
            raise ValueError(f"evidence has unexpected fields: {sorted(unexpected_fields)}")

        return cls(
            event_id=payload["event_id"],
            model_prediction=payload["model_prediction"],
            model_score=payload["model_score"],
            anomaly_prediction=payload["anomaly_prediction"],
            anomaly_score=payload["anomaly_score"],
            protocol=payload["protocol"],
            service=payload["service"],
            state=payload["state"],
            features=_normalize_features(payload["features"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a newly allocated, JSON-compatible Student-2 record."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            "event_id": self.event_id,
            "model_prediction": self.model_prediction,
            "model_score": self.model_score,
            "anomaly_prediction": self.anomaly_prediction,
            "anomaly_score": self.anomaly_score,
            "protocol": self.protocol,
            "service": self.service,
            "state": self.state,
            "features": dict(self.features),
        }


@dataclass(frozen=True)
class CapabilityRequest:
    """One proposed agent action evaluated entirely in memory."""

    agent_id: str
    identity: str
    tool: str
    action: CapabilityAction
    resource: str

    def __post_init__(self) -> None:
        """Reject incomplete or malformed requests before policy evaluation."""

        _validate_identifier("agent_id", self.agent_id)
        _validate_identifier("identity", self.identity)
        _validate_identifier("tool", self.tool)
        _validate_capability_action(self.action)
        _validate_identifier("resource", self.resource)


@dataclass(frozen=True)
class CapabilityRule:
    """An exact, allow-listed capability in a controlled environment."""

    agent_id: str
    identity: str
    tool: str
    action: CapabilityAction
    resource: str
    requires_human_approval: bool = False

    def __post_init__(self) -> None:
        """Validate a non-executable policy rule."""

        _validate_identifier("agent_id", self.agent_id)
        _validate_identifier("identity", self.identity)
        _validate_identifier("tool", self.tool)
        _validate_capability_action(self.action)
        _validate_identifier("resource", self.resource)
        if type(self.requires_human_approval) is not bool:
            raise TypeError("requires_human_approval must be a bool")
        if self.action is not CapabilityAction.READ and not self.requires_human_approval:
            raise ValueError("non-read capabilities require human approval")

    def matches(self, request: CapabilityRequest) -> bool:
        """Return whether this exact rule grants the supplied request."""

        return (
            self.agent_id == request.agent_id
            and self.identity == request.identity
            and self.tool == request.tool
            and self.action is request.action
            and self.resource == request.resource
        )


@dataclass(frozen=True)
class SecurityDecision:
    """A deterministic authorization decision without action execution."""

    request: CapabilityRequest
    state: DecisionState
    reason: str
    matched_rule_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the decision."""

        return asdict(self)


@dataclass(frozen=True)
class CapabilityPolicy:
    """Evaluate validated rules with immutable, deny-by-default semantics."""

    rules: tuple[CapabilityRule, ...]

    def __post_init__(self) -> None:
        """Reject executable, ambiguous, or mutable policy configuration."""

        if type(self.rules) is not tuple:
            raise TypeError("rules must be a tuple of CapabilityRule values")

        seen_capabilities: set[tuple[str, str, str, CapabilityAction, str]] = set()
        for rule in self.rules:
            if type(rule) is not CapabilityRule:
                raise TypeError("rules must contain only CapabilityRule values")
            capability = (
                rule.agent_id,
                rule.identity,
                rule.tool,
                rule.action,
                rule.resource,
            )
            if capability in seen_capabilities:
                raise ValueError("duplicate capability rule")
            seen_capabilities.add(capability)

    def evaluate(self, request: CapabilityRequest) -> SecurityDecision:
        """Evaluate a request without performing the requested operation."""

        if type(request) is not CapabilityRequest:
            raise TypeError("request must be a CapabilityRequest")

        matches = tuple(rule for rule in self.rules if rule.matches(request))
        if not matches:
            return SecurityDecision(
                request=request,
                state=DecisionState.BLOCKED,
                reason="capability_not_granted",
                matched_rule_count=0,
            )

        if matches[0].requires_human_approval:
            return SecurityDecision(
                request=request,
                state=DecisionState.REVIEW_REQUIRED,
                reason="human_approval_required",
                matched_rule_count=1,
            )

        return SecurityDecision(
            request=request,
            state=DecisionState.ALLOWED,
            reason="explicit_capability_grant",
            matched_rule_count=1,
        )


@dataclass(frozen=True)
class DetectionReviewTrigger:
    """A review request derived from model evidence, never a security proof."""

    event_id: str
    review_required: bool
    reasons: tuple[str, ...]
    evidence_contract_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the review trigger."""

        return asdict(self)


def review_trigger_from_evidence(evidence: DetectionEvidenceV1) -> DetectionReviewTrigger:
    """Convert sealed detection evidence into a bounded review trigger.

    A positive model or anomaly signal requests investigation. It does not
    establish attack-path reachability, validate an attack, or authorize any
    external action.
    """

    if type(evidence) is not DetectionEvidenceV1:
        raise TypeError("evidence must be a DetectionEvidenceV1")

    reasons: list[str] = []
    if evidence.model_prediction == 1:
        reasons.append(f"supervised_model_flag(score={evidence.model_score:.4f})")
    if evidence.anomaly_prediction == 1:
        reasons.append(f"anomaly_model_flag(score={evidence.anomaly_score:.4f})")

    return DetectionReviewTrigger(
        event_id=evidence.event_id,
        review_required=bool(reasons),
        reasons=tuple(reasons),
        evidence_contract_version=evidence.CONTRACT_VERSION,
    )


def _validate_identifier(field_name: str, value: object) -> None:
    if type(value) is not str or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty, normalized identifier")


def _validate_capability_action(value: object) -> None:
    if type(value) is not CapabilityAction:
        raise TypeError("action must be a CapabilityAction")


def _validate_evidence_text(field_name: str, value: object) -> None:
    if type(value) is not str or not value or len(value) > _EVIDENCE_TEXT_LIMIT:
        raise ValueError(f"{field_name} must be a non-empty text value")
    if any(character.isspace() and character not in {" "} for character in value):
        raise ValueError(f"{field_name} must not contain control whitespace")


def _validate_binary_signal(field_name: str, value: object) -> None:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{field_name} must be the integer 0 or 1")


def _validate_finite_number(field_name: str, value: object, *, probability: bool) -> None:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    if probability and not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _normalize_features(
    features: object,
) -> tuple[tuple[str, str | int | float | bool], ...]:
    if type(features) is not dict:
        raise TypeError("features must be a built-in dict")

    normalized: list[tuple[str, str | int | float | bool]] = []
    for name, value in features.items():
        if type(name) is not str or not _IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError("feature names must be normalized identifiers")
        if type(value) not in {str, int, float, bool}:
            raise TypeError("feature values must be scalar built-in values")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("feature values must be finite")
        normalized.append((name, value))
    return tuple(sorted(normalized))


def _validate_normalized_features(
    features: object,
) -> None:
    if type(features) is not tuple:
        raise TypeError("features must be an immutable tuple")
    names: list[str] = []
    for item in features:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("features must contain name/value tuples")
        name, value = item
        if type(name) is not str or not _IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError("feature names must be normalized identifiers")
        if type(value) not in {str, int, float, bool}:
            raise TypeError("feature values must be scalar built-in values")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("feature values must be finite")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    if features != tuple(sorted(features)):
        raise ValueError("features must be sorted by name")

