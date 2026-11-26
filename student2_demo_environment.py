"""Strictly in-memory, simulation-only remediation environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    DECLINED = "declined"
    ESCALATED = "escalated"
    MORE_EVIDENCE_REQUESTED = "more_evidence_requested"


@dataclass(frozen=True)
class ApprovalEvent:
    approval_id: str
    recommendation_id: str
    investigation_id: str
    operator_identity: str
    decision: ApprovalDecision
    timestamp: str
    simulation_only: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.approval_id, self.recommendation_id, self.investigation_id, self.operator_identity,
            self.timestamp,
        )):
            raise ValueError("approval fields must be non-empty strings")
        if not isinstance(self.note, str) or len(self.note) > 500:
            raise ValueError("approval note must be text of at most 500 characters")
        if type(self.decision) is not ApprovalDecision or self.simulation_only is not True:
            raise ValueError("approval must be an explicit simulation-only decision")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RemediationAuditEvent:
    audit_id: str
    environment_id: str
    recommendation_id: str
    approval_id: str
    control_id: str
    before_state: bool
    after_state: bool
    timestamp: str
    simulation_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DemoEnvironment:
    """A mutable holder for bounded in-memory boolean controls only."""

    INITIAL_CONTROLS = {
        "ingress_restricted": False,
        "service_credentials_rotated": False,
        "egress_restricted": False,
    }

    def __init__(self, environment_id: str = "demo-environment") -> None:
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("environment_id must be a non-empty string")
        self.environment_id = environment_id
        self._initial_state = dict(self.INITIAL_CONTROLS)
        self._state = dict(self._initial_state)
        self._audits: list[RemediationAuditEvent] = []
        self._applied_recommendations: set[str] = set()
        self._version = 1

    @property
    def initial_state(self) -> dict[str, bool]:
        return dict(self._initial_state)

    @property
    def current_state(self) -> dict[str, bool]:
        return dict(self._state)

    @property
    def audit_history(self) -> tuple[RemediationAuditEvent, ...]:
        return tuple(self._audits)

    @property
    def applied_recommendation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._applied_recommendations))

    @property
    def version(self) -> int:
        return self._version

    def control_state(self, control_id: str) -> bool:
        if control_id not in self._state:
            raise ValueError("unknown simulated control")
        return self._state[control_id]

    def apply(self, recommendation: object, approval: ApprovalEvent) -> RemediationAuditEvent:
        """Apply one approved, scenario-bound recommendation in memory."""

        recommendation_id = getattr(recommendation, "recommendation_id", None)
        investigation_id = getattr(recommendation, "investigation_id", None)
        control_id = getattr(recommendation, "target_control_id", None)
        mode = getattr(recommendation, "execution_mode", None)
        if not all(isinstance(value, str) and value for value in (recommendation_id, investigation_id, control_id)):
            raise TypeError("recommendation must expose validated identifiers")
        if mode != "simulation_only":
            raise ValueError("only simulation-only recommendations are accepted")
        if approval.decision is not ApprovalDecision.APPROVED:
            raise PermissionError("an affirmative approval is required")
        if approval.recommendation_id != recommendation_id or approval.investigation_id != investigation_id:
            raise ValueError("approval does not match recommendation provenance")
        if recommendation_id in self._applied_recommendations:
            raise ValueError("recommendation has already been applied")
        if control_id not in self._state:
            raise ValueError("recommendation targets an unknown control")
        before = self._state[control_id]
        self._state[control_id] = True
        self._version += 1
        audit = RemediationAuditEvent(
            audit_id=f"audit:{len(self._audits) + 1:04d}",
            environment_id=self.environment_id,
            recommendation_id=recommendation_id,
            approval_id=approval.approval_id,
            control_id=control_id,
            before_state=before,
            after_state=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._audits.append(audit)
        self._applied_recommendations.add(recommendation_id)
        return audit

    def reset(self) -> None:
        self._state = dict(self._initial_state)
        self._audits.clear()
        self._applied_recommendations.clear()
        self._version += 1
