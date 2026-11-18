"""Independent validation of in-memory simulated remediation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum

from student2_blue_agent import DefensiveRecommendation
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment, RemediationAuditEvent
from student2_red_agent import RedAgentFinding, evaluate_reachability, scenario_definition


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    explanation: str


@dataclass(frozen=True)
class VerificationResult:
    verification_id: str
    investigation_id: str
    red_finding_id: str
    recommendation_id: str
    control_checked: str
    approval_id: str
    before_state_result: bool
    after_state_result: bool
    before_graph_reachable: bool
    after_graph_reachable: bool
    checks: tuple[VerificationCheck, ...]
    status: VerificationStatus
    explanation: str
    timestamp: str
    simulation_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DemoVerifier:
    def verify(self, finding: RedAgentFinding, recommendation: DefensiveRecommendation, approval: ApprovalEvent | None, audit: RemediationAuditEvent | None, environment: DemoEnvironment) -> VerificationResult:
        scenario = scenario_definition(finding.scenario_id)
        checks = (
            VerificationCheck("approval", approval is not None and approval.decision is ApprovalDecision.APPROVED and approval.recommendation_id == recommendation.recommendation_id, "affirmative approval matches recommendation"),
            VerificationCheck("audit", audit is not None and audit.recommendation_id == recommendation.recommendation_id and audit.approval_id == (approval.approval_id if approval else ""), "audit provenance matches approval and recommendation"),
            VerificationCheck("control", recommendation.target_control_id == scenario.control_id and environment.control_state(scenario.control_id), "intended simulated control is active"),
        )
        before = finding.candidate_path.reachable
        after = evaluate_reachability(scenario.graph, environment.current_state).reachable
        all_passed = all(check.passed for check in checks) and before and not after
        return VerificationResult(f"verification:{recommendation.recommendation_id}", finding.investigation_id, finding.finding_id, recommendation.recommendation_id, scenario.control_id, approval.approval_id if approval else "missing", bool(audit and not audit.before_state), bool(audit and audit.after_state), before, after, checks, VerificationStatus.VERIFIED if all_passed else VerificationStatus.FAILED, "Independent simulation verification succeeded." if all_passed else "Independent simulation verification failed.", datetime.now(timezone.utc).isoformat())
