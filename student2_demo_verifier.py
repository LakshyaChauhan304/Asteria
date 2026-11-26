"""Independent validation of in-memory simulated remediation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum

from student2_blue_agent import DefensiveRecommendation
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment, RemediationAuditEvent
from student2_red_agent import RedAgentFinding, evaluate_reachability, graph_exposure, scenario_definition


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"


class VerificationOutcome(StrEnum):
    BLOCKED = "blocked"
    STILL_REACHABLE = "still_reachable"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationStep:
    sequence: int
    name: str
    passed: bool
    detail: str


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
    outcome: str = VerificationOutcome.FAILED.value
    steps: tuple[VerificationStep, ...] = ()

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
        after_path = evaluate_reachability(scenario.graph, environment.current_state)
        after = after_path.reachable
        exposure = graph_exposure(scenario.graph, environment.current_state)
        applied_once = recommendation.recommendation_id in environment.applied_recommendation_ids
        consistent = applied_once and audit is not None and audit in environment.audit_history
        blocked_count, edge_count = len(exposure["blocked_edge_ids"]), len(scenario.graph.edges)
        route_count, decoy_count = len(exposure["route_node_ids"]), len(exposure["decoy_node_ids"])
        provenance_ok = all(check.passed for check in checks) and consistent
        steps = tuple(
            VerificationStep(number, name, passed, detail)
            for number, (name, passed, detail) in enumerate((
                ("approval", checks[0].passed, checks[0].explanation if checks[0].passed else "no matching affirmative approval was found"),
                ("audit_provenance", checks[1].passed, checks[1].explanation if checks[1].passed else "audit record is missing or does not match the approval"),
                ("control_state", checks[2].passed, f"{scenario.control_id} is {'active' if environment.control_state(scenario.control_id) else 'not active'} in the sandbox"),
                ("state_consistency", consistent, "sandbox audit history contains this remediation exactly as recorded" if consistent else "sandbox state and audit history disagree"),
                ("baseline_path", before, "the path was reachable when the Red agent assessed it" if before else "the path was not reachable at assessment time, so there was nothing to eliminate"),
                ("recompute_reachability", not after, f"graph recomputed from current controls: target is {'still reachable' if after else 'no longer reachable'}"),
                ("exposure_summary", not after, f"{blocked_count} of {edge_count} edges blocked; {route_count} nodes lie on a route to the target; {decoy_count} decoy nodes are unaffected"),
            ), 1)
        )
        if not provenance_ok:
            outcome = VerificationOutcome.FAILED
        elif after:
            outcome = VerificationOutcome.STILL_REACHABLE
        elif not before:
            outcome = VerificationOutcome.INCONCLUSIVE
        else:
            outcome = VerificationOutcome.BLOCKED
        verified = outcome is VerificationOutcome.BLOCKED
        explanations = {
            VerificationOutcome.BLOCKED: "Independent simulation verification succeeded.",
            VerificationOutcome.STILL_REACHABLE: "Independent simulation verification failed: the target is still reachable.",
            VerificationOutcome.INCONCLUSIVE: "Independent simulation verification was inconclusive: no reachable path existed at assessment time.",
            VerificationOutcome.FAILED: "Independent simulation verification failed.",
        }
        return VerificationResult(f"verification:{recommendation.recommendation_id}", finding.investigation_id, finding.finding_id, recommendation.recommendation_id, scenario.control_id, approval.approval_id if approval else "missing", bool(audit and not audit.before_state), bool(audit and audit.after_state), before, after, checks, VerificationStatus.VERIFIED if verified else VerificationStatus.FAILED, explanations[outcome], datetime.now(timezone.utc).isoformat(), outcome=outcome.value, steps=steps)
