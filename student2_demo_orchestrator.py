"""Stateful composition of the real Phase-1 demo and simulated response flow."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from investigator import Investigator, READ_ONLY_INVESTIGATION_TOOL, TrustedObservationContext
from security_layer import CapabilityAction, CapabilityPolicy, CapabilityRule
from student2_blue_agent import BlueAgent, DefensiveRecommendation
from student2_demo_data import DemoDataBundle, initialize_baseline_engine, load_demo_data
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment, RemediationAuditEvent
from student2_demo_verifier import DemoVerifier, VerificationResult, VerificationStatus
from student2_detection_facade import Student2DetectionFacade
from student2_detection_to_investigator import DetectionInvestigationResult, Student2DetectionToInvestigator
from student2_red_agent import RedAgent, RedAgentFinding
from student2_xgboost_runtime import XGBoostRuntime


class WorkflowState(StrEnum):
    CREATED = "created"
    MODELS_READY = "models_ready"
    DETECTED = "detected"
    INVESTIGATED = "investigated"
    RED_SIMULATED = "red_simulated"
    BLUE_RECOMMENDED = "blue_recommended"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    VERIFIED = "verified"
    FAILED = "failed"


class IncidentStatus(StrEnum):
    OPEN = "open"
    PARTIALLY_MITIGATED = "partially_mitigated"
    SIMULATION_VERIFIED = "simulation_verified"
    VERIFICATION_FAILED = "verification_failed"


class DemoOrchestrator:
    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.state = WorkflowState.CREATED
        self.incident_status = IncidentStatus.OPEN
        self.environment = DemoEnvironment()
        self._bundle: DemoDataBundle | None = None
        self._adapter: Student2DetectionToInvestigator | None = None
        self._results: tuple[DetectionInvestigationResult, ...] = ()
        self.selected: DetectionInvestigationResult | None = None
        self.finding: RedAgentFinding | None = None
        self.recommendations: tuple[DefensiveRecommendation, ...] = ()
        self.approval: ApprovalEvent | None = None
        self.audit: RemediationAuditEvent | None = None
        self.verification: VerificationResult | None = None

    def initialize(self, *, training_rows: int = 20_000, event_count: int = 20) -> None:
        self._require(WorkflowState.CREATED)
        self._bundle = load_demo_data(repository_root=self.repository_root, training_rows=training_rows, event_count=event_count)
        baseline = initialize_baseline_engine(self._bundle)
        runtime = XGBoostRuntime.load(repository_root=self.repository_root)
        contexts = tuple(TrustedObservationContext("demo-investigator", "demo-security-service", f"events/{event_id}") for event_id in self._bundle.event_ids)
        rules = tuple(CapabilityRule(context.agent_id, context.identity, READ_ONLY_INVESTIGATION_TOOL, CapabilityAction.READ, context.resource) for context in contexts)
        self._adapter = Student2DetectionToInvestigator(Student2DetectionFacade(baseline.engine, runtime), Investigator(CapabilityPolicy(rules)))
        self.state = WorkflowState.MODELS_READY

    def run_detection(self) -> tuple[DetectionInvestigationResult, ...]:
        self._require(WorkflowState.MODELS_READY)
        assert self._adapter is not None and self._bundle is not None
        contexts = tuple(TrustedObservationContext("demo-investigator", "demo-security-service", f"events/{event_id}") for event_id in self._bundle.event_ids)
        self._results = self._adapter.start_investigations(self._bundle.inference_frame, contexts)
        self.state = WorkflowState.DETECTED
        return self._results

    def start_investigation(self, event_id: str | None = None) -> DetectionInvestigationResult:
        self._require(WorkflowState.DETECTED)
        if event_id is None:
            selected = next((item for item in self._results if any((item.detection.logistic_prediction, item.detection.anomaly_prediction, item.detection.xgboost_prediction))), None)
        else:
            selected = next((item for item in self._results if item.detection.event_id == event_id), None)
        if selected is None:
            raise ValueError("a genuine detected event must be selected")
        self.selected = selected
        self.state = WorkflowState.INVESTIGATED
        return selected

    def run_red_simulation(self, scenario_id: str) -> RedAgentFinding:
        self._require(WorkflowState.INVESTIGATED)
        assert self.selected is not None
        self.finding = RedAgent().simulate(scenario_id, self.selected.investigation, self.environment)
        self.state = WorkflowState.RED_SIMULATED
        return self.finding

    def generate_blue_recommendations(self) -> tuple[DefensiveRecommendation, ...]:
        self._require(WorkflowState.RED_SIMULATED)
        assert self.finding is not None and self.selected is not None
        self.recommendations = BlueAgent().recommend(self.finding, self.selected.investigation)
        if not self.recommendations:
            self.state = WorkflowState.FAILED
            self.incident_status = IncidentStatus.VERIFICATION_FAILED
            return self.recommendations
        self.state = WorkflowState.BLUE_RECOMMENDED
        return self.recommendations

    def approve_recommendation(self, operator_identity: str, recommendation_id: str | None = None) -> ApprovalEvent:
        self._require(WorkflowState.BLUE_RECOMMENDED)
        recommendation = self._recommendation(recommendation_id)
        self.approval = ApprovalEvent(
            f"approval:{recommendation.recommendation_id}", recommendation.recommendation_id,
            recommendation.investigation_id, operator_identity, ApprovalDecision.APPROVED,
            datetime.now(timezone.utc).isoformat(),
        )
        self.state = WorkflowState.AWAITING_APPROVAL
        return self.approval

    def decline_recommendation(self, operator_identity: str, recommendation_id: str | None = None) -> ApprovalEvent:
        self._require(WorkflowState.BLUE_RECOMMENDED)
        recommendation = self._recommendation(recommendation_id)
        self.approval = ApprovalEvent(
            f"approval:{recommendation.recommendation_id}", recommendation.recommendation_id,
            recommendation.investigation_id, operator_identity, ApprovalDecision.DECLINED,
            datetime.now(timezone.utc).isoformat(),
        )
        return self.approval

    def apply_simulated_remediation(self) -> RemediationAuditEvent:
        self._require(WorkflowState.AWAITING_APPROVAL)
        recommendation = self._recommendation(None)
        assert self.approval is not None
        self.audit = self.environment.apply(recommendation, self.approval)
        self.state = WorkflowState.REMEDIATION_IN_PROGRESS
        return self.audit

    def verify_remediation(self) -> VerificationResult:
        self._require(WorkflowState.REMEDIATION_IN_PROGRESS)
        assert self.finding is not None and self.approval is not None
        self.verification = DemoVerifier().verify(self.finding, self._recommendation(None), self.approval, self.audit, self.environment)
        if self.verification.status is VerificationStatus.VERIFIED:
            self.state = WorkflowState.VERIFIED
            self.incident_status = IncidentStatus.SIMULATION_VERIFIED
        else:
            self.state = WorkflowState.FAILED
            self.incident_status = IncidentStatus.VERIFICATION_FAILED
        return self.verification

    def export_audit_report(self) -> dict[str, object]:
        report = {"workflow_state": self.state.value, "incident_status": self.incident_status.value, "environment": {"environment_id": self.environment.environment_id, "initial_state": self.environment.initial_state, "current_state": self.environment.current_state, "audit_history": [item.to_dict() for item in self.environment.audit_history]}, "telemetry": [] if self._bundle is None else list(self._bundle.telemetry_rows), "selected_event_id": None if self.selected is None else self.selected.detection.event_id, "investigation": None if self.selected is None else self.selected.investigation.to_dict(), "red_finding": None if self.finding is None else self.finding.to_dict(), "recommendations": [item.to_dict() for item in self.recommendations], "approval": None if self.approval is None else self.approval.to_dict(), "remediation_audit": None if self.audit is None else self.audit.to_dict(), "verification": None if self.verification is None else self.verification.to_dict()}
        json.dumps(report)
        return report

    def _require(self, state: WorkflowState) -> None:
        if self.state is not state:
            raise RuntimeError(f"invalid workflow transition from {self.state} to {state}")

    def _recommendation(self, recommendation_id: str | None) -> DefensiveRecommendation:
        if not self.recommendations:
            raise RuntimeError("no defensive recommendation is available")
        if recommendation_id is None:
            return self.recommendations[0]
        found = next((item for item in self.recommendations if item.recommendation_id == recommendation_id), None)
        if found is None:
            raise ValueError("unknown recommendation")
        return found
