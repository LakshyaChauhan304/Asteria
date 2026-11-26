"""Real engine reports at each workflow stage, for UI-module tests (no ML models needed)."""

from dataclasses import asdict
from datetime import datetime, timezone

from student2_blue_agent import BlueAgent
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment
from student2_demo_verifier import DemoVerifier
from student2_red_agent import RedAgent, graph_exposure, scenario_definition
from student2_phase2_support import record


def _base(stage, scenario="exposed_api", decision=ApprovalDecision.APPROVED, note=""):
    if stage == "declined":
        decision = ApprovalDecision.DECLINED
    """Build the same report shape DemoOrchestrator.export_audit_report() produces."""
    investigation, environment = record(), DemoEnvironment()
    report = {
        "workflow_state": "investigated", "incident_status": "open",
        "environment": {}, "telemetry": [], "detection_cases": [],
        "selected_event_id": "UNSW-1", "investigation": investigation.to_dict(),
        "red_finding": None, "attack_graph": {}, "recommendations": [], "approval": None,
        "remediation_audit": None, "verification": None, "decision_history": [],
        "scenario_suggestion": {
            "scenario_id": scenario, "rationale": "No credential-bearing service or outbound-heavy pattern was observed, so the exposed-service path is the default simulated scenario.",
            "basis": ["service=http", "protocol=tcp"], "inferred": True,
        },
        "graph_exposure": {},
    }
    finding = recommendation = approval = audit = verification = None
    history = []
    if stage != "investigated":
        finding = RedAgent().simulate(scenario, investigation, environment)
        report["workflow_state"] = "red_simulated"
    if stage not in ("investigated", "red"):
        recommendation = BlueAgent().recommend(finding, investigation)[0]
        report["workflow_state"] = "blue_recommended"
        report["recommendations"] = [recommendation.to_dict()]
    if stage in ("approved", "applied", "verified", "declined"):
        approval = ApprovalEvent(
            "approval:1", recommendation.recommendation_id, investigation.investigation_id,
            "demo-operator", decision, datetime.now(timezone.utc).isoformat(), note=note)
        history.append(approval)
        report["workflow_state"] = "awaiting_approval" if decision is ApprovalDecision.APPROVED else "blue_recommended"
        report["approval"] = approval.to_dict()
    if stage in ("applied", "verified"):
        audit = environment.apply(recommendation, approval)
        report["workflow_state"] = "remediation_in_progress"
        report["remediation_audit"] = audit.to_dict()
    if stage == "verified":
        verification = DemoVerifier().verify(finding, recommendation, approval, audit, environment)
        report["workflow_state"] = "verified"
        report["incident_status"] = "simulation_verified"
        report["verification"] = verification.to_dict()
    if finding is not None:
        sc = scenario_definition(scenario)
        report["red_finding"] = finding.to_dict()
        report["attack_graph"] = {
            "nodes": [asdict(n) for n in sc.graph.nodes], "edges": [asdict(e) for e in sc.graph.edges],
            "candidate_path": finding.to_dict()["candidate_path"], "control_id": sc.control_id}
        report["graph_exposure"] = graph_exposure(sc.graph, environment.current_state)
    report["decision_history"] = [h.to_dict() for h in history]
    report["environment"] = {
        "environment_id": environment.environment_id, "initial_state": environment.initial_state,
        "current_state": environment.current_state,
        "audit_history": [a.to_dict() for a in environment.audit_history]}
    return report


def stage_report(stage, **kw):
    return _base(stage, **kw)
