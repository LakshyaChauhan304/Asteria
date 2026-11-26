import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment
from student2_demo_orchestrator import DemoOrchestrator, WorkflowState
from student2_demo_verifier import DemoVerifier, VerificationOutcome, VerificationStatus
from student2_phase2_support import record
from student2_red_agent import (
    AttackGraph, AttackGraphEdge, AttackGraphNode, RedAgent, ScenarioDefinition,
    evaluate_reachability, graph_exposure, scenario_definition, scenario_ids,
)
from student2_scenario_selector import suggest_scenario


def node(name):
    return AttackGraphNode(name, name, "asset", "zone", "low", "demo")


def edge(name, source, target, controls=()):
    return AttackGraphEdge(name, source, target, "link", ("c",), tuple(controls))


def two_route_graph(second_route_controlled):
    nodes = tuple(node(name) for name in ("a", "b", "c", "t"))
    edges = (
        edge("e1", "a", "b", ("ctl",)), edge("e2", "b", "t", ("ctl",)),
        edge("e3", "a", "c", ("ctl",) if second_route_controlled else ()),
        edge("e4", "c", "t", ("ctl",) if second_route_controlled else ()),
    )
    return AttackGraph(nodes, edges, "a", "t")


class GraphEngineTests(unittest.TestCase):
    def test_unblocked_alternate_route_keeps_target_reachable(self):
        path = evaluate_reachability(two_route_graph(False), {"ctl": True})
        self.assertTrue(path.reachable)
        self.assertEqual(path.node_ids, ("a", "c", "t"))

    def test_all_routes_blocked_returns_structural_path_as_unreachable(self):
        path = evaluate_reachability(two_route_graph(True), {"ctl": True})
        self.assertFalse(path.reachable)
        self.assertEqual(path.node_ids[0], "a")
        self.assertEqual(path.node_ids[-1], "t")

    def test_structurally_disconnected_graph_is_rejected(self):
        graph = AttackGraph((node("a"), node("t")), (), "a", "t")
        with self.assertRaises(ValueError):
            evaluate_reachability(graph, {})

    def test_scenarios_have_alternate_routes_and_decoys(self):
        for scenario_id in scenario_ids():
            with self.subTest(scenario=scenario_id):
                scenario = scenario_definition(scenario_id)
                controls = DemoEnvironment().current_state
                before = graph_exposure(scenario.graph, controls)
                self.assertGreater(len(scenario.graph.nodes), 4)
                self.assertTrue(before["target_reachable"])
                self.assertTrue(before["decoy_node_ids"])
                self.assertGreater(len(before["route_node_ids"]), 4 - 1)
                controls[scenario.control_id] = True
                after = graph_exposure(scenario.graph, controls)
                self.assertFalse(after["target_reachable"])
                self.assertTrue(after["blocked_edge_ids"])
                self.assertFalse(evaluate_reachability(scenario.graph, controls).reachable)
                json.dumps(after)


class ScenarioSelectorTests(unittest.TestCase):
    def test_rules_are_explainable_and_use_flow_fields_only(self):
        self.assertEqual(suggest_scenario({"service": "ftp"}).scenario_id, "credential_misuse")
        outbound = suggest_scenario({"service": "-", "source_bytes": 9000, "destination_bytes": 100})
        self.assertEqual(outbound.scenario_id, "suspicious_egress")
        default = suggest_scenario({"service": "http", "protocol": "tcp", "source_bytes": 200, "destination_bytes": 300})
        self.assertEqual(default.scenario_id, "exposed_api")
        self.assertTrue(default.inferred)
        self.assertTrue(default.basis)

    def test_malformed_values_fall_back_to_default(self):
        self.assertEqual(suggest_scenario({"source_bytes": "n/a"}).scenario_id, "exposed_api")


class VerifierOutcomeTests(unittest.TestCase):
    def setup(self):
        from student2_blue_agent import BlueAgent
        investigation, environment = record(), DemoEnvironment()
        finding = RedAgent().simulate("exposed_api", investigation, environment)
        recommendation = BlueAgent().recommend(finding, investigation)[0]
        approval = ApprovalEvent("approved", recommendation.recommendation_id, investigation.investigation_id, "operator", ApprovalDecision.APPROVED, "now")
        return finding, recommendation, approval, environment

    def test_blocked_outcome_records_steps(self):
        finding, recommendation, approval, environment = self.setup()
        audit = environment.apply(recommendation, approval)
        result = DemoVerifier().verify(finding, recommendation, approval, audit, environment)
        self.assertEqual(result.outcome, VerificationOutcome.BLOCKED.value)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertGreaterEqual(len(result.steps), 6)
        self.assertTrue(all(step.passed for step in result.steps))
        json.dumps(result.to_dict())

    def test_missing_approval_fails_without_claiming_success(self):
        finding, recommendation, approval, environment = self.setup()
        audit = environment.apply(recommendation, approval)
        result = DemoVerifier().verify(finding, recommendation, None, audit, environment)
        self.assertEqual(result.outcome, VerificationOutcome.FAILED.value)
        self.assertFalse(result.steps[0].passed)

    def test_open_alternate_route_reports_still_reachable(self):
        finding, recommendation, approval, environment = self.setup()
        audit = environment.apply(recommendation, approval)
        base = scenario_definition("exposed_api")
        leaky_edges = base.graph.edges + (
            AttackGraphEdge("exposed_api:leak", base.graph.nodes[0].node_id, base.graph.nodes[3].node_id, "alternate_route", ("c",), ()),
        )
        leaky = ScenarioDefinition("exposed_api", AttackGraph(base.graph.nodes, leaky_edges, base.graph.entry_node_id, base.graph.target_node_id), base.control_id)
        with patch("student2_demo_verifier.scenario_definition", return_value=leaky):
            result = DemoVerifier().verify(finding, recommendation, approval, audit, environment)
        self.assertEqual(result.outcome, VerificationOutcome.STILL_REACHABLE.value)
        self.assertEqual(result.status, VerificationStatus.FAILED)


class DecisionFlowTests(unittest.TestCase):
    def workflow(self):
        flow = DemoOrchestrator(Path.cwd())
        flow.selected = SimpleNamespace(investigation=record(), detection=SimpleNamespace(event_id="UNSW-1"))
        flow.state = WorkflowState.INVESTIGATED
        flow.run_red_simulation("exposed_api")
        flow.generate_blue_recommendations()
        return flow

    def test_only_approval_changes_state_and_every_decision_is_logged(self):
        flow = self.workflow()
        before = flow.environment.current_state
        flow.record_decision("analyst", ApprovalDecision.MORE_EVIDENCE_REQUESTED, "need netflow")
        flow.record_decision("analyst", ApprovalDecision.ESCALATED, "tier 2")
        flow.decline_recommendation("analyst", note="not now")
        self.assertEqual(flow.state, WorkflowState.BLUE_RECOMMENDED)
        self.assertEqual(flow.environment.current_state, before)
        self.assertFalse(flow.environment.audit_history)
        self.assertEqual(flow.approval.decision, ApprovalDecision.DECLINED)
        flow.approve_recommendation("analyst", note="approved after review")
        self.assertEqual(flow.state, WorkflowState.AWAITING_APPROVAL)
        flow.apply_simulated_remediation()
        result = flow.verify_remediation()
        self.assertEqual(result.outcome, "blocked")
        report = flow.export_audit_report()
        self.assertEqual([d["decision"] for d in report["decision_history"]], ["more_evidence_requested", "escalated", "declined", "approved"])
        self.assertTrue(report["graph_exposure"]["blocked_edge_ids"])
        self.assertFalse(report["graph_exposure"]["target_reachable"])
        json.dumps(report)

    def test_overlong_note_is_rejected(self):
        flow = self.workflow()
        with self.assertRaises(ValueError):
            flow.approve_recommendation("analyst", note="x" * 501)
        self.assertEqual(flow.state, WorkflowState.BLUE_RECOMMENDED)

    def test_non_approval_cannot_be_applied(self):
        flow = self.workflow()
        flow.record_decision("analyst", ApprovalDecision.ESCALATED)
        with self.assertRaises(RuntimeError):
            flow.apply_simulated_remediation()


if __name__ == "__main__":
    unittest.main()
