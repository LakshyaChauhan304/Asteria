import unittest

from investigator import ObservationClass
from student2_demo_environment import DemoEnvironment
from student2_red_agent import AttackGraph, AttackGraphNode, RedAgent
from student2_phase2_support import record


class RedAgentTests(unittest.TestCase):
    def test_scenarios_are_reachable_and_recorded_as_inference(self):
        for scenario in ("exposed_api", "credential_misuse", "suspicious_egress"):
            with self.subTest(scenario=scenario):
                investigation = record()
                finding = RedAgent().simulate(scenario, investigation, DemoEnvironment())
                self.assertTrue(finding.candidate_path.reachable)
                self.assertTrue(finding.simulation_only)
                self.assertEqual(finding.investigation_id, investigation.investigation_id)
                self.assertEqual(investigation.observations[-1].classification, ObservationClass.INFERENCE)

    def test_malformed_graph_is_rejected(self):
        node = AttackGraphNode("a", "a", "asset", "zone", "low", "demo")
        with self.assertRaises(ValueError):
            AttackGraph((node,), (), "a", "missing")
