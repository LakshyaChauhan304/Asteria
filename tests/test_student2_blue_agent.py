import unittest

from student2_blue_agent import BlueAgent
from student2_demo_environment import DemoEnvironment
from student2_red_agent import RedAgent
from student2_phase2_support import record


class BlueAgentTests(unittest.TestCase):
    def test_each_scenario_maps_to_its_only_supported_control(self):
        expected = {"exposed_api": "ingress_restricted", "credential_misuse": "service_credentials_rotated", "suspicious_egress": "egress_restricted"}
        for scenario, control in expected.items():
            with self.subTest(scenario=scenario):
                investigation = record()
                finding = RedAgent().simulate(scenario, investigation, DemoEnvironment())
                recommendation = BlueAgent().recommend(finding, investigation)[0]
                self.assertEqual(recommendation.target_control_id, control)
                self.assertTrue(recommendation.supporting_evidence_ids)
                self.assertEqual(recommendation.execution_mode, "simulation_only")
