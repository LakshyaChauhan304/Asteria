import unittest

from student2_blue_agent import BlueAgent
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment
from student2_red_agent import RedAgent
from student2_phase2_support import record


class DemoEnvironmentTests(unittest.TestCase):
    def test_declined_approval_cannot_change_state_and_approved_is_audited(self):
        investigation, environment = record(), DemoEnvironment()
        recommendation = BlueAgent().recommend(RedAgent().simulate("exposed_api", investigation, environment), investigation)[0]
        declined = ApprovalEvent("declined", recommendation.recommendation_id, investigation.investigation_id, "operator", ApprovalDecision.DECLINED, "now")
        with self.assertRaises(PermissionError):
            environment.apply(recommendation, declined)
        self.assertFalse(environment.control_state("ingress_restricted"))
        approved = ApprovalEvent("approved", recommendation.recommendation_id, investigation.investigation_id, "operator", ApprovalDecision.APPROVED, "now")
        audit = environment.apply(recommendation, approved)
        self.assertTrue(environment.control_state("ingress_restricted"))
        self.assertEqual(audit.control_id, "ingress_restricted")
        with self.assertRaises(ValueError):
            environment.apply(recommendation, approved)
