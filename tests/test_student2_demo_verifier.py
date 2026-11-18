import unittest

from student2_blue_agent import BlueAgent
from student2_demo_environment import ApprovalDecision, ApprovalEvent, DemoEnvironment
from student2_demo_verifier import DemoVerifier, VerificationStatus
from student2_red_agent import RedAgent
from student2_phase2_support import record


class DemoVerifierTests(unittest.TestCase):
    def test_success_and_missing_approval_are_distinguished(self):
        investigation, environment = record(), DemoEnvironment()
        finding = RedAgent().simulate("exposed_api", investigation, environment)
        recommendation = BlueAgent().recommend(finding, investigation)[0]
        approval = ApprovalEvent("approved", recommendation.recommendation_id, investigation.investigation_id, "operator", ApprovalDecision.APPROVED, "now")
        audit = environment.apply(recommendation, approval)
        self.assertEqual(DemoVerifier().verify(finding, recommendation, approval, audit, environment).status, VerificationStatus.VERIFIED)
        self.assertEqual(DemoVerifier().verify(finding, recommendation, None, audit, environment).status, VerificationStatus.FAILED)
