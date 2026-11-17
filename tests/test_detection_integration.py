"""Tests for the additive Student-1-to-Student-2 integration bridge."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_evidence import DetectionEvidence
from detection_integration import adapt_student1_evidence, start_investigation_from_detection
from investigator import READ_ONLY_INVESTIGATION_TOOL, Investigator, TrustedObservationContext
from security_layer import CapabilityAction, CapabilityPolicy, CapabilityRule, DecisionState


def student1_payload() -> dict[str, object]:
    return DetectionEvidence(
        event_id="evt-integration",
        model_prediction=1,
        model_score=0.91,
        anomaly_prediction=0,
        anomaly_score=0.12,
        protocol="tcp",
        service="http",
        state="FIN",
        features={"dur": 0.1, "spkts": 4},
    ).to_dict()


class DetectionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-integration",
        )
        self.rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
        )

    def test_adapter_seals_the_established_student1_dictionary_contract(self) -> None:
        evidence = adapt_student1_evidence(student1_payload())

        self.assertEqual(evidence.event_id, "evt-integration")
        self.assertEqual(evidence.model_score, 0.91)
        self.assertEqual(evidence.features, (("dur", 0.1), ("spkts", 4)))

    def test_integration_starts_authorized_dry_run_investigation(self) -> None:
        record = start_investigation_from_detection(
            student1_payload(),
            self.context,
            Investigator(CapabilityPolicy(rules=(self.rule,))),
            investigation_id="AST-integration",
        )

        self.assertEqual(record.investigation_id, "AST-integration")
        self.assertEqual(record.evidence.event_id, "evt-integration")
        self.assertEqual(record.information_requests[0].decision.state, DecisionState.ALLOWED)
        self.assertEqual(record.authorization_history[0].execution_outcome.value, "not_attempted")

    def test_adapter_rejects_custom_evidence_objects_without_calling_them(self) -> None:
        class ExplodingPayload:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError(f"unexpected attribute access: {name}")

        with self.assertRaises(TypeError):
            adapt_student1_evidence(ExplodingPayload())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
