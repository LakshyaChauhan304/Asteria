"""Focused tests for Student-2 boundary hardening."""

import sys
import unittest
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_evidence import DetectionEvidence
from security_layer import (
    CapabilityAction,
    CapabilityPolicy,
    CapabilityRequest,
    CapabilityRule,
    DecisionState,
    DetectionEvidenceV1,
    review_trigger_from_evidence,
)


def student1_evidence_dict() -> dict[str, object]:
    """Return the exact shape emitted by the current Student-1 contract."""

    return DetectionEvidence(
        event_id="evt-001",
        model_prediction=1,
        model_score=0.92,
        anomaly_prediction=0,
        anomaly_score=0.18,
        protocol="tcp",
        service="http",
        state="FIN",
        features={"dur": 0.1, "spkts": 4},
    ).to_dict()


class EvidenceBoundaryTests(unittest.TestCase):
    """Verify immutable, exact, non-executable evidence ingestion."""

    def test_valid_student1_evidence_is_accepted(self) -> None:
        evidence = DetectionEvidenceV1.from_dict(student1_evidence_dict())

        self.assertEqual(evidence.event_id, "evt-001")
        self.assertEqual(evidence.features, (("dur", 0.1), ("spkts", 4)))
        self.assertEqual(
            evidence.to_dict()["contract_version"],
            DetectionEvidenceV1.CONTRACT_VERSION,
        )

    def test_missing_required_field_is_rejected(self) -> None:
        payload = student1_evidence_dict()
        del payload["state"]

        with self.assertRaisesRegex(ValueError, "missing fields"):
            DetectionEvidenceV1.from_dict(payload)

    def test_invalid_type_is_rejected(self) -> None:
        payload = student1_evidence_dict()
        payload["model_prediction"] = "1"

        with self.assertRaises(ValueError):
            DetectionEvidenceV1.from_dict(payload)

    def test_unexpected_field_is_rejected_safely(self) -> None:
        payload = student1_evidence_dict()
        payload["unexpected"] = "ignored is not safe"

        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            DetectionEvidenceV1.from_dict(payload)

    def test_original_evidence_is_not_mutated(self) -> None:
        payload = student1_evidence_dict()
        original_features = payload["features"]
        evidence = DetectionEvidenceV1.from_dict(payload)

        original_features["dur"] = 9.9
        payload["event_id"] = "changed-after-boundary"

        self.assertEqual(evidence.event_id, "evt-001")
        self.assertEqual(dict(evidence.features)["dur"], 0.1)

    def test_custom_mapping_and_provider_are_rejected_without_callbacks(self) -> None:
        class ExplodingMapping(dict):
            def __iter__(self):
                raise AssertionError("custom mapping was executed")

        class ExplodingProvider:
            def to_dict(self):
                raise AssertionError("custom provider was executed")

        with self.assertRaises(TypeError):
            DetectionEvidenceV1.from_dict(ExplodingMapping(student1_evidence_dict()))

        with self.assertRaises(TypeError):
            review_trigger_from_evidence(ExplodingProvider())

    def test_review_trigger_requires_sealed_evidence(self) -> None:
        evidence = DetectionEvidenceV1.from_dict(student1_evidence_dict())
        trigger = review_trigger_from_evidence(evidence)

        self.assertTrue(trigger.review_required)
        self.assertEqual(trigger.evidence_contract_version, DetectionEvidenceV1.CONTRACT_VERSION)
        self.assertEqual(trigger.reasons, ("supervised_model_flag(score=0.9200)",))

    def test_mapping_proxy_cannot_be_injected_as_evidence(self) -> None:
        with self.assertRaises(TypeError):
            DetectionEvidenceV1.from_dict(MappingProxyType(student1_evidence_dict()))


class PolicyBoundaryTests(unittest.TestCase):
    """Verify validated, exact-match, deny-by-default policy behavior."""

    def setUp(self) -> None:
        self.read_rule = CapabilityRule(
            agent_id="finance-agent",
            identity="finance-service",
            tool="read_documents",
            action=CapabilityAction.READ,
            resource="public/",
        )
        self.write_rule = CapabilityRule(
            agent_id="finance-agent",
            identity="finance-service",
            tool="write_file",
            action=CapabilityAction.WRITE,
            resource="public/",
            requires_human_approval=True,
        )
        self.policy = CapabilityPolicy(rules=(self.read_rule, self.write_rule))

    def test_valid_enum_action_is_accepted(self) -> None:
        request = CapabilityRequest(
            agent_id="finance-agent",
            identity="finance-service",
            tool="read_documents",
            action=CapabilityAction.READ,
            resource="public/",
        )

        self.assertEqual(self.policy.evaluate(request).state, DecisionState.ALLOWED)

    def test_invalid_raw_action_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            CapabilityRequest(
                agent_id="finance-agent",
                identity="finance-service",
                tool="read_documents",
                action="read",
                resource="public/",
            )

    def test_malformed_rule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapabilityRule(
                agent_id="finance agent",
                identity="finance-service",
                tool="read_documents",
                action=CapabilityAction.READ,
                resource="public/",
            )

        with self.assertRaises(ValueError):
            CapabilityRule(
                agent_id="finance-agent",
                identity="finance-service",
                tool="write_file",
                action=CapabilityAction.WRITE,
                resource="public/",
            )

    def test_unknown_capability_is_blocked(self) -> None:
        request = CapabilityRequest(
            agent_id="finance-agent",
            identity="finance-service",
            tool="read_documents",
            action=CapabilityAction.READ,
            resource="restricted/",
        )

        self.assertEqual(self.policy.evaluate(request).state, DecisionState.BLOCKED)

    def test_state_changing_capability_requires_review(self) -> None:
        request = CapabilityRequest(
            agent_id="finance-agent",
            identity="finance-service",
            tool="write_file",
            action=CapabilityAction.WRITE,
            resource="public/",
        )

        self.assertEqual(self.policy.evaluate(request).state, DecisionState.REVIEW_REQUIRED)

    def test_exact_match_is_preserved(self) -> None:
        request = CapabilityRequest(
            agent_id="finance-agent",
            identity="finance-service",
            tool="read_documents",
            action=CapabilityAction.READ,
            resource="public/index.md",
        )

        self.assertEqual(self.policy.evaluate(request).state, DecisionState.BLOCKED)

    def test_policy_rejects_mutable_rules_and_custom_rule_objects(self) -> None:
        class ExplodingRule:
            def matches(self, _request):
                raise AssertionError("custom rule was executed")

        with self.assertRaises(TypeError):
            CapabilityPolicy(rules=[self.read_rule])

        with self.assertRaises(TypeError):
            CapabilityPolicy(rules=(ExplodingRule(),))

    def test_duplicate_rules_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapabilityPolicy(rules=(self.read_rule, self.read_rule))

    def test_policy_rejects_non_request_objects(self) -> None:
        with self.assertRaises(TypeError):
            self.policy.evaluate({"agent_id": "finance-agent"})


if __name__ == "__main__":
    unittest.main()

