"""Focused tests for the deterministic Student-2 Investigator slice."""

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_evidence import DetectionEvidence
from investigator import (
    READ_ONLY_INVESTIGATION_TOOL,
    AssessmentStatus,
    AuthorizationPreview,
    EvidenceReference,
    ExecutionOutcome,
    HypothesisStatus,
    InformationRequestStatus,
    InvestigationObservation,
    InvestigationRecord,
    InvestigationStatus,
    Investigator,
    ObservationClass,
    TrustedObservationContext,
)
from security_layer import (
    CapabilityAction,
    CapabilityPolicy,
    CapabilityRequest,
    CapabilityRule,
    DecisionState,
    DetectionEvidenceV1,
    SecurityDecision,
)


def make_evidence(*, model_prediction: int = 1, anomaly_prediction: int = 0) -> DetectionEvidenceV1:
    return DetectionEvidenceV1.from_dict(
        DetectionEvidence(
            event_id="evt-001",
            model_prediction=model_prediction,
            model_score=0.92,
            anomaly_prediction=anomaly_prediction,
            anomaly_score=0.18,
            protocol="tcp",
            service="http",
            state="FIN",
            features={"dur": 0.1, "spkts": 4},
        ).to_dict()
    )


class TrustedContextTests(unittest.TestCase):
    def test_valid_context_is_immutable(self) -> None:
        context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-001",
        )

        self.assertEqual(context.identity, "security-service")
        with self.assertRaises(AttributeError):
            context.identity = "attacker"

    def test_invalid_context_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrustedObservationContext(
                agent_id="investigator agent",
                identity="security-service",
                resource="events/evt-001",
            )

    def test_investigator_cannot_override_trusted_identity_or_scope(self) -> None:
        context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-001",
        )
        policy = CapabilityPolicy(
            rules=(
                CapabilityRule(
                    agent_id=context.agent_id,
                    identity=context.identity,
                    tool=READ_ONLY_INVESTIGATION_TOOL,
                    action=CapabilityAction.READ,
                    resource=context.resource,
                ),
            )
        )

        preview = Investigator(policy).investigate(make_evidence(), context)

        self.assertIsNotNone(preview.proposed_capability)
        assert preview.proposed_capability is not None
        self.assertEqual(preview.proposed_capability.identity, context.identity)
        self.assertEqual(preview.proposed_capability.resource, context.resource)


class InvestigatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-001",
        )
        self.rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
        )

    def test_suspicious_evidence_produces_deterministic_preview(self) -> None:
        investigator = Investigator(CapabilityPolicy(rules=(self.rule,)))

        first = investigator.investigate(make_evidence(anomaly_prediction=1), self.context)
        second = investigator.investigate(make_evidence(anomaly_prediction=1), self.context)

        self.assertEqual(first, second)
        self.assertTrue(first.review_trigger.review_required)
        self.assertEqual(
            first.observations,
            ("supervised_model_detection_signal", "anomaly_detection_signal"),
        )
        self.assertEqual(
            first.hypotheses,
            ("additional read-only context may help assess this event",),
        )

    def test_normal_evidence_does_not_invent_an_incident(self) -> None:
        preview = Investigator(CapabilityPolicy(rules=())).investigate(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )

        self.assertFalse(preview.review_trigger.review_required)
        self.assertEqual(preview.observations, ("no_positive_detection_signal",))
        self.assertEqual(preview.hypotheses, ())
        self.assertIsNone(preview.proposed_capability)
        self.assertIsNone(preview.decision)

    def test_evidence_is_not_mutated(self) -> None:
        evidence = make_evidence()
        before = evidence.to_dict()

        Investigator(CapabilityPolicy(rules=(self.rule,))).investigate(evidence, self.context)

        self.assertEqual(evidence.to_dict(), before)

    def test_only_read_capabilities_are_generated(self) -> None:
        preview = Investigator(CapabilityPolicy(rules=(self.rule,))).investigate(
            make_evidence(), self.context
        )

        assert preview.proposed_capability is not None
        self.assertEqual(preview.proposed_capability.action, CapabilityAction.READ)
        self.assertEqual(preview.proposed_capability.tool, READ_ONLY_INVESTIGATION_TOOL)

    def test_no_execution_result_exists(self) -> None:
        preview = Investigator(CapabilityPolicy(rules=(self.rule,))).investigate(
            make_evidence(), self.context
        )

        self.assertIsInstance(preview, AuthorizationPreview)
        self.assertFalse(hasattr(preview, "execution_result"))

    def test_arbitrary_callable_cannot_be_added_to_capability_request(self) -> None:
        with self.assertRaises(TypeError):
            CapabilityRequest(
                agent_id=self.context.agent_id,
                identity=self.context.identity,
                tool=READ_ONLY_INVESTIGATION_TOOL,
                action=CapabilityAction.READ,
                resource=self.context.resource,
                handler=lambda: None,
            )


class InvestigatorPolicyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-001",
        )

    def test_generated_request_can_be_allowed(self) -> None:
        rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
        )

        preview = Investigator(CapabilityPolicy(rules=(rule,))).investigate(
            make_evidence(), self.context
        )

        assert preview.decision is not None
        self.assertEqual(preview.decision.state, DecisionState.ALLOWED)

    def test_ungranted_generated_request_is_blocked(self) -> None:
        preview = Investigator(CapabilityPolicy(rules=())).investigate(
            make_evidence(), self.context
        )

        assert preview.decision is not None
        self.assertEqual(preview.decision.state, DecisionState.BLOCKED)

    def test_review_required_policy_is_preserved(self) -> None:
        rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
            requires_human_approval=True,
        )

        preview = Investigator(CapabilityPolicy(rules=(rule,))).investigate(
            make_evidence(), self.context
        )

        assert preview.decision is not None
        self.assertEqual(preview.decision.state, DecisionState.REVIEW_REQUIRED)


class StatefulInvestigatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedObservationContext(
            agent_id="investigator-agent",
            identity="security-service",
            resource="events/evt-001",
        )
        self.rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
        )

    def assert_state_unchanged(
        self,
        record: InvestigationRecord,
        error_type: type[Exception],
        operation: object,
    ) -> None:
        before = record.to_dict()
        with self.assertRaises(error_type):
            operation()
        self.assertEqual(record.to_dict(), before)

    def test_start_creates_stateful_record_with_competing_hypotheses(self) -> None:
        record = Investigator(CapabilityPolicy(rules=(self.rule,))).start_investigation(
            make_evidence(), self.context, investigation_id="AST-1000"
        )

        self.assertEqual(record.investigation_id, "AST-1000")
        self.assertEqual(record.status, InvestigationStatus.AWAITING_INFORMATION)
        self.assertEqual(record.evidence.event_id, "evt-001")
        self.assertEqual(len(record.hypotheses), 2)
        self.assertEqual(len(record.hypothesis_history), 2)
        self.assertTrue(record.observations)
        self.assertTrue(record.timeline)
        self.assertEqual(
            record.information_requests[0].status,
            InformationRequestStatus.AUTHORIZED,
        )
        self.assertEqual(
            record.authorization_history[0].execution_outcome,
            ExecutionOutcome.NOT_ATTEMPTED,
        )
        snapshot = record.to_dict()
        self.assertEqual(snapshot["initial_detection"]["reasons"], [
            "supervised_model_flag(score=0.9200)"
        ])

    def test_normal_start_has_no_authorization_request(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )

        self.assertEqual(record.status, InvestigationStatus.OPEN)
        self.assertEqual(record.information_requests, ())
        self.assertEqual(len(record.hypotheses), 2)
        self.assertTrue(any("not evidence" in item for item in record.limitations))

    def test_request_read_only_context_uses_bound_context(self) -> None:
        investigator = Investigator(CapabilityPolicy(rules=(self.rule,)))
        record = investigator.start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )

        request = investigator.request_read_only_context(record.investigation_id)

        self.assertEqual(request.capability.action, CapabilityAction.READ)
        self.assertEqual(request.capability.identity, self.context.identity)
        self.assertEqual(request.capability.resource, self.context.resource)
        self.assertEqual(request.status, InformationRequestStatus.AUTHORIZED)

    def test_record_rejects_forged_policy_preview_and_identity(self) -> None:
        investigator = Investigator(CapabilityPolicy(rules=(self.rule,)))
        evidence = make_evidence()
        preview = investigator.investigate(evidence, self.context)
        forged_request = CapabilityRequest(
            agent_id="attacker-agent",
            identity="attacker-identity",
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource="events/other",
        )
        forged_preview = AuthorizationPreview(
            evidence_event_id=evidence.event_id,
            evidence_contract_version=evidence.CONTRACT_VERSION,
            review_trigger=preview.review_trigger,
            observations=preview.observations,
            hypotheses=preview.hypotheses,
            proposed_capability=forged_request,
            decision=SecurityDecision(
                forged_request,
                DecisionState.ALLOWED,
                "forged",
                1,
            ),
        )

        with self.assertRaises(ValueError):
            InvestigationRecord(
                "AST-forged",
                evidence,
                self.context,
                forged_preview,
                investigator.policy,
            )

    def test_record_rejects_every_request_substitution(self) -> None:
        investigator = Investigator(CapabilityPolicy(rules=(self.rule,)))
        evidence = make_evidence()
        preview = investigator.investigate(evidence, self.context)
        substitutions = {
            "agent_id": "attacker-agent",
            "identity": "attacker-identity",
            "tool": "attacker_tool",
            "action": CapabilityAction.WRITE,
            "resource": "events/other",
        }

        for field_name, replacement in substitutions.items():
            with self.subTest(field_name=field_name):
                fields = {
                    "agent_id": self.context.agent_id,
                    "identity": self.context.identity,
                    "tool": READ_ONLY_INVESTIGATION_TOOL,
                    "action": CapabilityAction.READ,
                    "resource": self.context.resource,
                }
                fields[field_name] = replacement
                forged_request = CapabilityRequest(**fields)
                forged_preview = AuthorizationPreview(
                    evidence_event_id=evidence.event_id,
                    evidence_contract_version=evidence.CONTRACT_VERSION,
                    review_trigger=preview.review_trigger,
                    observations=preview.observations,
                    hypotheses=preview.hypotheses,
                    proposed_capability=forged_request,
                    decision=SecurityDecision(
                        forged_request,
                        DecisionState.ALLOWED,
                        "forged",
                        1,
                    ),
                )

                with self.assertRaises(ValueError):
                    InvestigationRecord(
                        f"AST-{field_name}",
                        evidence,
                        self.context,
                        forged_preview,
                        investigator.policy,
                    )

    def test_record_policy_snapshot_is_stable_after_caller_mutation(self) -> None:
        policy = CapabilityPolicy(rules=(self.rule,))
        investigator = Investigator(policy)
        record = investigator.start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )

        object.__setattr__(policy, "rules", ())

        request = investigator.request_read_only_context(record.investigation_id)
        self.assertEqual(request.decision.state, DecisionState.ALLOWED)
        self.assertEqual(request.capability.identity, self.context.identity)

    def test_record_public_request_api_rejects_caller_supplied_decisions(self) -> None:
        record = Investigator(CapabilityPolicy(rules=(self.rule,))).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        forged_request = CapabilityRequest(
            "attacker-agent",
            "attacker-identity",
            "attacker_tool",
            CapabilityAction.READ,
            "events/other",
        )
        forged_decision = SecurityDecision(
            forged_request,
            DecisionState.ALLOWED,
            "forged",
            1,
        )

        self.assert_state_unchanged(
            record,
            TypeError,
            lambda: record.request_information(forged_request, forged_decision),
        )

    def test_failed_fulfillment_does_not_append_an_observation(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        reference = EvidenceReference(
            "atomic:evt-001",
            "trusted.context",
            "evt-001",
            "atomicity test",
        )
        observation = InvestigationObservation(
            "O-atomic-1",
            "evt-001",
            "trusted.context",
            ObservationClass.FACT,
            "must not persist when fulfillment fails",
            (reference,),
        )
        self.assert_state_unchanged(
            record,
            KeyError,
            lambda: record.add_observation(observation, request_id="request:9999"),
        )

    def test_blocked_and_review_requests_cannot_be_fulfilled(self) -> None:
        blocked = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(), self.context
        )
        blocked_request = blocked.information_requests[0]
        self.assert_state_unchanged(
            blocked,
            ValueError,
            lambda: blocked.fulfill_information_request(
                blocked_request.request_id,
                blocked.observations[0].observation_id,
            ),
        )

        review_rule = CapabilityRule(
            agent_id=self.context.agent_id,
            identity=self.context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=self.context.resource,
            requires_human_approval=True,
        )
        review = Investigator(CapabilityPolicy(rules=(review_rule,))).start_investigation(
            make_evidence(), self.context
        )
        review_request = review.information_requests[0]
        self.assert_state_unchanged(
            review,
            ValueError,
            lambda: review.fulfill_information_request(
                review_request.request_id,
                review.observations[0].observation_id,
            ),
        )

    def test_observation_failure_paths_preserve_complete_state(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        reference = EvidenceReference(
            "atomic:evt-001",
            "trusted.context",
            "evt-001",
            "atomicity test",
        )
        observation = InvestigationObservation(
            "O-atomic-1",
            "evt-001",
            "trusted.context",
            ObservationClass.FACT,
            "valid observation",
            (reference,),
        )
        self.assert_state_unchanged(
            record,
            KeyError,
            lambda: record.fulfill_information_request(
                "request:9999", record.observations[0].observation_id
            ),
        )
        record.add_observation(observation)
        self.assert_state_unchanged(record, ValueError, lambda: record.add_observation(observation))

        invalid = InvestigationObservation(
            "O-invalid-1",
            "evt-001",
            "trusted.context",
            ObservationClass.FACT,
            "mutated after construction",
            (reference,),
        )
        object.__setattr__(invalid, "references", ())
        self.assert_state_unchanged(record, ValueError, lambda: record.add_observation(invalid))

        conflicting_reference = EvidenceReference(
            "atomic:evt-001",
            "other.context",
            "evt-001",
            "conflicting provenance",
        )
        conflicting = InvestigationObservation(
            "O-conflict-1",
            "evt-001",
            "other.context",
            ObservationClass.FACT,
            "conflicting evidence reference",
            (conflicting_reference,),
        )
        self.assert_state_unchanged(
            record,
            ValueError,
            lambda: record.add_observation(conflicting),
        )

    def test_unknown_hypothesis_evidence_is_rejected_without_state_change(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        hypothesis = record.hypotheses[0]

        self.assert_state_unchanged(
            record,
            ValueError,
            lambda: record.update_hypothesis(
                hypothesis.hypothesis_id,
                HypothesisStatus.SUPPORTED,
                supporting_evidence=("unknown:reference",),
            ),
        )

    def test_timeline_is_chronological_and_correlation_is_inference(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        reference = EvidenceReference(
            "telemetry:evt-002",
            "trusted.telemetry",
            "evt-002",
            "trusted observation",
        )
        later = InvestigationObservation(
            "O-external-1",
            "evt-002",
            "trusted.telemetry",
            ObservationClass.FACT,
            "later observation",
            (reference,),
            datetime(2026, 1, 2, tzinfo=UTC),
        )
        earlier = InvestigationObservation(
            "O-external-2",
            "evt-002",
            "trusted.telemetry",
            ObservationClass.FACT,
            "earlier observation",
            (reference,),
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        record.add_observation(later)
        record.add_observation(earlier)
        correlation = record.correlate_observations(
            (later.observation_id, earlier.observation_id),
            "two observations share the event scope",
        )

        known_times = [entry.occurred_at for entry in record.timeline if entry.occurred_at]
        self.assertEqual(known_times, sorted(known_times))
        self.assertEqual(correlation.classification, ObservationClass.INFERENCE)
        self.assertEqual(correlation.source, "investigator.correlation")

    def test_hypothesis_revision_preserves_history(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(), self.context
        )
        first = record.hypotheses[0]
        updated = record.update_hypothesis(
            first.hypothesis_id,
            HypothesisStatus.SUPPORTED,
            supporting_evidence=first.supporting_evidence,
        )

        self.assertEqual(updated.revision, 2)
        self.assertEqual(len(record.hypothesis_history), 3)
        self.assertEqual(record.hypothesis_history[0].status, HypothesisStatus.UNTESTED)

    def test_fulfilled_request_records_observation_without_execution(self) -> None:
        record = Investigator(CapabilityPolicy(rules=(self.rule,))).start_investigation(
            make_evidence(), self.context
        )
        request = record.information_requests[0]
        reference = EvidenceReference(
            "context:evt-001",
            "trusted.context",
            "evt-001",
            "externally supplied context",
        )
        observation = InvestigationObservation(
            "O-context-1",
            "evt-001",
            "trusted.context",
            ObservationClass.FACT,
            "context was supplied by an authorized source",
            (reference,),
        )

        record.add_observation(observation, request_id=request.request_id)

        self.assertEqual(
            record.information_requests[0].status,
            InformationRequestStatus.FULFILLED,
        )
        self.assertEqual(record.status, InvestigationStatus.OPEN)
        self.assertEqual(
            record.authorization_history[-1].execution_outcome,
            ExecutionOutcome.NOT_ATTEMPTED,
        )

    def test_state_transitions_and_assessment(self) -> None:
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            make_evidence(model_prediction=0, anomaly_prediction=0), self.context
        )
        basis = record.observations[0].observation_id
        assessment = record.finalize(
            AssessmentStatus.INCONCLUSIVE,
            "The available signals do not establish an incident.",
            basis_ids=(basis,),
            limitations=("No independent context was available.",),
        )

        self.assertEqual(assessment.status, AssessmentStatus.INCONCLUSIVE)
        self.assertEqual(record.status, InvestigationStatus.ASSESSED)
        record.close()
        self.assertEqual(record.status, InvestigationStatus.CLOSED)
        with self.assertRaises(ValueError):
            record.finalize(AssessmentStatus.SUPPORTED, "late assessment")
        with self.assertRaises(ValueError):
            record.transition_to(InvestigationStatus.OPEN)

    def test_contradictory_model_signal_is_recorded_as_limitation(self) -> None:
        evidence = DetectionEvidenceV1.from_dict(
            DetectionEvidence(
                event_id="evt-001",
                model_prediction=1,
                model_score=0.1,
                anomaly_prediction=0,
                anomaly_score=0.18,
                protocol="tcp",
                service="http",
                state="FIN",
                features={"dur": 0.1},
            ).to_dict()
        )
        record = Investigator(CapabilityPolicy(rules=())).start_investigation(
            evidence, self.context
        )

        self.assertTrue(any("inconsistent" in item for item in record.limitations))

    def test_malformed_stateful_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Investigator(CapabilityPolicy(rules=())).start_investigation(
                {"event_id": "evt-001"}, self.context  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            InvestigationObservation(
                "O-empty",
                "evt-001",
                "trusted.source",
                ObservationClass.FACT,
                "invalid empty references",
                (),
            )


if __name__ == "__main__":
    unittest.main()
