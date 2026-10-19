"""Deterministic, proposal-only Student-2 investigation primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Final

from security_layer import (
    CapabilityAction,
    CapabilityPolicy,
    CapabilityRequest,
    DecisionState,
    DetectionEvidenceV1,
    DetectionReviewTrigger,
    SecurityDecision,
    review_trigger_from_evidence,
)

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}")
READ_ONLY_INVESTIGATION_TOOL: Final[str] = "read_detection_context"


class InvestigationStatus(StrEnum):
    OPEN = "open"
    AWAITING_INFORMATION = "awaiting_information"
    ASSESSED = "assessed"
    CLOSED = "closed"


class ObservationClass(StrEnum):
    FACT = "fact"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


class HypothesisStatus(StrEnum):
    UNTESTED = "untested"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class InformationRequestStatus(StrEnum):
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"
    AUTHORIZED = "authorized"
    FULFILLED = "fulfilled"


class ExecutionOutcome(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AssessmentStatus(StrEnum):
    SIGNALS_PRESENT = "signals_present"
    INCONCLUSIVE = "inconclusive"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class TrustedObservationContext:
    """Trusted identity and resource scope supplied outside the investigator."""

    agent_id: str
    identity: str
    resource: str

    def __post_init__(self) -> None:
        _validate_identifier("agent_id", self.agent_id)
        _validate_identifier("identity", self.identity)
        _validate_identifier("resource", self.resource)


@dataclass(frozen=True)
class AuthorizationPreview:
    """Investigation observations and authorization, with no execution state."""

    evidence_event_id: str
    evidence_contract_version: str
    review_trigger: DetectionReviewTrigger
    observations: tuple[str, ...]
    hypotheses: tuple[str, ...]
    proposed_capability: CapabilityRequest | None
    decision: SecurityDecision | None

    def __post_init__(self) -> None:
        _validate_text("evidence_event_id", self.evidence_event_id)
        _validate_text("evidence_contract_version", self.evidence_contract_version)
        if type(self.review_trigger) is not DetectionReviewTrigger:
            raise TypeError("review_trigger must be a DetectionReviewTrigger")
        if self.review_trigger.event_id != self.evidence_event_id:
            raise ValueError("review trigger must reference the evidence event")
        if self.review_trigger.evidence_contract_version != self.evidence_contract_version:
            raise ValueError("review trigger must reference the evidence contract")
        _validate_text_tuple("observations", self.observations)
        _validate_text_tuple("hypotheses", self.hypotheses)
        if (
            self.proposed_capability is not None
            and type(self.proposed_capability) is not CapabilityRequest
        ):
            raise TypeError("proposed_capability must be a CapabilityRequest or None")
        if self.decision is not None and type(self.decision) is not SecurityDecision:
            raise TypeError("decision must be a SecurityDecision or None")
        if self.proposed_capability is None and self.decision is not None:
            raise ValueError("decision requires a proposed capability")
        if self.proposed_capability is not None and self.decision is not None:
            if self.decision.request != self.proposed_capability:
                raise ValueError("decision must describe the proposed capability")


@dataclass(frozen=True)
class EvidenceReference:
    reference_id: str
    source: str
    event_id: str
    description: str

    def __post_init__(self) -> None:
        _validate_identifier("reference_id", self.reference_id)
        _validate_identifier("source", self.source)
        _validate_identifier("event_id", self.event_id)
        _validate_text("description", self.description)


@dataclass(frozen=True)
class InvestigationObservation:
    observation_id: str
    event_id: str
    source: str
    classification: ObservationClass
    summary: str
    references: tuple[EvidenceReference, ...]
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_identifier("observation_id", self.observation_id)
        _validate_identifier("event_id", self.event_id)
        _validate_identifier("source", self.source)
        if type(self.classification) is not ObservationClass:
            raise TypeError("classification must be an ObservationClass")
        _validate_text("summary", self.summary)
        _validate_reference_tuple(self.references)
        _validate_timestamp(self.occurred_at)


@dataclass(frozen=True)
class TimelineEntry:
    sequence: int
    observation_id: str
    event_id: str
    source: str
    summary: str
    evidence_reference_ids: tuple[str, ...]
    occurred_at: datetime | None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        _validate_identifier("observation_id", self.observation_id)
        _validate_identifier("event_id", self.event_id)
        _validate_identifier("source", self.source)
        _validate_text("summary", self.summary)
        _validate_identifier_tuple("evidence_reference_ids", self.evidence_reference_ids)
        _validate_timestamp(self.occurred_at)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    classification: ObservationClass
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    revision: int = 1

    def __post_init__(self) -> None:
        _validate_identifier("hypothesis_id", self.hypothesis_id)
        _validate_text("statement", self.statement)
        if type(self.classification) is not ObservationClass:
            raise TypeError("classification must be an ObservationClass")
        _validate_identifier_tuple("supporting_evidence", self.supporting_evidence)
        _validate_identifier_tuple("contradicting_evidence", self.contradicting_evidence)
        if type(self.status) is not HypothesisStatus:
            raise TypeError("status must be a HypothesisStatus")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be a positive integer")


@dataclass(frozen=True)
class InformationRequest:
    request_id: str
    capability: CapabilityRequest
    decision: SecurityDecision
    status: InformationRequestStatus
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier("request_id", self.request_id)
        if type(self.capability) is not CapabilityRequest:
            raise TypeError("capability must be a CapabilityRequest")
        if type(self.decision) is not SecurityDecision:
            raise TypeError("decision must be a SecurityDecision")
        if self.decision.request != self.capability:
            raise ValueError("decision must describe the capability")
        if type(self.status) is not InformationRequestStatus:
            raise TypeError("status must be an InformationRequestStatus")
        _validate_identifier_tuple("observation_ids", self.observation_ids)
        allowed_statuses = {
            DecisionState.BLOCKED: {InformationRequestStatus.BLOCKED},
            DecisionState.REVIEW_REQUIRED: {InformationRequestStatus.REVIEW_REQUIRED},
            DecisionState.ALLOWED: {
                InformationRequestStatus.AUTHORIZED,
                InformationRequestStatus.FULFILLED,
            },
        }[self.decision.state]
        if self.status not in allowed_statuses:
            raise ValueError("request status does not match its security decision")
        if self.status is not InformationRequestStatus.FULFILLED and self.observation_ids:
            raise ValueError("only fulfilled requests may reference observations")


@dataclass(frozen=True)
class AuthorizationRecord:
    request_id: str
    request: CapabilityRequest
    decision: SecurityDecision
    execution_outcome: ExecutionOutcome = ExecutionOutcome.NOT_ATTEMPTED

    def __post_init__(self) -> None:
        _validate_identifier("request_id", self.request_id)
        if type(self.request) is not CapabilityRequest:
            raise TypeError("request must be a CapabilityRequest")
        if type(self.decision) is not SecurityDecision:
            raise TypeError("decision must be a SecurityDecision")
        if self.decision.request != self.request:
            raise ValueError("decision must describe the request")
        if type(self.execution_outcome) is not ExecutionOutcome:
            raise TypeError("execution_outcome must be an ExecutionOutcome")


@dataclass(frozen=True)
class FinalAssessment:
    status: AssessmentStatus
    statement: str
    basis_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not AssessmentStatus:
            raise TypeError("status must be an AssessmentStatus")
        _validate_text("statement", self.statement)
        _validate_identifier_tuple("basis_ids", self.basis_ids)
        _validate_text_tuple("limitations", self.limitations)


class InvestigationRecord:
    """Append-only investigation state with explicit, non-executing transitions."""

    def __init__(
        self,
        investigation_id: str,
        evidence: DetectionEvidenceV1,
        context: TrustedObservationContext,
        preview: AuthorizationPreview,
    ) -> None:
        _validate_identifier("investigation_id", investigation_id)
        if type(evidence) is not DetectionEvidenceV1:
            raise TypeError("evidence must be a DetectionEvidenceV1")
        if type(context) is not TrustedObservationContext:
            raise TypeError("context must be a TrustedObservationContext")
        if type(preview) is not AuthorizationPreview:
            raise TypeError("preview must be an AuthorizationPreview")
        if preview.evidence_event_id != evidence.event_id:
            raise ValueError("preview must reference the evidence")

        self._investigation_id = investigation_id
        self._evidence = evidence
        self._context = context
        self._preview = preview
        self._status = InvestigationStatus.OPEN
        self._evidence_references: dict[str, EvidenceReference] = {}
        self._observations: dict[str, InvestigationObservation] = {}
        self._timeline: list[TimelineEntry] = []
        self._hypotheses: dict[str, Hypothesis] = {}
        self._hypothesis_history: list[Hypothesis] = []
        self._information_requests: dict[str, InformationRequest] = {}
        self._information_request_history: list[InformationRequest] = []
        self._authorization_history: list[AuthorizationRecord] = []
        self._unresolved_questions: list[str] = []
        self._limitations: list[str] = []
        self._final_assessment: FinalAssessment | None = None
        self._seed_from_evidence()
        if preview.proposed_capability is not None and preview.decision is not None:
            self.request_information(preview.proposed_capability, preview.decision)

    @property
    def investigation_id(self) -> str:
        return self._investigation_id

    @property
    def status(self) -> InvestigationStatus:
        return self._status

    @property
    def evidence(self) -> DetectionEvidenceV1:
        return self._evidence

    @property
    def context(self) -> TrustedObservationContext:
        return self._context

    @property
    def initial_detection(self) -> DetectionReviewTrigger:
        return self._preview.review_trigger

    @property
    def evidence_references(self) -> tuple[EvidenceReference, ...]:
        return tuple(self._evidence_references.values())

    @property
    def observations(self) -> tuple[InvestigationObservation, ...]:
        return tuple(self._observations.values())

    @property
    def timeline(self) -> tuple[TimelineEntry, ...]:
        return tuple(
            sorted(
                self._timeline,
                key=lambda item: (
                    item.occurred_at is None,
                    item.occurred_at or datetime.max,
                    item.sequence,
                ),
            )
        )

    @property
    def hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(self._hypotheses.values())

    @property
    def hypothesis_history(self) -> tuple[Hypothesis, ...]:
        return tuple(self._hypothesis_history)

    @property
    def information_requests(self) -> tuple[InformationRequest, ...]:
        return tuple(self._information_requests.values())

    @property
    def information_request_history(self) -> tuple[InformationRequest, ...]:
        return tuple(self._information_request_history)

    @property
    def authorization_history(self) -> tuple[AuthorizationRecord, ...]:
        return tuple(self._authorization_history)

    @property
    def unresolved_questions(self) -> tuple[str, ...]:
        return tuple(self._unresolved_questions)

    @property
    def limitations(self) -> tuple[str, ...]:
        return tuple(self._limitations)

    @property
    def final_assessment(self) -> FinalAssessment | None:
        return self._final_assessment

    def transition_to(self, status: InvestigationStatus) -> None:
        if type(status) is not InvestigationStatus:
            raise TypeError("status must be an InvestigationStatus")
        if status is self._status:
            return
        transitions = {
            InvestigationStatus.OPEN: {
                InvestigationStatus.AWAITING_INFORMATION,
                InvestigationStatus.ASSESSED,
            },
            InvestigationStatus.AWAITING_INFORMATION: {
                InvestigationStatus.OPEN,
                InvestigationStatus.ASSESSED,
            },
            InvestigationStatus.ASSESSED: {InvestigationStatus.CLOSED},
            InvestigationStatus.CLOSED: set(),
        }
        if status not in transitions[self._status]:
            raise ValueError(f"invalid investigation transition: {self._status} -> {status}")
        self._status = status

    def add_observation(
        self,
        observation: InvestigationObservation,
        request_id: str | None = None,
    ) -> None:
        if type(observation) is not InvestigationObservation:
            raise TypeError("observation must be an InvestigationObservation")
        self._ensure_open_for_mutation()
        if observation.observation_id in self._observations:
            raise ValueError("observation_id must be unique")
        for reference in observation.references:
            self._evidence_references.setdefault(reference.reference_id, reference)
        self._observations[observation.observation_id] = observation
        self._timeline.append(
            TimelineEntry(
                sequence=len(self._timeline) + 1,
                observation_id=observation.observation_id,
                event_id=observation.event_id,
                source=observation.source,
                summary=observation.summary,
                evidence_reference_ids=tuple(ref.reference_id for ref in observation.references),
                occurred_at=observation.occurred_at,
            )
        )
        if request_id is not None:
            self.fulfill_information_request(request_id, observation.observation_id)

    def correlate_observations(
        self,
        observation_ids: tuple[str, ...],
        summary: str,
    ) -> InvestigationObservation:
        _validate_identifier_tuple("observation_ids", observation_ids)
        if len(observation_ids) < 2:
            raise ValueError("correlation requires at least two observations")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("correlation observation IDs must be unique")
        observations = [self._observations.get(item) for item in observation_ids]
        if any(item is None for item in observations):
            raise ValueError("correlation references an unknown observation")
        refs = tuple(
            reference
            for observation in observations
            for reference in observation.references  # type: ignore[union-attr]
        )
        correlation = InvestigationObservation(
            observation_id=f"correlation:{len(self._observations) + 1:04d}",
            event_id=self._evidence.event_id,
            source="investigator.correlation",
            classification=ObservationClass.INFERENCE,
            summary=summary,
            references=refs,
        )
        self.add_observation(correlation)
        return correlation

    def add_hypothesis(
        self,
        statement: str,
        classification: ObservationClass = ObservationClass.HYPOTHESIS,
    ) -> Hypothesis:
        self._ensure_open_for_mutation()
        hypothesis = Hypothesis(
            hypothesis_id=f"H-{len(self._hypotheses) + 1:04d}",
            statement=statement,
            classification=classification,
        )
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._hypothesis_history.append(hypothesis)
        return hypothesis

    def update_hypothesis(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        supporting_evidence: tuple[str, ...] | None = None,
        contradicting_evidence: tuple[str, ...] | None = None,
    ) -> Hypothesis:
        self._ensure_open_for_mutation()
        current = self._hypotheses.get(hypothesis_id)
        if current is None:
            raise KeyError(hypothesis_id)
        updated = replace(
            current,
            status=status,
            supporting_evidence=(
                current.supporting_evidence if supporting_evidence is None else supporting_evidence
            ),
            contradicting_evidence=(
                current.contradicting_evidence
                if contradicting_evidence is None
                else contradicting_evidence
            ),
            revision=current.revision + 1,
        )
        self._hypotheses[hypothesis_id] = updated
        self._hypothesis_history.append(updated)
        return updated

    def request_information(
        self,
        capability: CapabilityRequest,
        decision: SecurityDecision,
    ) -> InformationRequest:
        self._ensure_open_for_mutation()
        if type(capability) is not CapabilityRequest:
            raise TypeError("capability must be a CapabilityRequest")
        if type(decision) is not SecurityDecision:
            raise TypeError("decision must be a SecurityDecision")
        if decision.request != capability:
            raise ValueError("decision must describe the capability")
        if capability.action is not CapabilityAction.READ:
            raise ValueError("the first Investigator vertical slice is read-only")
        request_id = f"request:{len(self._information_request_history) + 1:04d}"
        status = {
            DecisionState.BLOCKED: InformationRequestStatus.BLOCKED,
            DecisionState.REVIEW_REQUIRED: InformationRequestStatus.REVIEW_REQUIRED,
            DecisionState.ALLOWED: InformationRequestStatus.AUTHORIZED,
        }[decision.state]
        request = InformationRequest(request_id, capability, decision, status)
        self._information_requests[request_id] = request
        self._information_request_history.append(request)
        self._authorization_history.append(AuthorizationRecord(request_id, capability, decision))
        if decision.state in {DecisionState.ALLOWED, DecisionState.REVIEW_REQUIRED}:
            if self._status is InvestigationStatus.OPEN:
                self.transition_to(InvestigationStatus.AWAITING_INFORMATION)
        return request

    def fulfill_information_request(self, request_id: str, observation_id: str) -> None:
        self._ensure_open_for_mutation()
        _validate_identifier("request_id", request_id)
        _validate_identifier("observation_id", observation_id)
        request = self._information_requests.get(request_id)
        if request is None:
            raise KeyError(request_id)
        if request.status is not InformationRequestStatus.AUTHORIZED:
            raise ValueError("only an authorized request can be fulfilled")
        if observation_id not in self._observations:
            raise ValueError("fulfillment must reference an existing observation")
        fulfilled = replace(
            request,
            status=InformationRequestStatus.FULFILLED,
            observation_ids=(observation_id,),
        )
        self._information_requests[request_id] = fulfilled
        self._information_request_history.append(fulfilled)
        self._authorization_history.append(
            AuthorizationRecord(request_id, request.capability, request.decision)
        )
        if self._status is InvestigationStatus.AWAITING_INFORMATION:
            self.transition_to(InvestigationStatus.OPEN)

    def add_unresolved_question(self, question: str) -> None:
        self._ensure_open_for_mutation()
        _validate_text("question", question)
        if question not in self._unresolved_questions:
            self._unresolved_questions.append(question)

    def resolve_question(self, question: str) -> None:
        self._ensure_open_for_mutation()
        _validate_text("question", question)
        if question not in self._unresolved_questions:
            raise KeyError(question)
        self._unresolved_questions.remove(question)

    def add_limitation(self, limitation: str) -> None:
        self._ensure_open_for_mutation()
        _validate_text("limitation", limitation)
        if limitation not in self._limitations:
            self._limitations.append(limitation)

    def finalize(
        self,
        status: AssessmentStatus,
        statement: str,
        basis_ids: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
    ) -> FinalAssessment:
        self._ensure_open_for_mutation()
        if self._final_assessment is not None:
            raise ValueError("investigation has already been assessed")
        known_ids = set(self._evidence_references) | set(self._observations) | set(self._hypotheses)
        if any(item not in known_ids for item in basis_ids):
            raise ValueError("assessment basis contains an unknown reference")
        assessment = FinalAssessment(status, statement, basis_ids, limitations)
        for limitation in limitations:
            self.add_limitation(limitation)
        self._final_assessment = assessment
        self.transition_to(InvestigationStatus.ASSESSED)
        return assessment

    def close(self) -> None:
        if self._final_assessment is None:
            raise ValueError("investigation must be assessed before closing")
        self.transition_to(InvestigationStatus.CLOSED)

    def to_dict(self) -> dict[str, object]:
        """Return a data-only audit snapshot; it contains no executable values."""

        return {
            "investigation_id": self._investigation_id,
            "status": self._status.value,
            "evidence": self._evidence.to_dict(),
            "context": {
                "agent_id": self._context.agent_id,
                "identity": self._context.identity,
                "resource": self._context.resource,
            },
            "initial_detection": _review_trigger_to_dict(self._preview.review_trigger),
            "evidence_references": [
                _evidence_reference_to_dict(item) for item in self.evidence_references
            ],
            "observations": [_observation_to_dict(item) for item in self.observations],
            "timeline": [_timeline_to_dict(item) for item in self.timeline],
            "hypotheses": [_hypothesis_to_dict(item) for item in self.hypotheses],
            "hypothesis_history": [
                _hypothesis_to_dict(item) for item in self.hypothesis_history
            ],
            "information_requests": [
                _information_request_to_dict(item) for item in self.information_requests
            ],
            "information_request_history": [
                _information_request_to_dict(item)
                for item in self.information_request_history
            ],
            "authorization_history": [
                _authorization_to_dict(item) for item in self.authorization_history
            ],
            "unresolved_questions": list(self.unresolved_questions),
            "limitations": list(self.limitations),
            "final_assessment": (
                None
                if self._final_assessment is None
                else _assessment_to_dict(self._final_assessment)
            ),
        }

    def _seed_from_evidence(self) -> None:
        reference = EvidenceReference(
            reference_id=f"evidence:{self._evidence.event_id}",
            source="student1.detection_evidence_v1",
            event_id=self._evidence.event_id,
            description="sealed Student-1 detection evidence",
        )
        self._evidence_references[reference.reference_id] = reference
        trigger = self._preview.review_trigger
        signal_observations = _observations_from_evidence(self._evidence)
        for index, summary in enumerate(signal_observations, start=1):
            observation = InvestigationObservation(
                observation_id=f"observation:{index:04d}",
                event_id=self._evidence.event_id,
                source="student1.detection_evidence_v1",
                classification=ObservationClass.FACT,
                summary=summary,
                references=(reference,),
            )
            self.add_observation(observation)
        if trigger.review_required:
            self._limitations.append(
                "Detection predictions and scores are signals, not proof of compromise."
            )
            self.add_unresolved_question(
                "Was the activity authorized for the bound identity and resource?"
            )
        else:
            self._limitations.append(
                "Absence of a positive detection signal is not evidence that no incident occurred."
            )
        self._limitations.append("No tool execution was performed by the Investigator.")
        if self._evidence.model_prediction == 1 and self._evidence.model_score < 0.5:
            self._limitations.append("Model prediction and model score are inconsistent.")
        if self._evidence.model_prediction == 0 and self._evidence.model_score >= 0.5:
            self._limitations.append("Model prediction and model score are inconsistent.")
        self.add_unresolved_question(
            "What independent context confirms or contradicts the detection signals?"
        )
        self.add_hypothesis(
            "The observed event reflects suspicious activity.",
            ObservationClass.HYPOTHESIS,
        )
        self.add_hypothesis(
            "The observed event is benign or contextually authorized.",
            ObservationClass.HYPOTHESIS,
        )

    def _ensure_open_for_mutation(self) -> None:
        if self._status in {InvestigationStatus.ASSESSED, InvestigationStatus.CLOSED}:
            raise ValueError("assessed or closed investigations are immutable")


@dataclass(frozen=True)
class Investigator:
    """Create read-only proposals and evaluate them without executing anything."""

    policy: CapabilityPolicy
    _investigations: dict[str, InvestigationRecord] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if type(self.policy) is not CapabilityPolicy:
            raise TypeError("policy must be a CapabilityPolicy")

    def investigate(
        self,
        evidence: DetectionEvidenceV1,
        context: TrustedObservationContext,
    ) -> AuthorizationPreview:
        """Return a deterministic authorization preview for sealed evidence."""

        _validate_investigation_inputs(evidence, context)
        trigger = review_trigger_from_evidence(evidence)
        observations = _observations_from_evidence(evidence)
        if not trigger.review_required:
            return AuthorizationPreview(
                evidence_event_id=evidence.event_id,
                evidence_contract_version=evidence.CONTRACT_VERSION,
                review_trigger=trigger,
                observations=observations,
                hypotheses=(),
                proposed_capability=None,
                decision=None,
            )
        request = self._request_from_context(context)
        decision = self.policy.evaluate(request)
        return AuthorizationPreview(
            evidence_event_id=evidence.event_id,
            evidence_contract_version=evidence.CONTRACT_VERSION,
            review_trigger=trigger,
            observations=observations,
            hypotheses=("additional read-only context may help assess this event",),
            proposed_capability=request,
            decision=decision,
        )

    def start_investigation(
        self,
        evidence: DetectionEvidenceV1,
        context: TrustedObservationContext,
        investigation_id: str | None = None,
    ) -> InvestigationRecord:
        _validate_investigation_inputs(evidence, context)
        preview = self.investigate(evidence, context)
        if investigation_id is None:
            investigation_id = f"AST-{len(self._investigations) + 1:04d}"
        _validate_identifier("investigation_id", investigation_id)
        if investigation_id in self._investigations:
            raise ValueError("investigation_id must be unique")
        record = InvestigationRecord(investigation_id, evidence, context, preview)
        self._investigations[investigation_id] = record
        return record

    def get_investigation(self, investigation_id: str) -> InvestigationRecord:
        _validate_identifier("investigation_id", investigation_id)
        try:
            return self._investigations[investigation_id]
        except KeyError as error:
            raise KeyError(investigation_id) from error

    def request_read_only_context(self, investigation_id: str) -> InformationRequest:
        record = self.get_investigation(investigation_id)
        request = self._request_from_context(record.context)
        decision = self.policy.evaluate(request)
        return record.request_information(request, decision)

    def record_observation(
        self,
        investigation_id: str,
        observation: InvestigationObservation,
        request_id: str | None = None,
    ) -> InvestigationRecord:
        record = self.get_investigation(investigation_id)
        record.add_observation(observation, request_id=request_id)
        return record

    @staticmethod
    def _request_from_context(context: TrustedObservationContext) -> CapabilityRequest:
        if type(context) is not TrustedObservationContext:
            raise TypeError("context must be a TrustedObservationContext")
        return CapabilityRequest(
            agent_id=context.agent_id,
            identity=context.identity,
            tool=READ_ONLY_INVESTIGATION_TOOL,
            action=CapabilityAction.READ,
            resource=context.resource,
        )


def _validate_investigation_inputs(
    evidence: DetectionEvidenceV1,
    context: TrustedObservationContext,
) -> None:
    if type(evidence) is not DetectionEvidenceV1:
        raise TypeError("evidence must be a DetectionEvidenceV1")
    if type(context) is not TrustedObservationContext:
        raise TypeError("context must be a TrustedObservationContext")


def _observations_from_evidence(evidence: DetectionEvidenceV1) -> tuple[str, ...]:
    observations: list[str] = []
    if evidence.model_prediction == 1:
        observations.append("supervised_model_detection_signal")
    if evidence.anomaly_prediction == 1:
        observations.append("anomaly_detection_signal")
    if not observations:
        observations.append("no_positive_detection_signal")
    return tuple(observations)


def _validate_identifier(field_name: str, value: object) -> None:
    if type(value) is not str or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty, normalized identifier")


def _validate_text(field_name: str, value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _validate_text_tuple(field_name: str, value: object) -> None:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise TypeError(f"{field_name} must be a tuple of non-empty text values")


def _validate_identifier_tuple(field_name: str, value: object) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of identifiers")
    for item in value:
        _validate_identifier(field_name, item)


def _validate_reference_tuple(value: object) -> None:
    if type(value) is not tuple or not value:
        raise ValueError("references must be a non-empty tuple")
    if any(type(item) is not EvidenceReference for item in value):
        raise TypeError("references must contain only EvidenceReference values")


def _validate_timestamp(value: object) -> None:
    if value is None:
        return
    if type(value) is not datetime or value.tzinfo is None:
        raise TypeError("timestamps must be timezone-aware datetime values")


def _review_trigger_to_dict(trigger: DetectionReviewTrigger) -> dict[str, object]:
    return {
        "event_id": trigger.event_id,
        "evidence_contract_version": trigger.evidence_contract_version,
        "review_required": trigger.review_required,
        "reasons": list(trigger.reasons),
    }


def _evidence_reference_to_dict(reference: EvidenceReference) -> dict[str, str]:
    return {
        "reference_id": reference.reference_id,
        "source": reference.source,
        "event_id": reference.event_id,
        "description": reference.description,
    }


def _observation_to_dict(observation: InvestigationObservation) -> dict[str, object]:
    return {
        "observation_id": observation.observation_id,
        "event_id": observation.event_id,
        "source": observation.source,
        "classification": observation.classification.value,
        "summary": observation.summary,
        "references": [_evidence_reference_to_dict(item) for item in observation.references],
        "occurred_at": (
            None
            if observation.occurred_at is None
            else observation.occurred_at.isoformat()
        ),
    }


def _timeline_to_dict(entry: TimelineEntry) -> dict[str, object]:
    return {
        "sequence": entry.sequence,
        "observation_id": entry.observation_id,
        "event_id": entry.event_id,
        "source": entry.source,
        "summary": entry.summary,
        "evidence_reference_ids": list(entry.evidence_reference_ids),
        "occurred_at": None if entry.occurred_at is None else entry.occurred_at.isoformat(),
    }


def _hypothesis_to_dict(hypothesis: Hypothesis) -> dict[str, object]:
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "statement": hypothesis.statement,
        "classification": hypothesis.classification.value,
        "supporting_evidence": list(hypothesis.supporting_evidence),
        "contradicting_evidence": list(hypothesis.contradicting_evidence),
        "status": hypothesis.status.value,
        "revision": hypothesis.revision,
    }


def _information_request_to_dict(request: InformationRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "capability": {
            "agent_id": request.capability.agent_id,
            "identity": request.capability.identity,
            "tool": request.capability.tool,
            "action": request.capability.action.value,
            "resource": request.capability.resource,
        },
        "decision": {
            "state": request.decision.state.value,
            "reason": request.decision.reason,
            "matched_rule_count": request.decision.matched_rule_count,
        },
        "status": request.status.value,
        "observation_ids": list(request.observation_ids),
    }


def _authorization_to_dict(record: AuthorizationRecord) -> dict[str, object]:
    return {
        "request_id": record.request_id,
        "request": {
            "agent_id": record.request.agent_id,
            "identity": record.request.identity,
            "tool": record.request.tool,
            "action": record.request.action.value,
            "resource": record.request.resource,
        },
        "decision": {
            "state": record.decision.state.value,
            "reason": record.decision.reason,
            "matched_rule_count": record.decision.matched_rule_count,
        },
        "execution_outcome": record.execution_outcome.value,
    }


def _assessment_to_dict(assessment: FinalAssessment) -> dict[str, object]:
    return {
        "status": assessment.status.value,
        "statement": assessment.statement,
        "basis_ids": list(assessment.basis_ids),
        "limitations": list(assessment.limitations),
    }
