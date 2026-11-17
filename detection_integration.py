"""Student-2-only bridge from sealed Student-1 evidence into investigation.

The caller supplies the plain built-in dictionary produced by
``DetectionEvidence.to_dict()``. This adapter intentionally does not accept a
Student-1 object or invoke an arbitrary ``to_dict`` method: doing so would make
the evidence boundary executable again.
"""

from __future__ import annotations

from typing import Any

from investigator import InvestigationRecord, Investigator, TrustedObservationContext
from security_layer import DetectionEvidenceV1


def adapt_student1_evidence(payload: dict[str, Any]) -> DetectionEvidenceV1:
    """Seal a plain Student-1 evidence dictionary as ``DetectionEvidenceV1``."""

    return DetectionEvidenceV1.from_dict(payload)


def start_investigation_from_detection(
    payload: dict[str, Any],
    context: TrustedObservationContext,
    investigator: Investigator,
    *,
    investigation_id: str | None = None,
) -> InvestigationRecord:
    """Start a bounded, non-executing Student-2 investigation from detection.

    This is an additive integration point. It performs validation, policy
    evaluation, and in-memory investigation creation only; it never changes
    Student-1 detection behavior or invokes a tool.
    """

    if type(context) is not TrustedObservationContext:
        raise TypeError("context must be a TrustedObservationContext")
    if type(investigator) is not Investigator:
        raise TypeError("investigator must be an Investigator")
    evidence = adapt_student1_evidence(payload)
    return investigator.start_investigation(evidence, context, investigation_id)
