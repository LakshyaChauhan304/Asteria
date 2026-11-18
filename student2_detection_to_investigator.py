"""Additive bridge from validated detector signals into investigations.

This module keeps the Student-1 evidence schema intact: Logistic Regression and
Isolation Forest populate its established fields, while XGBoost is retained as
an explicitly sourced Student-2 FACT observation.  No detector output grants a
capability or executes a tool.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real

import pandas as pd
from detection_integration import adapt_student1_evidence
from investigator import (
    EvidenceReference,
    InvestigationObservation,
    InvestigationRecord,
    Investigator,
    ModelSignalProvenance,
    ObservationClass,
    TrustedObservationContext,
)
from student2_detection_facade import Student2DetectionFacade
from student2_xgboost_runtime import XGBoostDetectionResult

_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}")
_FEATURE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_REQUIRED_EVENT_COLUMNS = ("proto", "service", "state")


class DetectionInvestigationIntegrationError(ValueError):
    """Raised when detector output cannot safely become investigation evidence."""


@dataclass(frozen=True)
class DetectionInvestigationResult:
    """One detection result paired with its sealed evidence and investigation."""

    detection: XGBoostDetectionResult
    investigation: InvestigationRecord


@dataclass(frozen=True)
class Student2DetectionToInvestigator:
    """Start bounded investigations from validated facade results.

    Context is supplied by the trusted caller as an immutable value per event;
    it is never selected from event data or any model output.
    """

    facade: Student2DetectionFacade
    investigator: Investigator

    def __post_init__(self) -> None:
        if type(self.facade) is not Student2DetectionFacade:
            raise TypeError("facade must be a Student2DetectionFacade")
        if type(self.investigator) is not Investigator:
            raise TypeError("investigator must be an Investigator")

    def start_investigations(
        self,
        raw_frame: pd.DataFrame,
        contexts: tuple[TrustedObservationContext, ...],
        *,
        event_id_column: str = "event_id",
    ) -> tuple[DetectionInvestigationResult, ...]:
        """Validate a batch, then create one in-memory record per event.

        All input conversion and provenance validation completes before any
        investigation record is created.  This fails closed for malformed rows,
        duplicate IDs, result misalignment, or untrusted context objects.
        """

        if type(raw_frame) is not pd.DataFrame:
            raise TypeError("raw_frame must be a pandas DataFrame")
        if type(contexts) is not tuple:
            raise TypeError("contexts must be a tuple of TrustedObservationContext values")
        if type(event_id_column) is not str or not _FEATURE_NAME.fullmatch(event_id_column):
            raise DetectionInvestigationIntegrationError(
                "event_id_column must be a valid identifier"
            )
        if event_id_column not in raw_frame.columns:
            raise DetectionInvestigationIntegrationError(
                f"raw_frame is missing event ID column: {event_id_column}"
            )
        _validate_raw_columns(raw_frame.columns)
        missing = [column for column in _REQUIRED_EVENT_COLUMNS if column not in raw_frame.columns]
        if missing:
            raise DetectionInvestigationIntegrationError(
                f"raw_frame is missing required event fields: {missing}"
            )
        if len(contexts) != len(raw_frame):
            raise DetectionInvestigationIntegrationError(
                "contexts must align one-to-one with input rows"
            )
        if any(type(context) is not TrustedObservationContext for context in contexts):
            raise TypeError("contexts must contain only TrustedObservationContext values")

        detections = self.facade.predict(raw_frame, event_id_column=event_id_column)
        if len(detections) != len(raw_frame):
            raise DetectionInvestigationIntegrationError(
                "detection facade output length must match the input rows"
            )
        candidates = tuple(
            _candidate_from_row(raw_frame.iloc[index], detections[index], event_id_column)
            for index in range(len(raw_frame))
        )
        event_ids = tuple(candidate.event_id for candidate in candidates)
        if len(set(event_ids)) != len(event_ids):
            raise DetectionInvestigationIntegrationError("event IDs must be unique within a batch")

        prepared_inputs = tuple(
            (adapt_student1_evidence(candidate.evidence_payload), context)
            for candidate, context in zip(candidates, contexts, strict=True)
        )
        records = self.investigator._prepare_investigations(prepared_inputs)
        results: list[DetectionInvestigationResult] = []
        for candidate, record in zip(candidates, records, strict=True):
            record.add_observation(candidate.xgboost_observation)
            record.add_model_signal(candidate.xgboost_signal)
            record.add_limitation(
                "XGBoost output is a detection signal, not proof of compromise or authorization."
            )
            if candidate.detection.xgboost_prediction == 1:
                record.add_unresolved_question(
                    "What independent context confirms or contradicts the XGBoost detection signal?"
                )
                if not record.information_requests:
                    record.request_information()
            results.append(DetectionInvestigationResult(candidate.detection, record))
        self.investigator._commit_prepared_investigations(records)
        return tuple(results)


@dataclass(frozen=True)
class _Candidate:
    event_id: str
    detection: XGBoostDetectionResult
    evidence_payload: dict[str, object]
    xgboost_observation: InvestigationObservation
    xgboost_signal: ModelSignalProvenance


def _candidate_from_row(
    row: pd.Series,
    detection: object,
    event_id_column: str,
) -> _Candidate:
    if type(detection) is not XGBoostDetectionResult:
        raise DetectionInvestigationIntegrationError(
            "detection facade must return XGBoostDetectionResult values"
        )
    event_id = _event_id(row[event_id_column])
    if detection.event_id != event_id:
        raise DetectionInvestigationIntegrationError("detection result event ID is misaligned")
    protocol = _event_text(row["proto"], "proto")
    service = _event_text(row["service"], "service")
    state = _event_text(row["state"], "state")
    features = {
        str(column): _scalar_feature(row[column], str(column))
        for column in row.index
        if column != event_id_column
    }
    if any(not _FEATURE_NAME.fullmatch(name) for name in features):
        raise DetectionInvestigationIntegrationError(
            "raw feature names must be normalized identifiers"
        )
    evidence_payload: dict[str, object] = {
        "event_id": event_id,
        "model_prediction": _binary(detection.logistic_prediction, "logistic_prediction"),
        "model_score": _probability(detection.logistic_score, "logistic_score"),
        "anomaly_prediction": _binary(detection.anomaly_prediction, "anomaly_prediction"),
        "anomaly_score": _finite(detection.anomaly_score, "anomaly_score"),
        "protocol": protocol,
        "service": service,
        "state": state,
        "features": features,
    }
    xgboost_probability = _probability(detection.xgboost_probability, "xgboost_probability")
    xgboost_prediction = _binary(detection.xgboost_prediction, "xgboost_prediction")
    xgboost_threshold = _probability(detection.threshold, "xgboost_threshold")
    xgboost_signal = ModelSignalProvenance(
        signal_id=f"signal:xgboost:{event_id}",
        event_id=event_id,
        model_identifier=_identifier(detection.model_identifier, "model_identifier"),
        model_metadata_identifier=detection.model_metadata_identifier,
        model_version=detection.model_version,
        feature_schema_identifier=detection.feature_schema_identifier,
        probability=xgboost_probability,
        prediction=xgboost_prediction,
        threshold=xgboost_threshold,
        threshold_status=_event_text(detection.threshold_status, "threshold_status"),
        source_identifier=_identifier(detection.source_identifier, "source_identifier"),
    )
    reference = EvidenceReference(
        reference_id=f"xgboost:{event_id}",
        source=xgboost_signal.source_identifier,
        event_id=event_id,
        description="validated XGBoost supervised detection output",
    )
    observation = InvestigationObservation(
        observation_id=f"observation:xgboost:{event_id}",
        event_id=event_id,
        source=xgboost_signal.source_identifier,
        classification=ObservationClass.FACT,
        summary=(
            "XGBoost supervised detection signal "
            f"prediction={xgboost_prediction} "
            f"probability={xgboost_probability:.4f}"
        ),
        references=(reference,),
    )
    return _Candidate(event_id, detection, evidence_payload, observation, xgboost_signal)


def _validate_raw_columns(columns: pd.Index) -> None:
    labels = tuple(columns.tolist())
    if any(type(label) is not str or not _FEATURE_NAME.fullmatch(label) for label in labels):
        raise DetectionInvestigationIntegrationError(
            "raw DataFrame columns must be normalized identifiers"
        )
    if len(set(labels)) != len(labels):
        raise DetectionInvestigationIntegrationError("raw DataFrame columns must be unique")


def _event_id(value: object) -> str:
    if type(value) is not str or not _EVENT_ID.fullmatch(value):
        raise DetectionInvestigationIntegrationError("event IDs must be normalized strings")
    return value


def _identifier(value: object, field: str) -> str:
    if type(value) is not str or not _EVENT_ID.fullmatch(value):
        raise DetectionInvestigationIntegrationError(f"{field} must be a normalized identifier")
    return value


def _event_text(value: object, field: str) -> str:
    if type(value) is not str or not value or any(character.isspace() for character in value):
        raise DetectionInvestigationIntegrationError(
            f"{field} must be a non-empty event text value"
        )
    return value


def _scalar_feature(value: object, name: str) -> str | int | float | bool:
    if type(value) is str:
        return value
    if type(value) is bool:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise DetectionInvestigationIntegrationError(f"feature {name} must be a finite scalar value")


def _binary(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1):
        raise DetectionInvestigationIntegrationError(f"{name} must be a binary integer")
    return int(value)


def _probability(value: object, name: str) -> float:
    finite = _finite(value, name)
    if not 0.0 <= finite <= 1.0:
        raise DetectionInvestigationIntegrationError(f"{name} must be between 0 and 1")
    return finite


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise DetectionInvestigationIntegrationError(f"{name} must be finite")
    return float(value)
