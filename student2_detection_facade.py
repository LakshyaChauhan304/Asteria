"""Student-2 composition layer for independent Asteria detection signals.

The facade invokes the existing trained detection engine and the saved XGBoost
pipeline independently.  It neither changes their inputs nor combines their
signals into an alert, authorization decision, or execution request.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd

from student2_xgboost_runtime import XGBoostDetectionResult, XGBoostPredictionBatch

_BASELINE_KEYS = (
    "logistic_prediction",
    "logistic_probability",
    "isolation_prediction",
    "isolation_score",
)


class DetectionFacadeError(ValueError):
    """Raised when independent detector outputs cannot be safely aligned."""


@dataclass(frozen=True)
class Student2DetectionFacade:
    """Compose two trusted, already-initialized detector interfaces.

    ``detection_engine`` is the existing Student-1 engine interface and
    ``xgboost_runtime`` is the Student-2 inference interface.  The facade does
    not import or mutate Student-1 implementation modules.
    """

    detection_engine: object
    xgboost_runtime: object

    def __post_init__(self) -> None:
        _require_callable(self.detection_engine, "predict", "detection_engine")
        _require_callable(self.xgboost_runtime, "predict", "xgboost_runtime")

    def predict(
        self, raw_frame: pd.DataFrame, *, event_id_column: str = "event_id"
    ) -> tuple[XGBoostDetectionResult, ...]:
        """Return separate detector signals, aligned to input order and IDs."""

        if type(raw_frame) is not pd.DataFrame:
            raise TypeError("raw_frame must be a pandas DataFrame")
        if type(event_id_column) is not str or not event_id_column:
            raise DetectionFacadeError("event_id_column must be a non-empty string")
        if event_id_column not in raw_frame.columns:
            raise DetectionFacadeError(f"raw_frame is missing event ID column: {event_id_column}")
        event_ids = tuple(raw_frame[event_id_column].tolist())

        # Student-1's detection engine is trained on model features only.  The
        # event identifier is transport metadata, not a model feature; keeping
        # it out of this call prevents the existing preprocessor from treating
        # it as a numerical feature.  XGBoost owns its complete preprocessing
        # pipeline and receives the untouched raw frame below.
        baseline_frame = raw_frame.drop(columns=[event_id_column])
        baseline_output = self.detection_engine.predict(baseline_frame)
        xgboost_batch = self.xgboost_runtime.predict(
            raw_frame, event_id_column=event_id_column
        )
        baseline = _validate_baseline_output(baseline_output, len(raw_frame))
        _validate_xgboost_batch(xgboost_batch, event_ids)

        return tuple(
            XGBoostDetectionResult(
                event_id=event_ids[index],
                logistic_prediction=baseline.logistic_predictions[index],
                logistic_score=baseline.logistic_probabilities[index],
                anomaly_prediction=baseline.isolation_predictions[index],
                anomaly_score=baseline.isolation_scores[index],
                xgboost_prediction=xgboost_batch.predictions[index],
                xgboost_probability=xgboost_batch.probabilities[index],
            )
            for index in range(len(raw_frame))
        )


@dataclass(frozen=True)
class _BaselineSignals:
    logistic_predictions: tuple[int, ...]
    logistic_probabilities: tuple[float, ...]
    isolation_predictions: tuple[int, ...]
    isolation_scores: tuple[float, ...]


def _require_callable(component: object, attribute: str, name: str) -> Callable[..., object]:
    if component is None:
        raise DetectionFacadeError(f"{name} must be initialized")
    method = getattr(component, attribute, None)
    if not callable(method):
        raise DetectionFacadeError(f"{name} must provide callable {attribute}()")
    return method


def _validate_baseline_output(output: object, expected_rows: int) -> _BaselineSignals:
    if type(output) is not dict:
        raise DetectionFacadeError("detection_engine.predict() must return a plain dictionary")
    missing = [key for key in _BASELINE_KEYS if key not in output]
    if missing:
        raise DetectionFacadeError(f"detection engine output is missing keys: {missing}")
    logistic_predictions = _binary_values(
        output["logistic_prediction"], expected_rows, "logistic_prediction"
    )
    logistic_probabilities = _probability_values(
        output["logistic_probability"], expected_rows, "logistic_probability"
    )
    isolation_predictions = _binary_values(
        output["isolation_prediction"], expected_rows, "isolation_prediction"
    )
    isolation_scores = _finite_values(
        output["isolation_score"], expected_rows, "isolation_score"
    )
    return _BaselineSignals(
        logistic_predictions=logistic_predictions,
        logistic_probabilities=logistic_probabilities,
        isolation_predictions=isolation_predictions,
        isolation_scores=isolation_scores,
    )


def _validate_xgboost_batch(batch: object, event_ids: tuple[object, ...]) -> None:
    if type(batch) is not XGBoostPredictionBatch:
        raise DetectionFacadeError("xgboost_runtime.predict() must return XGBoostPredictionBatch")
    if batch.event_ids != event_ids:
        raise DetectionFacadeError("XGBoost event IDs do not match the raw input order")
    expected_rows = len(event_ids)
    _binary_values(batch.predictions, expected_rows, "xgboost predictions")
    _probability_values(batch.probabilities, expected_rows, "xgboost probabilities")


def _as_values(values: object, expected_rows: int, name: str) -> tuple[Any, ...]:
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise DetectionFacadeError(f"{name} must be a one-dimensional sequence") from error
    if len(result) != expected_rows:
        raise DetectionFacadeError(f"{name} must contain {expected_rows} values")
    return result


def _binary_values(values: object, expected_rows: int, name: str) -> tuple[int, ...]:
    result = _as_values(values, expected_rows, name)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1)
        for value in result
    ):
        raise DetectionFacadeError(f"{name} must contain only binary integer values")
    return tuple(int(value) for value in result)


def _probability_values(values: object, expected_rows: int, name: str) -> tuple[float, ...]:
    result = _finite_values(values, expected_rows, name)
    if any(not 0.0 <= value <= 1.0 for value in result):
        raise DetectionFacadeError(f"{name} values must be between 0 and 1")
    return result


def _finite_values(values: object, expected_rows: int, name: str) -> tuple[float, ...]:
    result = _as_values(values, expected_rows, name)
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
        for value in result
    ):
        raise DetectionFacadeError(f"{name} must contain finite numeric values")
    return tuple(float(value) for value in result)
