"""Bounded Student-2 runtime for the partner's saved XGBoost pipeline.

This module performs inference only.  It deliberately accepts raw pandas data
and passes it straight to the persisted sklearn pipeline, which owns its own
preprocessing.  Its results are detection signals, never authorization or
final investigation evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/xgboost_asteria_pipeline.pkl")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class XGBoostRuntimeError(ValueError):
    """Raised when the bounded inference contract cannot be satisfied."""


@dataclass(frozen=True)
class XGBoostModelMetadata:
    """Validated, inference-relevant subset of the partner model metadata."""

    model_name: str
    model_type: str
    input_features: tuple[str, ...]
    threshold: float
    threshold_status: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> XGBoostModelMetadata:
        """Validate the JSON object without accepting executable mappings."""

        if type(payload) is not dict:
            raise TypeError("model metadata must be a plain dictionary")
        required = {
            "model_name",
            "model_type",
            "input_features",
            "number_of_input_features",
            "threshold",
            "threshold_status",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise XGBoostRuntimeError(f"model metadata is missing fields: {missing}")
        model_name = payload["model_name"]
        model_type = payload["model_type"]
        feature_values = payload["input_features"]
        feature_count = payload["number_of_input_features"]
        threshold = payload["threshold"]
        threshold_status = payload["threshold_status"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise XGBoostRuntimeError("model_name must be a non-empty string")
        if model_type != "XGBClassifier":
            raise XGBoostRuntimeError("model_type must be XGBClassifier")
        if type(feature_values) is not list or not feature_values:
            raise XGBoostRuntimeError("input_features must be a non-empty list")
        if type(feature_count) is not int or feature_count != len(feature_values):
            raise XGBoostRuntimeError(
                "number_of_input_features must equal the number of input_features"
            )
        if any(
            type(feature) is not str or not _IDENTIFIER.fullmatch(feature)
            for feature in feature_values
        ):
            raise XGBoostRuntimeError("input_features contains an invalid feature identifier")
        if len(set(feature_values)) != len(feature_values):
            raise XGBoostRuntimeError("input_features must not contain duplicates")
        if type(threshold) not in (int, float) or isinstance(threshold, bool):
            raise XGBoostRuntimeError("threshold must be a numeric value")
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
            raise XGBoostRuntimeError("threshold must be finite and between 0 and 1")
        if not isinstance(threshold_status, str) or not threshold_status.strip():
            raise XGBoostRuntimeError("threshold_status must be a non-empty string")
        return cls(
            model_name=model_name,
            model_type=model_type,
            input_features=tuple(feature_values),
            threshold=threshold_value,
            threshold_status=threshold_status,
        )


@dataclass(frozen=True)
class XGBoostDetectionResult:
    """Per-event XGBoost signal, kept distinct from other detector outputs."""

    event_id: object
    xgboost_probability: float
    xgboost_prediction: int
    logistic_prediction: int | None = None
    logistic_score: float | None = None
    anomaly_prediction: int | None = None
    anomaly_score: float | None = None
    model_identifier: str = "XGBClassifier"
    model_metadata_identifier: str | None = None
    model_version: str | None = None
    feature_schema_identifier: str | None = None
    threshold: float = 0.5
    threshold_status: str = "Provisional"
    source_identifier: str = "student2.xgboost_runtime"


@dataclass(frozen=True)
class XGBoostPredictionBatch:
    """Ordered output from one raw-data inference call."""

    event_ids: tuple[object, ...]
    probabilities: tuple[float, ...]
    predictions: tuple[int, ...]
    threshold: float
    threshold_status: str
    model_identifier: str = "XGBClassifier"
    model_metadata_identifier: str | None = None
    model_version: str | None = None
    feature_schema_identifier: str | None = None
    source_identifier: str = "student2.xgboost_runtime"

    def with_existing_signals(
        self,
        *,
        logistic_predictions: Sequence[int] | None = None,
        logistic_scores: Sequence[float] | None = None,
        anomaly_predictions: Sequence[int] | None = None,
        anomaly_scores: Sequence[float] | None = None,
    ) -> tuple[XGBoostDetectionResult, ...]:
        """Pair independent baseline signals by row without conflating them."""

        count = len(self.event_ids)
        values = (
            ("logistic_predictions", logistic_predictions),
            ("logistic_scores", logistic_scores),
            ("anomaly_predictions", anomaly_predictions),
            ("anomaly_scores", anomaly_scores),
        )
        for name, sequence in values:
            if sequence is not None and len(sequence) != count:
                raise XGBoostRuntimeError(f"{name} must have {count} values")
        return tuple(
            XGBoostDetectionResult(
                event_id=self.event_ids[index],
                xgboost_probability=self.probabilities[index],
                xgboost_prediction=self.predictions[index],
                logistic_prediction=(
                    None if logistic_predictions is None else logistic_predictions[index]
                ),
                logistic_score=None if logistic_scores is None else logistic_scores[index],
                anomaly_prediction=(
                    None if anomaly_predictions is None else anomaly_predictions[index]
                ),
                anomaly_score=None if anomaly_scores is None else anomaly_scores[index],
                model_identifier=self.model_identifier,
                model_metadata_identifier=self.model_metadata_identifier,
                model_version=self.model_version,
                feature_schema_identifier=self.feature_schema_identifier,
                threshold=self.threshold,
                threshold_status=self.threshold_status,
                source_identifier=self.source_identifier,
            )
            for index in range(count)
        )


class XGBoostRuntime:
    """Inference-only wrapper around a trusted, complete sklearn pipeline."""

    def __init__(self, pipeline: object, metadata: XGBoostModelMetadata) -> None:
        if type(metadata) is not XGBoostModelMetadata:
            raise TypeError("metadata must be XGBoostModelMetadata")
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise XGBoostRuntimeError("saved pipeline must provide predict_proba")
        self._pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        *,
        repository_root: Path | None = None,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> XGBoostRuntime:
        """Load trusted repository-local metadata and a pipeline artifact.

        ``model_path`` and ``metadata_path`` are application configuration, not
        agent input.  Both must remain under the repository's ``models`` folder,
        which prevents path traversal or arbitrary absolute artifact loading.
        """

        root = (repository_root or Path(__file__).resolve().parent).resolve()
        metadata_file = _trusted_model_file(root, metadata_path, ".json")
        model_file = _trusted_model_file(root, model_path, ".pkl")
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise XGBoostRuntimeError("model metadata is not valid JSON") from error
        metadata = XGBoostModelMetadata.from_dict(payload)
        try:
            pipeline = joblib.load(model_file)
        except Exception as error:  # joblib exposes several implementation errors.
            raise XGBoostRuntimeError(f"could not load saved pipeline: {model_file}") from error
        return cls(pipeline, metadata)

    def predict(
        self, frame: pd.DataFrame, *, event_id_column: str = "event_id"
    ) -> XGBoostPredictionBatch:
        """Produce class-1 probabilities from raw rows, preserving row order."""

        if type(frame) is not pd.DataFrame:
            raise TypeError("frame must be a pandas DataFrame")
        if type(event_id_column) is not str or not _IDENTIFIER.fullmatch(event_id_column):
            raise XGBoostRuntimeError("event_id_column must be a valid identifier")
        if event_id_column not in frame.columns:
            raise XGBoostRuntimeError(f"input frame is missing event ID column: {event_id_column}")
        missing = [
            feature for feature in self.metadata.input_features if feature not in frame.columns
        ]
        if missing:
            raise XGBoostRuntimeError(f"input frame is missing required features: {missing}")
        event_ids = tuple(frame[event_id_column].tolist())
        raw_features = frame.loc[:, list(self.metadata.input_features)]
        if raw_features.empty:
            return XGBoostPredictionBatch(
                event_ids=event_ids,
                probabilities=(),
                predictions=(),
                threshold=self.metadata.threshold,
                threshold_status=self.metadata.threshold_status,
                model_identifier=self.metadata.model_type,
                model_metadata_identifier=self.metadata.model_name,
            )
        raw_probabilities = self._pipeline.predict_proba(raw_features)
        probabilities = _class_one_probabilities(raw_probabilities, len(frame))
        predictions = tuple(
            int(probability >= self.metadata.threshold) for probability in probabilities
        )
        return XGBoostPredictionBatch(
            event_ids=event_ids,
            probabilities=probabilities,
            predictions=predictions,
            threshold=self.metadata.threshold,
            threshold_status=self.metadata.threshold_status,
            model_identifier=self.metadata.model_type,
            model_metadata_identifier=self.metadata.model_name,
        )


def _trusted_model_file(repository_root: Path, configured_path: Path, suffix: str) -> Path:
    if not isinstance(configured_path, Path) or configured_path.is_absolute():
        raise XGBoostRuntimeError("artifact paths must be relative pathlib.Path values")
    models_root = (repository_root / "models").resolve()
    candidate = (repository_root / configured_path).resolve()
    if candidate.suffix != suffix or models_root not in candidate.parents:
        raise XGBoostRuntimeError(f"artifact must be a {suffix} file under models/")
    if not candidate.is_file():
        raise FileNotFoundError(f"required model artifact is unavailable: {candidate}")
    return candidate


def _class_one_probabilities(raw_probabilities: object, expected_rows: int) -> tuple[float, ...]:
    table = pd.DataFrame(raw_probabilities)
    if table.shape != (expected_rows, 2):
        raise XGBoostRuntimeError(
            "predict_proba must return one binary probability pair per input row"
        )
    try:
        values = tuple(float(value) for value in table.iloc[:, 1].tolist())
    except (TypeError, ValueError) as error:
        raise XGBoostRuntimeError("class-1 probabilities must be numeric") from error
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise XGBoostRuntimeError("class-1 probabilities must be finite and between 0 and 1")
    return values
