"""Reproducible, additive data preparation for the real Asteria demo.

This module does not alter Student-1 preprocessing or model artifacts.  It
loads repository datasets, prepares raw model-feature frames, and initializes
the existing in-memory baseline engine for one bounded demonstration run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from student2_xgboost_runtime import XGBoostModelMetadata

_ID_COLUMN = "id"
_LABEL_COLUMN = "label"
_ATTACK_CATEGORY_COLUMN = "attack_cat"
_EVENT_PREFIX = "UNSW-"


class DemoDataError(ValueError):
    """Raised when real demo data cannot satisfy the established contracts."""


@dataclass(frozen=True)
class DemoDataBundle:
    """Validated raw data needed by the three-model demo, without labels in inference."""

    training_features: pd.DataFrame
    training_labels: pd.Series
    inference_frame: pd.DataFrame
    ground_truth: pd.DataFrame
    telemetry_rows: tuple[dict[str, object], ...]
    metadata: XGBoostModelMetadata
    training_row_count: int
    dataset_training_row_count: int
    dataset_testing_row_count: int
    missing_feature_diagnostics: tuple[str, ...]

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self.inference_frame["event_id"].tolist())


@dataclass(frozen=True)
class BaselineInitialization:
    """The existing Student-1 engine and its measured in-memory initialization."""

    engine: object
    training_duration_seconds: float


def load_demo_data(
    *,
    repository_root: Path | None = None,
    training_rows: int = 25_000,
    event_count: int = 50,
    random_state: int = 42,
) -> DemoDataBundle:
    """Load real UNSW-NB15 CSVs and prepare feature-only inference data.

    Labels and attack categories are deliberately preserved outside the
    inference frame; they are available only for optional post-run inspection.
    """

    if type(training_rows) is not int or training_rows < 2:
        raise DemoDataError("training_rows must be an integer of at least 2")
    if type(event_count) is not int or not 1 <= event_count <= 100:
        raise DemoDataError("event_count must be an integer between 1 and 100")
    if type(random_state) is not int:
        raise DemoDataError("random_state must be an integer")

    root = (repository_root or Path(__file__).resolve().parent).resolve()
    metadata = _load_metadata(root)
    training = _read_dataset(root / "data" / "UNSW_NB15_training-set.csv", "training")
    testing = _read_dataset(root / "data" / "UNSW_NB15_testing-set.csv", "testing")
    _validate_dataset_columns(training, metadata, "training")
    _validate_dataset_columns(testing, metadata, "testing")
    if event_count > len(testing):
        raise DemoDataError("event_count exceeds available testing rows")

    training_subset = _stratified_subset(training, training_rows, random_state)
    selected = testing.sample(n=event_count, random_state=random_state).sort_index()
    feature_names = list(metadata.input_features)
    training_features = training_subset.loc[:, feature_names].copy()
    training_labels = training_subset[_LABEL_COLUMN].copy()
    inference_features = selected.loc[:, feature_names].copy()
    event_ids = selected[_ID_COLUMN].map(_event_id_from_dataset_id)
    inference_frame = inference_features.copy()
    inference_frame.insert(0, "event_id", event_ids.tolist())
    ground_truth = selected.loc[:, [_ID_COLUMN, _LABEL_COLUMN, _ATTACK_CATEGORY_COLUMN]].copy()
    ground_truth.insert(0, "event_id", event_ids.tolist())
    telemetry_columns = ("event_id", "proto", "service", "state", "dur", "sbytes", "dbytes")
    telemetry_frame = selected.loc[:, ["proto", "service", "state", "dur", "sbytes", "dbytes"]].copy()
    telemetry_frame.insert(0, "event_id", event_ids.tolist())
    telemetry_rows = tuple(
        {
            "event_id": str(row.event_id),
            "protocol": str(row.proto),
            "service": str(row.service),
            "state": str(row.state),
            "duration": f"{float(row.dur):.6f}",
            "source_bytes": int(row.sbytes),
            "destination_bytes": int(row.dbytes),
        }
        for row in telemetry_frame.loc[:, telemetry_columns].itertuples(index=False)
    )

    return DemoDataBundle(
        training_features=training_features,
        training_labels=training_labels,
        inference_frame=inference_frame,
        ground_truth=ground_truth,
        telemetry_rows=telemetry_rows,
        metadata=metadata,
        training_row_count=len(training_features),
        dataset_training_row_count=len(training),
        dataset_testing_row_count=len(testing),
        missing_feature_diagnostics=(),
    )


def initialize_baseline_engine(bundle: DemoDataBundle) -> BaselineInitialization:
    """Train the unmodified Student-1 baseline engine in memory only."""

    if type(bundle) is not DemoDataBundle:
        raise TypeError("bundle must be a DemoDataBundle")
    # Imported only at the integration edge; no Student-1 implementation is changed.
    from detection_engine import AsteriaDetectionEngine

    engine = AsteriaDetectionEngine()
    started = time.perf_counter()
    engine.train(bundle.training_features, bundle.training_labels)
    return BaselineInitialization(
        engine=engine,
        training_duration_seconds=time.perf_counter() - started,
    )


def _load_metadata(repository_root: Path) -> XGBoostModelMetadata:
    metadata_path = repository_root / "models" / "model_metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required demo metadata is unavailable: {metadata_path}") from error
    except json.JSONDecodeError as error:
        raise DemoDataError("demo model metadata is not valid JSON") from error
    return XGBoostModelMetadata.from_dict(payload)


def _read_dataset(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required {name} dataset is unavailable: {path}")
    return pd.read_csv(path)


def _validate_dataset_columns(
    frame: pd.DataFrame, metadata: XGBoostModelMetadata, name: str
) -> None:
    required = {_ID_COLUMN, _LABEL_COLUMN, _ATTACK_CATEGORY_COLUMN, *metadata.input_features}
    missing = tuple(sorted(required.difference(frame.columns)))
    if missing:
        raise DemoDataError(f"{name} dataset is missing required columns: {list(missing)}")
    if frame.columns.duplicated().any():
        raise DemoDataError(f"{name} dataset contains duplicate column names")
    if frame[_LABEL_COLUMN].isna().any() or frame[_LABEL_COLUMN].nunique() < 2:
        raise DemoDataError(f"{name} dataset must contain two non-null label classes")


def _stratified_subset(frame: pd.DataFrame, requested_rows: int, random_state: int) -> pd.DataFrame:
    if requested_rows >= len(frame):
        return frame.copy()
    try:
        subset, _ = train_test_split(
            frame,
            train_size=requested_rows,
            random_state=random_state,
            stratify=frame[_LABEL_COLUMN],
        )
    except ValueError as error:
        raise DemoDataError("could not create deterministic stratified training subset") from error
    return subset.sort_index()


def _event_id_from_dataset_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DemoDataError("dataset id values must be numeric")
    integer_value = int(value)
    if integer_value != value or integer_value < 0:
        raise DemoDataError("dataset id values must be non-negative integers")
    return f"{_EVENT_PREFIX}{integer_value}"
