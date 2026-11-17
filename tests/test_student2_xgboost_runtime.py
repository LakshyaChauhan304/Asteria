"""Tests for the additive, inference-only Student-2 XGBoost runtime."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_evidence import DetectionEvidence
from detection_integration import adapt_student1_evidence
from student2_xgboost_runtime import (
    XGBoostModelMetadata,
    XGBoostRuntime,
    XGBoostRuntimeError,
)


def metadata_payload(*, threshold: float = 0.5) -> dict[str, object]:
    return {
        "model_name": "Asteria XGBoost Network Intrusion Detector",
        "model_type": "XGBClassifier",
        "input_features": ["dur", "proto"],
        "number_of_input_features": 2,
        "threshold": threshold,
        "threshold_status": "Provisional",
    }


class FakeCompletePipeline:
    def __init__(self, probabilities: list[list[float]]) -> None:
        self.probabilities = probabilities
        self.frames: list[pd.DataFrame] = []
        self.fit_calls = 0

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        self.frames.append(frame.copy())
        return self.probabilities

    def fit(self, *_: object) -> None:
        self.fit_calls += 1
        raise AssertionError("inference must not train the pipeline")


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["evt-2", "evt-1"],
            "proto": ["udp", "tcp"],
            "dur": [0.2, 0.1],
            "unrelated": [9, 8],
        }
    )


class XGBoostRuntimeTests(unittest.TestCase):
    def runtime(
        self, probabilities: list[list[float]] | None = None, *, threshold: float = 0.5
    ) -> tuple[XGBoostRuntime, FakeCompletePipeline]:
        pipeline = FakeCompletePipeline(probabilities or [[0.51, 0.49], [0.4, 0.6]])
        metadata = XGBoostModelMetadata.from_dict(metadata_payload(threshold=threshold))
        return XGBoostRuntime(pipeline, metadata), pipeline

    def test_loads_valid_repository_local_pipeline_and_metadata(self) -> None:
        pipeline = FakeCompletePipeline([[0.5, 0.5]])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = root / "models"
            models.mkdir()
            (models / "model_metadata.json").write_text(
                json.dumps(metadata_payload()), encoding="utf-8"
            )
            (models / "xgboost_asteria_pipeline.pkl").write_bytes(b"trusted-test-placeholder")
            with patch("student2_xgboost_runtime.joblib.load", return_value=pipeline) as loader:
                runtime = XGBoostRuntime.load(repository_root=root)

        self.assertIs(runtime._pipeline, pipeline)
        self.assertEqual(runtime.metadata.input_features, ("dur", "proto"))
        loader.assert_called_once()

    def test_missing_artifact_has_a_useful_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "models").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "required model artifact"):
                XGBoostRuntime.load(repository_root=root)

    def test_malformed_metadata_is_rejected_before_pipeline_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = root / "models"
            models.mkdir()
            (models / "model_metadata.json").write_text("not-json", encoding="utf-8")
            (models / "xgboost_asteria_pipeline.pkl").write_bytes(b"placeholder")
            with self.assertRaisesRegex(XGBoostRuntimeError, "not valid JSON"):
                XGBoostRuntime.load(repository_root=root)

    def test_missing_input_columns_are_rejected_without_prediction(self) -> None:
        runtime, pipeline = self.runtime()
        with self.assertRaisesRegex(XGBoostRuntimeError, r"missing required features: \['proto'\]"):
            runtime.predict(pd.DataFrame({"event_id": ["evt"], "dur": [0.1]}))
        self.assertEqual(pipeline.frames, [])

    def test_feature_order_is_normalized_for_the_complete_raw_pipeline(self) -> None:
        runtime, pipeline = self.runtime()
        runtime.predict(raw_rows())

        self.assertEqual(list(pipeline.frames[0].columns), ["dur", "proto"])
        self.assertEqual(pipeline.frames[0].iloc[0].to_dict(), {"dur": 0.2, "proto": "udp"})

    def test_probability_and_provisional_threshold_are_exposed(self) -> None:
        runtime, _ = self.runtime([[0.5, 0.5], [0.51, 0.49]], threshold=0.5)
        batch = runtime.predict(raw_rows())

        self.assertEqual(batch.probabilities, (0.5, 0.49))
        self.assertEqual(batch.predictions, (1, 0))
        self.assertEqual(batch.threshold_status, "Provisional")

    def test_multiple_rows_preserve_original_event_order(self) -> None:
        runtime, _ = self.runtime()
        batch = runtime.predict(raw_rows())

        self.assertEqual(batch.event_ids, ("evt-2", "evt-1"))
        self.assertEqual(batch.probabilities, (0.49, 0.6))
        results = batch.with_existing_signals(anomaly_scores=[-0.1, 0.2])
        self.assertEqual(results[1].event_id, "evt-1")
        self.assertIsNone(results[0].logistic_prediction)
        self.assertEqual(results[0].anomaly_score, -0.1)

    def test_empty_correctly_shaped_input_never_calls_the_pipeline(self) -> None:
        runtime, pipeline = self.runtime()
        empty = pd.DataFrame(columns=["event_id", "dur", "proto"])
        batch = runtime.predict(empty)

        self.assertEqual(batch.event_ids, ())
        self.assertEqual(batch.probabilities, ())
        self.assertEqual(pipeline.frames, [])

    def test_inference_does_not_retrain_or_double_preprocess(self) -> None:
        runtime, pipeline = self.runtime()
        runtime.predict(raw_rows())

        self.assertEqual(pipeline.fit_calls, 0)
        self.assertEqual(len(pipeline.frames), 1)
        self.assertIsInstance(pipeline.frames[0], pd.DataFrame)

    def test_pipeline_without_predict_proba_is_rejected(self) -> None:
        with self.assertRaisesRegex(XGBoostRuntimeError, "predict_proba"):
            XGBoostRuntime(object(), XGBoostModelMetadata.from_dict(metadata_payload()))

    def test_existing_detection_adapter_remains_compatible_and_schema_is_unchanged(self) -> None:
        evidence = DetectionEvidence(
            event_id="evt-schema",
            model_prediction=1,
            model_score=0.8,
            anomaly_prediction=0,
            anomaly_score=0.1,
            protocol="tcp",
            service="http",
            state="FIN",
            features={"dur": 0.1},
        )
        sealed = adapt_student1_evidence(evidence.to_dict())

        self.assertEqual(sealed.event_id, "evt-schema")
        self.assertEqual(
            set(DetectionEvidence.__dataclass_fields__),
            {
                "event_id", "model_prediction", "model_score", "anomaly_prediction",
                "anomaly_score", "protocol", "service", "state", "features",
            },
        )

    def test_rejects_untrusted_or_escaping_model_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = root / "models"
            models.mkdir()
            (models / "model_metadata.json").write_text(
                json.dumps(metadata_payload()), encoding="utf-8"
            )
            with self.assertRaisesRegex(XGBoostRuntimeError, "under models"):
                XGBoostRuntime.load(
                    repository_root=root,
                    model_path=Path("../untrusted.pkl"),
                )


if __name__ == "__main__":
    unittest.main()
