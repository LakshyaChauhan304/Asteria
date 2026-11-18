"""Tests for the additive Student-2 detector-composition facade."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection_evidence import DetectionEvidence
from student2_detection_facade import DetectionFacadeError, Student2DetectionFacade
from student2_xgboost_runtime import XGBoostPredictionBatch


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["evt-2", "evt-1"],
            "dur": [0.2, 0.1],
            "proto": ["udp", "tcp"],
        }
    )


class FakeDetectionEngine:
    def __init__(self, output: object | None = None, error: Exception | None = None) -> None:
        self.output = output or {
            "logistic_prediction": [1, 0],
            "logistic_probability": [0.9, 0.1],
            "isolation_prediction": [0, 1],
            "isolation_score": [0.2, -0.3],
        }
        self.error = error
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> object:
        self.frames.append(frame)
        if self.error is not None:
            raise self.error
        return self.output


class FakeXGBoostRuntime:
    def __init__(self, batch: XGBoostPredictionBatch | None = None) -> None:
        self.batch = batch or XGBoostPredictionBatch(
            event_ids=("evt-2", "evt-1"),
            probabilities=(0.8, 0.4),
            predictions=(1, 0),
            threshold=0.5,
            threshold_status="Provisional",
        )
        self.calls: list[tuple[pd.DataFrame, str]] = []

    def predict(self, frame: pd.DataFrame, *, event_id_column: str) -> XGBoostPredictionBatch:
        self.calls.append((frame, event_id_column))
        return self.batch


class Student2DetectionFacadeTests(unittest.TestCase):
    def test_baseline_receives_feature_frame_and_xgboost_receives_original_raw_frame(self) -> None:
        frame = raw_rows()
        engine = FakeDetectionEngine()
        runtime = FakeXGBoostRuntime()

        results = Student2DetectionFacade(engine, runtime).predict(frame)

        self.assertEqual(list(engine.frames[0].columns), ["dur", "proto"])
        self.assertEqual(engine.frames[0].to_dict(orient="list"), {
            "dur": [0.2, 0.1],
            "proto": ["udp", "tcp"],
        })
        self.assertNotIn("event_id", engine.frames[0].columns)
        self.assertIs(runtime.calls[0][0], frame)
        self.assertEqual(runtime.calls[0][1], "event_id")
        self.assertEqual(len(engine.frames), 1)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual([result.event_id for result in results], ["evt-2", "evt-1"])

    def test_keeps_each_model_signal_distinct_and_aligned(self) -> None:
        facade = Student2DetectionFacade(FakeDetectionEngine(), FakeXGBoostRuntime())
        results = facade.predict(raw_rows())

        self.assertEqual(results[0].logistic_prediction, 1)
        self.assertEqual(results[0].logistic_score, 0.9)
        self.assertEqual(results[0].anomaly_prediction, 0)
        self.assertEqual(results[0].anomaly_score, 0.2)
        self.assertEqual(results[0].xgboost_prediction, 1)
        self.assertEqual(results[0].xgboost_probability, 0.8)

    def test_missing_or_uninitialized_components_are_rejected(self) -> None:
        with self.assertRaisesRegex(DetectionFacadeError, "detection_engine must be initialized"):
            Student2DetectionFacade(None, FakeXGBoostRuntime())
        with self.assertRaisesRegex(DetectionFacadeError, "xgboost_runtime must be initialized"):
            Student2DetectionFacade(FakeDetectionEngine(), None)

    def test_malformed_baseline_prediction_lengths_are_rejected(self) -> None:
        engine = FakeDetectionEngine(
            {
                "logistic_prediction": [1],
                "logistic_probability": [0.9, 0.1],
                "isolation_prediction": [0, 1],
                "isolation_score": [0.2, -0.3],
            }
        )
        message = "logistic_prediction must contain 2 values"
        with self.assertRaisesRegex(DetectionFacadeError, message):
            Student2DetectionFacade(engine, FakeXGBoostRuntime()).predict(raw_rows())

    def test_mismatched_xgboost_event_ids_are_rejected(self) -> None:
        runtime = FakeXGBoostRuntime(
            XGBoostPredictionBatch(
                event_ids=("wrong", "evt-1"),
                probabilities=(0.8, 0.4),
                predictions=(1, 0),
                threshold=0.5,
                threshold_status="Provisional",
            )
        )
        with self.assertRaisesRegex(DetectionFacadeError, "event IDs do not match"):
            Student2DetectionFacade(FakeDetectionEngine(), runtime).predict(raw_rows())

    def test_empty_input_returns_no_results_after_both_components_run(self) -> None:
        empty = pd.DataFrame(columns=["event_id", "dur", "proto"])
        engine = FakeDetectionEngine(
            {
                "logistic_prediction": [],
                "logistic_probability": [],
                "isolation_prediction": [],
                "isolation_score": [],
            }
        )
        runtime = FakeXGBoostRuntime(
            XGBoostPredictionBatch((), (), (), threshold=0.5, threshold_status="Provisional")
        )

        self.assertEqual(Student2DetectionFacade(engine, runtime).predict(empty), ())
        self.assertEqual(len(engine.frames), 1)
        self.assertEqual(len(runtime.calls), 1)

    def test_component_failures_propagate_without_partial_result(self) -> None:
        engine = FakeDetectionEngine(error=RuntimeError("engine is not trained"))
        runtime = FakeXGBoostRuntime()

        with self.assertRaisesRegex(RuntimeError, "not trained"):
            Student2DetectionFacade(engine, runtime).predict(raw_rows())
        self.assertEqual(runtime.calls, [])

    def test_student1_evidence_schema_and_authorization_boundary_are_unchanged(self) -> None:
        facade = Student2DetectionFacade(FakeDetectionEngine(), FakeXGBoostRuntime())
        result = facade.predict(raw_rows())[0]

        self.assertFalse(hasattr(result, "decision"))
        self.assertFalse(hasattr(result, "execution_outcome"))
        self.assertEqual(
            set(DetectionEvidence.__dataclass_fields__),
            {
                "event_id", "model_prediction", "model_score", "anomaly_prediction",
                "anomaly_score", "protocol", "service", "state", "features",
            },
        )


if __name__ == "__main__":
    unittest.main()
