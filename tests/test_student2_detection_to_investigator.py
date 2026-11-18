"""Integration tests from validated facade output into the Investigator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investigator import (
    READ_ONLY_INVESTIGATION_TOOL,
    Investigator,
    ModelSignalProvenance,
    ObservationClass,
    TrustedObservationContext,
)
from security_layer import (
    CapabilityAction,
    CapabilityPolicy,
    CapabilityRule,
    DecisionState,
)
from student2_detection_facade import Student2DetectionFacade
from student2_detection_to_investigator import (
    DetectionInvestigationIntegrationError,
    Student2DetectionToInvestigator,
)
from student2_xgboost_runtime import XGBoostPredictionBatch


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["evt-2", "evt-1"],
            "proto": ["udp", "tcp"],
            "service": ["dns", "http"],
            "state": ["CON", "FIN"],
            "dur": [0.2, 0.1],
            "spkts": [3, 4],
        }
    )


class FakeEngine:
    def predict(self, frame: pd.DataFrame) -> dict[str, list[float] | list[int]]:
        outputs = {
            0.2: (0, 0.2, 0, 0.1),
            0.1: (1, 0.9, 0, 0.2),
        }
        self.last_frame = frame.copy()
        rows = [outputs[duration] for duration in frame["dur"]]
        return {
            "logistic_prediction": [row[0] for row in rows],
            "logistic_probability": [row[1] for row in rows],
            "isolation_prediction": [row[2] for row in rows],
            "isolation_score": [row[3] for row in rows],
        }


class FakeRuntime:
    def predict(self, frame: pd.DataFrame, *, event_id_column: str) -> XGBoostPredictionBatch:
        outputs = {"evt-2": (0.8123456789, 1), "evt-1": (0.3, 0)}
        rows = [outputs[event_id] for event_id in frame[event_id_column]]
        return XGBoostPredictionBatch(
            event_ids=tuple(frame[event_id_column].tolist()),
            probabilities=tuple(row[0] for row in rows),
            predictions=tuple(row[1] for row in rows),
            threshold=0.5,
            threshold_status="Provisional",
        )


class CountingEngine(FakeEngine):
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: pd.DataFrame) -> dict[str, list[float] | list[int]]:
        self.calls += 1
        return super().predict(frame)


class NeutralEngine:
    def predict(self, frame: pd.DataFrame) -> dict[str, list[float] | list[int]]:
        count = len(frame)
        return {
            "logistic_prediction": [0] * count,
            "logistic_probability": [0.1] * count,
            "isolation_prediction": [0] * count,
            "isolation_score": [0.1] * count,
        }


class NegativeRuntime:
    def predict(self, frame: pd.DataFrame, *, event_id_column: str) -> XGBoostPredictionBatch:
        count = len(frame)
        return XGBoostPredictionBatch(
            event_ids=tuple(frame[event_id_column].tolist()),
            probabilities=(0.2,) * count,
            predictions=(0,) * count,
            threshold=0.5,
            threshold_status="Provisional",
        )


def contexts() -> tuple[TrustedObservationContext, ...]:
    return (
        TrustedObservationContext("investigator-agent", "security-service", "events/evt-2"),
        TrustedObservationContext("investigator-agent", "security-service", "events/evt-1"),
    )


def workflow(policy: CapabilityPolicy | None = None) -> Student2DetectionToInvestigator:
    facade = Student2DetectionFacade(FakeEngine(), FakeRuntime())
    return Student2DetectionToInvestigator(facade, Investigator(policy or CapabilityPolicy(())))


class DetectionToInvestigatorTests(unittest.TestCase):
    def test_multiple_events_preserve_signals_provenance_and_identity(self) -> None:
        results = workflow().start_investigations(raw_rows(), contexts())

        self.assertEqual([item.detection.event_id for item in results], ["evt-2", "evt-1"])
        self.assertEqual(
            [item.investigation.evidence.event_id for item in results], ["evt-2", "evt-1"]
        )
        first = results[0].investigation
        xgboost = [item for item in first.observations if item.source == "student2.xgboost_runtime"]
        self.assertEqual(len(xgboost), 1)
        self.assertEqual(xgboost[0].classification, ObservationClass.FACT)
        self.assertEqual(xgboost[0].references[0].event_id, "evt-2")
        self.assertEqual(results[0].detection.xgboost_probability, 0.8123456789)

    def test_xgboost_only_signal_requests_read_only_context_but_policy_denies(self) -> None:
        result = workflow().start_investigations(
            raw_rows().iloc[[0]].reset_index(drop=True), contexts()[:1]
        )[0]

        self.assertEqual(result.detection.xgboost_prediction, 1)
        self.assertEqual(len(result.investigation.information_requests), 1)
        self.assertEqual(
            result.investigation.information_requests[0].decision.state,
            DecisionState.BLOCKED,
        )
        self.assertEqual(
            result.investigation.information_requests[0].capability.action,
            CapabilityAction.READ,
        )
        self.assertTrue(any("not proof" in item for item in result.investigation.limitations))

    def test_xgboost_only_signal_preserves_review_required_policy(self) -> None:
        context = contexts()[0]
        rule = CapabilityRule(
            context.agent_id,
            context.identity,
            READ_ONLY_INVESTIGATION_TOOL,
            CapabilityAction.READ,
            context.resource,
            requires_human_approval=True,
        )
        result = workflow(CapabilityPolicy((rule,))).start_investigations(
            raw_rows().iloc[[0]].reset_index(drop=True), (context,)
        )[0]

        self.assertEqual(
            result.investigation.information_requests[0].decision.state,
            DecisionState.REVIEW_REQUIRED,
        )

    def test_xgboost_negative_does_not_request_context_without_other_signal(self) -> None:
        adapter = Student2DetectionToInvestigator(
            Student2DetectionFacade(NeutralEngine(), NegativeRuntime()),
            Investigator(CapabilityPolicy(())),
        )
        result = adapter.start_investigations(
            raw_rows().iloc[[0]].reset_index(drop=True), contexts()[:1]
        )[0]

        self.assertEqual(result.detection.xgboost_prediction, 0)
        self.assertEqual(result.investigation.information_requests, ())

    def test_existing_policy_still_governs_logistic_or_anomaly_review(self) -> None:
        context = contexts()[1]
        rule = CapabilityRule(
            context.agent_id,
            context.identity,
            READ_ONLY_INVESTIGATION_TOOL,
            CapabilityAction.READ,
            context.resource,
        )
        result = workflow(CapabilityPolicy((rule,))).start_investigations(
            raw_rows().iloc[[1]].reset_index(drop=True), (context,)
        )[0]

        self.assertEqual(
            result.investigation.information_requests[0].decision.state.value, "allowed"
        )
        self.assertEqual(result.investigation.evidence.model_prediction, 1)
        self.assertEqual(result.detection.xgboost_prediction, 0)

    def test_duplicate_event_ids_fail_before_investigations_are_created(self) -> None:
        duplicate = raw_rows()
        duplicate.loc[1, "event_id"] = "evt-2"
        adapter = workflow()

        with self.assertRaisesRegex(DetectionInvestigationIntegrationError, "must be unique"):
            adapter.start_investigations(duplicate, contexts())
        with self.assertRaises(KeyError):
            adapter.investigator.get_investigation("AST-0001")

    def test_invalid_event_data_fails_closed_before_investigation_creation(self) -> None:
        invalid = raw_rows().iloc[[0]].reset_index(drop=True)
        invalid.loc[0, "service"] = "not valid"
        adapter = workflow()

        with self.assertRaisesRegex(DetectionInvestigationIntegrationError, "service"):
            adapter.start_investigations(invalid, contexts()[:1])
        with self.assertRaises(KeyError):
            adapter.investigator.get_investigation("AST-0001")

    def test_misaligned_facade_result_fails_closed(self) -> None:
        class MisalignedRuntime(FakeRuntime):
            def predict(
                self, frame: pd.DataFrame, *, event_id_column: str
            ) -> XGBoostPredictionBatch:
                return XGBoostPredictionBatch(
                    ("wrong",), (0.8,), (1,), threshold=0.5, threshold_status="Provisional"
                )

        adapter = Student2DetectionToInvestigator(
            Student2DetectionFacade(FakeEngine(), MisalignedRuntime()),
            Investigator(CapabilityPolicy(())),
        )
        with self.assertRaisesRegex(Exception, "event IDs do not match"):
            adapter.start_investigations(
                raw_rows().iloc[[0]].reset_index(drop=True), contexts()[:1]
            )

    def test_duplicate_raw_columns_fail_before_the_facade_is_invoked(self) -> None:
        for duplicate_column in ("event_id", "service", "dur"):
            with self.subTest(duplicate_column=duplicate_column):
                frame = raw_rows().iloc[[0]].reset_index(drop=True)
                frame.insert(1, duplicate_column, frame[duplicate_column], allow_duplicates=True)
                engine = CountingEngine()
                adapter = Student2DetectionToInvestigator(
                    Student2DetectionFacade(engine, FakeRuntime()),
                    Investigator(CapabilityPolicy(())),
                )

                with self.assertRaisesRegex(
                    DetectionInvestigationIntegrationError, "columns must be unique"
                ):
                    adapter.start_investigations(frame, contexts()[:1])
                self.assertEqual(engine.calls, 0)

    def test_model_signal_preserves_exact_probability_and_provenance_in_serialization(self) -> None:
        result = workflow().start_investigations(
            raw_rows().iloc[[0]].reset_index(drop=True), contexts()[:1]
        )[0]
        serialized = result.investigation.to_dict()["model_signals"][0]

        self.assertEqual(serialized["probability"], 0.8123456789)
        self.assertEqual(serialized["threshold"], 0.5)
        self.assertEqual(serialized["prediction"], 1)
        self.assertEqual(serialized["event_id"], "evt-2")
        self.assertEqual(serialized["source_identifier"], "student2.xgboost_runtime")
        self.assertIsNone(serialized["model_metadata_identifier"])
        self.assertEqual(ModelSignalProvenance.from_dict(serialized).to_dict(), serialized)

    def test_invalid_model_signal_values_and_provenance_are_rejected(self) -> None:
        valid = {
            "signal_id": "signal:xgboost:evt-2",
            "event_id": "evt-2",
            "model_identifier": "XGBClassifier",
            "model_metadata_identifier": None,
            "model_version": None,
            "feature_schema_identifier": None,
            "probability": 0.8,
            "prediction": 1,
            "threshold": 0.5,
            "threshold_status": "Provisional",
            "source_identifier": "student2.xgboost_runtime",
        }
        invalid_values = (
            ("probability", 1.1),
            ("threshold", -0.1),
            ("source_identifier", "bad source"),
        )
        for field, value in invalid_values:
            with self.subTest(field=field):
                invalid = dict(valid)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    ModelSignalProvenance.from_dict(invalid)

    def test_later_preparation_failure_publishes_no_partial_batch(self) -> None:
        adapter = workflow()
        original = __import__("investigator").InvestigationRecord.add_model_signal

        def fail_second(record: object, signal: ModelSignalProvenance) -> None:
            if record.evidence.event_id == "evt-1":
                raise RuntimeError("simulated later preparation failure")
            original(record, signal)

        with (
            patch("investigator.InvestigationRecord.add_model_signal", new=fail_second),
            self.assertRaisesRegex(RuntimeError, "later preparation failure"),
        ):
            adapter.start_investigations(raw_rows(), contexts())
        with self.assertRaises(KeyError):
            adapter.investigator.get_investigation("AST-0001")
        with self.assertRaises(KeyError):
            adapter.investigator.get_investigation("AST-0002")


if __name__ == "__main__":
    unittest.main()
