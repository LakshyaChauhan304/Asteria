"""Run the real three-model detection-to-investigation Asteria demo.

No tool is executed: the final decision is an authorization preview only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from investigator import (
    READ_ONLY_INVESTIGATION_TOOL,
    Investigator,
    TrustedObservationContext,
)
from security_layer import CapabilityAction, CapabilityPolicy, CapabilityRule
from student2_demo_data import initialize_baseline_engine, load_demo_data
from student2_detection_facade import Student2DetectionFacade
from student2_detection_to_investigator import Student2DetectionToInvestigator
from student2_xgboost_runtime import XGBoostRuntime


def run_demo(*, repository_root: Path, training_rows: int, event_count: int) -> dict[str, object]:
    """Execute actual data, loaded XGBoost, baseline training, and investigation creation."""

    started = time.perf_counter()
    bundle = load_demo_data(
        repository_root=repository_root,
        training_rows=training_rows,
        event_count=event_count,
    )
    baseline = initialize_baseline_engine(bundle)
    xgboost_runtime = XGBoostRuntime.load(repository_root=repository_root)
    facade = Student2DetectionFacade(baseline.engine, xgboost_runtime)
    contexts = tuple(
        TrustedObservationContext(
            "demo-investigator",
            "demo-security-service",
            f"events/{event_id}",
        )
        for event_id in bundle.event_ids
    )
    policy = CapabilityPolicy(
        tuple(
            CapabilityRule(
                context.agent_id,
                context.identity,
                READ_ONLY_INVESTIGATION_TOOL,
                CapabilityAction.READ,
                context.resource,
            )
            for context in contexts
        )
    )
    adapter = Student2DetectionToInvestigator(facade, Investigator(policy))
    inference_started = time.perf_counter()
    results = adapter.start_investigations(bundle.inference_frame, contexts)
    inference_duration = time.perf_counter() - inference_started

    detections = tuple(result.detection for result in results)
    positive = lambda item: item.logistic_prediction == 1 or item.anomaly_prediction == 1 or item.xgboost_prediction == 1
    selected_index = next((index for index, item in enumerate(detections) if positive(item)), None)
    selected_as_candidate = False
    if selected_index is None:
        selected_index = max(range(len(detections)), key=lambda index: detections[index].xgboost_probability)
        selected_as_candidate = True
    selected = results[selected_index]
    return {
        "model_load": {
            "xgboost_model": xgboost_runtime.metadata.model_name,
            "xgboost_threshold": xgboost_runtime.metadata.threshold,
            "baseline_initialized": True,
        },
        "dataset": {
            "training_rows_available": bundle.dataset_training_row_count,
            "testing_rows_available": bundle.dataset_testing_row_count,
            "training_rows_used": bundle.training_row_count,
            "selected_event_count": len(bundle.event_ids),
            "missing_feature_diagnostics": list(bundle.missing_feature_diagnostics),
        },
        "durations_seconds": {
            "baseline_training": round(baseline.training_duration_seconds, 6),
            "inference_and_investigation": round(inference_duration, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "events": [
            {
                "event_id": item.event_id,
                "logistic_prediction": item.logistic_prediction,
                "logistic_probability": item.logistic_score,
                "isolation_prediction": item.anomaly_prediction,
                "isolation_score": item.anomaly_score,
                "xgboost_prediction": item.xgboost_prediction,
                "xgboost_probability": item.xgboost_probability,
            }
            for item in detections
        ],
        "signal_counts": {
            "logistic_positive": sum(item.logistic_prediction for item in detections),
            "isolation_anomaly": sum(item.anomaly_prediction for item in detections),
            "xgboost_positive": sum(item.xgboost_prediction for item in detections),
            "model_disagreements": sum(
                len({item.logistic_prediction, item.anomaly_prediction, item.xgboost_prediction}) > 1
                for item in detections
            ),
        },
        "selected_investigation": {
            "event_id": selected.detection.event_id,
            "investigation_id": selected.investigation.investigation_id,
            "selected_as_highest_xgboost_candidate": selected_as_candidate,
            "evidence_count": len(selected.investigation.evidence_references),
            "hypotheses": [hypothesis.statement for hypothesis in selected.investigation.hypotheses],
            "limitations": list(selected.investigation.limitations),
            "authorization_preview": [
                {
                    "tool": request.capability.tool,
                    "action": request.capability.action.value,
                    "decision": request.decision.state.value,
                }
                for request in selected.investigation.information_requests
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the actual Asteria three-model demo")
    parser.add_argument("--training-rows", type=int, default=20_000)
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(
        run_demo(
            repository_root=args.repository_root.resolve(),
            training_rows=args.training_rows,
            event_count=args.events,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
