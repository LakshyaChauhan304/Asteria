"""Exercise real detections and the isolated response workflow before deployment."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Support running this file directly from any current working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student2_demo_environment import ApprovalDecision  # noqa: E402
from student2_demo_orchestrator import DemoOrchestrator, WorkflowState  # noqa: E402
from student2_demo_verifier import DemoVerifier, VerificationStatus  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    step = "Load sample and run detection"
    try:
        batch = DemoOrchestrator(ROOT)
        batch.initialize(training_rows=20_000, event_count=20)
        results = batch.run_detection()
        require(len(results) >= 2, "Two detected events are needed for independent cases")
        primary = next(
            (item for item in results if item.detection.event_id == "UNSW-8318"), results[0]
        )
        secondary = next(
            item for item in results if item.detection.event_id != primary.detection.event_id
        )
        detection = primary.detection
        print(f"Event: {detection.event_id}")
        print(
            f"Logistic regression: class={detection.logistic_prediction}, "
            f"probability={detection.logistic_score}"
        )
        print(
            f"Isolation forest: class={detection.anomaly_prediction}, "
            f"score={detection.anomaly_score}"
        )
        print(
            f"XGBoost: class={detection.xgboost_prediction}, "
            f"probability={detection.xgboost_probability}"
        )
        print(f"PASS: {step}")

        step = "Exposed API red and blue proposal"
        approved = batch.open_case(str(detection.event_id))
        before = approved.environment.current_state
        finding = approved.run_red_simulation("exposed_api")
        recommendations = approved.generate_blue_recommendations()
        require(finding.candidate_path.reachable, "Expected reachable simulated path")
        require(bool(recommendations), "Blue agent returned no recommendation")
        require(approved.environment.current_state == before, "Controls changed before approval")
        require(not approved.environment.audit_history, "Unexpected remediation audit")
        print(f"Red finding: {finding.explanation}")
        print(f"Blue recommendation: {recommendations[0].title}")
        print(f"PASS: {step}")

        step = "Declined credential-misuse proposal"
        declined = batch.open_case(str(secondary.detection.event_id))
        declined_before = declined.environment.current_state
        declined_finding = declined.run_red_simulation("credential_misuse")
        declined_recommendation = declined.generate_blue_recommendations()[0]
        decision = declined.decline_recommendation("predeploy-operator")
        require(decision.decision is ApprovalDecision.DECLINED, "Decision was not declined")
        require(declined.environment.current_state == declined_before, "Declined controls changed")
        require(
            declined.audit is None and not declined.environment.audit_history, "Unexpected audit"
        )
        for operation in (declined.apply_simulated_remediation, declined.verify_remediation):
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Declined workflow unexpectedly allowed apply or verification")
        declined_verification = DemoVerifier().verify(
            declined_finding,
            declined_recommendation,
            decision,
            declined.audit,
            declined.environment,
        )
        require(
            declined_verification.status is not VerificationStatus.VERIFIED,
            "Declined verification passed",
        )
        require(
            declined.environment.current_state == declined_before, "Verification changed controls"
        )
        require(
            declined.audit is None and not declined.environment.audit_history,
            "Unexpected audit after verification",
        )
        print(
            f"Declined event: {secondary.detection.event_id}; "
            f"verifier={declined_verification.status.value}"
        )
        print(f"PASS: {step}")

        step = "Approve, apply, and independently verify exposed API control"
        approval = approved.approve_recommendation("predeploy-operator")
        require(approval.decision is ApprovalDecision.APPROVED, "Decision was not approved")
        require(approved.environment.current_state == before, "Controls changed before apply")
        require(
            approved.audit is None and not approved.environment.audit_history,
            "Audit created before apply",
        )
        audit = approved.apply_simulated_remediation()
        require(
            audit.control_id == recommendations[0].target_control_id, "Unexpected control applied"
        )
        require(approved.environment.control_state(audit.control_id), "Control is not active")
        verification = approved.verify_remediation()
        require(
            approved.state is WorkflowState.VERIFIED, "Orchestrator did not reach verified state"
        )
        require(
            verification.status is VerificationStatus.VERIFIED, "Verifier did not report success"
        )
        require(verification.before_graph_reachable, "Initial path was not reachable")
        require(not verification.after_graph_reachable, "Path remains reachable")
        require(
            bool(verification.checks) and all(check.passed for check in verification.checks),
            "Verifier checks failed",
        )
        print(
            f"Control: {audit.control_id}; reachable before={verification.before_graph_reachable}, "
            f"after={verification.after_graph_reachable}"
        )
        print(
            "Verifier checks: "
            + ", ".join(f"{check.name}={check.passed}" for check in verification.checks)
        )
        print(f"PASS: {step}")
        return 0
    except Exception:
        print(f"FAIL: {step}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
