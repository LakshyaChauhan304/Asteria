"""Actual-detection plus simulated-response smoke test; never touches infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

from student2_demo_orchestrator import DemoOrchestrator


def main() -> None:
    root = Path(__file__).resolve().parent
    workflow = DemoOrchestrator(root)
    workflow.initialize(training_rows=20_000, event_count=10)
    workflow.run_detection()
    workflow.start_investigation()
    workflow.run_red_simulation("exposed_api")
    workflow.generate_blue_recommendations()
    workflow.approve_recommendation("demo-operator")
    workflow.apply_simulated_remediation()
    workflow.verify_remediation()
    approved = workflow.export_audit_report()

    declined = DemoOrchestrator(root)
    declined.initialize(training_rows=20_000, event_count=5)
    declined.run_detection()
    declined.start_investigation()
    declined.run_red_simulation("credential_misuse")
    declined.generate_blue_recommendations()
    declined.decline_recommendation("demo-operator")
    print(json.dumps({"approved_flow": approved, "declined_flow": declined.export_audit_report()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
