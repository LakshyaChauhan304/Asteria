import json
import unittest
from pathlib import Path

from student2_demo_orchestrator import DemoOrchestrator, WorkflowState


class OrchestratorTests(unittest.TestCase):
    def test_invalid_transition_is_rejected_and_empty_report_is_serializable(self):
        workflow = DemoOrchestrator(Path.cwd())
        with self.assertRaises(RuntimeError):
            workflow.run_detection()
        self.assertEqual(workflow.state, WorkflowState.CREATED)
        json.dumps(workflow.export_audit_report())
