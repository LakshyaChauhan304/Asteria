import unittest

from student2_demo_qa import answer
from student2_demo_ui import graph_svg, pipeline_svg


class DemoUiTests(unittest.TestCase):
    def test_visual_helpers_reflect_backend_states_without_actions(self):
        self.assertIn("VERIFIED", pipeline_svg("verified"))
        self.assertIn("NO LIVE NETWORK CONNECTION", graph_svg(None, False))

    def test_qa_is_evidence_grounded(self):
        report = {"selected_event_id": "UNSW-1", "investigation": {"evidence": {"model_prediction": 1, "model_score": 0.9, "anomaly_prediction": 0, "anomaly_score": 0.2}, "model_signals": [{"prediction": 1, "probability": 0.8}], "evidence_references": []}}
        response = answer("Detection Analyst", "What is the XGBoost output?", report)
        self.assertIn("UNSW-1", response)
        self.assertIn("not proof", response)


if __name__ == "__main__":
    unittest.main()
