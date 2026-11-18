"""Exercise the public Streamlit entry point with the bundled data and models."""

import unittest
from importlib.util import find_spec
from pathlib import Path

from streamlit.testing.v1 import AppTest


@unittest.skipUnless(find_spec("xgboost"), "demo model runtime is unavailable")
class DemoWorkspaceTests(unittest.TestCase):
    def test_twenty_events_and_two_independent_cases(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "demo_app.py"),
            default_timeout=120,
        )
        app.run()

        def clean():
            self.assertFalse(app.exception)
            self.assertFalse(app.error, [item.value for item in app.error])

        def press(key):
            app.button(key=key).click().run(timeout=120)
            clean()

        def visit(page):
            app.radio(key="demo_view").set_value(page).run()
            clean()

        clean()
        self.assertEqual([button.key for button in app.button], ["initialize"])
        press("initialize")
        self.assertEqual(app.radio(key="demo_view").value, "Detection")
        self.assertEqual([button.key for button in app.button], ["detect"])
        self.assertEqual(len(app.dataframe[0].value), 20)
        self.assertNotIn("XGBoost probability", app.dataframe[0].value.columns)
        press("detect")
        rows = app.dataframe[0].value
        self.assertEqual(rows["Event"].nunique(), 20)
        self.assertTrue(rows["XGBoost probability"].between(0, 1).all())
        self.assertFalse(app.button)

        first, second = rows["Event"].iloc[:2]
        visit("Investigations")
        app.selectbox(key="case_candidate").select(first).run()
        press("open_case")
        self.assertEqual(app.dataframe[0].value.iloc[0]["event_id"], first)
        self.assertEqual(len(app.dataframe[1].value), 3)

        visit("Red Simulation")
        press("red")
        self.assertIn("Public API", app.get("html")[-1].proto.body)
        visit("Blue Response")
        press("blue")
        self.assertTrue(any("Restrict public ingress" in h.value for h in app.subheader))
        visit("Verification")
        self.assertFalse(app.button)
        visit("Blue Response")
        press("approve")
        visit("Verification")
        press("apply")
        press("verify")
        self.assertEqual([metric.value for metric in app.metric], ["Reachable", "Blocked"])
        self.assertTrue(app.dataframe[0].value["passed"].all())

        visit("Audit & Reports")
        self.assertEqual(len(app.get("download_button")), 2)
        self.assertTrue(any(first in caption.value for caption in app.caption))

        visit("Investigations")
        app.selectbox(key="case_candidate").select(second).run()
        press("open_case")
        self.assertEqual(app.dataframe[0].value.iloc[0]["event_id"], second)
        visit("Verification")
        self.assertFalse(app.metric)
        visit("Investigations")
        app.selectbox(key="case_candidate").select(first).run()
        press("open_case")
        visit("Verification")
        self.assertEqual([metric.value for metric in app.metric], ["Reachable", "Blocked"])


if __name__ == "__main__":
    unittest.main()
