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
        visit("Cases")
        def select_case(index):
            app.session_state["case_queue"] = {
                "selection": {"rows": [index], "columns": [], "cells": []}
            }
            app.run()
            clean()

        select_case(0)
        self.assertEqual(app.dataframe[1].value.iloc[0]["event_id"], first)
        self.assertEqual(len(app.dataframe[2].value), 3)

        press("run_red_simulation")
        self.assertEqual(app.session_state.demo_cases[first].state.value, "red_simulated")
        press("generate_blue_recommendation")
        self.assertTrue(any("Restrict public ingress" in h.value for h in app.subheader))
        press("approval_approve")
        self.assertFalse(app.button(key="apply_control").disabled)
        press("apply_control")
        press("run_verifier")
        self.assertEqual(
            app.session_state.demo_cases[first].export_audit_report()["verification"]["outcome"],
            "blocked",
        )

        visit("Command Center")
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Cases opened"], "1")
        self.assertEqual(metrics["Cases verified"], "1")

        visit("Audit & Reports")
        self.assertEqual(len(app.get("download_button")), 3)

        visit("Cases")
        select_case(1)
        self.assertEqual(app.dataframe[1].value.iloc[0]["event_id"], second)
        visit("Command Center")
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Cases opened"], "2")
        self.assertEqual(metrics["Cases verified"], "1")
        visit("Cases")
        self.assertTrue(app.button(key="apply_control").disabled)
        select_case(0)
        self.assertEqual(
            app.session_state.demo_cases[first].export_audit_report()["verification"]["outcome"],
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()
