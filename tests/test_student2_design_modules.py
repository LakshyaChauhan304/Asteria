import unittest

from student2_report_fixtures import stage_report
from student2_components import (affected_nodes, approval_card_html, decision_log_html, outcome_badge_html,
                                 patch_progress_html, sandbox_log_html, workflow_rail_html)
from student2_graph3d import fallback_table_html, graph_document_html, graph_payload, headline
from student2_report import SECTION_TITLES, render_report_html, render_report_pdf, report_sections

STAGES = ("investigated", "red", "blue", "approved", "declined", "applied", "verified")


class GraphTests(unittest.TestCase):
    def test_empty_before_red(self):
        self.assertIsNone(graph_payload(stage_report("investigated"), "red"))
        self.assertIsNone(graph_payload({}, "blue"))

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            graph_payload(stage_report("red"), "green")

    def test_payload_mirrors_backend_state(self):
        for stage in ("red", "applied", "verified"):
            report = stage_report(stage)
            payload = graph_payload(report, "blue")
            exposure = report["graph_exposure"]
            self.assertEqual(payload["blocked_edge_ids"], sorted(exposure["blocked_edge_ids"]))
            self.assertEqual({e["id"] for e in payload["edges"] if e["status"] == "blocked"}, set(exposure["blocked_edge_ids"]))
            self.assertEqual({n["id"] for n in payload["nodes"] if n["kind"] == "decoy"}, set(exposure["decoy_node_ids"]) - {payload["nodes"][0]["id"]})
            self.assertEqual(payload["target_reachable"], exposure["target_reachable"])
            self.assertEqual(len(payload["nodes"]), len(report["attack_graph"]["nodes"]))

    def test_headline_and_text_fallback_cover_everything(self):
        payload = graph_payload(stage_report("verified"), "blue")
        self.assertIn("no longer reachable", headline(payload))
        table = fallback_table_html(payload)
        for node in payload["nodes"]:
            self.assertIn(node["name"], table)
        self.assertEqual(table.count("BLOCKED"), len(payload["blocked_edge_ids"]))
        self.assertIn("no control has been applied", headline(graph_payload(stage_report("blue"), "blue")))

    def test_document_is_offline_and_escapes(self):
        html = graph_document_html(graph_payload(stage_report("red"), "red"))
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<script src", html)
        self.assertIn("prefers-reduced-motion", html)


class ReportTests(unittest.TestCase):
    def test_sections_in_order_and_never_omitted(self):
        for stage in STAGES:
            titles = [s["title"] for s in report_sections(stage_report(stage))]
            self.assertEqual(titles, list(SECTION_TITLES), stage)
            html = render_report_html(stage_report(stage))
            positions = [html.index(f"</span>{t}</h2>") for t in SECTION_TITLES]
            self.assertEqual(positions, sorted(positions))

    def test_pending_is_shown(self):
        html = render_report_html(stage_report("investigated"))
        self.assertGreaterEqual(html.count("Pending"), 4)

    def test_escaping(self):
        report = stage_report("approved", note="<script>alert(1)</script>")
        report["decision_history"][0]["operator_identity"] = "<img src=x onerror=1>"
        html = render_report_html(report)
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertEqual(render_report_pdf(report)[:4], b"%PDF")

    def test_declined_report_shows_no_remediation(self):
        html = render_report_html(stage_report("declined"))
        self.assertIn("declined", html)
        self.assertIn("applying the control has not happened", html)

    def test_pdf_for_every_stage(self):
        for stage in STAGES:
            self.assertGreater(len(render_report_pdf(stage_report(stage))), 1500)


class ComponentTests(unittest.TestCase):
    def test_outcome_badges_have_text(self):
        for outcome in ("blocked", "still_reachable", "inconclusive", "failed"):
            self.assertIn(outcome.replace("_", " ").upper(), outcome_badge_html(outcome))
        self.assertIn("PENDING", outcome_badge_html(None))

    def test_rail_follows_recorded_state(self):
        self.assertEqual(workflow_rail_html(stage_report("declined")).count("class='ast-step done'"), 4)
        self.assertEqual(workflow_rail_html(stage_report("approved")).count("class='ast-step done'"), 5)
        self.assertEqual(workflow_rail_html(stage_report("verified")).count("class='ast-step done'"), 7)

    def test_approval_card_facts(self):
        report = stage_report("blue")
        card = approval_card_html(report)
        self.assertIn("ingress_restricted", card)
        self.assertIn("no live rollback", card)
        self.assertIn("External Actor", card)
        self.assertEqual(set(affected_nodes(report)), {"External Actor", "Public API", "Workload Identity", "Sensitive Data Store", "Edge Proxy"})
        self.assertIn("No recommendation yet", approval_card_html(stage_report("red")))

    def test_declined_leaves_control_inactive(self):
        card = approval_card_html(stage_report("declined"))
        self.assertIn("<small>NOW</small><b>inactive</b>", card)
        self.assertIn("Declined", decision_log_html(stage_report("declined")))

    def test_progress_and_log(self):
        self.assertIn("4 of 4", patch_progress_html(stage_report("verified")))
        self.assertIn("1 of 4", patch_progress_html(stage_report("blue")))
        steps = stage_report("verified")["verification"]["steps"]
        log = sandbox_log_html(steps[:3], len(steps))
        self.assertEqual(log.count("PASS"), 3)
        self.assertIn("RUN", log)
        self.assertNotIn("RUN", sandbox_log_html(steps, len(steps), True))

    def test_escaping_in_components(self):
        report = stage_report("blue")
        report["recommendations"][0]["title"] = "<b>x</b>"
        self.assertNotIn("<b>x</b>", approval_card_html(report))


if __name__ == "__main__":
    unittest.main()
