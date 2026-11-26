"""Write static HTML previews of the design modules (no Streamlit needed).

Usage: python scripts/preview_design.py OUTPUT_DIR
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from student2_components import (approval_card_html, component_css, decision_log_html,  # noqa: E402
                                outcome_badge_html, patch_progress_html, sandbox_log_html, workflow_rail_html)
from student2_graph3d import graph_document_html, graph_payload  # noqa: E402
from student2_report import render_report_html  # noqa: E402
from student2_report_fixtures import stage_report  # noqa: E402


def page(body):
    return ("<!doctype html><meta charset='utf-8'><body style='margin:0;padding:22px;background:#060b15;font:14px system-ui'>"
            f"<style>{component_css()}</style>{body}")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    approved, verified = stage_report("blue"), stage_report("verified")
    (out / "approval.html").write_text(page(
        workflow_rail_html(approved) + approval_card_html(approved) + "<h4 style='color:#8096aa'>Decision log</h4>" + decision_log_html(stage_report("declined"))))
    steps = verified["verification"]["steps"]
    (out / "sandbox.html").write_text(page(
        workflow_rail_html(verified) + "<div style='margin:8px 0'>" + outcome_badge_html("blocked") + "</div>"
        + sandbox_log_html(steps, len(steps), True) + patch_progress_html(verified)))
    (out / "report.html").write_text(render_report_html(verified))
    for stage, mode in (("red", "red"), ("verified", "blue")):
        (out / f"graph_{mode}.html").write_text(graph_document_html(graph_payload(stage_report(stage), mode)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "preview_out")
