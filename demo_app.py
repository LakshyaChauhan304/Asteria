"""The Asteria demo workspace and its session-scoped controls."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from student2_components import (
    approval_card_html,
    component_css,
    decision_log_html,
    outcome_badge_html,
    patch_progress_html,
    replay_verifier_steps,
    sandbox_log_html,
    workflow_rail_html,
)
from student2_demo_data import initialize_baseline_engine, load_demo_data
from student2_demo_environment import ApprovalDecision
from student2_demo_orchestrator import DemoOrchestrator, WorkflowState
from student2_graph3d import render_graph_3d
from student2_red_agent import scenario_ids
from student2_report import render_report_html, render_report_pdf, report_filename
from student2_xgboost_runtime import XGBoostRuntime

ROOT = Path(__file__).resolve().parent
VIEWS = (
    "Command Center", "Cases", "Detection", "Data Sources", "Audit & Reports",
)
@st.cache_resource(show_spinner="Loading demo models and sample data...")
def load_demo_resources(repository_root: Path, training_rows: int, event_count: int):
    bundle = load_demo_data(
        repository_root=repository_root,
        training_rows=training_rows,
        event_count=event_count,
    )
    baseline = initialize_baseline_engine(bundle)
    runtime = XGBoostRuntime.load(repository_root=repository_root)
    return bundle, baseline, runtime


def initialize_demo(batch: DemoOrchestrator) -> None:
    batch.initialize(resource_loader=load_demo_resources)


def perform(operation, *args, destination=None):
    """Run a native widget callback and keep failures visible on the same page."""
    st.session_state.pop("demo_error", None)
    try:
        operation(*args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        st.session_state.demo_error = f"{type(error).__name__}: {error}"
        return
    if destination:
        st.session_state.demo_view = destination


def open_case(batch, event_id):
    cases = st.session_state.demo_cases
    if event_id not in cases:
        cases[event_id] = batch.open_case(event_id)
    st.session_state.demo_event = event_id


def models_disagree(detection):
    """Model predictions disagree when the three class predictions are not equal."""
    return len({
        detection["logistic_prediction"],
        detection["isolation_prediction"],
        detection["xgboost_prediction"],
    }) != 1


def home(batch):
    st.title("Asteria Command Center")
    detections = batch.export_audit_report()["detection_cases"]
    cases = st.session_state.demo_cases
    reports = [case.export_audit_report() for case in cases.values()]
    verified = sum(
        report["workflow_state"] == "verified"
        and (report["verification"] or {}).get("outcome") == "blocked"
        for report in reports
    )
    metrics = st.columns(5)
    for column, label, value in zip(
        metrics,
        ("Events scored", "2+ model signals", "Models disagree", "Cases opened", "Cases verified"),
        (
            len(detections),
            sum(item["signal_count"] >= 2 for item in detections),
            sum(models_disagree(item) for item in detections),
            len(cases),
            verified,
        ),
        strict=True,
    ):
        column.metric(label, value)
    if batch.state is WorkflowState.CREATED:
        st.button(
            "Initialize Models", key="initialize",
            on_click=perform, args=(initialize_demo, batch), kwargs={"destination": "Detection"},
            type="primary",
        )
    else:
        st.success("Models initialized. Open Detection from the sidebar.")


def grid(rows):
    st.dataframe(list(rows), height=35 * (len(rows) + 1) + 3, hide_index=True)


def detection(batch, report):
    st.title("Detection Engine")
    st.caption("Each model output is a signal. Compromise requires independent evidence.")
    if batch.state is WorkflowState.CREATED:
        st.info("Initialize the models on Command Center to load the 20 sampled flows.")
        return
    if batch.state is WorkflowState.MODELS_READY:
        st.button(
            "Detect",
            key="detect",
            on_click=perform,
            args=(batch.run_detection,),
            type="primary",
        )
    signals = {row["event_id"]: row for row in report["detection_cases"]}
    st.subheader("Detection results" if signals else "Loaded telemetry")
    if signals:
        st.caption(f"{len(signals)} scored events. Open Cases to inspect an event.")
    else:
        st.caption("Telemetry is ready. Select Detect to produce model scores.")
    rows = []
    for flow in report["telemetry"]:
        row = {
            "Event": flow["event_id"], "Protocol": flow["protocol"],
            "Service": flow["service"], "State": flow["state"],
            "Duration (s)": flow["duration"],
            "Source bytes": flow["source_bytes"], "Destination bytes": flow["destination_bytes"],
        }
        if signals:
            signal = signals[flow["event_id"]]
            row.update({
                "Logistic class": signal["logistic_prediction"],
                "Logistic probability": signal["logistic_probability"],
                "Isolation class": signal["isolation_prediction"],
                "Isolation score": signal["isolation_score"],
                "XGBoost class": signal["xgboost_prediction"],
                "XGBoost probability": signal["xgboost_probability"],
            })
        rows.append(row)
    grid(rows)


def investigations(batch, report, selected):
    st.title("Cases")
    detections = batch.export_audit_report()["detection_cases"]
    if not detections:
        st.info("Run Detect on the Detection page to populate the case queue.")
        return
    cases = st.session_state.demo_cases
    rows = []
    for detection in detections:
        event_id = detection["event_id"]
        case_report = cases[event_id].export_audit_report() if event_id in cases else None
        rows.append({
            "Event": event_id,
            "Model signals": detection["signal_count"],
            "Models disagree": "Models disagree" if models_disagree(detection) else "Agree",
            "Workflow state": case_report["workflow_state"] if case_report else "Not opened",
            "Incident status": case_report["incident_status"] if case_report else "Not opened",
        })
    selection = st.dataframe(
        rows,
        hide_index=True,
        key="case_queue",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = selection.selection.rows
    if selected_rows:
        candidate = rows[selected_rows[0]]["Event"]
        if candidate != selected:
            perform(open_case, batch, candidate)
            st.rerun()
    record = report["investigation"]
    if not record:
        st.info("Select a row to open its case and inspect the evidence.")


def run_case_operation(operation, *args) -> None:
    """The UI requests a recorded workflow operation; it never changes state itself."""
    perform(operation, *args)


def red_tab(case, report):
    suggestion = report["scenario_suggestion"]
    if suggestion:
        st.caption("INFERENCE · suggested simulated scenario")
        st.info(suggestion["rationale"])
        st.caption("Basis: " + ", ".join(suggestion["basis"]))
    suggested_id = suggestion["scenario_id"] if suggestion else scenario_ids()[0]
    scenario_id = st.selectbox(
        "Scenario for isolated simulation",
        scenario_ids(),
        index=scenario_ids().index(suggested_id),
        key="red_scenario",
        disabled=case.state is not WorkflowState.INVESTIGATED,
    )
    if case.state is WorkflowState.INVESTIGATED and st.button(
        "Run Red simulation", key="run_red_simulation", type="primary"
    ):
        run_case_operation(case.run_red_simulation, scenario_id)
        st.rerun()
    render_graph_3d(report, "red")
    if report["red_finding"]:
        st.caption(report["red_finding"]["explanation"])


def blue_tab(case, report):
    if case.state is WorkflowState.RED_SIMULATED and st.button(
        "Generate Blue recommendation", key="generate_blue_recommendation", type="primary"
    ):
        run_case_operation(case.generate_blue_recommendations)
        st.rerun()
    recommendations = report["recommendations"]
    if not recommendations:
        st.info("A recorded Red finding is required before a Blue recommendation can be generated.")
        return
    for recommendation in recommendations:
        st.subheader(recommendation["title"])
        st.write(recommendation["rationale"])
        st.caption(recommendation["expected_security_effect"])
    render_graph_3d(report, "blue")
    st.markdown(patch_progress_html(report), unsafe_allow_html=True)


def approval_tab(case, report):
    st.markdown(approval_card_html(report), unsafe_allow_html=True)
    if not report["recommendations"]:
        return
    operator = st.text_input("Operator", value="demo-operator", key="approval_operator")
    note = st.text_area("Optional note", max_chars=500, key="approval_note")
    decision_buttons = (
        ("Approve", "approval_approve", ApprovalDecision.APPROVED, ""),
        ("Approve with note", "approval_approve_note", ApprovalDecision.APPROVED, note),
        ("Decline", "approval_decline", ApprovalDecision.DECLINED, note),
        (
            "Request more evidence",
            "approval_more_evidence",
            ApprovalDecision.MORE_EVIDENCE_REQUESTED,
            note,
        ),
        ("Escalate", "approval_escalate", ApprovalDecision.ESCALATED, note),
    )
    buttons = st.columns(len(decision_buttons))
    for column, (label, key, decision, decision_note) in zip(
        buttons, decision_buttons, strict=True
    ):
        with column:
            if st.button(
                label,
                key=key,
                use_container_width=True,
                disabled=case.state is not WorkflowState.BLUE_RECOMMENDED,
            ):
                run_case_operation(case.record_decision, operator, decision, decision_note)
                st.rerun()
    st.markdown(decision_log_html(report), unsafe_allow_html=True)


def sandbox_tab(case, report):
    approval = report["approval"] or {}
    approved = approval.get("decision") == ApprovalDecision.APPROVED.value
    can_apply = case.state is WorkflowState.AWAITING_APPROVAL and approved
    if st.button("Apply control", key="apply_control", type="primary", disabled=not can_apply):
        run_case_operation(case.apply_simulated_remediation)
        st.rerun()
    can_verify = case.state is WorkflowState.REMEDIATION_IN_PROGRESS
    if st.button("Run verifier", key="run_verifier", disabled=not can_verify):
        run_case_operation(case.verify_remediation)
        st.rerun()
    verification = report["verification"]
    if not verification:
        if report["remediation_audit"]:
            st.info("Control applied. Verification has not run.")
        elif approved:
            st.info("Control approved but not applied.")
        else:
            st.info("Control has not been approved.")
        return
    st.caption("replayed from the verifier record")
    replay_key = f"replayed_verification:{verification['verification_id']}"
    slot = st.empty()
    if st.session_state.get(replay_key):
        slot.markdown(
            sandbox_log_html(verification["steps"], len(verification["steps"]), True),
            unsafe_allow_html=True,
        )
    else:
        replay_verifier_steps(report, slot, delay=0.3)
        st.session_state[replay_key] = True
    st.markdown(outcome_badge_html(verification["outcome"]), unsafe_allow_html=True)


def report_tab(report):
    if not report["investigation"]:
        st.info("Open a case in Cases to view its report and downloads.")
        return
    html = render_report_html(report)
    st.components.v1.html(html, height=1400, scrolling=True)
    downloads = st.columns(3)
    with downloads[0]:
        st.download_button(
            "Download HTML",
            html,
            report_filename(report, "html"),
            "text/html",
            key="download_html",
        )
    with downloads[1]:
        st.download_button(
            "Download PDF",
            render_report_pdf(report),
            report_filename(report, "pdf"),
            "application/pdf",
            key="download_pdf",
        )
    with downloads[2]:
        st.download_button(
            "Download JSON",
            json.dumps(report, indent=2),
            report_filename(report, "json"),
            "application/json",
            key="download_json",
        )


def case_workspace(case):
    report = case.export_audit_report()
    st.markdown(workflow_rail_html(report), unsafe_allow_html=True)
    evidence, red, blue, approval, sandbox, report_view = st.tabs(
        ("Evidence", "Red", "Blue", "Approval", "Sandbox", "Report")
    )
    with evidence:
        record = report["investigation"]
        st.subheader(record["investigation_id"])
        st.caption(
            f"Event {report['selected_event_id']} · "
            f"{report['workflow_state'].replace('_', ' ')}"
        )
        flow = next(
            row for row in report["telemetry"]
            if row["event_id"] == report["selected_event_id"]
        )
        grid([flow])
        signal = next(
            row for row in report["detection_cases"]
            if row["event_id"] == report["selected_event_id"]
        )
        grid([
            {
                "Model": "Logistic regression",
                "Class": signal["logistic_prediction"],
                "Score": signal["logistic_probability"],
                "Score type": "Probability",
            },
            {
                "Model": "Isolation forest",
                "Class": signal["isolation_prediction"],
                "Score": signal["isolation_score"],
                "Score type": "Anomaly score",
            },
            {
                "Model": "XGBoost",
                "Class": signal["xgboost_prediction"],
                "Score": signal["xgboost_probability"],
                "Score type": "Probability",
            },
        ])
        with st.expander("Evidence, observations and timeline"):
            keys = ("evidence_references", "observations", "model_signals", "timeline")
            st.json({key: record[key] for key in keys})
    with red:
        red_tab(case, report)
    with blue:
        blue_tab(case, report)
    with approval:
        approval_tab(case, report)
    with sandbox:
        sandbox_tab(case, report)
    with report_view:
        report_tab(report)


def main():
    st.set_page_config(page_title="Asteria | SOC", layout="wide", initial_sidebar_state="expanded")
    css = (ROOT / "assets" / "asteria_workspace.css").read_text(encoding="utf-8")
    st.html(f'<style>{css}</style><span data-asteria-style hidden></span>')
    st.markdown(f"<style>{component_css()}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.html('<div data-asteria-brand><span aria-hidden="true">✦</span>Asteria</div>')
        view = st.radio("Console navigation", VIEWS, label_visibility="collapsed", key="demo_view")

    st.warning(
        "Demo - isolated simulation. Public/synthetic data only. "
        "State is per session and not durable."
    )

    if "demo_batch" not in st.session_state:
        st.session_state.demo_batch = DemoOrchestrator(ROOT)
        st.session_state.demo_cases = {}
    batch = st.session_state.demo_batch
    event_id = st.session_state.get("demo_event")
    case = st.session_state.demo_cases.get(event_id)
    error = st.session_state.get("demo_error")
    if error:
        st.error(error)

    if view == "Command Center":
        home(batch)
    elif view == "Detection":
        detection(batch, batch.export_audit_report())
    elif view == "Cases":
        investigations(batch, (case or batch).export_audit_report(), event_id)
        if st.session_state.get("demo_event"):
            case = st.session_state.demo_cases[st.session_state.demo_event]
            case_workspace(case)
    elif view == "Data Sources":
        st.title("Data Sources")
        st.info("Not connected in this demo")
    elif view == "Audit & Reports":
        report_tab((case or batch).export_audit_report())


if __name__ == "__main__":
    main()
