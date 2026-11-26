"""The Asteria demo workspace and its session-scoped controls."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from student2_demo_data import initialize_baseline_engine, load_demo_data
from student2_demo_orchestrator import DemoOrchestrator, WorkflowState
from student2_demo_qa import answer
from student2_xgboost_runtime import XGBoostRuntime

ROOT = Path(__file__).resolve().parent
VIEWS = (
    "Command Center", "Cases", "Detection", "Data Sources", "Audit & Reports",
)
STEPS = {
    WorkflowState.INVESTIGATED: ("Generate simulated path", "red"),
    WorkflowState.RED_SIMULATED: ("Prepare defensive proposal", "blue"),
    WorkflowState.BLUE_RECOMMENDED: ("Approve simulated control", "approve"),
    WorkflowState.AWAITING_APPROVAL: ("Apply approved sandbox control", "apply"),
    WorkflowState.REMEDIATION_IN_PROGRESS: ("Run independent verifier", "verify"),
}


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
        st.button("Detect", key="detect", on_click=perform, args=(batch.run_detection,), type="primary")
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
        return
    st.subheader(record["investigation_id"])
    st.caption(f"Event {selected} · {report['workflow_state'].replace('_', ' ')}")
    flow = next(row for row in report["telemetry"] if row["event_id"] == selected)
    grid([flow])
    st.subheader("Model signals")
    signal = next(row for row in report["detection_cases"] if row["event_id"] == selected)
    grid([
        {
            "Model": "Logistic regression", "Class": signal["logistic_prediction"],
            "Score": signal["logistic_probability"], "Score type": "Probability",
        },
        {
            "Model": "Isolation forest", "Class": signal["isolation_prediction"],
            "Score": signal["isolation_score"], "Score type": "Anomaly score",
        },
        {
            "Model": "XGBoost", "Class": signal["xgboost_prediction"],
            "Score": signal["xgboost_probability"], "Score type": "Probability",
        },
    ])
    with st.expander("Evidence, observations and timeline"):
        keys = ("evidence_references", "observations", "model_signals", "timeline")
        st.json({key: record[key] for key in keys})


def stage_button(case, operation, *args):
    label, key = STEPS[case.state]
    st.button(label, key=key, on_click=perform, args=(operation, *args), type="primary")


def red_panel(case, report):
    st.title("Red Simulation")
    st.caption("Isolated scenario visualization")
    finding = report["red_finding"]
    if case.state is WorkflowState.INVESTIGATED:
        stage_button(case, case.run_red_simulation, "exposed_api")
    if not finding:
        st.info("Generate the simulated path to display it for this event.")
        return
    graph = report["attack_graph"]
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    cards = [
        "<article><strong>" + escape(nodes[node_id]["display_name"]) + "</strong></article>"
        for node_id in graph["candidate_path"]["node_ids"]
    ]
    connector = '<span class="connector" aria-hidden="true">→</span>'
    st.html('<div class="asteria-graph">' + connector.join(cards) + "</div>")
    st.write(finding["explanation"])
    for limitation in finding["limitations"]:
        st.caption(limitation)


def blue_panel(case, report):
    st.title("Blue Response")
    if case.state is WorkflowState.RED_SIMULATED:
        stage_button(case, case.generate_blue_recommendations)
    if not report["recommendations"]:
        st.info("Generate a path on Red Simulation first, then prepare a defensive proposal here.")
        return
    for item in report["recommendations"]:
        st.subheader(item["title"])
        st.write(item["rationale"])
        st.write(item["expected_security_effect"])
        st.caption(item["operational_impact"])
    if case.state is WorkflowState.BLUE_RECOMMENDED:
        stage_button(case, case.approve_recommendation, "demo-operator")
    if report["approval"]:
        st.success("Sandbox control approved. Continue in Verification.")


def verification_panel(case, report):
    st.title("Verification")
    if case.state is WorkflowState.AWAITING_APPROVAL and report["approval"]:
        stage_button(case, case.apply_simulated_remediation)
    elif case.state is WorkflowState.REMEDIATION_IN_PROGRESS:
        stage_button(case, case.verify_remediation)
    verification = report["verification"]
    if not verification:
        if report["remediation_audit"]:
            st.info("Control applied. Verification has not run.")
        elif (report["approval"] or {}).get("decision") == "approved":
            st.info("Control approved but not applied.")
        else:
            st.info("Control has not been approved.")
        return
    present = st.success if verification["status"] == "verified" else st.error
    present(verification["explanation"])
    before, after = st.columns(2)
    before.metric("Before simulated control", "Reachable" if verification["before_graph_reachable"] else "Blocked")
    after.metric("After simulated control", "Reachable" if verification["after_graph_reachable"] else "Blocked")
    grid(verification["checks"])


def audit(report):
    st.title("Audit & Reports")
    record = report["investigation"]
    if not record:
        st.info("Open a case in Cases to view its audit and downloads.")
        return
    details, exports = st.columns([3, 1])
    with details:
        st.subheader(record["investigation_id"])
        st.caption(f"Event {report['selected_event_id']} · {report['incident_status'].replace('_', ' ')}")
        st.subheader("Investigation timeline")
        grid(record["timeline"])
        for label, key in (
            ("Evidence and observations", "investigation"), ("Simulated path", "red_finding"),
            ("Defensive proposals", "recommendations"), ("Approval", "approval"),
            ("Sandbox application", "remediation_audit"), ("Verification", "verification"),
        ):
            with st.expander(label):
                if report[key]:
                    st.json(report[key])
                else:
                    st.caption("Pending")
    with exports:
        payload = json.dumps(report, indent=2)
        st.download_button("Download JSON", payload, "asteria-audit.json", "application/json")
        markdown = "# Asteria Incident Report\n\n" + chr(96) * 3 + "json\n" + payload + "\n" + chr(96) * 3 + "\n"
        st.download_button("Download Markdown", markdown, "asteria-audit.md", "text/markdown")
        question = st.text_input("Ask about this record")
        if question:
            st.info(answer("Security Engineer", question, report))


def main():
    st.set_page_config(page_title="Asteria | SOC", layout="wide", initial_sidebar_state="expanded")
    css = (ROOT / "assets" / "asteria_workspace.css").read_text(encoding="utf-8")
    st.html(f'<style>{css}</style><span data-asteria-style hidden></span>')
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
            panel_name = st.radio(
                "Case stage", ("Evidence", "Red Simulation", "Blue Response", "Verification"),
                horizontal=True, key="case_stage",
            )
            panel = {
                "Red Simulation": red_panel,
                "Blue Response": blue_panel,
                "Verification": verification_panel,
            }
            if panel_name != "Evidence":
                panel[panel_name](case, case.export_audit_report())
    elif view == "Data Sources":
        st.title("Data Sources")
        st.info("Not connected in this demo")
    elif view == "Audit & Reports":
        audit((case or batch).export_audit_report())


if __name__ == "__main__":
    main()
