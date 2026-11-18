"""Asteria's presentation demo, backed only by the isolated DemoOrchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from student2_demo_orchestrator import DemoOrchestrator, WorkflowState
from student2_demo_qa import answer
from student2_demo_ui import screen_html

ROOT = Path(__file__).resolve().parent
VIEWS = ("Command Center", "Detection Engine", "Investigations", "Red Simulation", "Blue Response", "Verification", "Audit & Reports")


def workflow() -> DemoOrchestrator:
    if "workflow" not in st.session_state:
        st.session_state.workflow = DemoOrchestrator(ROOT)
    return st.session_state.workflow


def action(label: str, allowed: bool, fn) -> None:
    if st.button(label, disabled=not allowed, use_container_width=True):
        try:
            fn()
            st.rerun()
        except (PermissionError, RuntimeError, TypeError, ValueError) as error:
            st.error(f"Simulation action failed safely: {error}")


def route_action(app: DemoOrchestrator, action_id: str) -> None:
    transitions = {
        "initialize": (WorkflowState.CREATED, lambda: app.initialize(training_rows=20_000, event_count=10)),
        "detect": (WorkflowState.MODELS_READY, app.run_detection),
        "investigate": (WorkflowState.DETECTED, app.start_investigation),
        "red": (WorkflowState.INVESTIGATED, lambda: app.run_red_simulation("exposed_api")),
        "blue": (WorkflowState.RED_SIMULATED, app.generate_blue_recommendations),
        "approve": (WorkflowState.BLUE_RECOMMENDED, lambda: app.approve_recommendation("demo-operator")),
        "apply": (WorkflowState.AWAITING_APPROVAL, app.apply_simulated_remediation),
        "verify": (WorkflowState.REMEDIATION_IN_PROGRESS, app.verify_remediation),
    }
    transition = transitions.get(action_id)
    if transition is not None and app.state is transition[0]:
        transition[1]()


def next_action(app: DemoOrchestrator) -> tuple[str, str] | None:
    actions = {
        WorkflowState.CREATED: ("initialize", "INITIALIZE MODELS"),
        WorkflowState.MODELS_READY: ("detect", "RUN DETECTION"),
        WorkflowState.DETECTED: ("investigate", "START INVESTIGATION"),
        WorkflowState.INVESTIGATED: ("red", "RUN RED SIMULATION"),
        WorkflowState.RED_SIMULATED: ("blue", "GENERATE BLUE PROPOSAL"),
        WorkflowState.BLUE_RECOMMENDED: ("approve", "APPROVE SANDBOX CONTROL"),
        WorkflowState.AWAITING_APPROVAL: ("apply", "APPLY SIMULATION"),
        WorkflowState.REMEDIATION_IN_PROGRESS: ("verify", "RUN VERIFIER"),
    }
    return actions.get(app.state)


def controls(app: DemoOrchestrator, view: str, report: dict[str, object]) -> None:
    st.markdown("<div class='control-label'>DEMO CONTROL PLANE · ACTIONS REMAIN IN THE ISOLATED IN-MEMORY ENVIRONMENT</div>", unsafe_allow_html=True)
    if view in {"Command Center", "Detection Engine", "Investigations"}:
        columns = st.columns(4)
        with columns[0]:
            action("01  INITIALIZE MODELS", app.state is WorkflowState.CREATED, lambda: app.initialize(training_rows=20_000, event_count=10))
        with columns[1]:
            action("02  RUN DETECTION", app.state is WorkflowState.MODELS_READY, app.run_detection)
        with columns[2]:
            action("03  START INVESTIGATION", app.state is WorkflowState.DETECTED, app.start_investigation)
        with columns[3]:
            action("04  RUN RED SIMULATION", app.state is WorkflowState.INVESTIGATED, lambda: app.run_red_simulation("exposed_api"))
    elif view == "Red Simulation":
        action("RUN EXPOSED-API SIMULATION", app.state is WorkflowState.INVESTIGATED, lambda: app.run_red_simulation("exposed_api"))
    elif view == "Blue Response":
        columns = st.columns(2)
        with columns[0]:
            action("GENERATE BLUE PROPOSAL", app.state is WorkflowState.RED_SIMULATED, app.generate_blue_recommendations)
        with columns[1]:
            operator = st.text_input("Operator identity", value="demo-operator", label_visibility="collapsed", placeholder="Operator identity")
            action("APPROVE SIMULATED CONTROL", app.state is WorkflowState.BLUE_RECOMMENDED, lambda: app.approve_recommendation(operator))
    elif view == "Verification":
        columns = st.columns(2)
        with columns[0]:
            action("APPLY APPROVED SIMULATION", app.state is WorkflowState.AWAITING_APPROVAL, app.apply_simulated_remediation)
        with columns[1]:
            action("RUN INDEPENDENT VERIFIER", app.state is WorkflowState.REMEDIATION_IN_PROGRESS, app.verify_remediation)
    elif view == "Audit & Reports":
        payload = json.dumps(report, indent=2, default=str)
        columns = st.columns(2)
        with columns[0]:
            st.download_button("DOWNLOAD AUDIT JSON", payload, "asteria-audit.json", "application/json", use_container_width=True)
        with columns[1]:
            st.download_button("DOWNLOAD AUDIT MARKDOWN", "# Asteria Incident Report\n\n" + payload, "asteria-audit.md", "text/markdown", use_container_width=True)
        question = st.text_input("Ask a record-grounded question", placeholder="What is the current verification state?")
        if question:
            st.info(answer("Security Engineer", question, report))


def main() -> None:
    st.set_page_config(page_title="ASTERIA | SOC", layout="wide", initial_sidebar_state="collapsed")
    css = (ROOT / "assets" / "asteria.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    app = workflow()
    requested_action = st.query_params.get("action")
    if isinstance(requested_action, str):
        route_action(app, requested_action)
        requested_route = st.query_params.get("view")
        st.query_params.clear()
        if requested_route in VIEWS:
            st.query_params["view"] = requested_route
        st.rerun()
    report = app.export_audit_report()
    requested_view = st.query_params.get("view")
    if requested_view in VIEWS:
        st.session_state.active_soc_view = requested_view
    view = st.session_state.get("active_soc_view", "Command Center")
    nav_column, action_column = st.columns([7, 2])
    with nav_column:
        selected_view = st.radio("SOC navigation", VIEWS, horizontal=True, label_visibility="collapsed", key="active_soc_view")
    with action_column:
        transition = next_action(app)
        if transition is not None and st.button(transition[1], use_container_width=True):
            route_action(app, transition[0])
            st.rerun()
    if selected_view != view:
        st.query_params["view"] = selected_view
        st.rerun()
    heights = {"Command Center": 1120, "Detection Engine": 820, "Investigations": 820, "Red Simulation": 970, "Blue Response": 850, "Verification": 990, "Audit & Reports": 780}
    components.html(screen_html(report, view), height=heights[view], scrolling=True)


if __name__ == "__main__":
    main()
