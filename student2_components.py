"""Presentational HTML components for the Asteria case workspace.

Every function turns backend report state into HTML. None of them contain a
button or can change state: Streamlit buttons stay in ``demo_app.py`` and call
the orchestrator. All text is escaped. Load ``assets/asteria_components.css``
once per run (see ``component_css``).
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence

CSS_PATH = Path(__file__).resolve().parent / "assets" / "asteria_components.css"

OUTCOMES = {
    "blocked": ("BLOCKED", "good", "The target is no longer reachable."),
    "still_reachable": ("STILL REACHABLE", "bad", "The target is still reachable after the control."),
    "inconclusive": ("INCONCLUSIVE", "warn", "The checks could not establish a result."),
    "failed": ("FAILED", "bad", "A verification check did not pass."),
}
DECISION_LABELS = {
    "approved": "Approved", "declined": "Declined", "escalated": "Escalated",
    "more_evidence_requested": "More evidence requested",
}
ROLLBACK_NOTE = (
    "Rollback: the demo environment has reset() only. There is no live rollback, "
    "because nothing outside the simulation is ever changed."
)


def component_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _e(value: object) -> str:
    return escape(str(value))


def outcome_badge_html(outcome: str | None) -> str:
    """Text + colour badge, so meaning never rests on colour alone."""
    if outcome not in OUTCOMES:
        return "<span class='ast-badge ast-pending'>PENDING</span>"
    label, tone, _ = OUTCOMES[outcome]
    mark = {"good": "✓", "bad": "✕", "warn": "!"}[tone]
    return f"<span class='ast-badge ast-{tone}'>{mark} {label}</span>"


def workflow_rail_html(report: Mapping[str, object]) -> str:
    """Pipeline strip. Each stage is done/current/pending, read from recorded report fields."""
    approval = report.get("approval") or {}
    verification = report.get("verification") or {}
    done = [
        bool(report.get("detection_cases")) or bool(report.get("investigation")),
        bool(report.get("investigation")),
        bool(report.get("red_finding")),
        bool(report.get("recommendations")),
        approval.get("decision") == "approved",
        bool(report.get("remediation_audit")),
        verification.get("outcome") == "blocked",
    ]
    names = ("Detect", "Investigate", "Red", "Blue", "Approval", "Sandbox", "Verified")
    current = next((i for i, d in enumerate(done) if not d), len(done))
    cells = []
    for i, (name, flag) in enumerate(zip(names, done, strict=True)):
        state = "done" if flag else ("current" if i == current else "todo")
        word = {"done": "done", "current": "next", "todo": "pending"}[state]
        cells.append(f"<li class='ast-step {state}'><b>{_e(name)}</b><small>{word}</small></li>")
    return f"<ol class='ast-rail' aria-label='Case workflow'>{''.join(cells)}</ol>"


def affected_nodes(report: Mapping[str, object]) -> list[str]:
    """Node names touched by edges the recommended control can block (from attack_graph only)."""
    graph = report.get("attack_graph") or {}
    control = graph.get("control_id")
    names = {n["node_id"]: n["display_name"] for n in graph.get("nodes", [])}
    seen: dict[str, None] = {}
    for edge in graph.get("edges", []):
        if control in edge.get("blocking_control_ids", []):
            seen[names.get(edge["source_node_id"], edge["source_node_id"])] = None
            seen[names.get(edge["target_node_id"], edge["target_node_id"])] = None
    return list(seen)


def approval_card_html(report: Mapping[str, object]) -> str:
    """Decision card: risk, affected nodes, before/after control state, rollback note, evidence."""
    recs = report.get("recommendations") or []
    if not recs:
        return "<div class='ast-card ast-empty'>No recommendation yet. Generate the Blue recommendation first.</div>"
    rec = recs[0]
    control = rec["target_control_id"]
    env = report.get("environment") or {}
    initial = (env.get("initial_state") or {}).get(control)
    current = (env.get("current_state") or {}).get(control)
    word = lambda v: "unknown" if v is None else ("active" if v else "inactive")  # noqa: E731
    nodes = affected_nodes(report)
    exposure = report.get("graph_exposure") or {}
    edges_total = len((report.get("attack_graph") or {}).get("edges", []))
    evidence = "".join(f"<li>{_e(i)}</li>" for i in rec.get("supporting_evidence_ids", []))
    priority = str(rec["priority"])
    after = "active (expected after approval)" if current is False else word(current)
    return (
        "<div class='ast-card'>"
        f"<div class='ast-card-head'><div><small>RECOMMENDED CONTROL (simulation only)</small><h3>{_e(rec['title'])}</h3></div>"
        f"<span class='ast-badge ast-risk-{_e(priority)}'>RISK {_e(priority.upper())}</span></div>"
        f"<p>{_e(rec['rationale'])}</p>"
        "<div class='ast-grid'>"
        f"<div><small>CONTROL</small><code>{_e(control)}</code></div>"
        f"<div><small>BEFORE</small><b>{_e(word(initial))}</b></div>"
        f"<div><small>NOW</small><b>{_e(word(current))}</b></div>"
        f"<div><small>AFTER APPROVAL</small><b>{_e(after)}</b></div>"
        f"<div><small>TARGET REACHABLE NOW</small><b>{'yes' if exposure.get('target_reachable') else 'no'}</b></div>"
        f"<div><small>EDGES THIS CONTROL CAN BLOCK</small><b>{len([1 for e in (report.get('attack_graph') or {}).get('edges', []) if control in e.get('blocking_control_ids', [])])} of {edges_total}</b></div>"
        "</div>"
        f"<div class='ast-row'><small>AFFECTED NODES</small><div class='ast-chips'>{''.join(f'<span>{_e(n)}</span>' for n in nodes) or '<span>none recorded</span>'}</div></div>"
        f"<div class='ast-row'><small>EXPECTED EFFECT</small><p>{_e(rec['expected_security_effect'])} {_e(rec['expected_verifier_condition'])}</p></div>"
        f"<div class='ast-row'><small>OPERATIONAL IMPACT</small><p>{_e(rec['operational_impact'])}</p></div>"
        f"<div class='ast-row'><small>SUPPORTING EVIDENCE</small><ul>{evidence}</ul></div>"
        f"<div class='ast-note'>{_e(ROLLBACK_NOTE)}</div>"
        "</div>"
    )


def decision_log_html(report: Mapping[str, object]) -> str:
    history = report.get("decision_history") or []
    if not history:
        return "<div class='ast-card ast-empty'>No operator decision recorded yet.</div>"
    rows = "".join(
        f"<tr><td>{i}</td><td>{_e(h['operator_identity'])}</td><td><b>{_e(DECISION_LABELS.get(h['decision'], h['decision']))}</b></td>"
        f"<td>{_e(h['timestamp'])}</td><td>{_e(h.get('note') or '-')}</td></tr>"
        for i, h in enumerate(history, 1)
    )
    return ("<table class='ast-table'><thead><tr><th>#</th><th>Operator</th><th>Decision</th><th>Time (UTC)</th><th>Note</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def patch_progress_html(report: Mapping[str, object]) -> str:
    """Blue patch progress: four stages, each read from a recorded report field."""
    approval = report.get("approval") or {}
    verification = report.get("verification") or {}
    stages = (
        ("Recommended", bool(report.get("recommendations")), "Blue recommendation recorded"),
        ("Approved", approval.get("decision") == "approved", DECISION_LABELS.get(approval.get("decision"), "no approval yet")),
        ("Applied in sandbox", bool(report.get("remediation_audit")), "control state changed in the simulation"),
        ("Verified", verification.get("outcome") == "blocked", OUTCOMES.get(verification.get("outcome"), ("", "", "not verified yet"))[2]),
    )
    first_open = next((i for i, s in enumerate(stages) if not s[1]), len(stages))
    items = []
    for i, (name, flag, detail) in enumerate(stages):
        state = "done" if flag else ("current" if i == first_open else "todo")
        mark = {"done": "✓", "current": "→", "todo": "·"}[state]
        items.append(f"<li class='ast-prog {state}'><span>{mark}</span><div><b>{_e(name)}</b><small>{_e(detail)}</small></div></li>")
    done = sum(1 for s in stages if s[1])
    return (f"<div class='ast-card'><small>PATCH PROGRESS · {done} of {len(stages)} stages recorded</small>"
            f"<ul class='ast-progress'>{''.join(items)}</ul></div>")


def sandbox_log_html(steps: Sequence[Mapping[str, object]], total: int | None = None, finished: bool = False) -> str:
    """Terminal-style log of verifier steps shown so far."""
    lines = "".join(
        f"<div class='ast-line {'ok' if s['passed'] else 'bad'}'><span class='n'>{int(s['sequence']):02d}</span>"
        f"<span class='r'>{'PASS' if s['passed'] else 'FAIL'}</span><span class='k'>{_e(s['name'])}</span>"
        f"<span class='d'>{_e(s['detail'])}</span></div>"
        for s in steps
    )
    if not finished and total is not None and len(steps) < total:
        lines += "<div class='ast-line run'><span class='n'>..</span><span class='r'>RUN</span><span class='k'>running</span><span class='d'>next check</span></div>"
    return f"<div class='ast-log' role='log' aria-live='polite'>{lines or '<div class=\"ast-line\">waiting for the verifier record</div>'}</div>"


def replay_verifier_steps(report: Mapping[str, object], placeholder, delay: float = 0.3) -> None:
    """Replay recorded verifier steps one by one into a ``st.empty()`` placeholder.

    Only the pacing is cosmetic; every line comes from ``verification['steps']``.
    ``delay`` is capped at 0.4 s per step.
    """
    import time

    verification = report.get("verification") or {}
    steps = list(verification.get("steps", []))
    delay = max(0.0, min(delay, 0.4))
    for count in range(1, len(steps) + 1):
        placeholder.markdown(sandbox_log_html(steps[:count], len(steps), finished=count == len(steps)), unsafe_allow_html=True)
        if count < len(steps):
            time.sleep(delay)
    if not steps:
        placeholder.markdown(sandbox_log_html([]), unsafe_allow_html=True)
