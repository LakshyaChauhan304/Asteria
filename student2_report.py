"""Exact, downloadable case report for the isolated Asteria demo.

One section model (``report_sections``) feeds BOTH renderers, so the on-screen
HTML, the HTML download and the PDF download always carry the same sections in
the same order with the same facts:

    Summary, Detection signals, Investigation timeline, Red finding and graph
    exposure, Blue recommendation, Decisions, Remediation audit,
    Verification steps, Limitations

Rules: every value comes from ``DemoOrchestrator.export_audit_report()``; a stage
that has not happened yet renders as "Pending" (never omitted); all text is
escaped; FACT / INFERENCE / HYPOTHESIS labels are kept separate.
"""

from __future__ import annotations

import io
from html import escape
from typing import Mapping
from xml.sax.saxutils import escape as xml_escape

from student2_graph3d import graph_payload

SECTION_TITLES = (
    "Summary", "Detection signals", "Investigation timeline", "Red finding and graph exposure",
    "Blue recommendation", "Decisions", "Remediation audit", "Verification steps", "Limitations",
)
PENDING = "Pending"
SIMULATION_NOTE = (
    "Simulation only. Nothing in this report describes a change to real infrastructure, "
    "and no live rollback exists in the demo environment (reset only)."
)
_OUTCOME_TEXT = {
    "blocked": "BLOCKED: the target is no longer reachable after the approved control.",
    "still_reachable": "STILL REACHABLE: the target remains reachable after the control.",
    "inconclusive": "INCONCLUSIVE: the checks could not establish a result.",
    "failed": "FAILED: a verification check did not pass.",
}


def _text(value: object) -> str:
    return PENDING if value in (None, "", [], {}) else str(value)


def _pending(title: str, what: str) -> list[tuple]:
    return [("note", f"{PENDING}: {what} has not happened yet for this case.", "pending")]


def _summary(r: Mapping) -> list[tuple]:
    verification = r.get("verification") or {}
    red = r.get("red_finding") or {}
    suggestion = r.get("scenario_suggestion") or {}
    outcome = verification.get("outcome")
    return [
        ("note", SIMULATION_NOTE, "info"),
        ("kv", [
            ("Event", _text(r.get("selected_event_id"))),
            ("Workflow state", _text(r.get("workflow_state"))),
            ("Incident status", _text(r.get("incident_status"))),
            ("Scenario simulated", _text(red.get("scenario_id"))),
            ("Scenario suggested (INFERENCE)", _text(suggestion.get("scenario_id"))),
            ("Verification outcome", _OUTCOME_TEXT.get(outcome, PENDING)),
        ]),
    ]


def _detection(r: Mapping) -> list[tuple]:
    event_id = r.get("selected_event_id")
    row = next((c for c in r.get("detection_cases", []) if str(c.get("event_id")) == str(event_id)), None)
    if row:
        rows = [
            ["Logistic regression", row["logistic_prediction"], row["logistic_probability"]],
            ["Isolation forest", row["isolation_prediction"], row["isolation_score"]],
            ["XGBoost", row["xgboost_prediction"], row["xgboost_probability"]],
        ]
        agree = len({row["logistic_prediction"], row["isolation_prediction"], row["xgboost_prediction"]}) == 1
        return [
            ("label", "FACT: recorded model outputs. Each output is a signal, not proof of compromise."),
            ("table", ["Model", "Predicted class", "Score"], rows),
            ("kv", [("Signals raised", row["signal_count"] if "signal_count" in row else _text(None)),
                    ("Models", "agree" if agree else "disagree")]),
        ]
    evidence = (r.get("investigation") or {}).get("evidence")
    if not evidence:
        return _pending("Detection signals", "detection")
    return [
        ("label", "FACT: sealed detection evidence recorded for this investigation."),
        ("kv", [("Supervised prediction", evidence.get("model_prediction")),
                ("Supervised score", evidence.get("model_score")),
                ("Anomaly prediction", evidence.get("anomaly_prediction")),
                ("Anomaly score", evidence.get("anomaly_score")),
                ("Protocol / service / state", f"{evidence.get('protocol')} / {evidence.get('service')} / {evidence.get('state')}")]),
    ]


def _timeline(r: Mapping) -> list[tuple]:
    inv = r.get("investigation")
    if not inv:
        return _pending("Investigation timeline", "the investigation")
    blocks: list[tuple] = []
    timeline = inv.get("timeline", [])
    if timeline:
        blocks.append(("table", ["#", "Observation", "Evidence"],
                       [[t["sequence"], t["summary"], ", ".join(t.get("evidence_reference_ids", []))] for t in timeline]))
    else:
        blocks.append(("note", f"{PENDING}: no timeline entries are recorded.", "pending"))
    hyps = inv.get("hypotheses", [])
    if hyps:
        blocks.append(("label", "HYPOTHESIS: statements under test. None is a confirmed finding."))
        blocks.append(("table", ["ID", "Hypothesis", "Status"], [[h["hypothesis_id"], h["statement"], h["status"]] for h in hyps]))
    return blocks


def _red(r: Mapping) -> list[tuple]:
    red, graph, exposure = r.get("red_finding"), r.get("attack_graph") or {}, r.get("graph_exposure") or {}
    blocks: list[tuple] = []
    suggestion = r.get("scenario_suggestion")
    if suggestion:
        blocks.append(("label", "INFERENCE: scenario suggested from the flow's displayed fields. The analyst may override it."))
        blocks.append(("kv", [("Suggested scenario", suggestion.get("scenario_id")),
                              ("Rationale", suggestion.get("rationale")),
                              ("Basis", ", ".join(suggestion.get("basis", [])) if isinstance(suggestion.get("basis"), (list, tuple)) else suggestion.get("basis"))]))
    if not red or not graph:
        return blocks + _pending("Red finding", "the Red simulation")
    names = {n["node_id"]: n["display_name"] for n in graph["nodes"]}
    path = red["candidate_path"]
    blocks.append(("label", "FACT: derived from the simulated graph and its current control state."))
    blocks.append(("kv", [
        ("Scenario", red["scenario_id"]),
        ("Candidate path", " > ".join(names.get(n, n) for n in path["node_ids"])),
        ("Target reachable now", "yes" if exposure.get("target_reachable") else "no"),
        ("Edges blocked", f"{len(exposure.get('blocked_edge_ids', []))} of {len(graph['edges'])}"),
        ("Decoy nodes", ", ".join(names.get(n, n) for n in exposure.get("decoy_node_ids", [])) or "none"),
        ("Explanation", red.get("explanation")),
    ]))
    blocks.append(("graph", None))
    blocks.append(("table", ["Edge", "From", "To", "Status"], [
        [e["edge_id"], names.get(e["source_node_id"]), names.get(e["target_node_id"]),
         "BLOCKED" if e["edge_id"] in exposure.get("blocked_edge_ids", []) else "OPEN"] for e in graph["edges"]]))
    return blocks


def _blue(r: Mapping) -> list[tuple]:
    recs = r.get("recommendations") or []
    if not recs:
        return _pending("Blue recommendation", "the Blue recommendation")
    out: list[tuple] = []
    for rec in recs:
        out.append(("kv", [
            ("Recommendation", rec["title"]), ("Priority", rec["priority"]),
            ("Control", rec["target_control_id"]), ("Rationale", rec["rationale"]),
            ("Expected effect", rec["expected_security_effect"]),
            ("Verifier condition", rec["expected_verifier_condition"]),
            ("Operational impact", rec["operational_impact"]),
            ("Supporting evidence", ", ".join(rec.get("supporting_evidence_ids", []))),
            ("Execution mode", rec["execution_mode"]),
        ]))
    return out


def _decisions(r: Mapping) -> list[tuple]:
    history = r.get("decision_history") or []
    if not history:
        return _pending("Decisions", "an operator decision")
    return [("table", ["#", "Operator", "Decision", "Time (UTC)", "Note"],
             [[i, h["operator_identity"], h["decision"], h["timestamp"], h.get("note") or "-"] for i, h in enumerate(history, 1)])]


def _remediation(r: Mapping) -> list[tuple]:
    audit = r.get("remediation_audit")
    if not audit:
        return _pending("Remediation audit", "applying the control")
    return [("kv", [
        ("Audit ID", audit["audit_id"]), ("Control", audit["control_id"]),
        ("Before", audit["before_state"]), ("After", audit["after_state"]),
        ("Approval", audit["approval_id"]), ("Time (UTC)", audit["timestamp"]),
        ("Mode", "simulation only"),
    ])]


def _verification(r: Mapping) -> list[tuple]:
    v = r.get("verification")
    if not v:
        return _pending("Verification steps", "verification")
    return [
        ("note", _OUTCOME_TEXT.get(v.get("outcome"), _text(v.get("outcome"))), "ok" if v.get("outcome") == "blocked" else "warn"),
        ("table", ["#", "Step", "Result", "Detail"],
         [[s["sequence"], s["name"], "PASS" if s["passed"] else "FAIL", s["detail"]] for s in v.get("steps", [])]),
        ("kv", [("Reachable before", v.get("before_graph_reachable")), ("Reachable after", v.get("after_graph_reachable"))]),
    ]


def _limitations(r: Mapping) -> list[tuple]:
    items = [SIMULATION_NOTE, "Model outputs are signals; compromise needs independent evidence.",
             "The scenario suggestion is a heuristic inference, not a finding.",
             "Data is public or synthetic and state is per session, not durable."]
    items += list((r.get("red_finding") or {}).get("limitations", []))
    items += list((r.get("investigation") or {}).get("limitations", []))
    return [("list", list(dict.fromkeys(str(i) for i in items)))]


def report_sections(report: Mapping[str, object]) -> list[dict]:
    builders = (_summary, _detection, _timeline, _red, _blue, _decisions, _remediation, _verification, _limitations)
    return [{"title": t, "blocks": b(report)} for t, b in zip(SECTION_TITLES, builders, strict=True)]


# ---------------------------------------------------------------- HTML ----

_CSS = """
:root{--ink:#14202e;--muted:#5a6b7d;--line:#d6dde6;--bg:#fff;--soft:#f3f6fa;--ok:#0a7d5a;--warn:#a15c00;--info:#1d5fa8;--pend:#6b5a8a}
*{box-sizing:border-box}body{margin:0;background:#e9eef4;color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,Segoe UI,sans-serif}
.page{max-width:900px;margin:18px auto;background:var(--bg);padding:34px 40px;border-radius:10px;box-shadow:0 1px 8px #0002}
header{border-bottom:3px solid var(--ink);padding-bottom:12px;margin-bottom:6px}header h1{margin:0;font-size:22px;letter-spacing:.02em}
header p{margin:4px 0 0;color:var(--muted);font-size:12.5px}
section{margin-top:22px;break-inside:avoid-page}h2{font-size:15px;margin:0 0 8px;padding:5px 0;border-bottom:1px solid var(--line)}
h2 span{color:var(--muted);font-weight:400;margin-right:6px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--soft)}dl{display:grid;grid-template-columns:200px 1fr;gap:4px 14px;margin:8px 0}dt{color:var(--muted)}dd{margin:0;word-break:break-word}
.note{padding:8px 11px;border-left:4px solid;background:var(--soft);margin:8px 0;border-radius:0 4px 4px 0}
.note.info{border-color:var(--info)}.note.ok{border-color:var(--ok);color:var(--ok);font-weight:600}.note.warn{border-color:var(--warn);color:var(--warn);font-weight:600}.note.pending{border-color:var(--pend);color:var(--pend)}
.label{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.04em;color:var(--muted);margin:6px 0 2px}
svg{width:100%;height:auto;background:var(--soft);border:1px solid var(--line);border-radius:6px;margin:8px 0}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;padding:0}}
@media (max-width:640px){.page{padding:18px}dl{grid-template-columns:1fr}}
"""


def _svg(r: Mapping) -> str:
    payload = graph_payload(r, "red")
    if payload is None:
        return ""
    xs = [n["pos"][0] + n["pos"][2] * 0.45 for n in payload["nodes"]]
    ys = [-(n["pos"][1]) + n["pos"][2] * 0.35 for n in payload["nodes"]]
    x0, y0 = min(xs) - 1.3, min(ys) - 0.9
    sx, sy = 74, 64
    width, height = (max(xs) - x0 + 1.3) * sx, (max(ys) - y0 + 0.9) * sy
    pt = {n["id"]: ((xs[i] - x0) * sx, (ys[i] - y0) * sy) for i, n in enumerate(payload["nodes"])}
    parts = [f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='Attack graph. Dashed lines are blocked edges.'>"]
    for e in payload["edges"]:
        (ax, ay), (bx, by) = pt[e["source"]], pt[e["target"]]
        blocked = e["status"] == "blocked"
        color = "#0a7d5a" if blocked else ("#c0392b" if e["on_path"] else "#8796a8")
        parts.append(f"<line x1='{ax:.0f}' y1='{ay:.0f}' x2='{bx:.0f}' y2='{by:.0f}' stroke='{color}' stroke-width='{2.4 if e['on_path'] else 1.4}'"
                     + (" stroke-dasharray='6 4'" if blocked else "") + "/>")
        if blocked:
            mx, my = (ax + bx) / 2, (ay + by) / 2
            parts.append(f"<path d='M{mx-4:.0f} {my-4:.0f}l8 8m0-8l-8 8' stroke='{color}' stroke-width='2'/>")
    for n in payload["nodes"]:
        x, y = pt[n["id"]]
        fill = "#9aa7b6" if n["kind"] == "decoy" else ("#e0a21a" if n["kind"] == "target" else "#c0392b" if n["kind"] == "entry" else "#2c6fbb")
        if n["kind"] == "target":
            parts.append(f"<rect x='{x-9:.0f}' y='{y-9:.0f}' width='18' height='18' fill='{fill}'/>")
        elif n["kind"] == "decoy":
            parts.append(f"<path d='M{x:.0f} {y-9:.0f}L{x+9:.0f} {y:.0f}L{x:.0f} {y+9:.0f}L{x-9:.0f} {y:.0f}Z' fill='{fill}'/>")
        elif n["kind"] == "entry":
            parts.append(f"<path d='M{x:.0f} {y-10:.0f}L{x+10:.0f} {y+8:.0f}L{x-10:.0f} {y+8:.0f}Z' fill='{fill}'/>")
        else:
            parts.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='8' fill='{fill}'/>")
        parts.append(f"<text x='{x:.0f}' y='{y+24:.0f}' text-anchor='middle' font-size='11' fill='#14202e'>{escape(n['name'])}</text>")
    parts.append("</svg><p class='label'>Entry ▲ · Target ■ · Decoy ◆ · On a route ● · dashed with ✕ = blocked</p>")
    return "".join(parts)


def _html_block(block: tuple, report: Mapping) -> str:
    kind = block[0]
    if kind == "note":
        return f"<div class='note {block[2]}'>{escape(str(block[1]))}</div>"
    if kind == "label":
        return f"<div class='label'>{escape(block[1])}</div>"
    if kind == "kv":
        return "<dl>" + "".join(f"<dt>{escape(str(k))}</dt><dd>{escape(_text(v))}</dd>" for k, v in block[1]) + "</dl>"
    if kind == "table":
        head = "".join(f"<th>{escape(h)}</th>" for h in block[1])
        body = "".join("<tr>" + "".join(f"<td>{escape(_text(c))}</td>" for c in row) + "</tr>" for row in block[2])
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    if kind == "list":
        return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in block[1]) + "</ul>"
    if kind == "graph":
        return _svg(report)
    raise ValueError(kind)


def render_report_html(report: Mapping[str, object]) -> str:
    """One self-contained HTML document: shown on screen AND offered as the download."""
    sections = "".join(
        f"<section><h2><span>{i}.</span>{escape(s['title'])}</h2>" + "".join(_html_block(b, report) for b in s["blocks"]) + "</section>"
        for i, s in enumerate(report_sections(report), 1)
    )
    event = escape(_text(report.get("selected_event_id")))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Asteria case report {event}</title><style>{_CSS}</style></head><body><div class='page'>"
        f"<header><h1>Asteria case report</h1><p>Event {event} · isolated simulation · public/synthetic data only</p></header>"
        f"{sections}</div></body></html>"
    )


# ----------------------------------------------------------------- PDF ----

def render_report_pdf(report: Mapping[str, object]) -> bytes:
    """PDF with the same sections, in the same order, from the same section model."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=base["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#5a6b7d"))
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Asteria case report")
    event = xml_escape(_text(report.get("selected_event_id")))
    story = [Paragraph("Asteria case report", base["Title"]),
             Paragraph(f"Event {event} · isolated simulation · public/synthetic data only", small)]
    width = A4[0] - 32 * mm
    for i, section in enumerate(report_sections(report), 1):
        story.append(Paragraph(f"{i}. {xml_escape(section['title'])}", h2))
        for block in section["blocks"]:
            kind = block[0]
            if kind == "note":
                story.append(Paragraph(xml_escape(str(block[1])), body))
            elif kind == "label":
                story.append(Paragraph(f"<b>{xml_escape(block[1])}</b>", small))
            elif kind == "kv":
                rows = [[Paragraph(f"<b>{xml_escape(str(k))}</b>", cell), Paragraph(xml_escape(_text(v)), cell)] for k, v in block[1]]
                table = Table(rows, colWidths=[width * 0.3, width * 0.7])
                table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6dde6"))]))
                story.append(table)
            elif kind == "table":
                rows = [[Paragraph(f"<b>{xml_escape(h)}</b>", cell) for h in block[1]]]
                rows += [[Paragraph(xml_escape(_text(c)), cell) for c in row] for row in block[2]]
                table = Table(rows, colWidths=[width / len(block[1])] * len(block[1]), repeatRows=1)
                table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c3d0")),
                                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f6fa")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(table)
            elif kind == "list":
                story.append(ListFlowable([ListItem(Paragraph(xml_escape(item), body)) for item in block[1]], bulletType="bullet"))
            # "graph" is HTML-only; the edge table that follows carries the same facts.
            story.append(Spacer(1, 3))
    doc.build(story)
    return buffer.getvalue()


def report_filename(report: Mapping[str, object], extension: str) -> str:
    event = "".join(c for c in str(report.get("selected_event_id") or "case") if c.isalnum() or c in "-_")
    return f"asteria-report-{event}.{extension}"
