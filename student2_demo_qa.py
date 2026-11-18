"""Deterministic, evidence-grounded answers for the demo analyst panels."""

from __future__ import annotations


def answer(role: str, question: str, report: dict[str, object]) -> str:
    """Answer only from exported workflow records; never invent missing evidence."""

    event_id = report.get("selected_event_id") or "no selected event"
    investigation = report.get("investigation") or {}
    if not isinstance(investigation, dict):
        return "The current records are not available."
    evidence = investigation.get("evidence") or {}
    signals = investigation.get("model_signals") or []
    references = investigation.get("evidence_references") or []
    lower = question.lower()
    if role == "Detection Analyst":
        if not evidence:
            return "Detection has not run, so no model evidence is available."
        if "xgboost" in lower and signals:
            signal = signals[0]
            return f"For {event_id}, XGBoost recorded prediction={signal.get('prediction')} and probability={signal.get('probability')}; it is a detection signal, not proof."
        return f"For {event_id}, Logistic prediction={evidence.get('model_prediction')} score={evidence.get('model_score')}; Isolation prediction={evidence.get('anomaly_prediction')} score={evidence.get('anomaly_score')}."
    if "evidence" in lower or "source" in lower:
        ids = ", ".join(str(item.get("reference_id")) for item in references if isinstance(item, dict))
        return f"Security Engineer: {event_id} has these evidence references: {ids or 'none'}. The graph simulation is separate from telemetry evidence."
    finding = report.get("red_finding")
    if finding:
        return f"Security Engineer: the Red Agent result for {event_id} is an isolated simulation. It does not establish compromise; review the finding limitations and verification record."
    return "The records do not support a stronger conclusion. Run investigation and Red/Blue simulation first."
