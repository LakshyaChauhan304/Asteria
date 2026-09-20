from datetime import datetime


def generate_investigation_id(index: int) -> str:
    """
    Generate a unique Asteria investigation ID.
    """

    return f"AST-{index:04d}"


def create_investigation(
    agent_id,
    detection_score,
    severity,
    reasons,
    events
):
    """
    Create an evidence-backed investigation record.
    """

    investigation = {
        "investigation_id": None,
        "created_at": datetime.now().isoformat(),

        "agent_id": agent_id,

        "state": "DETECTED",

        "detection_score": detection_score,
        "severity": severity,

        "reasons": reasons,

        "event_count": len(events),

        "events": events.to_dict("records"),

        "evidence": []
    }

    
    for _, event in events.iterrows():

        investigation["evidence"].append({
            "event_id": event["event_id"],
            "timestamp": str(event["timestamp"]),
            "tool": event["tool"],
            "action": event["action"],
            "resource": event["resource"],
            "result": event["result"]
        })

    return investigation