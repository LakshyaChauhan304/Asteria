from detection_evidence import DetectionEvidence


def build_detection_evidence(
    event_id,
    row,
    logistic_prediction,
    logistic_probability,
    isolation_prediction,
    isolation_score
):
    """
    Convert ML outputs and telemetry into a structured
    Asteria DetectionEvidence object.
    """

    features = {
    "dur": float(row["dur"]),
    "spkts": int(row["spkts"]),
    "dpkts": int(row["dpkts"]),
    "sbytes": int(row["sbytes"]),
    "dbytes": int(row["dbytes"]),
    "rate": float(row["rate"]),
    "sttl": int(row["sttl"]),
    "dttl": int(row["dttl"]),
    "sload": float(row["sload"]),
    "dload": float(row["dload"]),
    "sloss": int(row["sloss"]),
    "dloss": int(row["dloss"]),
    "ct_srv_src": int(row["ct_srv_src"]),
    "ct_state_ttl": int(row["ct_state_ttl"]),
    "ct_dst_ltm": int(row["ct_dst_ltm"]),
    "ct_src_ltm": int(row["ct_src_ltm"])
}

    evidence = DetectionEvidence(
        event_id=str(event_id),

        model_prediction=int(
            logistic_prediction
        ),

        model_score=float(
            logistic_probability
        ),

        anomaly_prediction=int(
            isolation_prediction
        ),

        anomaly_score=float(
            isolation_score
        ),

        protocol=str(
            row["proto"]
        ),

        service=str(
            row["service"]
        ),

        state=str(
            row["state"]
        ),

        features=features
    )

    return evidence