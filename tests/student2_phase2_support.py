from security_layer import CapabilityPolicy, DetectionEvidenceV1
from investigator import Investigator, TrustedObservationContext


def record():
    evidence = DetectionEvidenceV1.from_dict({
        "event_id": "UNSW-1", "model_prediction": 1, "model_score": 0.9,
        "anomaly_prediction": 0, "anomaly_score": 0.2, "protocol": "tcp",
        "service": "http", "state": "FIN", "features": {"dur": 0.1},
    })
    return Investigator(CapabilityPolicy(())).start_investigation(
        evidence, TrustedObservationContext("demo-agent", "demo-identity", "events/UNSW-1")
    )
