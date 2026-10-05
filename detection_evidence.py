from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class DetectionEvidence:
    """
    Structured detection evidence produced by Asteria's
    machine-learning detection layer.
    """

    event_id: str

    model_prediction: int
    model_score: float

    anomaly_prediction: int
    anomaly_score: float

    protocol: str
    service: str
    state: str

    features: Dict[str, Any]

    def to_dict(self):
        """
        Convert evidence object into a dictionary.
        """

        return asdict(self)