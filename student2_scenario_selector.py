"""Heuristic, explainable mapping from one flow record to a simulated scenario.

The mapping is an inference from displayed telemetry fields only. It never uses
labels or attack categories, and an analyst can always override the suggestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

AUTH_SERVICES = frozenset({"ftp", "ftp-data", "ssh", "pop3", "smtp", "radius"})
EGRESS_MIN_SOURCE_BYTES = 1_000
EGRESS_RATIO = 5


@dataclass(frozen=True)
class ScenarioSuggestion:
    scenario_id: str
    rationale: str
    basis: tuple[str, ...]
    inferred: bool = True

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["basis"] = list(self.basis)
        return result


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def suggest_scenario(flow: Mapping[str, object]) -> ScenarioSuggestion:
    service = str(flow.get("service", "")).strip().lower()
    protocol = str(flow.get("protocol", "")).strip().lower()
    source = _number(flow.get("source_bytes"))
    destination = _number(flow.get("destination_bytes"))
    if service in AUTH_SERVICES:
        return ScenarioSuggestion(
            "credential_misuse",
            f"The flow uses the credential-bearing service '{service}', so a credential-misuse path is the closest simulated scenario.",
            (f"service={service}", f"protocol={protocol or 'unknown'}"),
        )
    if source >= EGRESS_MIN_SOURCE_BYTES and source >= EGRESS_RATIO * max(destination, 1.0):
        return ScenarioSuggestion(
            "suspicious_egress",
            "Far more bytes left the source than came back, so an outbound-data path is the closest simulated scenario.",
            (f"source_bytes={int(source)}", f"destination_bytes={int(destination)}"),
        )
    return ScenarioSuggestion(
        "exposed_api",
        "No credential-bearing service or outbound-heavy pattern was observed, so the exposed-service path is the default simulated scenario.",
        (f"service={service or 'unknown'}", f"protocol={protocol or 'unknown'}"),
    )
