"""Deterministic recommendation mapping for simulation-only Red findings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from investigator import InvestigationRecord
from student2_red_agent import RedAgentFinding, scenario_definition


@dataclass(frozen=True)
class DefensiveRecommendation:
    recommendation_id: str
    investigation_id: str
    red_finding_id: str
    scenario_id: str
    target_control_id: str
    title: str
    rationale: str
    supporting_evidence_ids: tuple[str, ...]
    expected_security_effect: str
    expected_verifier_condition: str
    operational_impact: str
    priority: str
    execution_mode: str = "simulation_only"

    def __post_init__(self) -> None:
        if self.execution_mode != "simulation_only" or not self.supporting_evidence_ids:
            raise ValueError("recommendations must be evidence-backed simulations")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BlueAgent:
    _TITLES = {
        "exposed_api": "Restrict public ingress to approved sources",
        "credential_misuse": "Rotate and invalidate the simulated service credential",
        "suspicious_egress": "Restrict unapproved external egress",
    }

    def recommend(self, finding: RedAgentFinding, investigation: InvestigationRecord) -> tuple[DefensiveRecommendation, ...]:
        if type(finding) is not RedAgentFinding or type(investigation) is not InvestigationRecord:
            raise TypeError("blue recommendations require a RedAgentFinding and InvestigationRecord")
        if finding.investigation_id != investigation.investigation_id or not finding.simulation_only:
            raise ValueError("finding provenance does not match investigation")
        if not finding.candidate_path.reachable:
            return ()
        scenario = scenario_definition(finding.scenario_id)
        evidence_ids = tuple(reference.reference_id for reference in investigation.evidence_references)
        return (DefensiveRecommendation(
            f"recommendation:{finding.finding_id}", finding.investigation_id, finding.finding_id,
            finding.scenario_id, scenario.control_id, self._TITLES[finding.scenario_id],
            "A reachable path exists only in the isolated simulated graph.", evidence_ids,
            f"Set simulated {scenario.control_id} to true.", "Recomputed simulated path is blocked.",
            "Simulation-only control transition; no infrastructure is changed.", "high",
        ),)
