"""Deterministic, graph-only adversarial simulation for the Asteria demo."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from investigator import (
    EvidenceReference,
    HypothesisStatus,
    InvestigationObservation,
    InvestigationRecord,
    ObservationClass,
)
from student2_demo_environment import DemoEnvironment


@dataclass(frozen=True)
class AttackGraphNode:
    node_id: str
    display_name: str
    asset_type: str
    trust_zone: str
    criticality: str
    description: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in asdict(self).values()):
            raise ValueError("attack graph node fields must be non-empty strings")


@dataclass(frozen=True)
class AttackGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship: str
    required_conditions: tuple[str, ...]
    blocking_control_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.edge_id, self.source_node_id, self.target_node_id, self.relationship,
        )):
            raise ValueError("attack graph edge identifiers must be non-empty strings")
        if type(self.required_conditions) is not tuple or type(self.blocking_control_ids) is not tuple:
            raise TypeError("edge conditions and controls must be tuples")
        if any(not isinstance(value, str) or not value for value in self.required_conditions + self.blocking_control_ids):
            raise ValueError("edge conditions and controls must be non-empty strings")


@dataclass(frozen=True)
class AttackGraph:
    nodes: tuple[AttackGraphNode, ...]
    edges: tuple[AttackGraphEdge, ...]
    entry_node_id: str
    target_node_id: str

    def __post_init__(self) -> None:
        if type(self.nodes) is not tuple or type(self.edges) is not tuple:
            raise TypeError("graph nodes and edges must be tuples")
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("attack graph nodes and edges must be unique")
        if self.entry_node_id not in node_ids or self.target_node_id not in node_ids:
            raise ValueError("graph entry and target nodes must exist")
        if any(edge.source_node_id not in node_ids or edge.target_node_id not in node_ids for edge in self.edges):
            raise ValueError("attack graph edge references an unknown node")


@dataclass(frozen=True)
class AttackPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    reachable: bool


@dataclass(frozen=True)
class RedAgentFinding:
    finding_id: str
    investigation_id: str
    event_id: str
    scenario_id: str
    candidate_path: AttackPath
    preconditions: tuple[str, ...]
    supporting_model_signal_ids: tuple[str, ...]
    supporting_evidence_reference_ids: tuple[str, ...]
    explanation: str
    limitations: tuple[str, ...]
    simulation_only: bool = True

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["candidate_path"]["node_ids"] = list(self.candidate_path.node_ids)
        result["candidate_path"]["edge_ids"] = list(self.candidate_path.edge_ids)
        return result


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    graph: AttackGraph
    control_id: str


def scenario_definition(scenario_id: str) -> ScenarioDefinition:
    scenarios = {
        "exposed_api": ("External Actor", "Public API", "Workload Identity", "Sensitive Data Store", "ingress_restricted"),
        "credential_misuse": ("Untrusted Client", "Legacy Service", "Service Identity", "Internal Resource", "service_credentials_rotated"),
        "suspicious_egress": ("Workload Identity", "Internal Data Store", "Egress Proxy", "External Sink", "egress_restricted"),
    }
    if scenario_id not in scenarios:
        raise ValueError("unknown simulated scenario")
    names = scenarios[scenario_id]
    nodes = tuple(AttackGraphNode(f"{scenario_id}:n{index}", name, "simulated_asset", "demo", "high", name) for index, name in enumerate(names[:4], 1))
    edges = tuple(AttackGraphEdge(f"{scenario_id}:e{index}", nodes[index - 1].node_id, nodes[index].node_id, "simulated_reachability", ("simulation_precondition",), (names[4],)) for index in range(1, 4))
    return ScenarioDefinition(scenario_id, AttackGraph(nodes, edges, nodes[0].node_id, nodes[-1].node_id), names[4])


def evaluate_reachability(graph: AttackGraph, controls: dict[str, bool], *, max_path_length: int = 8, max_visited: int = 64) -> AttackPath:
    """Bounded BFS; controls block traversal but never invoke external systems."""

    if max_path_length < 1 or max_visited < 1:
        raise ValueError("traversal limits must be positive")
    adjacency: dict[str, list[AttackGraphEdge]] = {}
    for edge in graph.edges:
        if any(control not in controls for control in edge.blocking_control_ids):
            raise ValueError("graph references an unknown environment control")
        adjacency.setdefault(edge.source_node_id, []).append(edge)
    queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque([(graph.entry_node_id, (graph.entry_node_id,), ())])
    visited = {graph.entry_node_id}
    candidate: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    while queue and len(visited) <= max_visited:
        node, nodes, edges = queue.popleft()
        if node == graph.target_node_id:
            candidate = (nodes, edges)
            break
        if len(edges) >= max_path_length:
            continue
        for edge in adjacency.get(node, []):
            if edge.target_node_id not in visited:
                visited.add(edge.target_node_id)
                queue.append((edge.target_node_id, nodes + (edge.target_node_id,), edges + (edge.edge_id,)))
    if candidate is None:
        raise ValueError("no bounded candidate path exists")
    reachable = all(not controls[control] for edge_id in candidate[1] for edge in graph.edges if edge.edge_id == edge_id for control in edge.blocking_control_ids)
    return AttackPath(candidate[0], candidate[1], reachable)


class RedAgent:
    def simulate(self, scenario_id: str, investigation: InvestigationRecord, environment: DemoEnvironment) -> RedAgentFinding:
        if type(investigation) is not InvestigationRecord or type(environment) is not DemoEnvironment:
            raise TypeError("red simulation requires an InvestigationRecord and DemoEnvironment")
        scenario = scenario_definition(scenario_id)
        path = evaluate_reachability(scenario.graph, environment.current_state)
        finding_id = f"red:{scenario_id}:{investigation.investigation_id}"
        references = (EvidenceReference(f"{finding_id}:evidence", "student2.red_agent", investigation.evidence.event_id, "simulated attack-graph assessment"),)
        finding = RedAgentFinding(finding_id, investigation.investigation_id, investigation.evidence.event_id, scenario_id, path, ("isolated_in_memory_graph",), tuple(signal.signal_id for signal in investigation.model_signals), tuple(reference.reference_id for reference in investigation.evidence_references), "Candidate path is reachable in the isolated graph." if path.reachable else "Candidate path is blocked in the isolated graph by an active simulated control.", ("Simulation only; not evidence of compromise in real infrastructure.",))
        observation = InvestigationObservation(f"observation:{finding_id}", finding.event_id, "student2.red_agent", ObservationClass.INFERENCE, finding.explanation, references)
        investigation.add_observation(observation)
        hypothesis = investigation.add_hypothesis("A simulated path associated with this investigation may require defensive review.")
        investigation.update_hypothesis(
            hypothesis.hypothesis_id,
            HypothesisStatus.SUPPORTED if path.reachable else HypothesisStatus.CONTRADICTED,
            supporting_evidence=(references[0].reference_id,),
        )
        investigation.add_limitation("Red Agent result is an isolated attack-graph simulation, not live attack activity.")
        return finding
