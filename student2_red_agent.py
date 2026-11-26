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


# Node tuples: (name, asset_type, trust_zone, criticality, description).
# The first four nodes form the primary entry-to-target chain; later nodes are
# alternate routes and decoys. Extra edges are 1-based (source, target, controlled).
_SCENARIOS = {
    "exposed_api": {
        "control": "ingress_restricted",
        "nodes": (
            ("External Actor", "external_entity", "untrusted", "low", "Untrusted origin on the public internet"),
            ("Public API", "api_gateway", "dmz", "high", "Internet-facing API endpoint"),
            ("Workload Identity", "identity", "internal", "high", "Service identity used by the API workload"),
            ("Sensitive Data Store", "data_store", "restricted", "critical", "Protected resource holding sensitive records"),
            ("Edge Proxy", "proxy", "dmz", "medium", "Alternate edge route into the public API"),
            ("Bastion Host", "host", "management", "medium", "Management host with no route to the data store"),
            ("Admin Console", "application", "management", "medium", "Admin interface reachable only through the bastion"),
            ("Audit Log Service", "service", "internal", "low", "Log sink fed by the workload identity"),
        ),
        "extra_edges": ((1, 5, True), (5, 2, True), (1, 6, False), (6, 7, False), (3, 8, False)),
    },
    "credential_misuse": {
        "control": "service_credentials_rotated",
        "nodes": (
            ("Untrusted Client", "external_entity", "untrusted", "low", "Client without a trusted identity"),
            ("Legacy Service", "service", "dmz", "high", "Older service that still accepts static credentials"),
            ("Service Identity", "identity", "internal", "high", "Long-lived service credential"),
            ("Internal Resource", "data_store", "restricted", "critical", "Protected internal resource"),
            ("Auth Gateway", "gateway", "dmz", "medium", "Alternate authentication route into the service identity"),
            ("Batch Scheduler", "service", "internal", "low", "Scheduler with no route to the internal resource"),
            ("Secrets Cache", "data_store", "internal", "medium", "Cache reachable only through the scheduler"),
        ),
        "extra_edges": ((1, 5, False), (5, 3, True), (2, 6, False), (6, 7, False)),
    },
    "suspicious_egress": {
        "control": "egress_restricted",
        "nodes": (
            ("Workload Identity", "identity", "internal", "high", "Workload originating the outbound traffic"),
            ("Internal Data Store", "data_store", "restricted", "critical", "Store holding data that could leave the network"),
            ("Egress Proxy", "proxy", "dmz", "high", "Outbound gateway to external networks"),
            ("External Sink", "external_entity", "untrusted", "low", "External destination outside the trust boundary"),
            ("DNS Resolver", "service", "dmz", "medium", "Alternate outbound route through DNS"),
            ("Metrics Agent", "service", "internal", "low", "Agent with no route to the external sink"),
            ("Backup Target", "data_store", "internal", "medium", "Internal backup with no external route"),
        ),
        "extra_edges": ((1, 5, False), (5, 4, True), (1, 6, False), (2, 7, False)),
    },
}


def scenario_ids() -> tuple[str, ...]:
    return tuple(_SCENARIOS)


def scenario_definition(scenario_id: str) -> ScenarioDefinition:
    if scenario_id not in _SCENARIOS:
        raise ValueError("unknown simulated scenario")
    spec = _SCENARIOS[scenario_id]
    control = spec["control"]
    nodes = tuple(
        AttackGraphNode(f"{scenario_id}:n{index}", name, asset_type, zone, criticality, description)
        for index, (name, asset_type, zone, criticality, description) in enumerate(spec["nodes"], 1)
    )
    chain = tuple((index, index + 1, True) for index in range(1, 4))
    edges = tuple(
        AttackGraphEdge(
            f"{scenario_id}:e{number}", nodes[source - 1].node_id, nodes[target - 1].node_id,
            "simulated_reachability" if number <= 3 else ("alternate_route" if controlled else "adjacent_service"),
            ("simulation_precondition",), (control,) if controlled else (),
        )
        for number, (source, target, controlled) in enumerate(chain + spec["extra_edges"], 1)
    )
    return ScenarioDefinition(scenario_id, AttackGraph(nodes, edges, nodes[0].node_id, nodes[3].node_id), control)


def _blocked_edge_ids(graph: AttackGraph, controls: dict[str, bool]) -> set[str]:
    for edge in graph.edges:
        if any(control not in controls for control in edge.blocking_control_ids):
            raise ValueError("graph references an unknown environment control")
    return {edge.edge_id for edge in graph.edges if any(controls[control] for control in edge.blocking_control_ids)}


def _shortest_path(
    graph: AttackGraph, blocked: set[str], max_path_length: int, max_visited: int
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    adjacency: dict[str, list[AttackGraphEdge]] = {}
    for edge in graph.edges:
        if edge.edge_id not in blocked:
            adjacency.setdefault(edge.source_node_id, []).append(edge)
    queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque([(graph.entry_node_id, (graph.entry_node_id,), ())])
    visited = {graph.entry_node_id}
    while queue and len(visited) <= max_visited:
        node, nodes, edges = queue.popleft()
        if node == graph.target_node_id:
            return nodes, edges
        if len(edges) >= max_path_length:
            continue
        for edge in adjacency.get(node, []):
            if edge.target_node_id not in visited:
                visited.add(edge.target_node_id)
                queue.append((edge.target_node_id, nodes + (edge.target_node_id,), edges + (edge.edge_id,)))
    return None


def evaluate_reachability(graph: AttackGraph, controls: dict[str, bool], *, max_path_length: int = 8, max_visited: int = 64) -> AttackPath:
    """Bounded BFS that honours active controls; it never invokes external systems.

    The target is reachable only if some route avoids every blocked edge. When all
    routes are blocked the structural path is returned with reachable=False so it
    can still be displayed.
    """

    if max_path_length < 1 or max_visited < 1:
        raise ValueError("traversal limits must be positive")
    blocked = _blocked_edge_ids(graph, controls)
    open_path = _shortest_path(graph, blocked, max_path_length, max_visited)
    if open_path is not None:
        return AttackPath(open_path[0], open_path[1], True)
    structural = _shortest_path(graph, set(), max_path_length, max_visited)
    if structural is None:
        raise ValueError("no bounded candidate path exists")
    return AttackPath(structural[0], structural[1], False)


def graph_exposure(graph: AttackGraph, controls: dict[str, bool]) -> dict[str, object]:
    """Describe which nodes and edges are open, blocked, on a route or decoys.

    The result is derived only from the graph and the current control state, so a
    visualisation can reflect backend state without inventing any facts.
    """

    blocked = _blocked_edge_ids(graph, controls)

    def reach(edges, start, reverse=False):
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
            source, target = (edge.target_node_id, edge.source_node_id) if reverse else (edge.source_node_id, edge.target_node_id)
            adjacency.setdefault(source, []).append(target)
        seen, stack = {start}, [start]
        while stack:
            for neighbour in adjacency.get(stack.pop(), []):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen

    open_edges = [edge for edge in graph.edges if edge.edge_id not in blocked]
    reachable_now = reach(open_edges, graph.entry_node_id)
    from_entry = reach(graph.edges, graph.entry_node_id)
    to_target = reach(graph.edges, graph.target_node_id, reverse=True)
    on_route = from_entry & to_target
    return {
        "target_reachable": graph.target_node_id in reachable_now,
        "reachable_node_ids": [node.node_id for node in graph.nodes if node.node_id in reachable_now],
        "route_node_ids": [node.node_id for node in graph.nodes if node.node_id in on_route],
        "decoy_node_ids": [node.node_id for node in graph.nodes if node.node_id not in on_route],
        "blocked_edge_ids": [edge.edge_id for edge in graph.edges if edge.edge_id in blocked],
        "open_edge_ids": [edge.edge_id for edge in open_edges],
    }


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
