from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable, Sequence

import networkx as nx
import torch


REL_PARALLEL = 0
REL_DIRECT_FWD = 1
REL_INDIRECT_FWD = 2
REL_DIRECT_REV = 3
REL_INDIRECT_REV = 4
REL_IGNORE = -100
NUM_RELATIONS = 5


@dataclass(frozen=True)
class GraphSignature:
    tools: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    tool_ids: tuple[int, ...]
    tool_target: torch.Tensor
    relation_target: torch.Tensor
    reachability: torch.Tensor
    group_id: str
    dag_key: str


def graph_key(tools: Sequence[str], edges: Iterable[tuple[int, int]]) -> str:
    labelled = sorted((tools[u], tools[v]) for u, v in edges)
    raw = repr((tuple(sorted(tools)), tuple(labelled)))
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_signature(
    tools: Sequence[str],
    edges: Sequence[tuple[int, int]],
    vocabulary: Sequence[str],
    group_id: str,
) -> GraphSignature:
    vocab_index = {name: i for i, name in enumerate(vocabulary)}
    local_ids = tuple(vocab_index[name] for name in tools)
    n_vocab = len(vocabulary)
    target = torch.zeros(n_vocab, dtype=torch.float32)
    if local_ids:
        target[list(local_ids)] = 1.0

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(tools)))
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Campaign row is not a DAG")
    closure = nx.transitive_closure_dag(graph)

    reach = torch.zeros((n_vocab, n_vocab), dtype=torch.bool)
    direct = torch.zeros((n_vocab, n_vocab), dtype=torch.bool)
    for u, v in graph.edges():
        direct[local_ids[u], local_ids[v]] = True
    for u, v in closure.edges():
        reach[local_ids[u], local_ids[v]] = True

    relations = torch.full((n_vocab, n_vocab), REL_IGNORE, dtype=torch.long)
    active = sorted(local_ids)
    for offset, left in enumerate(active):
        for right in active[offset + 1 :]:
            if direct[left, right]:
                rel = REL_DIRECT_FWD
            elif reach[left, right]:
                rel = REL_INDIRECT_FWD
            elif direct[right, left]:
                rel = REL_DIRECT_REV
            elif reach[right, left]:
                rel = REL_INDIRECT_REV
            else:
                rel = REL_PARALLEL
            relations[left, right] = rel

    return GraphSignature(
        tools=tuple(tools),
        edges=tuple(sorted(edges)),
        tool_ids=local_ids,
        tool_target=target,
        relation_target=relations,
        reachability=reach,
        group_id=str(group_id),
        dag_key=graph_key(tools, edges),
    )


def invariant_node_features(
    num_nodes: int,
    edge_index: torch.Tensor,
    mode: str = "combined",
) -> torch.Tensor:
    """Permutation-equivariant DAG features; never uses topological-sort rank."""
    if num_nodes == 0:
        return torch.zeros((0, 6), dtype=torch.float32, device=edge_index.device)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    graph.add_edges_from(edges)
    order = list(nx.topological_sort(graph))
    source_depth = {node: 0 for node in order}
    for node in order:
        preds = list(graph.predecessors(node))
        if preds:
            source_depth[node] = 1 + max(source_depth[p] for p in preds)
    sink_depth = {node: 0 for node in reversed(order)}
    for node in reversed(order):
        succs = list(graph.successors(node))
        if succs:
            sink_depth[node] = 1 + max(sink_depth[s] for s in succs)

    denom = max(1, num_nodes - 1)
    rows = []
    for node in range(num_nodes):
        rows.append([
            source_depth[node] / denom,
            sink_depth[node] / denom,
            graph.in_degree(node) / denom,
            graph.out_degree(node) / denom,
            float(graph.in_degree(node) == 0),
            float(graph.out_degree(node) == 0),
        ])
    features = torch.tensor(rows, dtype=torch.float32, device=edge_index.device)
    if mode == "none":
        return torch.zeros_like(features)
    if mode == "path":
        return torch.zeros_like(features)
    if mode == "depth":
        features[:, 2:] = 0
    elif mode == "degree":
        features[:, :2] = 0
    return features


def relation_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Normalized Hamming distance over pair relations active in both graphs."""
    mask = (left != REL_IGNORE) & (right != REL_IGNORE)
    count = mask.sum().clamp(min=1)
    return ((left != right) & mask).float().sum() / count


def directed_relation_bias(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Dense relation IDs: self=0, direct fwd/rev=1/2, ancestor/desc=3/4, other=5."""
    bias = torch.full((num_nodes, num_nodes), 5, dtype=torch.long, device=edge_index.device)
    bias.fill_diagonal_(0)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    closure = nx.transitive_closure_dag(graph)
    for u, v in closure.edges():
        bias[u, v] = 3
        bias[v, u] = 4
    for u, v in graph.edges():
        bias[u, v] = 1
        bias[v, u] = 2
    return bias
