"""One-node / empty-edge GNN behaviour for GCN and directed encoders."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomic_zero_shot import build_one_node_dag, pyg_one_node  # noqa: E402
from data_synth import build_dag, dag_to_pyg  # noqa: E402
from encoders import DirectedGraphEncoder, GCNGraphEncoder  # noqa: E402


def _forward(encoder, data_list):
    batch = Batch.from_data_list(data_list)
    encoder.eval()
    with torch.no_grad():
        return encoder(
            batch.x, batch.edge_index, batch.batch,
            topo_pos=getattr(batch, "topo_pos", None),
        )


def test_gcn_one_node_empty_edges_cpu():
    enc = GCNGraphEncoder(num_tools=15, hidden_dim=16, embed_dim=8, num_layers=2)
    data = pyg_one_node("db_read", bidirectional=True)
    assert data.edge_index.numel() == 0
    z = _forward(enc, [data])
    assert z.shape == (1, 8)
    assert torch.isfinite(z).all()


def test_directed_one_node_uses_w_self_only():
    enc = DirectedGraphEncoder(num_tools=15, hidden_dim=16, embed_dim=8, num_layers=2)
    data = pyg_one_node("db_write", bidirectional=False)
    z = _forward(enc, [data])
    assert z.shape == (1, 8)
    assert torch.isfinite(z).all()


def test_batch_mix_one_node_and_chain():
    enc = DirectedGraphEncoder(num_tools=15, hidden_dim=16, embed_dim=8, num_layers=2)
    one = dag_to_pyg(build_one_node_dag("check_status"), bidirectional=False)
    chain = dag_to_pyg(build_dag(["db_read", "create_ticket"], [(0, 1)]), bidirectional=False)
    z = _forward(enc, [one, chain])
    assert z.shape == (2, 8)
    assert torch.isfinite(z).all()


def test_tiny_batch_and_incomplete_list():
    enc = GCNGraphEncoder(num_tools=15, hidden_dim=16, embed_dim=8, num_layers=1)
    graphs = [pyg_one_node(t) for t in ("db_read", "scan_malware", "log_audit_event")]
    z = _forward(enc, graphs[:1])
    assert z.shape[0] == 1
    z3 = _forward(enc, graphs)
    assert z3.shape[0] == 3


def test_isolated_two_node_no_edges_is_valid_pyg():
    G = build_dag(["db_read", "db_write"], [])
    data = dag_to_pyg(G, bidirectional=True)
    assert data.edge_index.size(1) == 0
    enc = GCNGraphEncoder(num_tools=15, hidden_dim=16, embed_dim=8, num_layers=2)
    z = _forward(enc, [data])
    assert torch.isfinite(z).all()
