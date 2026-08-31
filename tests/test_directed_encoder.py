"""Directed GNN vs bidirectional GCN: directionality is not a no-op."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_synth import build_dag, dag_to_pyg  # noqa: E402
from encoders import DirectedGraphEncoder, GCNGraphEncoder, resolve_graph_encoder_settings  # noqa: E402
from train import TrainConfig  # noqa: E402


def _encode(encoder, G, bidirectional: bool) -> torch.Tensor:
    data = dag_to_pyg(G, bidirectional=bidirectional)
    batch = Batch.from_data_list([data])
    encoder.eval()
    with torch.no_grad():
        z = encoder(batch.x, batch.edge_index, batch.batch, topo_pos=None)
    return z


def test_resolve_settings_default_is_bidirectional_gcn():
    cfg = TrainConfig()
    gtype, tie, bidir = resolve_graph_encoder_settings(cfg)
    assert gtype == "gcn"
    assert tie is False
    assert bidir is True


def test_resolve_settings_directed_and_tied():
    cfg = TrainConfig(graph_direction="directed")
    assert resolve_graph_encoder_settings(cfg) == ("directed", False, False)
    cfg2 = TrainConfig(graph_direction="tied_in_out")
    assert resolve_graph_encoder_settings(cfg2) == ("directed", True, False)


def test_tied_in_out_shares_weights():
    enc = DirectedGraphEncoder(num_tools=15, hidden_dim=32, embed_dim=16, num_layers=2, tie_in_out=True)
    assert enc.W_out is None
    for i in range(len(enc.W_in)):
        assert enc._out_linear(i) is enc.W_in[i]


def test_untied_has_independent_in_out():
    enc = DirectedGraphEncoder(num_tools=15, hidden_dim=32, embed_dim=16, num_layers=2, tie_in_out=False)
    assert enc.W_out is not None
    for i in range(len(enc.W_in)):
        assert enc._out_linear(i) is enc.W_out[i]
        assert enc.W_out[i] is not enc.W_in[i]


def test_reverse_edge_changes_directed_embedding_not_gcn():
    torch.manual_seed(0)
    fwd = build_dag(["db_read", "db_write"], [(0, 1)])
    rev = build_dag(["db_read", "db_write"], [(1, 0)])
    directed = DirectedGraphEncoder(num_tools=15, hidden_dim=32, embed_dim=16, num_layers=2)
    gcn = GCNGraphEncoder(num_tools=15, hidden_dim=32, embed_dim=16, num_layers=2)
    z_dir_f = _encode(directed, fwd, bidirectional=False)
    z_dir_r = _encode(directed, rev, bidirectional=False)
    assert not torch.allclose(z_dir_f, z_dir_r, atol=1e-5)

    z_gcn_f = _encode(gcn, fwd, bidirectional=True)
    z_gcn_r = _encode(gcn, rev, bidirectional=True)
    # Bidirectionalized 2-node graphs with identical labels are the same PyG graph.
    assert torch.allclose(z_gcn_f, z_gcn_r, atol=1e-5)


def test_dag_to_pyg_default_still_bidirectional():
    G = build_dag(["db_read", "db_write"], [(0, 1)])
    data = dag_to_pyg(G)
    assert data.edge_index.size(1) == 2
    data_dir = dag_to_pyg(G, bidirectional=False)
    assert data_dir.edge_index.size(1) == 1
