"""
encoders_v2.py — Campaign v4 Corrected LEGR Architecture
==========================================================

Addresses defects D-001, D-002, D-003 from architecture audit:
  - D-001/D-002: Uses frozen MiniLM text embeddings for node features
    (instead of arbitrary integer nn.Embedding)
  - D-003: Uses DirectedGraphEncoder by default (preserves edge direction)

This module provides drop-in replacements that are backward-compatible
with the LEGRDualEncoder interface. The legacy encoders.py is preserved
for reproducibility of prior experiments.

Architecture:
  Query Tower: query text → MiniLM → projection → normalized embedding
  Graph Tower: tool-name text → MiniLM (frozen) → node features
               → DirectedGNN → pooling → projection → normalized embedding
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from transformers import AutoModel, AutoTokenizer

try:
    from encoders import TextEncoder
except ImportError:
    from src.encoders import TextEncoder


class TextNodeFeatureEncoder(nn.Module):
    """Encodes tool names into dense vectors using a frozen MiniLM backbone.

    Caches embeddings for all known tool names to avoid recomputation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 64,
    ):
        super().__init__()
        self._model_name = model_name
        self._backbone = AutoModel.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._hidden_size = self._backbone.config.hidden_size

        for p in self._backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self._hidden_size, output_dim)

        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_tool_name(self, name: str, device: torch.device) -> torch.Tensor:
        """Encode a single tool name to a dense vector."""
        self._backbone.to(device)
        readable = name.replace("_", " ") if name else ""
        tokens = self._tokenizer(
            readable,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self._backbone(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.squeeze(0)

    def precompute_cache(self, tool_names: List[str], device: torch.device):
        """Pre-cache embeddings for all tool names."""
        self._backbone.to(device)
        for name in tool_names:
            if name not in self._cache:
                self._cache[name] = self._encode_tool_name(name, device)

    def get_features(
        self,
        tool_names: List[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """Get projected features for a batch of tool name lists.

        Args:
            tool_names: List of lists of tool names (one per graph in batch).
            device: Target device.

        Returns:
            Tensor of shape (total_nodes, output_dim).
        """
        all_raw = []
        for names in tool_names:
            for name in names:
                if not name:
                    all_raw.append(torch.zeros(self._hidden_size, device=device))
                    continue
                if name not in self._cache:
                    self._cache[name] = self._encode_tool_name(name, device)
                all_raw.append(self._cache[name].to(device))

        if not all_raw:
            return torch.zeros(0, self.proj.out_features, device=device)

        raw_tensor = torch.stack(all_raw).to(device)
        return self.proj(raw_tensor)


# ═══════════════════════════════════════════════════════════════════════════
#  Directed Graph Encoder with Text Node Features (D-001/D-002/D-003 fix)
# ═══════════════════════════════════════════════════════════════════════════

class DirectedTextGraphEncoder(nn.Module):
    """Directed GNN with text-based node features.

    Node features are dense vectors from frozen MiniLM applied to tool names.
    Message passing uses separate W_in and W_out transformations for
    incoming vs outgoing edges (or tied if tie_in_out=True).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 3,
        node_feature_dim: int = 64,
        max_topo_pos: int = 32,
        tie_in_out: bool = False,
    ):
        super().__init__()
        self.max_topo_pos = max_topo_pos
        self.tie_in_out = bool(tie_in_out)

        self.topo_embedding = nn.Embedding(max_topo_pos + 1, node_feature_dim)
        input_dim = node_feature_dim * 2

        self.W_self = nn.ModuleList()
        self.W_in = nn.ModuleList()
        self.W_out = nn.ModuleList() if not self.tie_in_out else None
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.W_self.append(nn.Linear(in_dim, hidden_dim))
            self.W_in.append(nn.Linear(in_dim, hidden_dim, bias=False))
            if not self.tie_in_out:
                self.W_out.append(nn.Linear(in_dim, hidden_dim, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def _out_linear(self, layer: int) -> nn.Linear:
        if self.tie_in_out:
            return self.W_in[layer]
        return self.W_out[layer]

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: Optional[torch.Tensor] = None,
        project: bool = True,
    ) -> torch.Tensor:
        if topo_pos is not None:
            topo_emb = self.topo_embedding(topo_pos.clamp(max=self.max_topo_pos))
        else:
            topo_emb = torch.zeros_like(node_features)

        h = torch.cat([node_features, topo_emb], dim=-1)

        src = edge_index[0]
        dst = edge_index[1]
        n_nodes = h.size(0)

        for i in range(len(self.W_self)):
            w_self = self.W_self[i]
            w_in = self.W_in[i]
            w_out = self._out_linear(i)

            h_new = w_self(h)
            if src.numel() > 0:
                in_agg = h.new_zeros(n_nodes, h_new.size(-1))
                out_agg = h.new_zeros(n_nodes, h_new.size(-1))
                in_agg.index_add_(0, dst, w_in(h)[src])
                out_agg.index_add_(0, src, w_out(h)[dst])
                h_new = h_new + in_agg + out_agg
            h = F.relu(self.norms[i](h_new))

        graph_emb = global_mean_pool(h, batch)
        if project:
            return self.proj(graph_emb)
        return graph_emb


# ═══════════════════════════════════════════════════════════════════════════
#  Corrected Dual Encoder
# ═══════════════════════════════════════════════════════════════════════════

class LEGRDualEncoderV2(nn.Module):
    """Corrected LEGR dual-encoder with text-based node features and directed GNN.

    This is the `legr_directed_toolname` architecture variant.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 3,
        node_feature_dim: int = 64,
        max_topo_pos: int = 32,
        freeze_text_backbone: bool = False,
        num_frozen_layers: int = 0,
        tie_in_out: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.graph_encoder_type = "directed_text"
        self.tie_in_out = tie_in_out

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            num_frozen_layers=num_frozen_layers,
        )

        self.node_feature_encoder = TextNodeFeatureEncoder(
            model_name=text_model_name,
            output_dim=node_feature_dim,
        )

        self.graph_encoder = DirectedTextGraphEncoder(
            hidden_dim=graph_hidden_dim,
            embed_dim=embed_dim,
            num_layers=graph_num_layers,
            node_feature_dim=node_feature_dim,
            max_topo_pos=max_topo_pos,
            tie_in_out=tie_in_out,
        )

    def precompute_tool_features(self, tool_names: List[str], device: torch.device):
        """Pre-cache tool name embeddings (call once before training)."""
        self.node_feature_encoder.precompute_cache(tool_names, device)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.text_encoder(input_ids, attention_mask)
        return F.normalize(z, p=2, dim=-1)

    def _node_features_from_tool_ids(
        self,
        tool_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map integer tool indices from dag_to_pyg onto MiniLM tool-name features."""
        try:
            import data_synth as ds
        except ImportError:
            from src import data_synth as ds

        ids = tool_ids.detach().reshape(-1).tolist()
        names = []
        for raw in ids:
            idx = int(raw)
            if 0 <= idx < len(ds.TOOL_VOCAB):
                names.append(ds.TOOL_VOCAB[idx])
            else:
                names.append("")
        return self.node_feature_encoder.get_features(
            [names], device=tool_ids.device,
        )

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.dtype in (torch.int32, torch.int64, torch.long):
            node_features = self._node_features_from_tool_ids(node_features)
        z = self.graph_encoder(node_features, edge_index, batch, topo_pos=topo_pos)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_node_features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        graph_topo_pos: Optional[torch.Tensor] = None,
    ):
        z_text = self.encode_text(input_ids, attention_mask)
        z_graph = self.encode_graph(
            graph_node_features, graph_edge_index, graph_batch,
            topo_pos=graph_topo_pos,
        )
        return z_text, z_graph


class LEGRDualEncoderV3(nn.Module):
    """Tied MiniLM + node-set attention pool concatenated with directed GNN.

    Query MiniLM backbone is shared with tool-name node encoding (no second
    frozen copy). Graph embedding is fuse([z_gnn, z_set]) so tool identity
    does not have to survive mean-pooled GNN mixing alone.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 3,
        node_feature_dim: int = 64,
        max_topo_pos: int = 32,
        freeze_text_backbone: bool = False,
        num_frozen_layers: int = 0,
        tie_in_out: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.graph_encoder_type = "setgnn_tied"
        self.tie_in_out = tie_in_out
        self.node_feature_dim = node_feature_dim

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            num_frozen_layers=num_frozen_layers,
        )
        self._node_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self._hidden_size = self.text_encoder.backbone.config.hidden_size
        self.node_proj = nn.Linear(self._hidden_size, node_feature_dim)
        self.set_attn = nn.Linear(node_feature_dim, 1)
        self.graph_encoder = DirectedTextGraphEncoder(
            hidden_dim=graph_hidden_dim,
            embed_dim=embed_dim,
            num_layers=graph_num_layers,
            node_feature_dim=node_feature_dim,
            max_topo_pos=max_topo_pos,
            tie_in_out=tie_in_out,
        )
        self.fuse = nn.Linear(graph_hidden_dim + node_feature_dim, embed_dim)

    def precompute_tool_features(self, tool_names: List[str], device: torch.device):
        """Warm the shared backbone on the tool vocabulary (optional)."""
        if not tool_names:
            return
        self._encode_tool_names_raw(list(tool_names), device)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.text_encoder(input_ids, attention_mask)
        return F.normalize(z, p=2, dim=-1)

    def _encode_tool_names_raw(
        self,
        names: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        if not names:
            return torch.zeros(0, self._hidden_size, device=device)
        readable = [n.replace("_", " ") if n else "" for n in names]
        tokens = self._node_tokenizer(
            readable,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.text_encoder.backbone(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _node_features_from_tool_ids(self, tool_ids: torch.Tensor) -> torch.Tensor:
        try:
            import data_synth as ds
        except ImportError:
            from src import data_synth as ds

        flat = tool_ids.detach().reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        names = []
        for raw in unique.tolist():
            idx = int(raw)
            if 0 <= idx < len(ds.TOOL_VOCAB):
                names.append(ds.TOOL_VOCAB[idx])
            else:
                names.append("")
        raw = self._encode_tool_names_raw(names, tool_ids.device)
        return self.node_proj(raw)[inverse]

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_features.dtype in (torch.int32, torch.int64, torch.long):
            node_features = self._node_features_from_tool_ids(node_features)
        attn = self.set_attn(node_features).squeeze(-1)
        alpha = pyg_softmax(attn, batch)
        z_set = global_add_pool(node_features * alpha.unsqueeze(-1), batch)
        z_gnn = self.graph_encoder(
            node_features, edge_index, batch, topo_pos=topo_pos, project=False,
        )
        z = self.fuse(torch.cat([z_gnn, z_set], dim=-1))
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_node_features: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        graph_topo_pos: Optional[torch.Tensor] = None,
    ):
        z_text = self.encode_text(input_ids, attention_mask)
        z_graph = self.encode_graph(
            graph_node_features, graph_edge_index, graph_batch,
            topo_pos=graph_topo_pos,
        )
        return z_text, z_graph
