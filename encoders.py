"""
encoders.py — LEGR Dual-Encoder Architecture
===============================================

Implements the Latent Execution-Graph Routing (LEGR) dual-encoder model:

    - **Text encoder**: Pre-trained transformer (e.g. all-MiniLM-L6-v2) that
      maps natural-language queries to dense vectors.
    - **Graph encoder**: GCN or GAT over execution-DAG structures with
      integer-coded tool-node features, topological position embeddings,
      and optional text-based node features.
    - **LEGRDualEncoder**: Wraps both towers for joint training, producing
      aligned (z_text, z_graph) embeddings in a shared latent space.

The dual-encoder is trained with ``GraphAwareContrastiveLoss`` (see loss.py)
to bring matching (query, DAG) pairs closer while pushing non-matching
pairs apart, weighted by Graph Edit Distance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer


def get_tokenizer(model_name: str):
    """Load a HuggingFace ``AutoTokenizer`` for the given pretrained model."""
    return AutoTokenizer.from_pretrained(model_name)


def _freeze_transformer_backbone_layers(backbone: nn.Module, num_layers: int) -> None:
    """Freeze the first ``num_layers`` blocks of a HF encoder (BERT / RoBERTa / DistilBERT)."""
    if num_layers <= 0:
        return
    encoder = getattr(backbone, "encoder", None)
    layer_stack = getattr(encoder, "layer", None) if encoder is not None else None
    if layer_stack is None:
        transformer = getattr(backbone, "transformer", None)
        layer_stack = getattr(transformer, "layer", None) if transformer is not None else None
    if layer_stack is None:
        return
    for i in range(min(num_layers, len(layer_stack))):
        for p in layer_stack[i].parameters():
            p.requires_grad = False


# ═══════════════════════════════════════════════════════════════════════════
#  Text Encoder
# ═══════════════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """Transformer-based text encoder with mean pooling."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        num_frozen_layers: int = 0,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif num_frozen_layers > 0:
            _freeze_transformer_backbone_layers(self.backbone, num_frozen_layers)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """AdamW param groups: backbone vs projection head (differential LR)."""
        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("proj."):
                head_params.append(p)
            else:
                backbone_params.append(p)
        groups: list[dict] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state  # (B, seq_len, hidden)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_emb * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_emb / count  # (B, hidden)

        return self.proj(mean_pooled)  # (B, embed_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Encoder (GCN)
# ═══════════════════════════════════════════════════════════════════════════

class GCNGraphEncoder(nn.Module):
    """GCN-based graph encoder for execution DAGs."""

    def __init__(
        self,
        num_tools: int,
        tool_embed_dim: int = 64,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 3,
        max_topo_pos: int = 32,
        use_text_node_features: bool = False,
        text_feature_dim: int = 0,
    ):
        super().__init__()
        self.max_topo_pos = max_topo_pos
        self.tool_embedding = nn.Embedding(num_tools + 1, tool_embed_dim, padding_idx=0)
        self.topo_embedding = nn.Embedding(max_topo_pos + 1, tool_embed_dim)

        input_dim = tool_embed_dim * 2
        if use_text_node_features and text_feature_dim > 0:
            self.text_proj = nn.Linear(text_feature_dim, tool_embed_dim)
            input_dim += tool_embed_dim
        else:
            self.text_proj = None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tool_emb = self.tool_embedding(x.squeeze(-1))

        if topo_pos is not None:
            topo_emb = self.topo_embedding(topo_pos.clamp(max=self.max_topo_pos))
        else:
            topo_emb = torch.zeros_like(tool_emb)

        h = torch.cat([tool_emb, topo_emb], dim=-1)

        if self.text_proj is not None:
            text_feat = torch.zeros(h.size(0), self.text_proj.in_features,
                                    device=h.device)
            h = torch.cat([h, self.text_proj(text_feat)], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

        graph_emb = global_mean_pool(h, batch)  # (num_graphs, hidden)
        return self.proj(graph_emb)  # (num_graphs, embed_dim)


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Encoder (GAT)
# ═══════════════════════════════════════════════════════════════════════════

class GATGraphEncoder(nn.Module):
    """GAT-based graph encoder for execution DAGs."""

    def __init__(
        self,
        num_tools: int,
        tool_embed_dim: int = 64,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        max_topo_pos: int = 32,
        use_text_node_features: bool = False,
        text_feature_dim: int = 0,
    ):
        super().__init__()
        self.max_topo_pos = max_topo_pos
        self.tool_embedding = nn.Embedding(num_tools + 1, tool_embed_dim, padding_idx=0)
        self.topo_embedding = nn.Embedding(max_topo_pos + 1, tool_embed_dim)

        input_dim = tool_embed_dim * 2
        if use_text_node_features and text_feature_dim > 0:
            self.text_proj = nn.Linear(text_feature_dim, tool_embed_dim)
            input_dim += tool_embed_dim
        else:
            self.text_proj = None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GATConv(in_dim, hidden_dim // num_heads,
                                      heads=num_heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tool_emb = self.tool_embedding(x.squeeze(-1))

        if topo_pos is not None:
            topo_emb = self.topo_embedding(topo_pos.clamp(max=self.max_topo_pos))
        else:
            topo_emb = torch.zeros_like(tool_emb)

        h = torch.cat([tool_emb, topo_emb], dim=-1)

        if self.text_proj is not None:
            text_feat = torch.zeros(h.size(0), self.text_proj.in_features,
                                    device=h.device)
            h = torch.cat([h, self.text_proj(text_feat)], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

        graph_emb = global_mean_pool(h, batch)
        return self.proj(graph_emb)


# ═══════════════════════════════════════════════════════════════════════════
#  Dual Encoder
# ═══════════════════════════════════════════════════════════════════════════

class LEGRDualEncoder(nn.Module):
    """Wraps both encoder towers for joint training and inference.

    Produces L2-normalised embeddings z_text and z_graph in a shared
    latent space of dimension ``embed_dim``.
    """

    def __init__(
        self,
        num_tools: int = 15,
        embed_dim: int = 256,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_encoder_type: str = "gcn",
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 3,
        graph_num_heads: int = 4,
        tool_embed_dim: int = 64,
        max_topo_pos: int = 32,
        freeze_text_backbone: bool = False,
        num_frozen_layers: int = 0,
        use_text_node_features: bool = False,
        text_feature_dim: int = 0,
        # Aliases used by ``train.py`` (override the graph_* / freeze_* names above)
        gcn_layers: int | None = None,
        node_embed_dim: int | None = None,
        gcn_hidden: int | None = None,
        freeze_text: bool | None = None,
    ):
        if gcn_layers is not None:
            graph_num_layers = gcn_layers
        if node_embed_dim is not None:
            tool_embed_dim = node_embed_dim
        if gcn_hidden is not None:
            graph_hidden_dim = gcn_hidden
        if freeze_text is not None:
            freeze_text_backbone = freeze_text

        super().__init__()
        self.embed_dim = embed_dim

        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            num_frozen_layers=num_frozen_layers,
        )

        GraphEncoderClass = GATGraphEncoder if graph_encoder_type == "gat" else GCNGraphEncoder
        gnn_kwargs = dict(
            num_tools=num_tools,
            tool_embed_dim=tool_embed_dim,
            hidden_dim=graph_hidden_dim,
            embed_dim=embed_dim,
            num_layers=graph_num_layers,
            max_topo_pos=max_topo_pos,
            use_text_node_features=use_text_node_features,
            text_feature_dim=text_feature_dim,
        )
        if graph_encoder_type == "gat":
            gnn_kwargs["num_heads"] = graph_num_heads

        self.graph_encoder = GraphEncoderClass(**gnn_kwargs)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.text_encoder(input_ids, attention_mask)
        return F.normalize(z, p=2, dim=-1)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.graph_encoder(x, edge_index, batch, topo_pos=topo_pos)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        graph_topo_pos: torch.Tensor | None = None,
    ):
        z_text = self.encode_text(input_ids, attention_mask)
        z_graph = self.encode_graph(
            graph_x, graph_edge_index, graph_batch, topo_pos=graph_topo_pos,
        )
        return z_text, z_graph
