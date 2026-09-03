from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from .structures import directed_relation_bias


def _aggregate(messages: torch.Tensor, index: torch.Tensor, count: int, reduce: str = "mean") -> torch.Tensor:
    if messages.numel() == 0:
        return messages.new_zeros((count, messages.size(-1)))
    result = messages.new_zeros((count, messages.size(-1)))
    result.index_add_(0, index, messages)
    if reduce == "mean":
        degree = messages.new_zeros(count)
        degree.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
        result = result / degree.clamp(min=1).unsqueeze(-1)
    return result


class DirectedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kind: str, dropout: float):
        super().__init__()
        self.kind = kind
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.in_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        extra = hidden_dim * 2 if kind == "pna" else hidden_dim
        self.mix = nn.Linear(hidden_dim + extra, hidden_dim)
        self.gate = nn.GRUCell(hidden_dim, hidden_dim) if kind == "gated" else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        incoming = _aggregate(self.in_linear(h)[src], dst, h.size(0))
        outgoing = _aggregate(self.out_linear(h)[dst], src, h.size(0))
        if self.kind == "pna":
            neighborhood = torch.cat([incoming, outgoing], dim=-1)
        else:
            neighborhood = incoming + outgoing
        update = F.gelu(self.mix(torch.cat([self.self_linear(h), neighborhood], dim=-1)))
        if self.gate is not None:
            update = self.gate(update, h)
        return self.norm(h + self.dropout(update))


class DirectedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.relation_bias = nn.Embedding(6, heads)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        output = h.clone()
        for graph_id in batch.unique(sorted=True).tolist():
            nodes = torch.where(batch == graph_id)[0]
            local_map = {int(node): i for i, node in enumerate(nodes.tolist())}
            keep = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
            local_edges = edge_index[:, keep].clone()
            if local_edges.numel():
                local_edges[0] = torch.tensor([local_map[int(v)] for v in local_edges[0]], device=h.device)
                local_edges[1] = torch.tensor([local_map[int(v)] for v in local_edges[1]], device=h.device)
            relation = directed_relation_bias(len(nodes), local_edges)
            bias = self.relation_bias(relation).permute(2, 0, 1).contiguous()
            x = h[nodes].unsqueeze(0)
            attended, _ = self.attn(x, x, x, attn_mask=bias, need_weights=False)
            x = self.norm1(x + self.dropout(attended))
            x = self.norm2(x + self.dropout(self.ffn(x)))
            output[nodes] = x.squeeze(0)
        return output


class Readout(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, kind: str):
        super().__init__()
        self.kind = kind
        self.tool_attn = nn.Linear(hidden_dim, 1)
        self.struct_attn = nn.Linear(hidden_dim, 1)
        self.set2set = Set2Set(hidden_dim, processing_steps=3) if kind == "set2set" else None
        source_dim = {
            "dual_attention": hidden_dim * 2,
            "set2set": hidden_dim * 2,
            "concat": hidden_dim * 3,
        }.get(kind, hidden_dim)
        self.proj = nn.Linear(source_dim, embed_dim)

    @staticmethod
    def _attention_pool(h, batch, scorer):
        alpha = pyg_softmax(scorer(h).squeeze(-1), batch)
        return global_add_pool(h * alpha.unsqueeze(-1), batch)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.kind == "dual_attention":
            pooled = torch.cat([
                self._attention_pool(h, batch, self.tool_attn),
                self._attention_pool(h, batch, self.struct_attn),
            ], dim=-1)
        elif self.kind == "virtual":
            pooled = self._attention_pool(h, batch, self.struct_attn)
        elif self.kind == "set2set":
            pooled = self.set2set(h, batch)
        elif self.kind == "concat":
            pooled = torch.cat([
                global_mean_pool(h, batch), global_max_pool(h, batch),
                global_add_pool(h, batch) / torch.bincount(batch).clamp(min=1).unsqueeze(-1),
            ], dim=-1)
        else:
            pooled = global_mean_pool(h, batch)
        return F.normalize(self.proj(pooled), dim=-1)


class GraphAdapter(nn.Module):
    """Model-only extension consuming V3 tool-name node features."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        embed_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        graph_kind: str,
        readout_kind: str,
    ):
        super().__init__()
        self.graph_kind = graph_kind
        self.input_proj = nn.Linear(node_dim + 6, hidden_dim)
        local_kind = graph_kind if graph_kind in {"gated", "pna"} else "residual"
        self.local = nn.ModuleList([
            DirectedResidualBlock(hidden_dim, local_kind, dropout) for _ in range(layers)
        ])
        self.global_blocks = nn.ModuleList([
            DirectedTransformerBlock(hidden_dim, heads, dropout) for _ in range(layers)
        ]) if graph_kind in {"graphormer", "gps"} else nn.ModuleList()
        self.readout = Readout(hidden_dim, embed_dim, readout_kind)

    def forward(self, node_features, structural_features, edge_index, batch):
        h = self.input_proj(torch.cat([node_features, structural_features], dim=-1))
        if self.graph_kind == "graphormer":
            for block in self.global_blocks:
                h = block(h, edge_index, batch)
        else:
            for index, block in enumerate(self.local):
                h = block(h, edge_index)
                if self.graph_kind == "gps":
                    h = self.global_blocks[index](h, edge_index, batch)
        return self.readout(h, batch), h


class CrossGraphReranker(nn.Module):
    def __init__(self, query_dim: int, graph_dim: int, hidden_dim: int, heads: int, layers: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, batch_first=True) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_tokens, query_mask, node_states, graph_batch, top_indices):
        batch_size, candidate_count = top_indices.shape
        result = query_tokens.new_zeros((batch_size, candidate_count))
        query = self.query_proj(query_tokens)
        for row in range(batch_size):
            q = query[row : row + 1]
            qmask = ~query_mask[row : row + 1].bool()
            qpool = q.masked_fill(qmask.unsqueeze(-1), 0).sum(1) / (~qmask).sum(1, keepdim=True).clamp(min=1)
            for col, candidate in enumerate(top_indices[row].tolist()):
                nodes = self.graph_proj(node_states[graph_batch == candidate]).unsqueeze(0)
                x = q
                for attn, norm in zip(self.blocks, self.norms):
                    updated, _ = attn(x, nodes, nodes, need_weights=False)
                    x = norm(x + updated)
                xpool = x.masked_fill(qmask.unsqueeze(-1), 0).sum(1) / (~qmask).sum(1, keepdim=True).clamp(min=1)
                npool = nodes.mean(1)
                result[row, col] = self.score(torch.cat([qpool, xpool, npool], dim=-1)).squeeze()
        return result
