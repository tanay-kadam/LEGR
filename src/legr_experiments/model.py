from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from encoders_v2 import LEGRDualEncoderV3
    from sbert_ft_baseline import SBERTFineTuneDualEncoder
except ImportError:  # pragma: no cover
    from src.encoders_v2 import LEGRDualEncoderV3
    from src.sbert_ft_baseline import SBERTFineTuneDualEncoder

from .config import ModelConfig
from .graph_encoders import CrossGraphReranker, GraphAdapter
from .structures import REL_IGNORE


def _load_state(module: nn.Module, checkpoint: str | Path | None) -> dict:
    if not checkpoint:
        return {"loaded": False, "missing": [], "unexpected": []}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    result = module.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


class LEGRResearchModel(nn.Module):
    """Non-destructive composition of existing V3/SBERT with new adapters."""

    def __init__(
        self,
        config: ModelConfig,
        vocabulary: Sequence[str],
        v3_checkpoint: str | Path | None = None,
        sbert_checkpoint: str | Path | None = None,
    ):
        super().__init__()
        self.config = config
        self.vocabulary = list(vocabulary)
        self.base_legr = LEGRDualEncoderV3(
            embed_dim=config.embed_dim,
            text_model_name=config.text_model,
            graph_hidden_dim=256,
            graph_num_layers=3,
            node_feature_dim=64,
            max_topo_pos=16,
            num_frozen_layers=4,
        )
        self.load_reports = {"v3": _load_state(self.base_legr, v3_checkpoint)}
        hidden = self.base_legr.text_encoder.backbone.config.hidden_size
        self.graph_adapter = GraphAdapter(
            node_dim=64, hidden_dim=config.hidden_dim, embed_dim=config.embed_dim,
            layers=config.graph_layers, heads=config.attention_heads,
            dropout=config.dropout, graph_kind=config.graph_kind,
            readout_kind=config.readout_kind,
        )
        self.tool_queries = nn.Parameter(torch.empty(len(vocabulary), hidden))
        nn.init.normal_(self.tool_queries, std=0.02)
        self.tool_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.tool_head = nn.Linear(hidden, 1)
        relation_dim = min(128, hidden)
        self.relation_proj = nn.Linear(hidden, relation_dim)
        self.relation_head = nn.Sequential(
            nn.Linear(relation_dim * 4, relation_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(relation_dim, 5),
        )
        self.struct_query_proj = nn.Linear(hidden, config.embed_dim)
        self.fusion_gate = nn.Linear(hidden, 5)
        self.expert_scale = nn.Parameter(torch.ones(5))

        self.semantic_expert = None
        if config.use_semantic_expert:
            self.semantic_expert = SBERTFineTuneDualEncoder(
                embed_dim=config.embed_dim, text_model_name=config.text_model,
                num_frozen_layers=4, tied=False,
            )
            self.load_reports["sbert"] = _load_state(self.semantic_expert, sbert_checkpoint)
            for parameter in self.semantic_expert.parameters():
                parameter.requires_grad = False

        self.reranker = CrossGraphReranker(
            query_dim=hidden, graph_dim=config.hidden_dim, hidden_dim=config.hidden_dim,
            heads=4, layers=config.rerank_layers,
        ) if config.use_reranker and config.rerank_k > 0 else None
        self._apply_unfreeze_policy(config.unfreeze)

    def _apply_unfreeze_policy(self, policy: str) -> None:
        backbone = self.base_legr.text_encoder.backbone
        if policy == "frozen":
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        elif policy == "last2":
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            layers = getattr(backbone, "encoder", None)
            layers = getattr(layers, "layer", [])
            for layer in list(layers)[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def initialize_tool_queries(self, device: torch.device) -> None:
        with torch.no_grad():
            raw = self.base_legr._encode_tool_names_raw(self.vocabulary, device)
            if raw.shape == self.tool_queries.shape:
                self.tool_queries.copy_(raw)

    @staticmethod
    def _masked_mean(states, mask):
        expanded = mask.unsqueeze(-1).float()
        return (states * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

    def _query_outputs(self, input_ids, attention_mask):
        result = self.base_legr.text_encoder.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        token_states = result.last_hidden_state
        pooled = self._masked_mean(token_states, attention_mask)
        query_embedding = F.normalize(self.struct_query_proj(pooled), dim=-1)
        tool_queries = self.tool_queries.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        contexts, _ = self.tool_attention(
            tool_queries, token_states, token_states,
            key_padding_mask=~attention_mask.bool(), need_weights=False,
        )
        tool_logits = self.tool_head(contexts).squeeze(-1)
        relation_context = self.relation_proj(contexts)
        left = relation_context.unsqueeze(2).expand(-1, -1, len(self.vocabulary), -1)
        right = relation_context.unsqueeze(1).expand(-1, len(self.vocabulary), -1, -1)
        relation_features = torch.cat([left, right, left - right, left * right], dim=-1)
        relation_logits = self.relation_head(relation_features)
        return token_states, pooled, query_embedding, tool_logits, relation_logits

    @staticmethod
    def _relation_scores(relation_logits, candidate_targets):
        log_probs = F.log_softmax(relation_logits, dim=-1)
        columns = []
        for candidate in range(candidate_targets.size(0)):
            target = candidate_targets[candidate]
            positions = torch.where(target != REL_IGNORE)
            if positions[0].numel() == 0:
                columns.append(log_probs.new_zeros(log_probs.size(0)))
                continue
            labels = target[positions]
            selected = log_probs[:, positions[0], positions[1], :]
            values = selected.gather(-1, labels.view(1, -1, 1).expand(log_probs.size(0), -1, 1))
            columns.append(values.squeeze(-1).mean(-1))
        return torch.stack(columns, dim=1)

    def _fuse(self, pooled, expert_scores):
        kind = self.config.fusion_kind
        if kind == "graph":
            return expert_scores[..., 2]
        if kind == "semantic":
            return expert_scores[..., 0]
        scaled = expert_scores * self.expert_scale.view(1, 1, -1)
        if kind == "fixed":
            return scaled.mean(-1)
        if kind == "scalar":
            weights = F.softmax(self.expert_scale, dim=0)
            return (expert_scores * weights.view(1, 1, -1)).sum(-1)
        weights = F.softmax(self.fusion_gate(pooled), dim=-1)
        return (scaled * weights.unsqueeze(1)).sum(-1)

    def score_batches(self, query_batch: dict, candidate_batch: dict) -> dict:
        device = next(self.parameters()).device
        ids = query_batch["input_ids"].to(device)
        mask = query_batch["attention_mask"].to(device)
        gx = candidate_batch["graph_x"].to(device)
        ge = candidate_batch["graph_edge_index"].to(device)
        gb = candidate_batch["graph_batch"].to(device)
        gs = candidate_batch["graph_struct_x"].to(device)
        tool_targets = candidate_batch["tool_targets"].to(device)
        relation_targets = candidate_batch["relation_targets"].to(device)

        tokens, pooled, z_query, tool_logits, relation_logits = self._query_outputs(ids, mask)
        node_features = self.base_legr._node_features_from_tool_ids(gx)
        z_graph, node_states = self.graph_adapter(node_features, gs, ge, gb)
        z_v3 = self.base_legr.encode_graph(gx, ge, gb, topo_pos=None)
        structural_score = z_query @ z_graph.t()
        v3_query = F.normalize(self.base_legr.text_encoder.proj(pooled), dim=-1)
        v3_score = v3_query @ z_v3.t()
        tool_score = F.normalize(torch.sigmoid(tool_logits), dim=-1) @ F.normalize(tool_targets, dim=-1).t()
        relation_score = self._relation_scores(relation_logits, relation_targets)

        semantic_score = structural_score.new_zeros(structural_score.shape)
        if self.semantic_expert is not None:
            with torch.no_grad():
                z_sem_q = self.semantic_expert.encode_query(ids, mask)
                z_sem_d = self.semantic_expert.encode_document(
                    candidate_batch["doc_input_ids"].to(device),
                    candidate_batch["doc_attention_mask"].to(device),
                )
            semantic_score = z_sem_q @ z_sem_d.t()

        experts = torch.stack([
            semantic_score, v3_score, structural_score, tool_score, relation_score,
        ], dim=-1)
        scores = self._fuse(pooled, experts)
        if self.reranker is not None and scores.size(1) > 1:
            k = min(self.config.rerank_k, scores.size(1))
            top = scores.detach().topk(k, dim=1).indices
            rerank = self.reranker(tokens, mask, node_states, gb, top)
            scores = scores.scatter_add(1, top, rerank)
        return {
            "scores": scores,
            "query_embedding": z_query,
            "graph_embedding": z_graph,
            "tool_logits": tool_logits,
            "relation_logits": relation_logits,
            "expert_scores": experts,
        }

    def forward(self, batch: dict) -> dict:
        return self.score_batches(batch, batch)
