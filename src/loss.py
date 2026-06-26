"""
loss.py — Graph-Aware Contrastive Loss for LEGR
==================================================

Implements the training objective for the LEGR dual-encoder:

    L = InfoNCE(z_text, z_graph) + λ · GED_weighting

The loss pulls matching (query, DAG) pairs together and pushes non-matching
pairs apart, with the repulsion strength modulated by the Graph Edit
Distance (GED) between the positive and negative DAGs.  This teaches the
model that structurally similar DAGs should have similar embeddings.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_alignment_metrics(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
) -> Dict[str, float]:
    """Retrieval @1 and mean positive cosine for L2-normalised embeddings.

    ``recall_at_1`` is the mean of text→graph and graph→text top-1 accuracy
    within the batch (diagonal positives in the similarity matrix).
    """
    b = z_text.size(0)
    if b == 0:
        return {"recall_at_1": 0.0, "pos_cos_mean": 0.0}

    sim = z_text @ z_graph.t()
    device = z_text.device
    labels = torch.arange(b, device=device, dtype=torch.long)

    t2g = (sim.argmax(dim=-1) == labels).float().mean()
    g2t = (sim.argmax(dim=0) == labels).float().mean()
    recall_at_1 = float((t2g + g2t) / 2.0)
    pos_cos_mean = float(sim.diag().mean().item())

    return {
        "recall_at_1": recall_at_1,
        "pos_cos_mean": pos_cos_mean,
    }


class GraphAwareContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss with GED-based margin weighting.

    ``temperature`` is a learnable scalar (initialized from ``temperature_init``)
    so it can be optimised with the rest of the criterion.

    Parameters
    ----------
    temperature_init
        Initial softmax temperature for InfoNCE.
    lambda_ged
        Weight on the GED-weighted auxiliary term.
    ged_scale
        Divides raw GED values before normalisation (numerical scale).
    ged_margin
        Minimum GED below which negative pairs are not penalised.
    """

    def __init__(
        self,
        temperature_init: float = 0.07,
        lambda_ged: float = 0.15,
        ged_scale: float = 2.0,
        ged_margin: float = 0.0,
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        self.lambda_ged = lambda_ged
        self.ged_scale = max(float(ged_scale), 1e-8)
        self.ged_margin = ged_margin

    def forward(
        self,
        z_text: torch.Tensor,
        z_graph: torch.Tensor,
        ged_matrix: Optional[torch.Tensor] = None,
        *,
        dag_ids: Optional[torch.Tensor] = None,
        ged_max: Optional[float] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and metrics for a batch of (text, graph) pairs.

        Parameters
        ----------
        z_text, z_graph
            (B, D) L2-normalised embeddings.
        ged_matrix
            (B, B) pairwise GED submatrix for this batch (same order as the batch).
        ged_max
            Optional global max GED (e.g. from the full dataset matrix) for
            normalising ``ged_matrix``.
        dag_ids
            Accepted for API compatibility with callers; indexing uses
            ``ged_matrix`` as-is when it is already a batch submatrix.
        """
        B = z_text.size(0)
        device = z_text.device
        tau = self.temperature.clamp(min=1e-8, max=100.0)

        # Cosine similarity matrix (already L2-normed)
        sim = torch.mm(z_text, z_graph.t()) / tau  # (B, B)

        labels = torch.arange(B, device=device)

        loss_t2g = F.cross_entropy(sim, labels)
        loss_g2t = F.cross_entropy(sim.t(), labels)
        loss = (loss_t2g + loss_g2t) / 2.0

        if ged_matrix is not None and self.lambda_ged > 0:
            ged_sub = ged_matrix.float() / self.ged_scale
            ged_weights = torch.clamp(ged_sub - self.ged_margin, min=0.0)

            if ged_max is not None and ged_max > 0:
                ged_weights = ged_weights / (float(ged_max) / self.ged_scale + 1e-8)
            else:
                max_ged = ged_weights.max()
                if max_ged > 0:
                    ged_weights = ged_weights / max_ged

            neg_mask = 1.0 - torch.eye(B, device=device)
            inv_ged = 1.0 - ged_weights
            weighted_neg_sim = sim * neg_mask * inv_ged

            pos = torch.exp(torch.diag(sim))
            denom = pos + weighted_neg_sim.sum(dim=1).clamp(min=1e-8)
            ged_loss = (-torch.log(pos / denom)).mean()

            loss = loss + self.lambda_ged * ged_loss

        metrics: Dict[str, float] = {
            "loss_total": float(loss.detach().item()),
            "temperature": float(tau.detach().item()),
        }
        return loss, metrics
