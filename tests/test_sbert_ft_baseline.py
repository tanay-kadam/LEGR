"""Tests for fine-tuned SBERT baseline helpers (no MiniLM download, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from action_type_mapping import TOOL_ACTION_TYPE  # noqa: E402
from data_synth import _FULL_TOOL_VOCAB, dag_to_text  # noqa: E402
from sbert_ft_baseline import (  # noqa: E402
    SBERTFineTuneDualEncoder,
    default_split_paths,
    load_config_from_legr_checkpoint,
    make_sbert_collate_fn,
)
from train import TrainConfig, _build_graph, _dag_hash, _parse_edges, _parse_tools  # noqa: E402


class TinyEnc(nn.Module):
    def __init__(self, model_name="x", embed_dim=8, freeze_backbone=False, num_frozen_layers=0):
        super().__init__()
        self.backbone = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, embed_dim)
        if freeze_backbone or num_frozen_layers > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_param_groups(self, backbone_lr: float, head_lr: float):
        backbone, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith("proj.") else backbone).append(p)
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        if head:
            groups.append({"params": head, "lr": head_lr})
        return groups

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids.clamp(min=0, max=31))
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.proj(pooled)


class DummyTok:
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        n = len(texts)
        ids = torch.ones(n, 4, dtype=torch.long)
        mask = torch.ones(n, 4, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = min(31, max(1, len(t) % 31))
        return {"input_ids": ids, "attention_mask": mask}


def labelled_hashes(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path)
    out = set()
    for _, row in df.iterrows():
        tools = _parse_tools(row["tools"])
        edges = _parse_edges(row["edges"])
        if not tools:
            continue
        G = _build_graph(tools, edges)
        out.add(_dag_hash(G))
    return out


def test_mapping_covers_full_legr_vocab():
    missing = [t for t in _FULL_TOOL_VOCAB if t not in TOOL_ACTION_TYPE]
    assert missing == []


def test_collate_tokenizes_dag_strings():
    G = _build_graph(["db_read", "create_ticket"], [(0, 1)])
    texts = [dag_to_text(G)]
    collate = make_sbert_collate_fn(DummyTok(), texts, max_length=128)
    batch = collate([{"query": "reset my password", "dag_id": 0, "graph": None}])
    assert batch["input_ids"].shape[0] == 1
    assert batch["doc_input_ids"].shape[0] == 1
    assert batch["dag_ids"].tolist() == [0]
    assert texts[0] == "db_read -> create_ticket"


def test_tied_vs_untied_parameter_identity():
    tied = SBERTFineTuneDualEncoder(embed_dim=8, tied=True, encoder_cls=TinyEnc)
    untied = SBERTFineTuneDualEncoder(embed_dim=8, tied=False, encoder_cls=TinyEnc)
    assert tied.query_encoder is tied.doc_encoder
    assert untied.query_encoder is not untied.doc_encoder
    tied_n = sum(p.numel() for p in tied.parameters())
    untied_n = sum(p.numel() for p in untied.parameters())
    assert untied_n == 2 * tied_n


def test_freeze_policy_on_both_towers():
    model = SBERTFineTuneDualEncoder(
        embed_dim=8, tied=False, num_frozen_layers=4, encoder_cls=TinyEnc,
    )
    assert all(not p.requires_grad for p in model.query_encoder.backbone.parameters())
    assert all(not p.requires_grad for p in model.doc_encoder.backbone.parameters())
    assert any(p.requires_grad for p in model.query_encoder.proj.parameters())


def test_loss_runs_on_dummy_batch():
    from loss import GraphAwareContrastiveLoss

    model = SBERTFineTuneDualEncoder(embed_dim=8, tied=False, encoder_cls=TinyEnc)
    criterion = GraphAwareContrastiveLoss(lambda_ged=0.0)
    tok = DummyTok()
    collate = make_sbert_collate_fn(tok, ["a", "b"], max_length=16)
    batch = collate([
        {"query": "q1", "dag_id": 0},
        {"query": "q2", "dag_id": 1},
    ])
    zq, zd = model(
        batch["input_ids"], batch["attention_mask"],
        batch["doc_input_ids"], batch["doc_attention_mask"],
    )
    ged = torch.zeros(2, 2)
    loss, metrics = criterion(zq, zd, ged)
    assert torch.isfinite(loss)
    assert "loss_total" in metrics


def test_drop_last_omits_incomplete_batch():
    from torch.utils.data import DataLoader, Dataset

    class D(Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, i):
            return {"query": f"q{i}", "dag_id": i % 2}

    collate = make_sbert_collate_fn(DummyTok(), ["a", "b"], max_length=16)
    loader = DataLoader(D(), batch_size=3, shuffle=False, collate_fn=collate, drop_last=True)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 3
    loader_keep = DataLoader(D(), batch_size=3, shuffle=False, collate_fn=collate, drop_last=False)
    sizes = [b["input_ids"].shape[0] for b in loader_keep]
    assert sizes == [3, 1]


def test_config_copied_from_checkpoint(tmp_path):
    cfg = TrainConfig(lambda_ged=0.30, lr=1e-3, seed=42)
    ckpt = tmp_path / "legr.pt"
    torch.save({"config": vars(cfg), "model_state": {}}, ckpt)
    loaded = load_config_from_legr_checkpoint(str(ckpt), lambda_ged=0.0, checkpoint_dir="out")
    assert loaded.lambda_ged == 0.0
    assert loaded.lr == 1e-3
    assert loaded.seed == 42
    assert loaded.checkpoint_dir == "out"


def test_default_split_paths_30tool_exist():
    train, val, test = default_split_paths(30)
    assert train.exists()
    assert val.exists()
    assert test.exists()


def test_labelled_split_hashes_15_and_45_disjoint():
    t15 = ROOT / "upgraded" / "upgraded_15tools" / "train.csv"
    e15 = ROOT / "upgraded" / "upgraded_15tools" / "test_topology_heldout.csv"
    t45 = ROOT / "upgraded" / "upgraded_45tools" / "train.csv"
    e45 = ROOT / "upgraded" / "upgraded_45tools" / "test_topology_heldout.csv"
    assert labelled_hashes(t15).isdisjoint(labelled_hashes(e15))
    assert labelled_hashes(t45).isdisjoint(labelled_hashes(e45))


def test_30tool_labelled_overlap_is_measured():
    """Record labelled-DAG overlap; current upgraded/ 30-tool files are disjoint."""
    train = ROOT / "upgraded" / "upgraded_30tools" / "train.csv"
    test = ROOT / "upgraded" / "upgraded_30tools" / "test_topology_heldout.csv"
    overlap = labelled_hashes(train) & labelled_hashes(test)
    # Do not silently rewrite splits. Current packaged files have zero labelled overlap.
    assert overlap == set()
