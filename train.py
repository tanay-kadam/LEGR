"""
train.py — Training Loop for Latent Execution-Graph Routing
=============================================================

Trains the ``LEGRDualEncoder`` to align the text and graph encoder latent
spaces using the Graph-Aware Contrastive Loss (GACL).

Experiment tracking
~~~~~~~~~~~~~~~~~~~
All scalar metrics are logged to **Weights & Biases** on every training
step and at the end of each validation epoch.  The full hyper-parameter
configuration is captured in the W&B run config for reproducibility.

Checkpointing
~~~~~~~~~~~~~
The best model (lowest validation loss) and the final model are saved
under ``checkpoints/``.

Usage
~~~~~
::

    python train.py                          # defaults
    python train.py --epochs 100 --lr 2e-5   # override
    python train.py --wandb_project LEGR     # custom W&B project

Ablation (train without GED loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    python train.py --lambda_ged 0 \\
                    --checkpoint_dir checkpoints_no_ged \\
                    --wandb_run_name LEGR-NoGED

    Default GED hyperparameters: ``lambda_ged=0.30``, ``ged_scale=2.5``,
    ``ged_margin=0.05``.  Match ``LEGR_WITH_GED`` in
    ``results/legr_full_lambda_ged_default.csv`` with
    ``--lambda_ged 0.15 --ged_scale 2.0 --ged_margin 0``.  No GED:
    ``--lambda_ged 0``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb

from legr_tool_count import (
    add_tool_count_argument,
    bootstrap_tool_count_from_argv,
    get_active_tool_count,
)

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from data_synth import LEGRDataset, build_splits, dag_to_pyg, NUM_TOOLS
from encoders import LEGRDualEncoder, get_tokenizer
from loss import GraphAwareContrastiveLoss, compute_alignment_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # Data
    entity_variants: int = 20
    data_seed: int = 42
    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    tool_count: Optional[int] = None

    # Architecture
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 256
    gcn_layers: int = 3
    node_embed_dim: int = 64
    gcn_hidden: int = 256
    freeze_text: bool = False
    num_frozen_layers: int = 4
    max_topo_pos: int = 16
    graph_encoder_type: str = "gcn"

    # Loss — default GED prior (override to match older CSV baselines via CLI)
    temperature_init: float = 0.05
    lambda_ged: float = 0.30
    ged_scale: float = 2.5
    ged_margin: float = 0.05

    # Optimiser — differential LR keeps the pretrained backbone stable
    # while letting randomly-initialised heads and GCN learn faster.
    lr: float = 2e-4
    text_backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Training loop
    epochs: int = 100
    warmup_epochs: int = 3
    batch_size: int = 128
    num_workers: int = 0
    max_length: int = 128
    val_every: int = 1
    patience: int = 15

    # Infra
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "LEGR"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    seed: int = 42


@dataclass
class _CSVSample:
    query: str
    graph: object
    dag_id: int
    dag_nx: nx.DiGraph


def _resolve_cfg_tool_count(cfg: TrainConfig) -> int:
    """Ensure runtime config matches the already-active LEGR tier."""
    active_tool_count = get_active_tool_count()
    if cfg.tool_count is not None and cfg.tool_count != active_tool_count:
        raise ValueError(
            f"TrainConfig.tool_count={cfg.tool_count} does not match the active "
            f"LEGR tool count ({active_tool_count}). Re-run with "
            f"--tool_count {cfg.tool_count} before importing LEGR modules."
        )
    cfg.tool_count = active_tool_count
    return active_tool_count


LEGR_30TOOL_DEFAULT_DIR = Path("upgraded") / "upgraded_30tools"
LEGR_30TOOL_TRAIN_CSV = LEGR_30TOOL_DEFAULT_DIR / "train.csv"
LEGR_30TOOL_VAL_CSV = LEGR_30TOOL_DEFAULT_DIR / "dev.csv"


def _default_train_val_csv_paths(tool_count: int) -> tuple[Optional[str], Optional[str]]:
    """Return packaged CSV train/val defaults for the selected LEGR tier."""
    if tool_count != 30:
        return None, None
    return str(LEGR_30TOOL_TRAIN_CSV), str(LEGR_30TOOL_VAL_CSV)


def _resolve_train_val_csv_paths(cfg: TrainConfig) -> tuple[Optional[str], Optional[str]]:
    """Resolve explicit or default CSV train/val paths for the current run."""
    if cfg.train_csv or cfg.val_csv:
        if not (cfg.train_csv and cfg.val_csv):
            raise ValueError("Provide both --train_csv and --val_csv together.")
        return cfg.train_csv, cfg.val_csv

    tool_count = cfg.tool_count if cfg.tool_count is not None else get_active_tool_count()
    train_csv, val_csv = _default_train_val_csv_paths(tool_count)
    if train_csv is None or val_csv is None:
        return None, None

    missing = [
        str(path)
        for path in (Path(train_csv), Path(val_csv))
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Default 30-tool LEGR CSVs are missing: "
            f"{', '.join(missing)}. Run "
            "`python scripts/prepare_legr_30tool_dataset.py` "
            "or pass --train_csv/--val_csv explicitly."
        )

    cfg.train_csv = train_csv
    cfg.val_csv = val_csv
    return train_csv, val_csv


def _build_checkpoint_payload(
    *,
    epoch: int,
    model,
    criterion,
    cfg: TrainConfig,
    optimizer=None,
    scheduler=None,
    val_loss: float | None = None,
) -> Dict[str, object]:
    """Build a checkpoint payload with persisted tool-count metadata."""
    payload: Dict[str, object] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "criterion_state": criterion.state_dict(),
        "config": vars(cfg),
        "tool_count": cfg.tool_count,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if val_loss is not None:
        payload["val_loss"] = val_loss
    return payload


class CSVTrainDataset(torch.utils.data.Dataset):
    """CSV-backed dataset with LEGRDataset-compatible training interface."""

    def __init__(self):
        super().__init__()
        self.samples: list[_CSVSample] = []
        self._unique_dags: list[nx.DiGraph] = []
        self._dag_texts: list[str] = []
        self.ged_matrix = None
        self.num_unique_dags = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {"query": s.query, "graph": s.graph, "dag_id": s.dag_id}

    def get_ged_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.ged_matrix).float()


def _parse_tools(cell) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(";") if x.strip()]


def _parse_edges(cell) -> list[tuple[int, int]]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part or "->" not in part:
            continue
        s, d = part.split("->", 1)
        out.append((int(s), int(d)))
    return out


def _build_graph(tools: list[str], edges: list[tuple[int, int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, t in enumerate(tools):
        G.add_node(i, tool=t)
    G.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"Invalid DAG: tools={tools}, edges={edges}")
    return G


def _dag_hash(G: nx.DiGraph) -> str:
    node_labels = tuple(sorted(G.nodes[n]["tool"] for n in G.nodes()))
    edge_labels = tuple(sorted((G.nodes[u]["tool"], G.nodes[v]["tool"]) for u, v in G.edges()))
    return f"{node_labels}|{edge_labels}"


def _build_csv_train_val_datasets(
    train_csv: str,
    val_csv: str,
) -> tuple[CSVTrainDataset, CSVTrainDataset]:
    """Build train/val datasets from CSVs with shared DAG id space and GED matrix."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    required = {"query", "tools", "edges"}
    for name, df in (("train", train_df), ("val", val_df)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns {sorted(missing)}; found {list(df.columns)}")

    # Build shared unique DAG pool across train+val
    unique_dags: list[nx.DiGraph] = []
    dag_hash_to_id: dict[str, int] = {}
    dag_pyg_cache: dict[int, object] = {}
    dag_texts: list[str] = []

    def _process_rows(df: pd.DataFrame):
        samples: list[_CSVSample] = []
        for _, row in df.iterrows():
            tools = _parse_tools(row["tools"])
            edges = _parse_edges(row["edges"])
            if not tools:
                continue
            G = _build_graph(tools, edges)
            h = _dag_hash(G)
            if h not in dag_hash_to_id:
                dag_hash_to_id[h] = len(unique_dags)
                unique_dags.append(G)
                dag_texts.append(row["dag_text"] if "dag_text" in df.columns and pd.notna(row["dag_text"]) else "")
                dag_pyg_cache[dag_hash_to_id[h]] = dag_to_pyg(G)
            dag_id = dag_hash_to_id[h]
            samples.append(
                _CSVSample(
                    query=str(row["query"]),
                    graph=dag_pyg_cache[dag_id].clone(),
                    dag_id=dag_id,
                    dag_nx=G,
                )
            )
        return samples

    train_samples = _process_rows(train_df)
    val_samples = _process_rows(val_df)

    if not unique_dags:
        raise ValueError("No valid DAGs found in train/val CSVs.")

    # Exact GED via graph_edit_distance becomes prohibitively slow once the
    # unique DAG pool grows. For CSV-backed training we use a deterministic
    # structural distance surrogate that preserves relative topology similarity.
    def _fast_graph_distance(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
        n_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
        e_diff = abs(g1.number_of_edges() - g2.number_of_edges())

        t1 = sorted(g1.nodes[n]["tool"] for n in g1.nodes())
        t2 = sorted(g2.nodes[n]["tool"] for n in g2.nodes())
        # Bag mismatch cost
        i = j = common = 0
        while i < len(t1) and j < len(t2):
            if t1[i] == t2[j]:
                common += 1
                i += 1
                j += 1
            elif t1[i] < t2[j]:
                i += 1
            else:
                j += 1
        tool_bag_diff = (len(t1) - common) + (len(t2) - common)

        # Topology signature mismatch
        indeg1 = sorted(g1.in_degree(n) for n in g1.nodes())
        indeg2 = sorted(g2.in_degree(n) for n in g2.nodes())
        outdeg1 = sorted(g1.out_degree(n) for n in g1.nodes())
        outdeg2 = sorted(g2.out_degree(n) for n in g2.nodes())
        deg_diff = abs(sum(indeg1) - sum(indeg2)) + abs(sum(outdeg1) - sum(outdeg2))

        return float(n_diff + e_diff + tool_bag_diff + 0.25 * deg_diff)

    n_u = len(unique_dags)
    ged = torch.zeros((n_u, n_u), dtype=torch.float32).numpy()
    for i in range(n_u):
        for j in range(i + 1, n_u):
            d = _fast_graph_distance(unique_dags[i], unique_dags[j])
            ged[i, j] = d
            ged[j, i] = d

    train_ds = CSVTrainDataset()
    train_ds.samples = train_samples
    train_ds._unique_dags = unique_dags
    train_ds._dag_texts = dag_texts
    train_ds.ged_matrix = ged
    train_ds.num_unique_dags = len(unique_dags)

    val_ds = CSVTrainDataset()
    val_ds.samples = val_samples
    val_ds._unique_dags = unique_dags
    val_ds._dag_texts = dag_texts
    val_ds.ged_matrix = ged
    val_ds.num_unique_dags = len(unique_dags)

    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Collation
# ═══════════════════════════════════════════════════════════════════════════════

def make_collate_fn(tokenizer: AutoTokenizer, max_length: int = 128):
    """Return a collate function that tokenises queries and batches graphs.

    The GED sub-matrix for each batch is constructed at training time from
    the pre-computed full GED matrix (passed via the ``ged_full`` tensor
    stored on the dataset).
    """

    def collate(batch):
        queries = [b["query"] for b in batch]
        graphs = [b["graph"] for b in batch]
        dag_ids = torch.tensor([b["dag_id"] for b in batch], dtype=torch.long)

        encoded = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        graph_batch = Batch.from_data_list(graphs)

        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "graph_x": graph_batch.x,
            "graph_edge_index": graph_batch.edge_index,
            "graph_batch": graph_batch.batch,
            "dag_ids": dag_ids,
        }

        if hasattr(graph_batch, "topo_pos") and graph_batch.topo_pos is not None:
            result["graph_topo_pos"] = graph_batch.topo_pos

        return result

    return collate


def _ged_submatrix(
    dag_ids: torch.Tensor,
    ged_full: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Index into the full GED matrix to produce the (B, B) batch sub-matrix."""
    return ged_full[dag_ids][:, dag_ids].to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Training & validation steps
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: LEGRDualEncoder,
    criterion: GraphAwareContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    ged_full: torch.Tensor,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    ged_global_max: float | None = None,
) -> Dict[str, float]:
    """Run one training epoch; return averaged metrics."""
    model.train()
    criterion.train()

    accum = {}
    n_steps = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gx = batch["graph_x"].to(device)
        ge = batch["graph_edge_index"].to(device)
        gb = batch["graph_batch"].to(device)
        gtp = batch.get("graph_topo_pos")
        if gtp is not None:
            gtp = gtp.to(device)

        z_text, z_graph = model(ids, mask, gx, ge, gb, graph_topo_pos=gtp)

        dag_ids = batch["dag_ids"].to(device)
        ged_sub = _ged_submatrix(batch["dag_ids"], ged_full, device)
        loss, metrics = criterion(z_text, z_graph, ged_sub, dag_ids=dag_ids,
                                  ged_max=ged_global_max)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            max_grad_norm,
        )
        optimizer.step()

        # Accumulate
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_steps += 1

        pbar.set_postfix(loss=f"{metrics['loss_total']:.4f}")

    return {k: v / n_steps for k, v in accum.items()}


@torch.no_grad()
def validate(
    model: LEGRDualEncoder,
    criterion: GraphAwareContrastiveLoss,
    loader: DataLoader,
    ged_full: torch.Tensor,
    device: torch.device,
    ged_global_max: float | None = None,
) -> Dict[str, float]:
    """Run validation; return averaged loss + alignment metrics."""
    model.eval()
    criterion.eval()

    accum = {}
    n_steps = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gx = batch["graph_x"].to(device)
        ge = batch["graph_edge_index"].to(device)
        gb = batch["graph_batch"].to(device)
        gtp = batch.get("graph_topo_pos")
        if gtp is not None:
            gtp = gtp.to(device)

        z_text, z_graph = model(ids, mask, gx, ge, gb, graph_topo_pos=gtp)

        dag_ids = batch["dag_ids"].to(device)
        ged_sub = _ged_submatrix(batch["dag_ids"], ged_full, device)
        _, metrics = criterion(z_text, z_graph, ged_sub, dag_ids=dag_ids,
                               ged_max=ged_global_max)

        align = compute_alignment_metrics(z_text, z_graph)
        metrics.update(align)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in accum.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)
    _resolve_cfg_tool_count(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building dataset …")
    explicit_train_csv = cfg.train_csv
    explicit_val_csv = cfg.val_csv
    train_csv, val_csv = _resolve_train_val_csv_paths(cfg)
    if train_csv and val_csv:
        label = "CSV train/val datasets"
        if explicit_train_csv is None and explicit_val_csv is None:
            label = "default 30-tool CSV train/val datasets"
        print(f"  Using {label}:\n    train={train_csv}\n    val={val_csv}")
        train_ds, val_ds = _build_csv_train_val_datasets(train_csv, val_csv)
    else:
        train_ds, val_ds, _ = build_splits(
            entity_variants=cfg.entity_variants, seed=cfg.data_seed,
        )
    ged_full = train_ds.get_ged_tensor()
    ged_global_max = float(ged_full.max())

    tokenizer = get_tokenizer(cfg.text_model)
    collate = make_collate_fn(tokenizer, cfg.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # ── Model & loss ──────────────────────────────────────────────────────────
    device = torch.device(cfg.device)

    model = LEGRDualEncoder(
        num_tools=NUM_TOOLS,
        text_model_name=cfg.text_model,
        embed_dim=cfg.embed_dim,
        gcn_layers=cfg.gcn_layers,
        node_embed_dim=cfg.node_embed_dim,
        gcn_hidden=cfg.gcn_hidden,
        freeze_text=cfg.freeze_text,
        num_frozen_layers=cfg.num_frozen_layers,
        max_topo_pos=cfg.max_topo_pos,
        graph_encoder_type=getattr(cfg, "graph_encoder_type", "gcn"),
    ).to(device)

    criterion = GraphAwareContrastiveLoss(
        temperature_init=cfg.temperature_init,
        lambda_ged=cfg.lambda_ged,
        ged_scale=cfg.ged_scale,
        ged_margin=cfg.ged_margin,
    ).to(device)

    # ── Param groups with differential LR ─────────────────────────────────
    param_groups = model.text_encoder.get_param_groups(
        backbone_lr=cfg.text_backbone_lr,
        head_lr=cfg.lr,
    )
    param_groups.append({
        "params": list(model.graph_encoder.parameters()),
        "lr": cfg.lr,
    })
    param_groups.append({
        "params": list(criterion.parameters()),
        "lr": cfg.lr,
    })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.text_backbone_lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.warmup_epochs],
    )

    # ── Trainability diagnostics ──────────────────────────────────────────
    def _count_params(module: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, train

    txt_total, txt_train = _count_params(model.text_encoder)
    gcn_total, gcn_train = _count_params(model.graph_encoder)
    print(f"Text encoder : {txt_train:,} / {txt_total:,} params trainable")
    print(f"Graph encoder: {gcn_train:,} / {gcn_total:,} params trainable")
    print(f"LR  backbone={cfg.text_backbone_lr}  heads/GCN={cfg.lr}\n")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=vars(cfg),
    )
    wandb.watch(model, log="gradients", log_freq=50)

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training on {device}  |  "
          f"{len(train_ds)} train · {len(val_ds)} val  |  "
          f"{train_ds.num_unique_dags} unique DAGs\n")

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, criterion, optimizer, train_loader,
            ged_full, device, cfg.max_grad_norm, epoch,
            ged_global_max=ged_global_max,
        )
        scheduler.step()

        wandb.log({f"train/{k}": v for k, v in train_metrics.items()},
                  step=epoch)
        group_lrs = scheduler.get_last_lr()
        wandb.log({
            "lr/text_backbone": group_lrs[0],
            "lr/text_head": group_lrs[1] if len(group_lrs) > 1 else group_lrs[0],
            "lr/gcn": group_lrs[2] if len(group_lrs) > 2 else group_lrs[0],
        }, step=epoch)

        # Validation
        if epoch % cfg.val_every == 0:
            val_metrics = validate(
                model, criterion, val_loader, ged_full, device,
                ged_global_max=ged_global_max,
            )
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()},
                      step=epoch)

            val_loss = val_metrics["loss_total"]
            improved = val_loss < best_val_loss

            tag = " * best" if improved else ""
            print(
                f"Epoch {epoch:3d}  |  "
                f"train_loss {train_metrics['loss_total']:.4f}  |  "
                f"val_loss {val_loss:.4f}  |  "
                f"R@1 {val_metrics.get('recall_at_1', 0):.3f}  |  "
                f"tau {train_metrics['temperature']:.4f}{tag}"
            )

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    _build_checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        criterion=criterion,
                        cfg=cfg,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_loss=val_loss,
                    ),
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {cfg.patience} epochs).")
                    break

    # Save final model
    torch.save(
        _build_checkpoint_payload(
            epoch=epoch,
            model=model,
            criterion=criterion,
            cfg=cfg,
        ),
        ckpt_dir / "final_model.pt",
    )

    wandb.finish()
    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir.resolve()}")

    best_ckpt = ckpt_dir / "best_model.pt"
    final_ckpt = ckpt_dir / "final_model.pt"
    if best_ckpt.exists():
        return str(best_ckpt.resolve())
    return str(final_ckpt.resolve())


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the LEGR dual-encoder")
    add_tool_count_argument(p, default=_TOOL_COUNT_OVERRIDE)
    cfg = TrainConfig()
    for name, default in vars(cfg).items():
        if name == "tool_count":
            continue
        ty = type(default) if default is not None else str
        if ty is bool:
            p.add_argument(f"--{name}", action="store_true", default=default)
        else:
            p.add_argument(f"--{name}", type=ty, default=default)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
