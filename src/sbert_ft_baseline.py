"""
sbert_ft_baseline.py — Fine-tuned Sentence-BERT 2×2 (no GNN).

Two-tower bi-encoder over (query, dag_to_text(G)) pairs. Reuses LEGR's
TrainConfig recipe, GraphAwareContrastiveLoss, CSV splits, and eval metrics.

Cells
-----
1. untied text towers, lambda_ged=0
2. untied text towers, lambda_ged=0.30 (LEGR default GED knobs)
Tied-weights variant: shared TextEncoder for both towers (``--tied``).

Cells 3–4 (LEGR GNN) are not trained here.

Usage (from repo root)::

    python src/sbert_ft_baseline.py --tool_count 30 --lambda_ged 0 \\
        --checkpoint_dir artifacts/sbert_finetuned/checkpoints/cell1_untied_noged

    python src/sbert_ft_baseline.py --tool_count 30 --tied \\
        --checkpoint_dir artifacts/sbert_finetuned/checkpoints/tied_infonce
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from legr_tool_count import (
    add_tool_count_argument,
    bootstrap_tool_count_from_argv,
    get_active_tool_count,
)

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from data_synth import NUM_TOOLS, dag_to_text  # noqa: E402
from encoders import TextEncoder, get_tokenizer  # noqa: E402
from loss import GraphAwareContrastiveLoss, compute_alignment_metrics  # noqa: E402
from train import (  # noqa: E402
    TrainConfig,
    _build_checkpoint_payload,
    _build_csv_train_val_datasets,
    _ged_submatrix,
    _resolve_cfg_tool_count,
    _resolve_train_val_csv_paths,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# ═══════════════════════════════════════════════════════════════════════════
#  Dual text encoder
# ═══════════════════════════════════════════════════════════════════════════

class SBERTFineTuneDualEncoder(nn.Module):
    """Query tower + document tower (optional weight tying)."""

    def __init__(
        self,
        embed_dim: int = 256,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_text_backbone: bool = False,
        num_frozen_layers: int = 0,
        tied: bool = False,
        encoder_cls=TextEncoder,
    ):
        super().__init__()
        self.tied = bool(tied)
        self.embed_dim = embed_dim
        self.query_encoder = encoder_cls(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            num_frozen_layers=num_frozen_layers,
        )
        if self.tied:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = encoder_cls(
                model_name=text_model_name,
                embed_dim=embed_dim,
                freeze_backbone=freeze_text_backbone,
                num_frozen_layers=num_frozen_layers,
            )

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.query_encoder(input_ids, attention_mask)
        return F.normalize(z, p=2, dim=-1)

    def encode_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.doc_encoder(input_ids, attention_mask)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ):
        return (
            self.encode_query(input_ids, attention_mask),
            self.encode_document(doc_input_ids, doc_attention_mask),
        )

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        groups = self.query_encoder.get_param_groups(backbone_lr, head_lr)
        if self.tied:
            return groups
        groups.extend(self.doc_encoder.get_param_groups(backbone_lr, head_lr))
        return groups


def make_sbert_collate_fn(tokenizer: AutoTokenizer, dag_texts: list[str], max_length: int = 128):
    """Tokenise queries and flattened DAG strings; keep dag_ids for GED."""

    def collate(batch):
        queries = [b["query"] for b in batch]
        dag_ids = torch.tensor([b["dag_id"] for b in batch], dtype=torch.long)
        docs = [dag_texts[int(i)] for i in dag_ids.tolist()]
        q_enc = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        d_enc = tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": q_enc["input_ids"],
            "attention_mask": q_enc["attention_mask"],
            "doc_input_ids": d_enc["input_ids"],
            "doc_attention_mask": d_enc["attention_mask"],
            "dag_ids": dag_ids,
        }

    return collate


def unique_dag_texts(dataset) -> list[str]:
    """Canonical ``dag_to_text`` for every unique DAG (matches frozen SBERT)."""
    return [dag_to_text(dataset.get_unique_dag(i)) for i in range(dataset.num_unique_dags)]


def load_config_from_legr_checkpoint(
    checkpoint_path: Optional[str],
    **overrides: Any,
) -> TrainConfig:
    """Copy TrainConfig fields from a LEGR checkpoint; apply explicit overrides."""
    cfg = TrainConfig()
    source = "TrainConfig defaults"
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config_dict = ckpt.get("config", {})
        fields = TrainConfig.__dataclass_fields__
        filtered = {k: v for k, v in config_dict.items() if k in fields}
        cfg = TrainConfig(**{**{f: getattr(cfg, f) for f in fields}, **filtered})
        source = checkpoint_path
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg._config_source = source  # type: ignore[attr-defined]
    return cfg


def default_split_paths(tool_count: int) -> tuple[Path, Path, Path]:
    root = Path("upgraded") / f"upgraded_{tool_count}tools"
    return (
        root / "train.csv",
        root / "dev.csv",
        root / "test_topology_heldout.csv",
    )


def default_hard_negative_path(tool_count: int) -> Path:
    return Path("upgraded_data") / f"graph_{tool_count}tools" / "hard_negatives.csv"


# ═══════════════════════════════════════════════════════════════════════════
#  Train / val steps (text–text; same loss as LEGR)
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch_sbert(
    model: SBERTFineTuneDualEncoder,
    criterion: GraphAwareContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    ged_full: torch.Tensor,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    ged_global_max: float | None = None,
) -> Dict[str, float]:
    model.train()
    criterion.train()
    accum: Dict[str, float] = {}
    n_steps = 0
    pbar = tqdm(loader, desc=f"SBERT-FT epoch {epoch}", leave=False)
    for batch in pbar:
        z_q = model.encode_query(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        z_d = model.encode_document(
            batch["doc_input_ids"].to(device),
            batch["doc_attention_mask"].to(device),
        )
        dag_ids = batch["dag_ids"].to(device)
        ged_sub = _ged_submatrix(batch["dag_ids"], ged_full, device)
        loss, metrics = criterion(
            z_q, z_d, ged_sub, dag_ids=dag_ids, ged_max=ged_global_max,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()),
            max_grad_norm,
        )
        optimizer.step()
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_steps += 1
        pbar.set_postfix(loss=f"{metrics['loss_total']:.4f}")
    return {k: v / n_steps for k, v in accum.items()}


@torch.no_grad()
def validate_sbert(
    model: SBERTFineTuneDualEncoder,
    criterion: GraphAwareContrastiveLoss,
    loader: DataLoader,
    ged_full: torch.Tensor,
    device: torch.device,
    ged_global_max: float | None = None,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()
    accum: Dict[str, float] = {}
    n_steps = 0
    for batch in loader:
        z_q = model.encode_query(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        z_d = model.encode_document(
            batch["doc_input_ids"].to(device),
            batch["doc_attention_mask"].to(device),
        )
        dag_ids = batch["dag_ids"].to(device)
        ged_sub = _ged_submatrix(batch["dag_ids"], ged_full, device)
        _, metrics = criterion(
            z_q, z_d, ged_sub, dag_ids=dag_ids, ged_max=ged_global_max,
        )
        metrics.update(compute_alignment_metrics(z_q, z_d))
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        n_steps += 1
    return {k: v / max(n_steps, 1) for k, v in accum.items()}


@torch.no_grad()
def encode_all_queries_sbert(model, dataset, tokenizer, device, batch_size: int = 64):
    queries = [
        dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].query
        for i in range(len(dataset))
    ]
    out = []
    for i in range(0, len(queries), batch_size):
        enc = tokenizer(
            queries[i:i + batch_size],
            padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        out.append(model.encode_query(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_all_docs_sbert(model, dag_texts: list[str], tokenizer, device, batch_size: int = 64):
    out = []
    for i in range(0, len(dag_texts), batch_size):
        enc = tokenizer(
            dag_texts[i:i + batch_size],
            padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        out.append(
            model.encode_document(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu()
        )
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate_sbert_ft(
    model: SBERTFineTuneDualEncoder,
    dataset,
    tokenizer,
    device: torch.device,
) -> Dict[str, float]:
    from eval import compute_metrics

    q_embs = encode_all_queries_sbert(model, dataset, tokenizer, device)
    texts = unique_dag_texts(dataset) if hasattr(dataset, "get_unique_dag") else [
        dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)
    ]
    d_embs = encode_all_docs_sbert(model, texts, tokenizer, device)
    sim = torch.mm(q_embs, d_embs.t())
    topk = sim.topk(k=min(5, d_embs.size(0)), dim=1).indices
    gt = torch.tensor([
        dataset.samples[i]["dag_id"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].dag_id
        for i in range(len(dataset))
    ])
    return compute_metrics(topk, gt, dataset)


@torch.no_grad()
def evaluate_hard_negatives_sbert(
    model: SBERTFineTuneDualEncoder,
    hard_neg_df,
    tokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """Same 0.5 cosine threshold as ``eval.evaluate_hard_negatives``, text docs."""
    from data_synth import build_dag
    from train import _parse_edges, _parse_tools

    correct = 0
    total = 0
    false_positives = 0
    for _, row in hard_neg_df.iterrows():
        query = row.get("query", "")
        if not query:
            continue
        neg_tools = _parse_tools(row.get("neg_tools", ""))
        neg_edges = _parse_edges(row.get("neg_edges", ""))
        if not neg_tools:
            continue
        try:
            neg_G = build_dag(neg_tools, neg_edges)
        except Exception:
            continue
        enc_q = tokenizer(
            [str(query)], padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        enc_d = tokenizer(
            [dag_to_text(neg_G)], padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        z_q = model.encode_query(enc_q["input_ids"].to(device), enc_q["attention_mask"].to(device))
        z_d = model.encode_document(enc_d["input_ids"].to(device), enc_d["attention_mask"].to(device))
        sim = F.cosine_similarity(z_q, z_d).item()
        if sim < 0.5:
            correct += 1
        else:
            false_positives += 1
        total += 1
    return {
        "hardneg_pairs_evaluated": total,
        "hardneg_ranking_accuracy": round(correct / max(total, 1), 4),
        "hardneg_false_positive_rate": round(false_positives / max(total, 1), 4),
    }


def _count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


def write_run_metadata(out_dir: Path, cfg: TrainConfig, extra: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(cfg),
        "unavoidable_differences": [
            "Document tower is TextEncoder over dag_to_text, not a GNN.",
            "Collate tokenises DAG strings instead of batching PyG graphs.",
            "Param groups are two text towers (or one if --tied), not GCN.",
            "Hard-negative scoring uses the text document tower with the same 0.5 cosine threshold.",
        ],
        **extra,
    }
    (out_dir / "resolved_config.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )


def main(cfg: TrainConfig, *, tied: bool = False, eval_after: bool = True) -> str:
    torch.manual_seed(cfg.seed)
    _resolve_cfg_tool_count(cfg)
    tool_count = cfg.tool_count if cfg.tool_count is not None else get_active_tool_count()

    train_csv, val_csv, test_csv = default_split_paths(int(tool_count))
    if not cfg.train_csv:
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(f"Missing default splits under {train_csv.parent}")
        cfg.train_csv = str(train_csv)
        cfg.val_csv = str(val_csv)
    train_csv_s, val_csv_s = _resolve_train_val_csv_paths(cfg)

    print(f"  SBERT-FT tied={tied} lambda_ged={cfg.lambda_ged}")
    print(f"  train={train_csv_s}\n  val={val_csv_s}")
    train_ds, val_ds = _build_csv_train_val_datasets(train_csv_s, val_csv_s)
    dag_texts = unique_dag_texts(train_ds)
    ged_full = train_ds.get_ged_tensor()
    ged_global_max = float(ged_full.max())

    tokenizer = get_tokenizer(cfg.text_model)
    collate = make_sbert_collate_fn(tokenizer, dag_texts, cfg.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA: device=cuda but torch.cuda.is_available() is False")
    model = SBERTFineTuneDualEncoder(
        embed_dim=cfg.embed_dim,
        text_model_name=cfg.text_model,
        freeze_text_backbone=cfg.freeze_text,
        num_frozen_layers=cfg.num_frozen_layers,
        tied=tied,
    ).to(device)
    if device.type == "cuda" and next(model.parameters()).device.type != "cuda":
        raise RuntimeError(f"SBERT-FT parameters are not on CUDA: {next(model.parameters()).device}")
    print(
        f"  device={device}  cuda={torch.cuda.is_available()}  "
        f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'n/a'}"
    )
    criterion = GraphAwareContrastiveLoss(
        temperature_init=cfg.temperature_init,
        lambda_ged=cfg.lambda_ged,
        ged_scale=cfg.ged_scale,
        ged_margin=cfg.ged_margin,
    ).to(device)

    param_groups = model.get_param_groups(cfg.text_backbone_lr, cfg.lr)
    param_groups.append({"params": list(criterion.parameters()), "lr": cfg.lr})
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
        eta_min=cfg.text_backbone_lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.warmup_epochs],
    )

    q_tot, q_tr = _count_params(model.query_encoder)
    print(f"Query encoder : {q_tr:,} / {q_tot:,} trainable")
    if not tied:
        d_tot, d_tr = _count_params(model.doc_encoder)
        print(f"Doc encoder   : {d_tr:,} / {d_tot:,} trainable")
    else:
        print("Doc encoder   : tied to query encoder")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(ckpt_dir, cfg, {
        "tied": tied,
        "num_tools": NUM_TOOLS,
        "query_params_trainable": q_tr,
        "query_params_total": q_tot,
        "drop_last": True,
    })

    if wandb is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or f"sbert-ft-tied{tied}-ged{cfg.lambda_ged}",
            config={**vars(cfg), "tied": tied},
        )

    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch_sbert(
            model, criterion, optimizer, train_loader,
            ged_full, device, cfg.max_grad_norm, epoch, ged_global_max,
        )
        scheduler.step()
        if wandb is not None:
            wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
        if epoch % cfg.val_every == 0:
            val_metrics = validate_sbert(
                model, criterion, val_loader, ged_full, device, ged_global_max,
            )
            if wandb is not None:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            val_loss = val_metrics["loss_total"]
            improved = val_loss < best_val_loss
            print(
                f"Epoch {epoch:3d}  |  train {train_metrics['loss_total']:.4f}  |  "
                f"val {val_loss:.4f}  |  R@1 {val_metrics.get('recall_at_1', 0):.3f}"
                f"{' * best' if improved else ''}"
            )
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    _build_checkpoint_payload(
                        epoch=epoch, model=model, criterion=criterion, cfg=cfg,
                        optimizer=optimizer, scheduler=scheduler, val_loss=val_loss,
                    ) | {"tied": tied, "arch": "sbert_ft"},
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    torch.save(
        _build_checkpoint_payload(epoch=epoch, model=model, criterion=criterion, cfg=cfg)
        | {"tied": tied, "arch": "sbert_ft"},
        ckpt_dir / "final_model.pt",
    )
    if wandb is not None:
        wandb.finish()

    best = ckpt_dir / "best_model.pt"
    if eval_after and best.exists():
        ckpt_best = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_best["model_state"])
        print(f"  Reloaded best checkpoint for eval: {best} (epoch={ckpt_best.get('epoch')})")

    if eval_after:
        eval_path = Path(cfg.train_csv).with_name("test_topology_heldout.csv")
        if not eval_path.exists():
            eval_path = test_csv
        if eval_path.exists():
            from eval import CSVEvalDataset
            from utils import read_datafile
            ds = CSVEvalDataset(read_datafile(str(eval_path)))
            metrics = evaluate_sbert_ft(model, ds, tokenizer, device)
            (ckpt_dir / "eval_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8",
            )
            print(f"Eval {eval_path}: {metrics}")
            hn = default_hard_negative_path(int(tool_count))
            if hn.exists():
                import pandas as pd
                hn_m = evaluate_hard_negatives_sbert(
                    model, pd.read_csv(hn), tokenizer, device,
                )
                metrics.update(hn_m)
                (ckpt_dir / "eval_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8",
                )

    return str(best if best.exists() else ckpt_dir / "final_model.pt")


def parse_args() -> tuple[TrainConfig, dict]:
    p = argparse.ArgumentParser(description="Fine-tune Sentence-BERT on (query, DAG-string) pairs")
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
    p.add_argument("--tied", action="store_true", help="Share one TextEncoder for both towers")
    p.add_argument(
        "--legr_checkpoint", default=None,
        help="Optional LEGR best_model.pt whose config is copied",
    )
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()
    extras = {
        "tied": args.tied,
        "legr_checkpoint": args.legr_checkpoint,
        "skip_eval": args.skip_eval,
    }
    raw = vars(args)
    for k in ("tied", "legr_checkpoint", "skip_eval"):
        raw.pop(k, None)
    overrides = {k: v for k, v in raw.items()}
    cfg = load_config_from_legr_checkpoint(extras["legr_checkpoint"], **overrides)
    return cfg, extras


if __name__ == "__main__":
    cfg, extras = parse_args()
    main(cfg, tied=bool(extras["tied"]), eval_after=not extras["skip_eval"])
