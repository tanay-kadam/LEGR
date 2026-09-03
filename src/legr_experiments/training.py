from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ExperimentConfig
from .data import ResearchDataset, campaign_paths, make_collate
from .evaluation import evaluate_gallery
from .losses import CompositeRetrievalLoss
from .model import LEGRResearchModel
from .samplers import GroupAwareBatchSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExponentialMovingAverage:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.saved = {}

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self.saved = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.saved[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, parameter in model.named_parameters():
            if name in self.saved:
                parameter.copy_(self.saved[name])
        self.saved = {}


def checkpoint_paths(tier: int, seed: int) -> tuple[Path | None, Path | None]:
    root = Path("artifacts/campaign_v4/results")
    v3 = root / f"legr_setgnn_tied_no_ged_{tier}t_s{seed}" / "best_model.pt"
    sbert = root / f"sbert_ft_ged_{tier}t_s{seed}" / "best_model.pt"
    fallback_v3 = root / f"legr_setgnn_tied_no_ged_{tier}t_s42" / "best_model.pt"
    fallback_sbert = root / f"sbert_ft_ged_{tier}t_s42" / "best_model.pt"
    return (
        v3 if v3.exists() else (fallback_v3 if fallback_v3.exists() else None),
        sbert if sbert.exists() else (fallback_sbert if fallback_sbert.exists() else None),
    )


def _loader(dataset, tokenizer, cfg, training: bool):
    collate = make_collate(tokenizer, cfg.max_length)
    if training and cfg.group_aware:
        sampler = GroupAwareBatchSampler(dataset.samples, cfg.batch_size, cfg.seed, drop_last=False)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.num_workers)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=training,
        collate_fn=collate, num_workers=cfg.num_workers, drop_last=False,
    )


def _to_device_targets(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_experiment(
    config: ExperimentConfig,
    output_root: str | Path,
    vocabulary: list[str],
) -> dict:
    seed_everything(config.train.seed)
    output_dir = Path(output_root) / config.run_name()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8",
    )
    paths = campaign_paths(config.tier)
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model, local_files_only=True)
    train_data = ResearchDataset(paths["train"], vocabulary, config.model.structure_kind)
    dev_data = ResearchDataset(paths["dev"], vocabulary, config.model.structure_kind)
    dev_gallery = ResearchDataset([paths["dev"]], vocabulary, config.model.structure_kind)
    v3_checkpoint, sbert_checkpoint = checkpoint_paths(config.tier, config.train.seed)
    model = LEGRResearchModel(
        config.model, vocabulary, v3_checkpoint=v3_checkpoint,
        sbert_checkpoint=sbert_checkpoint,
    )
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    model.initialize_tool_queries(device)
    criterion = CompositeRetrievalLoss(config.loss).to(device)
    train_loader = _loader(train_data, tokenizer, config.train, True)
    backbone_parameters = [
        parameter for parameter in model.base_legr.text_encoder.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.train.backbone_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.train.lr})
    optimizer = AdamW(groups, weight_decay=config.train.weight_decay)
    if config.train.schedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.train.epochs))

    history = []
    best_metric = float("-inf")
    best_state = None
    stale = 0
    started = time.time()
    ema = ExponentialMovingAverage(model) if config.train.use_ema else None
    swa = AveragedModel(model) if config.train.use_swa else None
    original_twin_weight = criterion.config.twin_weight
    for epoch in range(config.train.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        if config.train.curriculum:
            progress = min(1.0, 2 * (epoch + 1) / max(1, config.train.epochs))
            criterion.config.twin_weight = original_twin_weight * progress
        totals = {}
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss_output = criterion(output, batch)
            total_loss = loss_output.total
            if config.train.online_hard_negatives:
                scores = output["scores"]
                dag_ids = batch["dag_ids"].to(scores.device)
                positive = torch.stack([
                    scores[row][dag_ids == dag_ids[row]].mean()
                    for row in range(scores.size(0))
                ])
                negative_scores = scores.masked_fill(
                    dag_ids[:, None].eq(dag_ids[None, :]), float("-inf"),
                )
                hardest = negative_scores.max(dim=1).values
                hard_loss = torch.nn.functional.softplus(
                    config.loss.margin - positive + hardest,
                ).mean()
                total_loss = total_loss + 0.25 * hard_loss
                loss_output.parts["online_hard"] = hard_loss
                loss_output.parts["total"] = total_loss
            total_loss.backward()
            clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            for name, value in loss_output.detached().items():
                totals[name] = totals.get(name, 0.0) + value
            steps += 1

        if swa is not None and epoch >= max(0, 2 * config.train.epochs // 3):
            swa.update_parameters(model)
        if ema is not None:
            ema.apply(model)
        dev_metrics, _ = evaluate_gallery(
            model, dev_data, dev_gallery, tokenizer,
            config.train.batch_size, device, seed=config.train.seed,
        )
        if ema is not None:
            ema.restore(model)
        record = {
            "epoch": epoch + 1,
            "train": {name: value / max(1, steps) for name, value in totals.items()},
            "dev": dev_metrics,
        }
        history.append(record)
        score = dev_metrics["recall@1"]
        if score > best_metric:
            best_metric = score
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if config.train.schedule == "plateau":
            scheduler.step(score)
        else:
            scheduler.step()
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= config.train.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if swa is not None and swa.n_averaged.item() > 0:
        model.load_state_dict(swa.module.state_dict())
    torch.save({
        "model_state": model.state_dict(), "config": config.as_dict(),
        "load_reports": model.load_reports, "best_dev_recall@1": best_metric,
    }, output_dir / "best_model.pt")
    final_metrics, _ = evaluate_gallery(
        model, dev_data, dev_gallery, tokenizer,
        config.train.batch_size, device, seed=config.train.seed,
    )
    summary = {
        "run_name": config.run_name(),
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "epochs_completed": len(history),
        "best_dev_recall@1": best_metric,
        "dev_metrics": final_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "load_reports": model.load_reports,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
