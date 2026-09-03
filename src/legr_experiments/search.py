from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig

from .config import ExperimentConfig
from .training import run_experiment


BACKBONES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
)


def backbone_available(name: str) -> bool:
    try:
        AutoConfig.from_pretrained(name, local_files_only=True)
        return True
    except Exception:
        return False


def mathematical_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    configs = []
    variants = [
        {"use_multi_positive": False, "use_twin": False, "use_tool": False, "use_relation": False, "use_distance": False},
        {"use_multi_positive": True, "use_twin": False, "use_tool": False, "use_relation": False, "use_distance": False},
        {"use_multi_positive": True, "use_twin": True, "use_tool": False, "use_relation": False, "use_distance": False},
        {"use_multi_positive": True, "use_twin": True, "use_tool": True, "use_relation": False, "use_distance": False},
        {"use_multi_positive": True, "use_twin": True, "use_tool": True, "use_relation": True, "use_distance": False},
        {"use_multi_positive": True, "use_twin": True, "use_tool": True, "use_relation": True, "use_distance": True},
    ]
    for index, values in enumerate(variants):
        cfg = deepcopy(base)
        cfg.name = f"math_{index}"
        cfg.model.graph_kind = "v3"
        cfg.model.use_reranker = False
        for key, value in values.items():
            setattr(cfg.loss, key, value)
        configs.append(cfg)
    for margin in (0.1, 0.2, 0.4):
        cfg = deepcopy(base)
        cfg.name = f"math_pair_m{str(margin).replace('.', '')}"
        cfg.model.graph_kind = "v3"
        cfg.model.use_reranker = False
        cfg.loss.rank_kind = "pairwise"
        cfg.loss.margin = margin
        configs.append(cfg)
    for scale in (0.5, 2.0):
        cfg = deepcopy(base)
        cfg.name = f"math_weights_{str(scale).replace('.', '')}"
        cfg.model.graph_kind = "v3"
        cfg.model.use_reranker = False
        cfg.loss.twin_weight *= scale
        cfg.loss.tool_weight *= scale
        cfg.loss.relation_weight *= scale
        configs.append(cfg)
    random_cfg = deepcopy(base)
    random_cfg.name = "math_random_batches"
    random_cfg.model.graph_kind = "v3"
    random_cfg.model.use_reranker = False
    random_cfg.train.group_aware = False
    configs.append(random_cfg)
    return configs


def architecture_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    configs = []
    for graph_kind in ("v3", "residual", "gated", "pna", "graphormer", "gps"):
        for structure_kind in ("none", "combined"):
            cfg = deepcopy(base)
            cfg.name = f"arch_{graph_kind}_{structure_kind}"
            cfg.model.graph_kind = graph_kind
            cfg.model.structure_kind = structure_kind
            cfg.model.use_reranker = False
            configs.append(cfg)
    for structure_kind in ("depth", "degree", "path"):
        cfg = deepcopy(base)
        cfg.name = f"arch_gps_{structure_kind}"
        cfg.model.graph_kind = "gps"
        cfg.model.structure_kind = structure_kind
        cfg.model.use_reranker = False
        configs.append(cfg)
    for readout in ("v3", "dual_attention", "virtual", "set2set", "concat"):
        cfg = deepcopy(base)
        cfg.name = f"readout_{readout}"
        cfg.model.readout_kind = readout
        cfg.model.use_reranker = False
        configs.append(cfg)
    return configs


def backbone_fusion_configs(base: ExperimentConfig) -> tuple[list[ExperimentConfig], list[dict]]:
    configs, skipped = [], []
    for backbone in BACKBONES:
        if not backbone_available(backbone):
            skipped.append({"backbone": backbone, "reason": "not available in local model cache"})
            continue
        for fusion in ("graph", "semantic", "fixed", "scalar", "gated"):
            cfg = deepcopy(base)
            short = backbone.split("/")[-1].replace("-", "_")
            cfg.name = f"fusion_{short}_{fusion}"
            cfg.model.text_model = backbone
            cfg.model.fusion_kind = fusion
            cfg.model.use_reranker = False
            configs.append(cfg)
    return configs, skipped


def reranker_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    configs = []
    for top_k in (10, 20, 40):
        for layers in (1, 2):
            cfg = deepcopy(base)
            cfg.name = f"rerank_k{top_k}_l{layers}"
            cfg.model.use_reranker = True
            cfg.model.rerank_k = top_k
            cfg.model.rerank_layers = layers
            configs.append(cfg)
    for unfreeze in ("frozen", "last2", "full"):
        cfg = deepcopy(base)
        cfg.name = f"optim_unfreeze_{unfreeze}"
        cfg.model.unfreeze = unfreeze
        cfg.model.use_reranker = False
        configs.append(cfg)
    for schedule in ("cosine", "plateau"):
        cfg = deepcopy(base)
        cfg.name = f"optim_schedule_{schedule}"
        cfg.train.schedule = schedule
        cfg.model.use_reranker = False
        configs.append(cfg)
    for option in ("ema", "swa", "hard_negatives", "curriculum"):
        cfg = deepcopy(base)
        cfg.name = f"optim_{option}"
        cfg.model.use_reranker = False
        if option == "ema":
            cfg.train.use_ema = True
        elif option == "swa":
            cfg.train.use_swa = True
        elif option == "hard_negatives":
            cfg.train.online_hard_negatives = True
        else:
            cfg.train.curriculum = True
        configs.append(cfg)
    return configs


class SearchController:
    def __init__(self, output_root: str | Path, vocabulary: list[str], budget_hours: float = 24):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.vocabulary = vocabulary
        self.deadline = time.time() + budget_hours * 3600
        self.records: list[dict] = []
        self.baseline_tool_f1: float | None = None

    def remaining_seconds(self) -> float:
        return self.deadline - time.time()

    def run_stage(self, stage: str, configs: list[ExperimentConfig], max_runs: int | None = None) -> ExperimentConfig:
        completed = []
        for index, config in enumerate(configs):
            if max_runs is not None and index >= max_runs:
                break
            if self.remaining_seconds() <= 300:
                break
            record = {"stage": stage, "run_name": config.run_name(), "config": config.as_dict()}
            try:
                summary = run_experiment(config, self.output_root, self.vocabulary)
                record.update(summary)
                completed.append((config, summary))
                if stage == "baseline":
                    self.baseline_tool_f1 = summary["dev_metrics"]["tool_set_f1"]
            except Exception as error:
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                torch.cuda.empty_cache()
            self.records.append(record)
            (self.output_root / "search_manifest.json").write_text(
                json.dumps(self.records, indent=2), encoding="utf-8",
            )
            torch.cuda.empty_cache()
        valid = [
            item for item in completed
            if item[1]["dev_metrics"]["p95_latency_ms"] <= 100
            and (
                self.baseline_tool_f1 is None
                or item[1]["dev_metrics"]["tool_set_f1"] >= self.baseline_tool_f1 - 0.002
            )
        ] or completed
        if not valid:
            raise RuntimeError(f"No successful configurations in stage {stage}")
        valid.sort(key=lambda item: (
            item[1]["dev_metrics"]["recall@1"],
            item[1]["dev_metrics"]["same_toolset_recall@1"],
            item[1]["dev_metrics"]["tool_set_f1"],
        ), reverse=True)
        return deepcopy(valid[0][0])

    def top_configs(self, limit: int) -> list[ExperimentConfig]:
        successful = [record for record in self.records if record.get("status") == "complete"]
        successful.sort(key=lambda record: (
            record["dev_metrics"]["recall@1"],
            record["dev_metrics"]["same_toolset_recall@1"],
            record["dev_metrics"]["tool_set_f1"],
        ), reverse=True)
        output = []
        for record in successful:
            raw = record["config"]
            config = ExperimentConfig(name=raw["name"], tier=raw["tier"])
            for key, value in raw["model"].items():
                setattr(config.model, key, value)
            for key, value in raw["loss"].items():
                setattr(config.loss, key, value)
            for key, value in raw["train"].items():
                setattr(config.train, key, value)
            output.append(config)
            if len(output) == limit:
                break
        return output
