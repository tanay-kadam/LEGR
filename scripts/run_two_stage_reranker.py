"""Train a two-stage SBERT-FT -> V3 structural reranker without data changes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data_synth import build_dag  # noqa: E402
from encoders import resolve_graph_encoder_settings  # noqa: E402
from eval import _load_model_and_tokenizer, encode_all_dags, encode_all_queries  # noqa: E402
from legr_experiments.data import ResearchDataset  # noqa: E402
from legr_experiments.final_evaluation import (  # noqa: E402
    dag_clustered_paired_bootstrap,
    deterministic_gallery,
    retrieval_diagnostics,
)
from legr_experiments.integrity import compare, snapshot  # noqa: E402
from legr_experiments.two_stage_reranker import (  # noqa: E402
    V3PairReranker,
    hierarchical_scores,
    same_toolset_pair_indices,
)
from sbert_ft_baseline import SBERTFineTuneDualEncoder  # noqa: E402
from src.data.tool_registry import get_tools  # noqa: E402


class QueryOnly:
    def __init__(self, samples):
        self.samples = [{"query": sample.query} for sample in samples]

    def __len__(self):
        return len(self.samples)


class DagOnly:
    def __init__(self, samples):
        self.dags = [build_dag(list(s.signature.tools), list(s.signature.edges)) for s in samples]
        self.num_unique_dags = len(self.dags)

    def get_unique_dag(self, index):
        return self.dags[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=(15, 30, 45), default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--v3-checkpoint")
    parser.add_argument("--output-root")
    return parser.parse_args()


def gold_indices(query_dataset, gallery):
    mapping = {sample.signature.dag_key: index for index, sample in enumerate(gallery.samples)}
    values, keys = [], []
    for sample in query_dataset.samples:
        key = sample.signature.dag_key
        if key not in mapping:
            raise ValueError(f"Gold DAG {key} absent from gallery")
        values.append(mapping[key])
        keys.append(key)
    return values, keys


def tools_for_queries(dataset):
    return torch.stack([sample.signature.tool_target for sample in dataset.samples])


def tools_for_gallery(gallery):
    return torch.stack([sample.signature.tool_target for sample in gallery.samples])


@torch.no_grad()
def encode_v3(model, tokenizer, config, query_dataset, gallery, device):
    _, _, bidirectional = resolve_graph_encoder_settings(config)
    query = encode_all_queries(
        model, QueryOnly(query_dataset.samples), tokenizer, device, batch_size=64
    )
    graph = encode_all_dags(
        model, DagOnly(gallery.samples), device, batch_size=64,
        bidirectional=bidirectional,
    )
    return query.cpu(), graph.cpu()


def load_sbert(device, tier):
    path = ROOT / f"artifacts/campaign_v4/results/sbert_ft_ged_{tier}t_s42/best_model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = SBERTFineTuneDualEncoder(
        embed_dim=config["embed_dim"],
        text_model_name=config["text_model"],
        num_frozen_layers=config["num_frozen_layers"],
        tied=payload.get("tied", False),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(config["text_model"], local_files_only=True)
    return model, tokenizer, path


@torch.no_grad()
def encode_sbert(model, tokenizer, query_dataset, gallery, device):
    query_values, graph_values = [], []
    for start in range(0, len(query_dataset), 64):
        queries = [s.query for s in query_dataset.samples[start : start + 64]]
        tokens = tokenizer(
            queries, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        query_values.append(model.encode_query(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        ).cpu())
    for start in range(0, len(gallery), 64):
        documents = [s.dag_text for s in gallery.samples[start : start + 64]]
        tokens = tokenizer(
            documents, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        graph_values.append(model.encode_document(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        ).cpu())
    return torch.cat(query_values), torch.cat(graph_values)


def score_model(model, query, graph, sbert_scores, candidate_tools, device):
    model.eval()
    with torch.no_grad():
        structural = model.score_matrix(query.to(device), graph.to(device)).cpu()
    hierarchical = hierarchical_scores(sbert_scores, structural, candidate_tools)
    return structural, hierarchical


def select_metrics(scores, gold, query_tools, candidate_tools):
    return retrieval_diagnostics(scores, gold, query_tools, candidate_tools, tolerance=1e-5)


def train_seed(
    seed,
    train_features,
    dev_features,
    args,
    device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = V3PairReranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tq, tg, train_gold, train_candidate_tools = train_features
    dq, dg, dev_sbert_scores, dev_gold, dev_qtools, dev_ctools = dev_features
    row_index, positive_index, negative_index = same_toolset_pair_indices(
        train_gold, train_candidate_tools
    )
    if len(row_index) == 0:
        raise ValueError("No same-tool-set training pairs")
    row_index = row_index.to(device)
    positive_index = positive_index.to(device)
    negative_index = negative_index.to(device)
    tq, tg = tq.to(device), tg.to(device)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(row_index), generator=generator).to(device)
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            rows = row_index[indices]
            positive = model(tq[rows], tg[positive_index[indices]])
            negative = model(tq[rows], tg[negative_index[indices]])
            loss = F.softplus(args.margin - positive + negative).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        structural, hierarchical = score_model(
            model, dq, dg, dev_sbert_scores, dev_ctools, device
        )
        structural_metrics, _ = select_metrics(
            structural, dev_gold, dev_qtools, dev_ctools
        )
        system_metrics, _ = select_metrics(
            hierarchical, dev_gold, dev_qtools, dev_ctools
        )
        key = (
            structural_metrics["true_twin_recall@1"],
            system_metrics["recall@1"],
            structural_metrics["hard_negative_pair_accuracy"],
        )
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "structural_true_twin_recall@1": key[0],
            "system_recall@1": key[1],
            "hard_negative_pair_accuracy": key[2],
        })
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += args.eval_every
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No reranker checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, best_key, history


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    args = parse_args()
    output_root = args.output_root or (
        f"artifacts/legr_model_search/two_stage_v3_reranker_{args.tier}t"
    )
    output = ROOT / output_root
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    before = snapshot()
    started = time.perf_counter()
    device = torch.device(args.device)
    vocabulary = list(get_tools(args.tier))
    base = ROOT / f"data/campaign_v4/campaign_v4_{args.tier}tools"
    train = ResearchDataset(base / "train.csv", vocabulary)
    dev = ResearchDataset(base / "dev.csv", vocabulary)
    test = ResearchDataset(base / "test_topology_heldout.csv", vocabulary)
    train_gallery = deterministic_gallery(train, 42)
    dev_gallery = deterministic_gallery(
        ResearchDataset([base / "dev.csv", base / "train.csv"], vocabulary), 42
    )
    test_gallery = deterministic_gallery(
        ResearchDataset(
            [base / "test_topology_heldout.csv", base / "candidate_corpus.csv"],
            vocabulary,
        ), 42
    )
    expected = {
        15: (248, 297, 322),
        30: (346, 415, 455),
        45: (498, 597, 650),
    }[args.tier]
    if (len(train_gallery), len(dev_gallery), len(test_gallery)) != expected:
        raise ValueError(
            f"Unexpected tier-{args.tier} gallery sizes: "
            f"{(len(train_gallery), len(dev_gallery), len(test_gallery))} != {expected}"
        )
    train_gold, _ = gold_indices(train, train_gallery)
    dev_gold, dev_keys = gold_indices(dev, dev_gallery)
    test_gold, test_keys = gold_indices(test, test_gallery)
    train_ctools, dev_ctools, test_ctools = map(
        tools_for_gallery, (train_gallery, dev_gallery, test_gallery)
    )
    train_qtools, dev_qtools, test_qtools = map(tools_for_queries, (train, dev, test))

    v3_checkpoint = ROOT / (
        args.v3_checkpoint
        or f"artifacts/campaign_v4/results/legr_setgnn_tied_no_ged_{args.tier}t_s42/best_model.pt"
    )
    if not v3_checkpoint.exists():
        raise FileNotFoundError(f"Missing tier-specific V3 checkpoint: {v3_checkpoint}")
    v3, v3_config, v3_tokenizer = _load_model_and_tokenizer(str(v3_checkpoint), device)
    train_v3_q, train_v3_g = encode_v3(v3, v3_tokenizer, v3_config, train, train_gallery, device)
    dev_v3_q, dev_v3_g = encode_v3(v3, v3_tokenizer, v3_config, dev, dev_gallery, device)
    test_v3_q, test_v3_g = encode_v3(v3, v3_tokenizer, v3_config, test, test_gallery, device)
    v3_test_scores = test_v3_q @ test_v3_g.t()
    v3_test_metrics, _ = select_metrics(
        v3_test_scores, test_gold, test_qtools, test_ctools
    )
    del v3
    torch.cuda.empty_cache()

    sbert, sbert_tokenizer, sbert_checkpoint = load_sbert(device, args.tier)
    train_sq, train_sg = encode_sbert(sbert, sbert_tokenizer, train, train_gallery, device)
    dev_sq, dev_sg = encode_sbert(sbert, sbert_tokenizer, dev, dev_gallery, device)
    test_sq, test_sg = encode_sbert(sbert, sbert_tokenizer, test, test_gallery, device)
    train_sbert_scores = train_sq @ train_sg.t()
    dev_sbert_scores = dev_sq @ dev_sg.t()
    test_sbert_scores = test_sq @ test_sg.t()
    del sbert
    torch.cuda.empty_cache()

    sbert_dev_metrics, _ = select_metrics(dev_sbert_scores, dev_gold, dev_qtools, dev_ctools)
    sbert_test_metrics, sbert_test_details = select_metrics(
        test_sbert_scores, test_gold, test_qtools, test_ctools
    )
    runs = []
    test_details = {}
    for seed in (42, 123, 2026):
        model, best_epoch, best_key, history = train_seed(
            seed,
            (train_v3_q, train_v3_g, train_gold, train_ctools),
            (dev_v3_q, dev_v3_g, dev_sbert_scores, dev_gold, dev_qtools, dev_ctools),
            args,
            device,
        )
        dev_structural, dev_system = score_model(
            model, dev_v3_q, dev_v3_g, dev_sbert_scores, dev_ctools, device
        )
        dev_struct_metrics, _ = select_metrics(
            dev_structural, dev_gold, dev_qtools, dev_ctools
        )
        dev_metrics, _ = select_metrics(dev_system, dev_gold, dev_qtools, dev_ctools)
        test_structural, test_system = score_model(
            model, test_v3_q, test_v3_g, test_sbert_scores, test_ctools, device
        )
        test_struct_metrics, _ = select_metrics(
            test_structural, test_gold, test_qtools, test_ctools
        )
        test_metrics, details = select_metrics(
            test_system, test_gold, test_qtools, test_ctools
        )
        torch.save({
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_key": best_key,
            "architecture": "SBERT-FT tool-set selector -> frozen V3 + two-layer residual MLP",
        }, output / f"reranker_seed{seed}.pt")
        (output / f"history_seed{seed}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output / f"test_scores_seed{seed}.npz",
            hierarchical=test_system.numpy(),
            structural=test_structural.numpy(),
            sbert=test_sbert_scores.numpy(),
            gold_indices=np.asarray(test_gold),
        )
        runs.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_key": list(best_key),
            "dev_structural": dev_struct_metrics,
            "dev_system": dev_metrics,
            "test_structural": test_struct_metrics,
            "test_system": test_metrics,
        })
        test_details[seed] = details

    exact_mean = np.mean([test_details[s]["exact_correct"] for s in (42, 123, 2026)], axis=0)
    tool_mean = np.mean([test_details[s]["tool_f1"] for s in (42, 123, 2026)], axis=0)
    twin_mean = np.mean([test_details[s]["twin_expected_correct"] for s in (42, 123, 2026)], axis=0)
    twin_mask = sbert_test_details["twin_eligible"]
    bootstrap = {
        "recall@1": dag_clustered_paired_bootstrap(
            exact_mean, sbert_test_details["exact_correct"], test_keys,
            args.bootstrap_samples, 42,
        ),
        "tool_set_f1": dag_clustered_paired_bootstrap(
            tool_mean, sbert_test_details["tool_f1"], test_keys,
            args.bootstrap_samples, 43,
        ),
        "true_twin_recall@1": dag_clustered_paired_bootstrap(
            twin_mean[twin_mask], sbert_test_details["twin_expected_correct"][twin_mask],
            np.asarray(test_keys)[twin_mask], args.bootstrap_samples, 44,
        ),
    }
    mean_metrics = {}
    for key in (
        "recall@1", "recall@3", "recall@5", "mrr@5", "tool_set_f1",
        "true_twin_recall@1", "hard_negative_pair_accuracy",
    ):
        values = np.asarray([run["test_system"][key] for run in runs])
        mean_metrics[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    summary = {
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "tier": args.tier,
        "evidence_notice": (
            "The 322-DAG gallery was inspected before this architecture was proposed; "
            "tier-15 results are exploratory, not pristine confirmatory test evidence."
            if args.tier == 15 else
            "The two-stage architecture was fixed on the 15-tool study before this tier's "
            "held-out scaling evaluation; no test-time checkpoint selection was performed."
        ),
        "architecture": {
            "stage_1": "frozen SBERT-FT selects exact tool set",
            "stage_2": "frozen V3 embeddings scored by zero-initialized two-layer residual MLP",
            "trained_parameters": sum(p.numel() for p in V3PairReranker().parameters()),
        },
        "galleries": {
            "train": len(train_gallery),
            "expanded_dev": len(dev_gallery),
            "final": len(test_gallery),
            "dev_twin_eligible_queries": int(sbert_dev_metrics["true_twin_queries"]),
        },
        "checkpoints": {
            "sbert_ft": sbert_checkpoint.relative_to(ROOT).as_posix(),
            "v3": v3_checkpoint.relative_to(ROOT).as_posix(),
        },
        "v3": {"test": v3_test_metrics},
        "sbert_ft": {"dev": sbert_dev_metrics, "test": sbert_test_metrics},
        "runs": runs,
        "test_mean": mean_metrics,
        "paired_bootstrap_vs_sbert_ft": bootstrap,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    lines = [
        f"# Two-stage SBERT-FT → V3 Reranker ({args.tier} tools)",
        "",
        summary["evidence_notice"],
        "",
        "| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| SBERT-FT | fixed | {sbert_test_metrics['recall@1']:.4f} | {sbert_test_metrics['recall@3']:.4f} | {sbert_test_metrics['recall@5']:.4f} | {sbert_test_metrics['tool_set_f1']:.4f} | {sbert_test_metrics['true_twin_recall@1']:.4f} |",
        f"| LEGR-V3 | fixed | {v3_test_metrics['recall@1']:.4f} | {v3_test_metrics['recall@3']:.4f} | {v3_test_metrics['recall@5']:.4f} | {v3_test_metrics['tool_set_f1']:.4f} | {v3_test_metrics['true_twin_recall@1']:.4f} |",
    ]
    for run in runs:
        metric = run["test_system"]
        lines.append(
            f"| Two-stage | {run['seed']} | {metric['recall@1']:.4f} | {metric['recall@3']:.4f} | {metric['recall@5']:.4f} | {metric['tool_set_f1']:.4f} | {metric['true_twin_recall@1']:.4f} |"
        )
    lines.extend(["", "## Paired bootstrap versus SBERT-FT", ""])
    for name, result in bootstrap.items():
        lines.append(
            f"- {name}: {result['delta']:+.4f}, 95% CI [{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}]"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = snapshot()
    drift = compare(before, after)
    (output / "integrity_report.json").write_text(
        json.dumps(drift, indent=2), encoding="utf-8"
    )
    if any(drift.values()):
        raise RuntimeError(f"Immutable inputs changed: {drift}")
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
