"""Fine-tune and evaluate one standard SentenceTransformer per tool tier.

This is intentionally separate from ``src/sbert_ft_baseline.py``.  It uses a
single ``SentenceTransformer`` object for both queries and DAG text, adds no
projection head, and does not load or modify any LEGR/dual-encoder checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        required = {"query", "dag_text", "dag_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
        clean = frame.dropna(subset=["query", "dag_text", "dag_id"])
        self.queries = clean["query"].astype(str).tolist()
        self.documents = clean["dag_text"].astype(str).tolist()
        self.labels = clean["dag_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.queries[index], self.documents[index]


class NoDuplicateDagBatchSampler(Sampler[list[int]]):
    """Balanced batches with at most one query for each positive DAG.

    Multiple-negatives ranking loss treats other documents in the batch as
    negatives. Keeping DAG IDs unique prevents a true positive from becoming
    a false in-batch negative while still using every training row each epoch.
    """

    def __init__(self, labels: list[str], batch_size: int, seed: int):
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        counts: dict[str, int] = defaultdict(int)
        for label in labels:
            counts[label] += 1
        self.num_batches = max(max(counts.values()), math.ceil(len(labels) / batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            grouped[label].append(index)
        for indices in grouped.values():
            rng.shuffle(indices)

        batches: list[list[int]] = [[] for _ in range(self.num_batches)]
        groups = list(grouped.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for indices in groups:
            available = sorted(range(self.num_batches), key=lambda i: (len(batches[i]), rng.random()))
            for index, batch_index in zip(indices, available):
                batches[batch_index].append(index)

        batches = [batch for batch in batches if batch]
        for batch in batches:
            if len(batch) > self.batch_size:
                raise RuntimeError("No-duplicate batch construction exceeded batch_size")
            rng.shuffle(batch)
        rng.shuffle(batches)
        yield from batches


def collate_pairs(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    queries, documents = zip(*batch)
    return list(queries), list(documents)


def move_features(features: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict[str, str]:
    result = {}
    for name in ("model.safetensors", "config_sentence_transformers.json", "modules.json"):
        path = model_dir / name
        if path.exists():
            result[name] = sha256_file(path)
    return result


def load_eval_helpers(tool_count: int):
    # LEGR's dataset registry is selected while eval.py is imported. Set the
    # explicit override before that import, without loading any LEGR model.
    from legr_tool_count import apply_tool_count_override

    apply_tool_count_override(tool_count)
    from eval import CSVEvalDataset, compute_metrics

    return CSVEvalDataset, compute_metrics


@torch.no_grad()
def evaluate(
    model: SentenceTransformer,
    csv_path: Path,
    tool_count: int,
    batch_size: int,
) -> tuple[dict[str, float], int, int]:
    CSVEvalDataset, compute_metrics = load_eval_helpers(tool_count)
    dataset = CSVEvalDataset(pd.read_csv(csv_path))
    queries = [sample["query"] for sample in dataset.samples]
    documents = [dataset.get_dag_text(i) for i in range(dataset.num_unique_dags)]
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()
    document_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()
    scores = query_embeddings @ document_embeddings.T
    topk = scores.topk(k=min(5, dataset.num_unique_dags), dim=1).indices
    ground_truth = torch.tensor([sample["dag_id"] for sample in dataset.samples])
    metrics = compute_metrics(topk, ground_truth, dataset)
    return metrics, len(dataset), dataset.num_unique_dags


@torch.no_grad()
def evaluate_id_retrieval(
    model: SentenceTransformer,
    query_frame: pd.DataFrame,
    gallery_frame: pd.DataFrame,
    batch_size: int,
    structural_metrics: bool = False,
) -> dict[str, float]:
    """Evaluate ID-based retrieval against a deduplicated DAG gallery."""
    gallery = gallery_frame.drop_duplicates("dag_id").reset_index(drop=True)
    id_to_index = {str(dag_id): index for index, dag_id in enumerate(gallery["dag_id"])}
    missing = sorted(
        {str(dag_id) for dag_id in query_frame["dag_id"]} - set(id_to_index)
    )
    if missing:
        raise ValueError(f"Gallery is missing {len(missing)} gold DAG IDs")

    query_embeddings = model.encode(
        query_frame["query"].astype(str).tolist(),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()
    document_embeddings = model.encode(
        gallery["dag_text"].astype(str).tolist(),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()
    topk = (query_embeddings @ document_embeddings.T).topk(
        k=min(5, len(gallery)), dim=1
    ).indices.tolist()
    ground_truth = [id_to_index[str(dag_id)] for dag_id in query_frame["dag_id"]]
    result: dict[str, float] = {}
    for k in (1, 3, 5):
        hits = 0
        reciprocal_rank = 0.0
        for gold, ranking in zip(ground_truth, topk):
            candidates = ranking[:k]
            if gold in candidates:
                hits += 1
                reciprocal_rank += 1.0 / (candidates.index(gold) + 1)
        result[f"recall@{k}"] = hits / len(ground_truth)
        result[f"mrr@{k}"] = reciprocal_rank / len(ground_truth)
    if structural_metrics:
        from data_synth import build_dag, compute_ged
        from train import _parse_edges, _parse_tools

        tool_f1_values = []
        ged_values = []
        graph_cache = {}
        ged_cache = {}

        def graph_at(index: int):
            if index not in graph_cache:
                row = gallery.iloc[index]
                graph_cache[index] = build_dag(
                    _parse_tools(row["tools"]), _parse_edges(row["edges"])
                )
            return graph_cache[index]

        for gold, ranking in zip(ground_truth, topk):
            predicted = ranking[0]
            gold_tools = set(_parse_tools(gallery.iloc[gold]["tools"]))
            predicted_tools = set(_parse_tools(gallery.iloc[predicted]["tools"]))
            overlap = len(gold_tools & predicted_tools)
            precision = overlap / len(predicted_tools) if predicted_tools else 0.0
            recall = overlap / len(gold_tools) if gold_tools else 0.0
            tool_f1_values.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            pair = (min(gold, predicted), max(gold, predicted))
            if pair not in ged_cache:
                ged_cache[pair] = (
                    0.0
                    if gold == predicted
                    else float(compute_ged(graph_at(gold), graph_at(predicted)))
                )
            ged_values.append(ged_cache[pair])

        result["exact_dag_accuracy"] = result["recall@1"]
        result["tool_set_f1"] = float(np.mean(tool_f1_values))
        result["mean_ged_error"] = float(np.mean(ged_values))
    result["num_queries"] = len(query_frame)
    result["gallery_unique_dags"] = len(gallery)
    return result


@torch.no_grad()
def evaluate_campaign_split(
    model: SentenceTransformer,
    test_csv: Path,
    candidate_csv: Path,
    batch_size: int,
) -> dict[str, float]:
    """Campaign-v4 full-gallery evaluation with one row per unique DAG.

    Campaign candidate files contain repeated query rows and omit test golds.
    The established full-gallery analysis therefore deduplicates candidates and
    unions in the test DAGs before retrieval.
    """
    test_frame = pd.read_csv(test_csv)
    candidate_frame = pd.read_csv(candidate_csv)
    gallery = pd.concat(
        [candidate_frame.drop_duplicates("dag_id"), test_frame.drop_duplicates("dag_id")],
        ignore_index=True,
    ).drop_duplicates("dag_id")
    return evaluate_id_retrieval(
        model, test_frame, gallery, batch_size, structural_metrics=True
    )


def train_tier(args: argparse.Namespace, tool_count: int) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "campaign_v4":
        split_dir = ROOT / "data" / "campaign_v4" / f"campaign_v4_{tool_count}tools"
    else:
        split_dir = ROOT / "upgraded" / f"upgraded_{tool_count}tools"
    train_csv = split_dir / "train.csv"
    dev_csv = split_dir / "dev.csv"
    heldout_csv = split_dir / "test_topology_heldout.csv"
    indomain_csv = split_dir / "test_indomain.csv"
    candidate_csv = split_dir / "candidate_corpus.csv"
    required_paths = [train_csv, dev_csv, heldout_csv]
    if args.dataset == "campaign_v4":
        required_paths.extend([indomain_csv, candidate_csv])
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir = Path(args.output_root) / f"{tool_count}tools"
    model_dir = out_dir / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing single-SBERT model: {model_dir}"
        )

    dataset = PairDataset(pd.read_csv(train_csv))
    sampler = NoDuplicateDagBatchSampler(dataset.labels, args.batch_size, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_pairs)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = SentenceTransformer(args.base_model, device=str(device))
    model.max_seq_length = args.max_length
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loss_fn = MultipleNegativesRankingLoss(model, scale=1.0 / args.temperature)
    if loss_fn.model is not model:
        raise RuntimeError("Loss is not using the single SentenceTransformer instance")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_recall = -1.0
    best_epoch = 0
    stale_epochs = 0
    print(
        f"[{tool_count} tools] one SentenceTransformer, {parameter_count:,} params, "
        f"{len(dataset)} pairs, {len(loader)} batches/epoch",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(
            loader,
            desc=f"single-SBERT {tool_count}t epoch {epoch}",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for queries, documents in progress:
            query_features = move_features(model.tokenize(queries), device)
            document_features = move_features(model.tokenize(documents), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn([query_features, document_features], torch.empty(0, device=device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP may skip an optimizer update if it detects overflow. Keep the
            # learning-rate schedule aligned with actual parameter updates.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

        model.eval()
        dev_frame = pd.read_csv(dev_csv)
        dev_metrics = evaluate_id_retrieval(
            model,
            dev_frame,
            dev_frame.drop_duplicates("dag_id"),
            args.eval_batch_size,
        )
        dev_count = int(dev_metrics["num_queries"])
        dev_dags = int(dev_metrics["gallery_unique_dags"])
        recall = float(dev_metrics["recall@1"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_metrics": dev_metrics,
        }
        history.append(record)
        improved = recall > best_recall + 1e-12
        print(
            f"[{tool_count} tools] epoch={epoch} loss={record['train_loss']:.4f} "
            f"dev_R@1={recall:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_recall = recall
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(str(model_dir))
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[{tool_count} tools] early stop at epoch {epoch}", flush=True)
                break

    del model, loss_fn, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    best_model = SentenceTransformer(str(model_dir), device=str(device))
    if args.dataset == "campaign_v4":
        test_results = {
            "test_indomain": evaluate_campaign_split(
                best_model, indomain_csv, candidate_csv, args.eval_batch_size
            ),
            "test_topology_heldout": evaluate_campaign_split(
                best_model, heldout_csv, candidate_csv, args.eval_batch_size
            ),
        }
    else:
        test_metrics, test_count, test_dags = evaluate(
            best_model, heldout_csv, tool_count, args.eval_batch_size
        )
        test_results = {
            "test_topology_heldout": {
                **test_metrics,
                "num_queries": test_count,
                "gallery_unique_dags": test_dags,
            }
        }
    payload = {
        "architecture": "single_standard_sentence_transformer",
        "dataset": args.dataset,
        "base_model": args.base_model,
        "tool_count": tool_count,
        "seed": args.seed,
        "single_model": True,
        "loss_uses_same_model_object": True,
        "custom_projection": False,
        "legr_or_gnn_loaded": False,
        "loss": "MultipleNegativesRankingLoss",
        "temperature": args.temperature,
        "parameter_count": parameter_count,
        "train_csv": str(train_csv.resolve()),
        "dev_csv": str(dev_csv.resolve()),
        "test_topology_heldout_csv": str(heldout_csv.resolve()),
        "train_pairs": len(dataset),
        "dev_examples": dev_count,
        "dev_unique_dags": dev_dags,
        "best_epoch": best_epoch,
        "best_dev_recall@1": best_recall,
        "test_results": test_results,
        "model_dir": str(model_dir.resolve()),
        "model_fingerprint": model_fingerprint(model_dir),
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{tool_count} tools] TEST {json.dumps(test_results, sort_keys=True)}", flush=True)
    return payload


def evaluate_saved_tier(args: argparse.Namespace, tool_count: int) -> dict:
    """Re-evaluate an existing standard SBERT checkpoint without training."""
    if args.dataset != "campaign_v4":
        raise ValueError("--eval_only currently supports --dataset campaign_v4")
    split_dir = ROOT / "data" / "campaign_v4" / f"campaign_v4_{tool_count}tools"
    candidate_csv = split_dir / "candidate_corpus.csv"
    out_dir = Path(args.output_root) / f"{tool_count}tools"
    model_dir = out_dir / "best_model"
    result_path = out_dir / "results.json"
    if not model_dir.exists() or not result_path.exists():
        raise FileNotFoundError(f"Missing trained single-SBERT artifacts in {out_dir}")

    model = SentenceTransformer(str(model_dir), device=args.device)
    modules = [module.__class__.__name__ for module in model]
    if modules != ["Transformer", "Pooling", "Normalize"]:
        raise RuntimeError(f"Unexpected SentenceTransformer modules: {modules}")
    test_results = {
        "test_indomain": evaluate_campaign_split(
            model, split_dir / "test_indomain.csv", candidate_csv, args.eval_batch_size
        ),
        "test_topology_heldout": evaluate_campaign_split(
            model,
            split_dir / "test_topology_heldout.csv",
            candidate_csv,
            args.eval_batch_size,
        ),
    }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["test_results"] = test_results
    payload["checkpoint_modules"] = modules
    payload["parameter_count_verified"] = sum(p.numel() for p in model.parameters())
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{tool_count} tools] EVAL {json.dumps(test_results, sort_keys=True)}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_counts", type=int, nargs="+", default=[15, 30, 45])
    parser.add_argument("--dataset", choices=["upgraded", "campaign_v4"], default="upgraded")
    parser.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_root", default=str(ROOT / "artifacts" / "sbert_single_model"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_root"] = str(output_root.resolve())
    config_name = "eval_config.json" if args.eval_only else "run_config.json"
    (output_root / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    runner = evaluate_saved_tier if args.eval_only else train_tier
    results = [runner(args, tool_count) for tool_count in args.tool_counts]
    summary = {str(result["tool_count"]): result["test_results"] for result in results}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
