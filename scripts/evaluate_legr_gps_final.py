"""Evaluate locked LEGR-GPS checkpoints on the unchanged 322-DAG gallery.

This entrypoint performs no training and writes only to a new output directory.
It evaluates three confirmed checkpoints, the identical frozen SBERT-FT expert,
and the existing standalone V3 checkpoint using one deterministic gallery.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data_synth import build_dag, compute_ged  # noqa: E402
from encoders import resolve_graph_encoder_settings  # noqa: E402
from eval import _load_model_and_tokenizer, encode_all_dags, encode_all_queries  # noqa: E402
from legr_experiments.data import ResearchDataset  # noqa: E402
from legr_experiments.final_evaluation import (  # noqa: E402
    build_candidate_cache,
    dag_clustered_paired_bootstrap,
    deterministic_gallery,
    retrieval_diagnostics,
    score_query_dataset,
    synchronized_cached_latency,
)
from legr_experiments.functional_clusters import load_research_model  # noqa: E402
from legr_experiments.integrity import compare, snapshot  # noqa: E402
from src.data.tool_registry import get_tools  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gallery-seed", type=int, default=42)
    parser.add_argument("--tie-tolerance", type=float, default=1e-5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument(
        "--output-root",
        default="artifacts/legr_model_search/final_322dag_eval",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def seed_from_checkpoint(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    seed = payload.get("config", {}).get("train", {}).get("seed")
    if seed is None:
        match = re.search(r"_s(\d+)_", str(path))
        if not match:
            raise ValueError(f"Cannot determine seed for {path}")
        seed = int(match.group(1))
    return int(seed)


def exact_ged_values(query_dataset, candidate_samples, predicted_indices):
    cache = {}
    values = []
    for query, predicted in zip(query_dataset.samples, predicted_indices):
        candidate = candidate_samples[int(predicted)]
        pair = (query.signature.dag_key, candidate.signature.dag_key)
        if pair not in cache:
            gold_graph = build_dag(list(query.signature.tools), list(query.signature.edges))
            predicted_graph = build_dag(
                list(candidate.signature.tools), list(candidate.signature.edges)
            )
            cache[pair] = float(compute_ged(gold_graph, predicted_graph))
        values.append(cache[pair])
    return np.asarray(values, dtype=np.float64)


class QueryOnly:
    def __init__(self, queries):
        self.samples = [{"query": query} for query in queries]

    def __len__(self):
        return len(self.samples)


class DagOnly:
    def __init__(self, samples):
        self.dags = [build_dag(list(s.signature.tools), list(s.signature.edges)) for s in samples]
        self.num_unique_dags = len(self.dags)

    def get_unique_dag(self, index):
        return self.dags[index]


@torch.no_grad()
def evaluate_v3(
    checkpoint: Path,
    queries,
    candidate_samples,
    device,
    gold_indices,
    query_tools,
    candidate_tools,
    tolerance,
):
    model, config, tokenizer = _load_model_and_tokenizer(str(checkpoint), device)
    _, _, bidirectional = resolve_graph_encoder_settings(config)
    query_embeddings = encode_all_queries(
        model, QueryOnly(queries), tokenizer, device, batch_size=64
    )
    candidate_embeddings = encode_all_dags(
        model, DagOnly(candidate_samples), device, batch_size=64,
        bidirectional=bidirectional,
    )
    scores = query_embeddings @ candidate_embeddings.t()
    metrics, details = retrieval_diagnostics(
        scores, gold_indices, query_tools, candidate_tools, tolerance=tolerance
    )
    ged = exact_ged_values_from_samples(details, candidate_samples)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, metrics, details, ged


def exact_ged_values_from_samples(details, candidate_samples, query_dataset=None):
    if query_dataset is None:
        return None
    return exact_ged_values(query_dataset, candidate_samples, details["predicted_indices"])


def per_query_frame(
    source_frame,
    query_dataset,
    cache,
    scores,
    details,
    gold_indices,
    ged_values,
):
    rows = []
    for index, sample in enumerate(query_dataset.samples):
        predicted = int(details["predicted_indices"][index])
        gold = int(gold_indices[index])
        source = source_frame.iloc[index]
        top5 = np.argsort(-scores[index].numpy(), kind="stable")[:5]
        rows.append({
            "query_index": index,
            "query": sample.query,
            "query_condition": source.get("query_condition", ""),
            "declared_split": source.get("split", "test_topology_heldout"),
            "gold_dag_key": cache.keys[gold],
            "predicted_dag_key": cache.keys[predicted],
            "gold_gallery_index": gold,
            "predicted_gallery_index": predicted,
            "rank": int(details["ranks"][index]),
            "best_tie_rank": int(details["best_tie_ranks"][index]),
            "worst_tie_rank": int(details["worst_tie_ranks"][index]),
            "exact_correct": float(details["exact_correct"][index]),
            "tie_expected_exact_correct": float(details["tie_expected_exact_correct"][index]),
            "tool_set_f1": float(details["tool_f1"][index]),
            "tie_expected_tool_set_f1": float(details["tie_expected_tool_f1"][index]),
            "true_twin_eligible": bool(details["twin_eligible"][index]),
            "true_twin_exact_correct": float(details["twin_exact_correct"][index]),
            "true_twin_expected_correct": float(details["twin_expected_correct"][index]),
            "hard_negative_pair_accuracy": float(details["hard_negative_pair_accuracy"][index])
            if np.isfinite(details["hard_negative_pair_accuracy"][index]) else "",
            "graph_edit_distance": float(ged_values[index]),
            "gold_score": float(scores[index, gold]),
            "predicted_score": float(scores[index, predicted]),
            "top5_dag_keys": ";".join(cache.keys[int(item)] for item in top5),
        })
    return pd.DataFrame(rows)


def mean_and_sd(records, metric):
    values = np.asarray([record["metrics"][metric] for record in records], dtype=np.float64)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def markdown_report(summary):
    lines = [
        "# Locked 322-DAG Final Evaluation",
        "",
        "This report evaluates the three preselected `confirm_r1` LEGR-GPS checkpoints on the unchanged Campaign-v4 15-tool topology-held-out queries and the combined 322-DAG gallery. No training or model selection used these results.",
        "",
        "## Gallery",
        "",
        f"- Queries: **{summary['gallery']['queries']}**",
        f"- Gold DAGs: **{summary['gallery']['gold_dags']}**",
        f"- Candidate-only DAGs: **{summary['gallery']['candidate_only_dags']}**",
        f"- Combined gallery: **{summary['gallery']['gallery_dags']}**",
        f"- Unique tool sets: **{summary['gallery']['unique_toolsets']}**",
        f"- Tie tolerance: `{summary['protocol']['tie_tolerance']}`",
        "",
        "## Results",
        "",
        "| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 | Tie-expected twin R@1 | Mean GED | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sb = summary["sbert_ft"]
    lines.append(
        f"| SBERT-FT | 42 checkpoint | {sb['metrics']['recall@1']:.4f} | {sb['metrics']['recall@3']:.4f} | {sb['metrics']['recall@5']:.4f} | {sb['metrics']['tool_set_f1']:.4f} | {sb['metrics']['true_twin_recall@1']:.4f} | {sb['metrics']['tie_expected_true_twin_recall@1']:.4f} | {sb['metrics']['mean_ged_error']:.4f} | {sb['latency']['p95_ms']:.3f} |"
    )
    v3 = summary["v3"]
    lines.append(
        f"| LEGR V3 | 42 | {v3['metrics']['recall@1']:.4f} | {v3['metrics']['recall@3']:.4f} | {v3['metrics']['recall@5']:.4f} | {v3['metrics']['tool_set_f1']:.4f} | {v3['metrics']['true_twin_recall@1']:.4f} | {v3['metrics']['tie_expected_true_twin_recall@1']:.4f} | {v3['metrics']['mean_ged_error']:.4f} | --- |"
    )
    for record in summary["legr_gps"]:
        m, latency = record["metrics"], record["latency"]
        lines.append(
            f"| LEGR-GPS | {record['seed']} | {m['recall@1']:.4f} | {m['recall@3']:.4f} | {m['recall@5']:.4f} | {m['tool_set_f1']:.4f} | {m['true_twin_recall@1']:.4f} | {m['tie_expected_true_twin_recall@1']:.4f} | {m['mean_ged_error']:.4f} | {latency['p95_ms']:.3f} |"
        )
    aggregate = summary["legr_gps_mean"]
    lines.append(
        f"| **LEGR-GPS mean** | 3 seeds | **{aggregate['recall@1']['mean']:.4f}** | {aggregate['recall@3']['mean']:.4f} | {aggregate['recall@5']['mean']:.4f} | **{aggregate['tool_set_f1']['mean']:.4f}** | **{aggregate['true_twin_recall@1']['mean']:.4f}** | {aggregate['tie_expected_true_twin_recall@1']['mean']:.4f} | {aggregate['mean_ged_error']['mean']:.4f} | {aggregate['p95_latency_ms']['mean']:.3f} |"
    )
    lines.extend(["", "## Paired DAG-clustered bootstrap: LEGR-GPS minus SBERT-FT", ""])
    for name, result in summary["paired_bootstrap"].items():
        lines.append(
            f"- {name}: delta **{result['delta']:+.4f}**, 95% CI **[{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}]** ({result['clusters']} gold-DAG clusters; {result['samples']} resamples)."
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        summary["paper_interpretation"],
        "",
        "Candidate embeddings were cached once per model. Latency is synchronized batch-size-one query encoding plus scoring over all 322 cached candidates; candidate-cache construction is reported separately in `summary.json`.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    output_root = ROOT / args.output_root
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing derived results: {output_root}. "
            "Pass a new --output-root for another run."
        )
    output_root.mkdir(parents=True)
    before = snapshot()
    (output_root / "immutable_before.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    vocabulary = list(get_tools(15))
    data_root = ROOT / "data/campaign_v4/campaign_v4_15tools"
    test_path = data_root / "test_topology_heldout.csv"
    candidate_path = data_root / "candidate_corpus.csv"
    query_dataset = ResearchDataset(test_path, vocabulary, structure_kind="degree")
    gallery_dataset = ResearchDataset(
        [test_path, candidate_path], vocabulary, structure_kind="degree"
    )
    gallery = deterministic_gallery(gallery_dataset, args.gallery_seed)
    source_frame = pd.read_csv(test_path)
    if len(query_dataset) != 300 or len(source_frame) != 300:
        raise ValueError(f"Expected 300 queries, found {len(query_dataset)}/{len(source_frame)}")
    if len(gallery) != 322:
        raise ValueError(f"Expected 322 gallery DAGs, found {len(gallery)}")
    gold_dags = len({sample.signature.dag_key for sample in query_dataset.samples})
    candidate_only = len({sample.signature.dag_key for sample in ResearchDataset(candidate_path, vocabulary).samples})
    query_tools = torch.stack([sample.signature.tool_target for sample in query_dataset.samples])
    candidate_tools = torch.stack([sample.signature.tool_target for sample in gallery.samples])
    toolset_count = len({tuple(row.tolist()) for row in candidate_tools})

    checkpoint_paths = sorted(
        (ROOT / "artifacts/legr_model_search").glob("confirm_r1_15t_s*_*/best_model.pt"),
        key=seed_from_checkpoint,
    )
    checkpoint_paths = [path for path in checkpoint_paths if seed_from_checkpoint(path) in (42, 123, 2026)]
    if len(checkpoint_paths) != 3:
        raise ValueError(f"Expected three confirm_r1 checkpoints, found {checkpoint_paths}")

    records = []
    detail_by_seed = {}
    first_semantic_scores = None
    first_semantic_details = None
    first_cache = None
    first_gold = None
    first_gold_keys = None
    semantic_latency = None
    started_all = time.perf_counter()
    for checkpoint in checkpoint_paths:
        seed = seed_from_checkpoint(checkpoint)
        model, config, payload = load_research_model(checkpoint, vocabulary, device)
        tokenizer = AutoTokenizer.from_pretrained(config.text_model, local_files_only=True)
        cache = build_candidate_cache(
            model, gallery, tokenizer, device, batch_size=args.batch_size
        )
        scores, experts, gold_indices, gold_keys = score_query_dataset(
            model, query_dataset, tokenizer, cache, device, batch_size=args.batch_size
        )
        metrics, details = retrieval_diagnostics(
            scores, gold_indices, query_tools, candidate_tools,
            tolerance=args.tie_tolerance,
        )
        ged_values = exact_ged_values(
            query_dataset, gallery.samples, details["predicted_indices"]
        )
        metrics["mean_ged_error"] = float(ged_values.mean())
        latency = synchronized_cached_latency(
            model,
            [sample.query for sample in query_dataset.samples],
            tokenizer,
            cache,
            device,
            warmup=args.latency_warmup,
        )
        frame = per_query_frame(
            source_frame, query_dataset, cache, scores, details,
            gold_indices, ged_values,
        )
        frame.to_csv(output_root / f"legr_gps_seed{seed}_per_query.csv", index=False)
        np.savez_compressed(
            output_root / f"legr_gps_seed{seed}_scores.npz",
            scores=scores.numpy(),
            experts=experts.numpy(),
            gold_indices=np.asarray(gold_indices),
            candidate_keys=np.asarray(cache.keys),
        )
        record = {
            "seed": seed,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_best_dev_recall@1": payload.get("best_dev_recall@1"),
            "metrics": metrics,
            "latency": latency,
        }
        records.append(record)
        detail_by_seed[seed] = details
        semantic_scores = experts[:, :, 0]
        semantic_metrics, semantic_details = retrieval_diagnostics(
            semantic_scores, gold_indices, query_tools, candidate_tools,
            tolerance=args.tie_tolerance,
        )
        if first_semantic_scores is None:
            first_semantic_scores = semantic_scores
            first_semantic_details = semantic_details
            first_cache = cache
            first_gold = gold_indices
            first_gold_keys = gold_keys
            semantic_ged = exact_ged_values(
                query_dataset, gallery.samples, semantic_details["predicted_indices"]
            )
            semantic_metrics["mean_ged_error"] = float(semantic_ged.mean())
            semantic_record = {
                "checkpoint": "artifacts/campaign_v4/results/sbert_ft_ged_15t_s42/best_model.pt",
                "metrics": semantic_metrics,
            }
            semantic_frame = per_query_frame(
                source_frame, query_dataset, cache, semantic_scores,
                semantic_details, gold_indices, semantic_ged,
            )
            semantic_frame.to_csv(output_root / "sbert_ft_per_query.csv", index=False)
        else:
            maximum_difference = float((semantic_scores - first_semantic_scores).abs().max())
            if maximum_difference > 1e-6:
                raise ValueError(
                    f"Frozen SBERT-FT scores differ across composite checkpoints: {maximum_difference}"
                )
        del model, cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Time the standalone frozen SBERT-FT query tower and dot product only.
    model, config, _ = load_research_model(checkpoint_paths[0], vocabulary, device)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model, local_files_only=True)
    semantic_candidates = first_cache.semantic if first_cache is not None else None
    latency_values = []
    queries = [sample.query for sample in query_dataset.samples]
    for index in range(args.latency_warmup):
        encoded = tokenizer([queries[index % len(queries)]], return_tensors="pt", truncation=True, max_length=128)
        z = model.semantic_expert.encode_query(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        _ = z @ semantic_candidates.t()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for query in queries:
        encoded = tokenizer([query], return_tensors="pt", truncation=True, max_length=128)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        z = model.semantic_expert.encode_query(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        _ = z @ semantic_candidates.t()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_values.append((time.perf_counter() - started) * 1000)
    latency_values = np.asarray(latency_values)
    semantic_latency = {
        "trials": len(latency_values),
        "warmup": args.latency_warmup,
        "batch_size": 1,
        "candidate_cache_size": len(gallery),
        "mean_ms": float(latency_values.mean()),
        "median_ms": float(np.median(latency_values)),
        "p95_ms": float(np.quantile(latency_values, 0.95)),
        "min_ms": float(latency_values.min()),
        "max_ms": float(latency_values.max()),
    }
    semantic_record["latency"] = semantic_latency
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Evaluate the existing V3 checkpoint under the same order and metrics.
    v3_checkpoint = ROOT / "artifacts/campaign_v4/results/legr_setgnn_tied_no_ged_15t_s42/best_model.pt"
    v3_model, v3_config, v3_tokenizer = _load_model_and_tokenizer(str(v3_checkpoint), device)
    _, _, bidirectional = resolve_graph_encoder_settings(v3_config)
    v3_queries = encode_all_queries(v3_model, QueryOnly(queries), v3_tokenizer, device, batch_size=64)
    v3_candidates = encode_all_dags(
        v3_model, DagOnly(gallery.samples), device, batch_size=64, bidirectional=bidirectional
    )
    v3_scores = v3_queries @ v3_candidates.t()
    v3_metrics, v3_details = retrieval_diagnostics(
        v3_scores, first_gold, query_tools, candidate_tools, tolerance=args.tie_tolerance
    )
    v3_ged = exact_ged_values(query_dataset, gallery.samples, v3_details["predicted_indices"])
    v3_metrics["mean_ged_error"] = float(v3_ged.mean())
    v3_frame = per_query_frame(
        source_frame, query_dataset, first_cache, v3_scores, v3_details,
        first_gold, v3_ged,
    )
    v3_frame.to_csv(output_root / "legr_v3_per_query.csv", index=False)
    del v3_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exact_mean = np.mean(
        [detail_by_seed[seed]["exact_correct"] for seed in sorted(detail_by_seed)], axis=0
    )
    tool_mean = np.mean(
        [detail_by_seed[seed]["tool_f1"] for seed in sorted(detail_by_seed)], axis=0
    )
    twin_mask = first_semantic_details["twin_eligible"]
    twin_mean = np.mean(
        [detail_by_seed[seed]["twin_expected_correct"] for seed in sorted(detail_by_seed)], axis=0
    )
    bootstrap = {
        "exact_recall@1": dag_clustered_paired_bootstrap(
            exact_mean,
            first_semantic_details["exact_correct"],
            first_gold_keys,
            args.bootstrap_samples,
            42,
        ),
        "tool_set_f1": dag_clustered_paired_bootstrap(
            tool_mean,
            first_semantic_details["tool_f1"],
            first_gold_keys,
            args.bootstrap_samples,
            43,
        ),
        "tie_expected_true_twin_recall@1": dag_clustered_paired_bootstrap(
            twin_mean[twin_mask],
            first_semantic_details["twin_expected_correct"][twin_mask],
            np.asarray(first_gold_keys)[twin_mask],
            args.bootstrap_samples,
            44,
        ),
    }
    passed = (
        bootstrap["exact_recall@1"]["ci95_low"] > 0
        and bootstrap["tool_set_f1"]["ci95_low"] > 0
        and bootstrap["tie_expected_true_twin_recall@1"]["delta"] > 0
    )
    if passed:
        interpretation = (
            "The locked LEGR-GPS checkpoints improve exact R@1 and tool-set F1 over SBERT-FT "
            "with DAG-clustered paired 95% confidence intervals excluding zero; the mean "
            "tie-aware true-twin result is also positive. These results support a final-test "
            "superiority claim under this 15-tool gallery."
        )
    else:
        interpretation = (
            "The predefined final-test success criterion was not fully met. The paper must report "
            "the measured trade-off and must not claim universal superiority over SBERT-FT."
        )

    aggregate_keys = [
        "recall@1", "recall@3", "recall@5", "mrr@5", "tool_set_f1",
        "true_twin_recall@1", "tie_expected_true_twin_recall@1",
        "hard_negative_pair_accuracy", "mean_ged_error",
    ]
    aggregate = {key: mean_and_sd(records, key) for key in aggregate_keys}
    aggregate["p95_latency_ms"] = {
        "mean": float(np.mean([record["latency"]["p95_ms"] for record in records])),
        "sample_sd": float(np.std([record["latency"]["p95_ms"] for record in records], ddof=1)),
    }
    summary = {
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "selection": "three locked confirm_r1 checkpoints; no test-time selection",
            "gallery_order": f"canonical-key sort followed by Python random shuffle seed {args.gallery_seed}",
            "tie_tolerance": args.tie_tolerance,
            "bootstrap_unit": "gold canonical DAG (six paraphrases retained as a cluster)",
            "bootstrap_samples": args.bootstrap_samples,
            "device": str(device),
        },
        "gallery": {
            "queries": len(query_dataset),
            "gold_dags": gold_dags,
            "candidate_only_dags": candidate_only,
            "gallery_dags": len(gallery),
            "unique_toolsets": toolset_count,
            "all_gold_present": True,
        },
        "sbert_ft": semantic_record,
        "v3": {"checkpoint": v3_checkpoint.relative_to(ROOT).as_posix(), "metrics": v3_metrics},
        "legr_gps": records,
        "legr_gps_mean": aggregate,
        "paired_bootstrap": bootstrap,
        "predefined_success_criterion_met": passed,
        "paper_interpretation": interpretation,
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    (output_root / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    (output_root / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    (output_root / "gallery_order.csv").write_text(
        "gallery_index,canonical_dag_key,tools,edges\n" + "\n".join(
            f'{index},{sample.signature.dag_key},"{";".join(sample.signature.tools)}","{";".join(f"{u}->{v}" for u, v in sample.signature.edges)}"'
            for index, sample in enumerate(gallery.samples)
        ),
        encoding="utf-8",
    )
    after = snapshot()
    drift = compare(before, after)
    (output_root / "immutable_after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "integrity_report.json").write_text(
        json.dumps(drift, indent=2), encoding="utf-8"
    )
    if any(drift.values()):
        raise RuntimeError(f"Immutable inputs changed: {drift}")
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
