"""Benchmark Campaign V4 V2/V3 indexing and online retrieval latency."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepaper_common import (
    ROOT,
    campaign_paths,
    checkpoint_manifest,
    create_output_dir,
    environment_snapshot,
    full_gallery_frame,
    repo_relative,
    sha256_file,
    validate_checkpoint_metadata,
    write_json,
)

sys.path.insert(0, str(ROOT / "src"))


def summarize_ms(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "throughput_queries_per_second": float(1000.0 / array.mean()),
        "n_measurements": int(len(array)),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def worker(args: argparse.Namespace) -> None:
    tier = int(args.worker_tier)
    from legr_tool_count import apply_tool_count_override

    apply_tool_count_override(tier)
    from encoders import resolve_graph_encoder_settings
    from eval import _load_model_and_tokenizer, encode_all_dags
    from prepaper_common import build_gallery_dataset

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gallery_frame, test_frame = full_gallery_frame(tier)
    gallery = build_gallery_dataset(gallery_frame)
    queries = test_frame["query"].astype(str).tolist()
    entries = checkpoint_manifest()[tier]
    results = []

    for entry in entries:
        checkpoint_path = Path(entry["checkpoint"])
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = validate_checkpoint_metadata(entry, checkpoint_payload, tier)
        model, cfg, tokenizer = _load_model_and_tokenizer(
            str(checkpoint_path), device, dataset_csv=str(campaign_paths(tier)["candidate"])
        )
        model.eval()
        _, _, bidirectional = resolve_graph_encoder_settings(cfg)

        # One untimed gallery pass warms kernels and allocator state.
        _ = encode_all_dags(model, gallery, device, batch_size=args.index_batch_size,
                            bidirectional=bidirectional)
        synchronize(device)
        indexing_ms: list[float] = []
        candidate_embeddings = None
        for _ in range(args.index_repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            candidate_embeddings = encode_all_dags(
                model, gallery, device, batch_size=args.index_batch_size,
                bidirectional=bidirectional,
            )
            synchronize(device)
            indexing_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
        assert candidate_embeddings is not None
        candidate_embeddings = candidate_embeddings.to(device)

        @torch.no_grad()
        def online_request(query: str) -> None:
            encoded = tokenizer(
                [query], padding=True, truncation=True, max_length=128,
                return_tensors="pt",
            )
            embedding = model.encode_text(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            scores = embedding @ candidate_embeddings.T
            _ = torch.argmax(scores, dim=1)

        for index in range(args.warmup):
            online_request(queries[index % len(queries)])
        synchronize(device)
        online_ms: list[float] = []
        for _ in range(args.passes):
            for query in queries:
                synchronize(device)
                start = time.perf_counter_ns()
                online_request(query)
                synchronize(device)
                online_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

        index_stats = summarize_ms(indexing_ms)
        index_stats["throughput_graphs_per_second"] = float(
            1000.0 * gallery.num_unique_dags / index_stats["mean_ms"]
        )
        result = {
            **{key: value for key, value in entry.items() if key != "checkpoint"},
            "tier": tier,
            "checkpoint": repo_relative(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            **metadata,
            "gallery_size": gallery.num_unique_dags,
            "query_count": len(queries),
            "online": summarize_ms(online_ms),
            "indexing": index_stats,
            "raw_online_ms": online_ms,
            "raw_indexing_ms": indexing_ms,
        }
        results.append(result)
        print(json.dumps({"model": entry["model_id"], "online": result["online"],
                          "indexing": result["indexing"]}), flush=True)
        del model, candidate_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(args.worker_output, {
        "tier": tier,
        "protocol": {
            "online_batch_size": 1,
            "warmup_requests": args.warmup,
            "passes": args.passes,
            "index_batch_size": args.index_batch_size,
            "index_warmup_passes": 1,
            "index_timed_repeats": args.index_repeats,
            "online_includes": ["tokenization", "host-to-device transfer", "query encoder", "cosine scoring", "argmax"],
            "online_excludes": ["checkpoint loading", "candidate graph encoding", "file I/O"],
        },
        "environment": environment_snapshot(device),
        "results": results,
    })


def write_outputs(output: Path, payload: dict) -> None:
    write_json(output / "latency_results.json", payload)
    flat_rows = []
    for result in payload["results"]:
        flat_rows.append({
            "model_id": result["model_id"],
            "architecture": result["architecture"],
            "objective": result["objective"],
            "tier": result["tier"],
            "gallery_size": result["gallery_size"],
            "queries": result["query_count"],
            "online_mean_ms": result["online"]["mean_ms"],
            "online_median_ms": result["online"]["median_ms"],
            "online_std_ms": result["online"]["std_ms"],
            "online_p95_ms": result["online"]["p95_ms"],
            "online_min_ms": result["online"]["min_ms"],
            "online_max_ms": result["online"]["max_ms"],
            "online_qps": result["online"]["throughput_queries_per_second"],
            "index_mean_ms": result["indexing"]["mean_ms"],
            "index_median_ms": result["indexing"]["median_ms"],
            "index_p95_ms": result["indexing"]["p95_ms"],
            "index_graphs_per_second": result["indexing"]["throughput_graphs_per_second"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
    with (output / "latency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Tools & Model & Gallery & Online mean (ms) & Online p95 (ms) & Indexing (ms) \\",
        r"\midrule",
    ]
    for row in flat_rows:
        label = f"{row['architecture']} {row['objective']}".replace("InfoNCE+GED", "+GED")
        lines.append(
            f"{row['tier']} & {label} & {row['gallery_size']} & "
            f"{row['online_mean_ms']:.2f} & {row['online_p95_ms']:.2f} & "
            f"{row['index_mean_ms']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "latency_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "environment.json", payload["environment"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/prepaper_v2_v3_latency_s42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--index-repeats", type=int, default=5)
    parser.add_argument("--index-batch-size", type=int, default=64)
    parser.add_argument("--worker-tier", type=int, choices=[15, 30, 45], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_tier is not None:
        worker(args)
        return

    output = create_output_dir(args.output)
    worker_payloads = []
    try:
        for tier in (15, 30, 45):
            worker_output = output / f"tier_{tier}_raw.json"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-tier", str(tier),
                "--worker-output", str(worker_output),
                "--device", args.device,
                "--warmup", str(args.warmup),
                "--passes", str(args.passes),
                "--index-repeats", str(args.index_repeats),
                "--index-batch-size", str(args.index_batch_size),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            worker_payloads.append(json.loads(worker_output.read_text(encoding="utf-8")))
    except Exception:
        (output / "INCOMPLETE.txt").write_text(
            "The benchmark did not complete. Existing partial files were retained for audit.\n",
            encoding="utf-8",
        )
        raise
    environments = [item["environment"] for item in worker_payloads]
    if any(value != environments[0] for value in environments[1:]):
        raise AssertionError("Worker hardware/software environments differ")
    results = [result for item in worker_payloads for result in item["results"]]
    if len(results) != 12:
        raise AssertionError(f"Expected 12 latency results, found {len(results)}")
    payload = {
        "dataset": "campaign_v4/test_topology_heldout",
        "gallery_protocol": "unique(candidate_corpus UNION test_topology_heldout)",
        "seed": 42,
        "environment": environments[0],
        "protocol": worker_payloads[0]["protocol"],
        "results": results,
    }
    write_outputs(output, payload)
    (output / "reproduce.txt").write_text(
        f"{sys.executable} scripts/benchmark_v2_v3_latency.py --device {args.device} "
        f"--warmup {args.warmup} --passes {args.passes} --index-repeats {args.index_repeats} "
        f"--output {repo_relative(output)}\n",
        encoding="utf-8",
    )
    print(f"Saved 12-model benchmark to {output}")


if __name__ == "__main__":
    main()
