"""Evaluate every Campaign-v4 LEGR checkpoint on deduplicated full galleries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[15, 30, 45], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Directory containing legr_*_{tier}t_s42 checkpoint directories. "
        "Defaults to artifacts/campaign_v4/results.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=None,
        help="Evaluate only this checkpoint directory (repeatable).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to artifacts/campaign_v4/results/legr_full_gallery_all_{tier}t.json",
    )
    args = parser.parse_args()

    from legr_tool_count import apply_tool_count_override

    apply_tool_count_override(args.tier)
    from data_synth import build_dag, compute_ged, dag_canonical_hash
    from encoders import resolve_graph_encoder_settings
    from eval import (
        _load_model_and_tokenizer,
        compute_metrics,
        encode_all_dags,
        encode_all_queries,
    )
    from train import _parse_edges, _parse_tools

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data_dir = ROOT / "data" / "campaign_v4" / f"campaign_v4_{args.tier}tools"
    candidate_path = data_dir / "candidate_corpus.csv"
    candidate_frame = pd.read_csv(candidate_path)
    result_root = ROOT / "artifacts" / "campaign_v4" / "results"
    checkpoint_root = (
        Path(args.checkpoint_root).resolve() if args.checkpoint_root else result_root
    )
    checkpoint_dirs = (
        [Path(path).resolve() for path in args.checkpoint_dir]
        if args.checkpoint_dir
        else sorted(checkpoint_root.glob(f"legr_*_{args.tier}t_s42"))
    )
    checkpoint_dirs = [path for path in checkpoint_dirs if (path / "best_model.pt").exists()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No Campaign-v4 LEGR checkpoints found for tier {args.tier}")

    class QueryDataset:
        def __init__(self, queries: list[str]):
            self.samples = [{"query": query} for query in queries]

        def __len__(self) -> int:
            return len(self.samples)

    class GalleryDataset:
        def __init__(self, frame: pd.DataFrame):
            self._dags = []
            self._hash_to_index = {}
            self._ged_cache = {}
            for _, row in frame.iterrows():
                graph = build_dag(_parse_tools(row["tools"]), _parse_edges(row["edges"]))
                key = dag_canonical_hash(graph)
                if key not in self._hash_to_index:
                    self._hash_to_index[key] = len(self._dags)
                    self._dags.append(graph)
            self.num_unique_dags = len(self._dags)

        def get_unique_dag(self, index: int):
            return self._dags[index]

        def get_ged(self, left: int, right: int) -> float:
            key = (min(left, right), max(left, right))
            if key not in self._ged_cache:
                self._ged_cache[key] = (
                    0.0
                    if left == right
                    else float(compute_ged(self._dags[left], self._dags[right]))
                )
            return self._ged_cache[key]

        def index_for_row(self, row: pd.Series) -> int:
            graph = build_dag(_parse_tools(row["tools"]), _parse_edges(row["edges"]))
            return self._hash_to_index[dag_canonical_hash(graph)]

    def evaluate_split(model, cfg, tokenizer, test_path: Path) -> dict:
        test_frame = pd.read_csv(test_path)
        gallery_frame = pd.concat(
            [
                candidate_frame.drop_duplicates("dag_id"),
                test_frame.drop_duplicates("dag_id"),
            ],
            ignore_index=True,
        ).drop_duplicates("dag_id")
        gallery = GalleryDataset(gallery_frame)
        queries = QueryDataset(test_frame["query"].astype(str).tolist())
        ground_truth = torch.tensor(
            [gallery.index_for_row(row) for _, row in test_frame.iterrows()],
            dtype=torch.long,
        )
        _, _, bidirectional = resolve_graph_encoder_settings(cfg)
        query_embeddings = encode_all_queries(model, queries, tokenizer, device)
        dag_embeddings = encode_all_dags(
            model, gallery, device, batch_size=64, bidirectional=bidirectional
        )
        if not torch.isfinite(query_embeddings).all() or not torch.isfinite(dag_embeddings).all():
            raise RuntimeError("Non-finite LEGR embeddings")
        topk = (query_embeddings @ dag_embeddings.T).topk(
            k=min(5, gallery.num_unique_dags), dim=1
        ).indices
        metrics = compute_metrics(topk, ground_truth, gallery)
        return {
            **{key: float(value) for key, value in metrics.items()},
            "exact_dag_accuracy": float(metrics["recall@1"]),
            "num_queries": len(test_frame),
            "gallery_unique_dags": gallery.num_unique_dags,
        }

    results = {}
    for checkpoint_dir in checkpoint_dirs:
        checkpoint = checkpoint_dir / "best_model.pt"
        print(f"[{args.tier} tools] loading {checkpoint_dir.name}", flush=True)
        model, cfg, tokenizer = _load_model_and_tokenizer(
            str(checkpoint), device, dataset_csv=str(candidate_path)
        )
        results[checkpoint_dir.name] = {
            "checkpoint_epoch": int(torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)),
            "graph_encoder_type": getattr(cfg, "graph_encoder_type", None),
            "graph_direction": getattr(cfg, "graph_direction", None),
            "lambda_ged": float(getattr(cfg, "lambda_ged", 0.0)),
            "test_indomain": evaluate_split(
                model, cfg, tokenizer, data_dir / "test_indomain.csv"
            ),
            "test_topology_heldout": evaluate_split(
                model, cfg, tokenizer, data_dir / "test_topology_heldout.csv"
            ),
        }
        print(json.dumps(results[checkpoint_dir.name], default=float), flush=True)
        del model
        torch.cuda.empty_cache()

    payload = {
        "dataset": "campaign_v4",
        "tier": args.tier,
        "seed": 42,
        "checkpoint_root": str(checkpoint_root),
        "gallery_protocol": "unique(candidate_corpus UNION evaluated_test_split)",
        "models": results,
    }
    output = Path(args.output) if args.output else result_root / f"legr_full_gallery_all_{args.tier}t.json"
    output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
