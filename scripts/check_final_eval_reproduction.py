"""Read-only check that cached scoring reproduces recorded seed-42 dev metrics."""

from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from legr_experiments.data import ResearchDataset  # noqa: E402
from legr_experiments.final_evaluation import (  # noqa: E402
    build_candidate_cache,
    deterministic_gallery,
    retrieval_diagnostics,
    score_query_dataset,
)
from legr_experiments.functional_clusters import load_research_model  # noqa: E402
from src.data.tool_registry import get_tools  # noqa: E402


def main():
    device = torch.device("cuda")
    vocabulary = list(get_tools(15))
    dev_path = ROOT / "data/campaign_v4/campaign_v4_15tools/dev.csv"
    dataset = ResearchDataset(dev_path, vocabulary, structure_kind="degree")
    gallery = deterministic_gallery(dataset, seed=42)
    checkpoint = next(
        (ROOT / "artifacts/legr_model_search").glob(
            "confirm_r1_15t_s42_*/best_model.pt"
        )
    )
    model, config, _ = load_research_model(checkpoint, vocabulary, device)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model, local_files_only=True)
    cache = build_candidate_cache(model, gallery, tokenizer, device, batch_size=64)
    scores, _, gold, _ = score_query_dataset(
        model, dataset, tokenizer, cache, device, batch_size=64
    )
    query_tools = torch.stack([sample.signature.tool_target for sample in dataset.samples])
    candidate_tools = torch.stack([sample.signature.tool_target for sample in gallery.samples])
    metrics, _ = retrieval_diagnostics(scores, gold, query_tools, candidate_tools)
    recorded = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "best_dev_recall@1"
    ]
    print({
        "recorded_recall@1": recorded,
        "cached_recall@1": metrics["recall@1"],
        "cached_tool_set_f1": metrics["tool_set_f1"],
        "gallery_size": metrics["gallery_size"],
    })
    if abs(metrics["recall@1"] - recorded) > 1e-12:
        raise RuntimeError("Cached scorer does not reproduce the checkpoint dev R@1")


if __name__ == "__main__":
    main()
