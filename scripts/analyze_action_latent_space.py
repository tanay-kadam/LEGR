"""
analyze_action_latent_space.py — Action-type structure in LEGR embeddings.

Does not retrain LEGR. Encodes unique eval DAGs with a frozen checkpoint
(or a synthetic random embedding path for tests / missing checkpoints).

Cluster diagnostics are computed in the original embedding space, not t-SNE.

Usage::

    python scripts/analyze_action_latent_space.py --tool_count 30 \\
        --checkpoint checkpoints_30tools/best_model.pt \\
        --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv \\
        --output artifacts/action_latent_space

    python scripts/analyze_action_latent_space.py --synthetic \\
        --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv \\
        --output artifacts/action_latent_space
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legr_tool_count import (  # noqa: E402
    add_tool_count_argument,
    bootstrap_tool_count_from_argv,
)

bootstrap_tool_count_from_argv(sys.argv)

from action_type_mapping import (  # noqa: E402
    classify_dag_action_group,
    mapping_covers,
)
from data_synth import dag_to_text  # noqa: E402
from latent_space_metrics import embedding_diagnostics  # noqa: E402
from utils.graph_utils import classify_topology  # noqa: E402

TSNE_SEED = 42


def _tools_of_graph(G) -> list[str]:
    return [G.nodes[n]["tool"] for n in sorted(G.nodes())]


def _edges_of_graph(G) -> list[tuple[int, int]]:
    return [(int(u), int(v)) for u, v in G.edges()]


def run_tsne(embeddings: np.ndarray, seed: int = TSNE_SEED) -> np.ndarray | None:
    n = embeddings.shape[0]
    if n < 4:
        return None
    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(2.0, (n - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(embeddings)


def load_dataset(dataset_csv: str):
    from eval import CSVEvalDataset
    from utils import read_datafile

    return CSVEvalDataset(read_datafile(dataset_csv))


def encode_or_synthetic(args, dataset, device):
    n = dataset.num_unique_dags
    if args.synthetic or not args.checkpoint:
        rng = np.random.default_rng(args.seed)
        dim = 256
        embs = rng.normal(size=(n, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        return embs, "synthetic_random"
    import torch
    from eval import _load_model_and_tokenizer, encode_all_dags
    from encoders import resolve_graph_encoder_settings

    model, cfg, _tok = _load_model_and_tokenizer(args.checkpoint, device)
    _, _, bidirectional = resolve_graph_encoder_settings(cfg)
    tensor = encode_all_dags(model, dataset, device, bidirectional=bidirectional)
    return tensor.numpy(), str(args.checkpoint)


def build_plot_frame(dataset, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(dataset.num_unique_dags):
        G = dataset.get_unique_dag(i)
        tools = _tools_of_graph(G)
        edges = _edges_of_graph(G)
        missing = mapping_covers(tools)
        if missing:
            raise KeyError(f"Unmapped tools in DAG {i}: {missing}")
        group = classify_dag_action_group(tools)
        topo = classify_topology(edges, len(tools))
        rows.append({
            "dag_id": i,
            "dag_text": dag_to_text(G),
            "tools": ";".join(tools),
            "n_nodes": len(tools),
            "n_edges": len(edges),
            "action_group": group,
            "topo_family": topo,
        })
    df = pd.DataFrame(rows)
    df["emb_dim"] = embeddings.shape[1]
    return df


def maybe_plot(df: pd.DataFrame, xy: np.ndarray | None, out_dir: Path) -> None:
    if xy is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    df = df.copy()
    df["tsne_x"] = xy[:, 0]
    df["tsne_y"] = xy[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    for ax, col, title in (
        (axes[0], "topo_family", "Topology family"),
        (axes[1], "action_group", "Action-type group"),
    ):
        for label, sub in df.groupby(col):
            ax.scatter(sub["tsne_x"], sub["tsne_y"], s=18, alpha=0.75, label=str(label))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(f"LEGR DAG embeddings (t-SNE seed={TSNE_SEED})")
    fig.tight_layout()
    fig.savefig(out_dir / "action_type_tsne.png")
    fig.savefig(out_dir / "action_type_tsne.pdf")
    plt.close(fig)
    df.to_csv(out_dir / "plotting_data.csv", index=False)


def write_report(out_dir: Path, diagnostics: dict, source: str, counts: dict) -> None:
    evidence = diagnostics["evidence"]
    omit = evidence != "STRONG SUPPORT"
    lines = [
        "# Action-type structure in LEGR latent space",
        "",
        f"**Status:** {'PENDING_CHECKPOINT (synthetic embeddings)' if source == 'synthetic_random' else 'COMPUTED'}",
        f"**Evidence class:** {evidence}",
        f"**Paper recommendation:** {'omit from main paper (appendix only if at all)' if omit else 'candidate appendix/main figure'}",
        "",
        "## Mapping",
        "",
        "See `src/action_type_mapping.py`. 15-tool Tool-Bound branches are the",
        "source of truth; remaining tools follow 45-tool Tool-Bound with the",
        "documented lifecycle/access rules. Unmapped tools abort the run.",
        "",
        "## Counts",
        "",
        json.dumps(counts, indent=2),
        "",
        "## Diagnostics (original embedding space, not t-SNE)",
        "",
        json.dumps({k: v for k, v in diagnostics.items() if k != "label_counts"}, indent=2, default=str),
        "",
        f"t-SNE random_state = {TSNE_SEED}.",
        "",
        "Do not infer clusters from the plot alone.",
        "",
    ]
    if source == "synthetic_random":
        lines += [
            "## Note",
            "",
            "This run used random embeddings because no checkpoint was provided.",
            "The evidence class is therefore not a scientific result about LEGR.",
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_tool_count_argument(p)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--dataset_csv",
        default="upgraded/upgraded_30tools/test_topology_heldout.csv",
    )
    p.add_argument("--output", default="artifacts/action_latent_space")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=TSNE_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.is_absolute():
        dataset_csv = ROOT / dataset_csv
    dataset = load_dataset(str(dataset_csv))
    device = __import__("torch").device(
        "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    embeddings, source = encode_or_synthetic(args, dataset, device)
    frame = build_plot_frame(dataset, embeddings)
    counts = frame["action_group"].value_counts().to_dict()
    diag = embedding_diagnostics(embeddings, frame["action_group"].tolist())
    diag["source"] = source
    (out_dir / "metrics.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    frame.to_csv(out_dir / "plotting_data.csv", index=False)
    xy = run_tsne(embeddings, seed=args.seed)
    if xy is not None:
        maybe_plot(frame, xy, out_dir)
        np.save(out_dir / "tsne_coords.npy", xy)
    write_report(out_dir, diag, source, counts)
    mapping_path = out_dir / "mapping_definition.md"
    mapping_path.write_text(
        Path(ROOT / "src" / "action_type_mapping.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(json.dumps({"evidence": diag["evidence"], "source": source, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
