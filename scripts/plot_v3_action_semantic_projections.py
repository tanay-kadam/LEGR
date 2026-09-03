"""Create reference-style PCA/t-SNE figures for action and semantic labels."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "legr_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
PALETTE = ("#ef4444", "#6abe6b", "#5796c5", "#8d6cab", "#d59a3a")


def create_output_dir(value: str) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    return output


def projected_frame(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    label_column: str,
    seed: int,
) -> pd.DataFrame:
    keep = frame[label_column].astype(str).str.casefold() != "mixed"
    subset = frame.loc[keep].copy().reset_index(drop=True)
    vectors = embeddings[keep.to_numpy()]
    if len(subset) < 4:
        raise AssertionError(f"Too few non-mixed points for {label_column}")
    pca = PCA(n_components=2).fit_transform(vectors)
    perplexity = min(30.0, max(2.0, (len(subset) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(vectors)
    subset["pca_x"] = pca[:, 0]
    subset["pca_y"] = pca[:, 1]
    subset["tsne_x"] = tsne[:, 0]
    subset["tsne_y"] = tsne[:, 1]
    return subset


def draw_figure(
    frame: pd.DataFrame,
    label_column: str,
    title: str,
    output_stem: Path,
    seed: int,
) -> None:
    labels = sorted(frame[label_column].unique())
    colors = {label: PALETTE[index] for index, label in enumerate(labels)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.9), dpi=200)
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)
    for axis, prefix, panel_title in (
        (axes[0], "pca", "PCA projection"),
        (axes[1], "tsne", "t-SNE projection"),
    ):
        for label in labels:
            subset = frame[frame[label_column] == label]
            color = colors[label]
            axis.scatter(
                subset[f"{prefix}_x"], subset[f"{prefix}_y"],
                s=32, alpha=0.82, color=color, linewidths=0,
                label=f"{label} (n={len(subset)})",
            )
            axis.scatter(
                subset[f"{prefix}_x"].mean(), subset[f"{prefix}_y"].mean(),
                marker="X", s=190, color=color, edgecolor="black",
                linewidth=1.5, zorder=5,
            )
        axis.set_title(panel_title, fontsize=17)
        axis.set_xlabel("Component 1", fontsize=12)
        axis.set_ylabel("Component 2", fontsize=12)
        axis.grid(True, alpha=0.18)
        axis.legend(frameon=False, fontsize=10, loc="best")
    fig.text(
        0.5, 0.025,
        f"Centroid markers: X  |  Seed {seed}  |  {len(frame)} non-tied unique graphs",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", default="artifacts/prepaper_v3_action_semantic_s42",
        help="Directory containing v3_graph_embeddings.npy and plotting_data.csv.",
    )
    parser.add_argument(
        "--output", default="artifacts/prepaper_v3_action_semantic_projection_s42"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analysis = Path(args.analysis_dir)
    if not analysis.is_absolute():
        analysis = ROOT / analysis
    embeddings = np.load(analysis / "v3_graph_embeddings.npy")
    labels = pd.read_csv(analysis / "plotting_data.csv")
    if len(labels) != len(embeddings):
        raise AssertionError("Embedding and label row counts differ")
    output = create_output_dir(args.output)

    specifications = (
        (
            "action_label",
            "LEGR V3 Graph Embeddings by Dominant Action Type",
            "v3_action_clusters_scatter",
        ),
        (
            "semantic_label",
            "LEGR V3 Graph Embeddings by Dominant Semantic Type",
            "v3_semantic_clusters_scatter",
        ),
    )
    summary_rows = []
    for column, title, stem in specifications:
        projected = projected_frame(embeddings, labels, column, args.seed)
        projected.to_csv(output / f"{stem}_points.csv", index=False)
        draw_figure(projected, column, title, output / stem, args.seed)
        for label, count in projected[column].value_counts().sort_index().items():
            summary_rows.append({"taxonomy": column, "label": label, "count": int(count)})
    pd.DataFrame(summary_rows).to_csv(output / "label_counts.csv", index=False)
    (output / "README.md").write_text(
        "# Reference-style V3 cluster projections\n\n"
        "Each figure contains PCA and t-SNE panels with projected-space centroids. "
        "DAGs labelled Mixed (exact plurality ties) are omitted to match the requested "
        "non-tied presentation. The action and semantic figures are projected separately "
        "because they retain different non-tied subsets; they are visualizations, not the "
        "source of the original-space quantitative metrics.\n",
        encoding="utf-8",
    )
    print(f"Saved action and semantic projection figures to {output}")


if __name__ == "__main__":
    main()
