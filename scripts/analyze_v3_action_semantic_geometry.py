"""Compare matched action and semantic structure in Campaign V4 V3 DAG embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "legr_matplotlib"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from prepaper_common import (
    ROOT,
    build_gallery_dataset,
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


SEMANTIC_GROUPS = {
    "Account & Subscription": (
        "read_user_profile",
        "read_subscription_status",
        "edit_username",
        "update_subscription_plan",
        "dispatch_message_to_usergroup",
    ),
    "Data & Access Management": (
        "read_database_record",
        "read_access_logs",
        "write_database_record",
        "reset_user_password",
        "escalate_case_to_human",
    ),
    "Service & Incident Operations": (
        "check_service_status",
        "scan_system_for_malware",
        "restart_service",
        "create_support_ticket",
        "route_task_by_condition",
    ),
}

ACTION_DISPLAY = {
    "DATA_RETRIEVAL": "Data Retrieval",
    "STATE_MODIFICATION": "State Modification",
    "ORCHESTRATION": "Orchestration",
}


def reverse_mapping(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group, tools in groups.items():
        for tool in tools:
            if tool in result:
                raise AssertionError(f"Tool {tool} appears in multiple semantic groups")
            result[tool] = group
    return result


def unique_plurality_label(tools: list[str], mapping: dict[str, str]) -> str:
    missing = sorted(set(tools) - set(mapping))
    if missing:
        raise KeyError(f"Unmapped tools: {missing}")
    counts = Counter(mapping[tool] for tool in tools)
    ranked = counts.most_common()
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return "Mixed"


def confidence_interval(values: np.ndarray, rng: np.random.Generator,
                        samples: int) -> list[float]:
    n = len(values)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = values[rng.integers(0, n, size=n)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def label_diagnostics(embeddings: np.ndarray, labels: np.ndarray, seed: int,
                      bootstrap: int) -> tuple[dict, dict[str, np.ndarray]]:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        raise AssertionError(f"Insufficient label support for diagnostics: {dict(zip(classes, counts))}")
    silhouette_by_point = silhouette_samples(embeddings, labels, metric="cosine")
    neighbors = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    neighbor_ids = neighbors.fit(embeddings).kneighbors(return_distance=False)
    purity_by_point = np.asarray([
        np.mean(labels[row[row != index][:5]] == labels[index])
        for index, row in enumerate(neighbor_ids)
    ])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    classifier = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")
    predictions = cross_val_predict(classifier, embeddings, labels, cv=splitter)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    rng = np.random.default_rng(seed)
    macro_boot = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        sampled = rng.integers(0, len(labels), size=len(labels))
        macro_boot[index] = f1_score(
            labels[sampled], predictions[sampled], labels=classes,
            average="macro", zero_division=0,
        )
    metrics = {
        "counts": {str(key): int(value) for key, value in zip(classes, counts)},
        "silhouette_cosine": float(silhouette_by_point.mean()),
        "silhouette_95ci": confidence_interval(silhouette_by_point, rng, bootstrap),
        "knn_k": 5,
        "knn_macro_f1": float(macro_f1),
        "knn_macro_f1_95ci": [float(np.percentile(macro_boot, 2.5)),
                               float(np.percentile(macro_boot, 97.5))],
        "knn_neighborhood_purity": float(purity_by_point.mean()),
        "knn_neighborhood_purity_95ci": confidence_interval(purity_by_point, rng, bootstrap),
    }
    raw = {
        "silhouette": silhouette_by_point,
        "purity": purity_by_point,
        "predictions": predictions,
    }
    return metrics, raw


def permutation_difference(
    embeddings: np.ndarray,
    action_labels: np.ndarray,
    semantic_labels: np.ndarray,
    observed_silhouette_difference: float,
    observed_purity_difference: float,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    neighbors = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    neighbor_ids = neighbors.fit(embeddings).kneighbors(return_distance=False)

    def purity(labels: np.ndarray) -> float:
        values = [
            np.mean(labels[row[row != index][:5]] == labels[index])
            for index, row in enumerate(neighbor_ids)
        ]
        return float(np.mean(values))

    rng = np.random.default_rng(seed)
    silhouette_extreme = 0
    purity_extreme = 0
    for _ in range(permutations):
        action_perm = rng.permutation(action_labels)
        semantic_perm = rng.permutation(semantic_labels)
        silhouette_diff = float(
            silhouette_samples(embeddings, action_perm, metric="cosine").mean()
            - silhouette_samples(embeddings, semantic_perm, metric="cosine").mean()
        )
        purity_diff = purity(action_perm) - purity(semantic_perm)
        silhouette_extreme += int(silhouette_diff >= observed_silhouette_difference)
        purity_extreme += int(purity_diff >= observed_purity_difference)
    return {
        "permutations": permutations,
        "alternative": "action_minus_semantic > null difference from independently permuted labels",
        "silhouette_difference_p_one_sided": (silhouette_extreme + 1) / (permutations + 1),
        "purity_difference_p_one_sided": (purity_extreme + 1) / (permutations + 1),
    }


def plot_embeddings(frame: pd.DataFrame, output: Path, seed: int) -> None:
    colors = ["#2166ac", "#b2182b", "#1b7837", "#777777"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=220)
    for axis, column, title in (
        (axes[0], "action_label", "Action-type labels"),
        (axes[1], "semantic_label", "Matched semantic labels"),
    ):
        for color, (label, subset) in zip(colors, sorted(frame.groupby(column))):
            axis.scatter(subset["tsne_x"], subset["tsne_y"], s=18, alpha=0.72,
                         color=color, label=f"{label} (n={len(subset)})", linewidths=0)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"Frozen LEGR V3 graph embeddings — Campaign V4 15-tool gallery (t-SNE seed={seed})")
    fig.tight_layout()
    fig.savefig(output / "v3_action_vs_semantic_tsne.png", bbox_inches="tight")
    fig.savefig(output / "v3_action_vs_semantic_tsne.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = checkpoint_manifest()[15][2]["checkpoint"]
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--output", default="artifacts/prepaper_v3_action_semantic_s42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    from legr_tool_count import apply_tool_count_override

    apply_tool_count_override(15)
    from data.tool_registry import TOOL_TO_CATEGORY, TOOLS_15
    from encoders import resolve_graph_encoder_settings
    from eval import _load_model_and_tokenizer, encode_all_dags

    output = create_output_dir(args.output)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = checkpoint_manifest()[15][2]
    metadata = validate_checkpoint_metadata(expected, checkpoint_payload, 15)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    semantic_map = reverse_mapping(SEMANTIC_GROUPS)
    if set(semantic_map) != set(TOOLS_15):
        raise AssertionError("Matched semantic taxonomy does not cover Campaign V4 TOOLS_15 exactly")
    action_map = {tool: ACTION_DISPLAY[TOOL_TO_CATEGORY[tool].value] for tool in TOOLS_15}
    for semantic_group, tools in SEMANTIC_GROUPS.items():
        composition = Counter(action_map[tool] for tool in tools)
        if sorted(composition.values()) != [1, 2, 2]:
            raise AssertionError(f"{semantic_group} action composition is not 2/2/1: {composition}")

    gallery_frame, _ = full_gallery_frame(15)
    gallery = build_gallery_dataset(gallery_frame)
    model, cfg, _ = _load_model_and_tokenizer(
        str(checkpoint_path), device, dataset_csv=str(campaign_paths(15)["candidate"])
    )
    model.eval()
    _, _, bidirectional = resolve_graph_encoder_settings(cfg)
    embeddings_tensor = encode_all_dags(model, gallery, device, batch_size=64,
                                         bidirectional=bidirectional)
    embeddings = embeddings_tensor.numpy()
    if embeddings.shape != (322, 256) or not np.isfinite(embeddings).all():
        raise AssertionError(f"Unexpected V3 embeddings: shape={embeddings.shape}")

    rows = []
    for index in range(gallery.num_unique_dags):
        graph = gallery.get_unique_dag(index)
        tools = [graph.nodes[node]["tool"] for node in sorted(graph.nodes())]
        rows.append({
            "gallery_index": index,
            "dag_id": gallery_frame.iloc[index]["dag_id"],
            "tools": ";".join(tools),
            "num_nodes": len(tools),
            "action_label": unique_plurality_label(tools, action_map),
            "semantic_label": unique_plurality_label(tools, semantic_map),
        })
    frame = pd.DataFrame(rows)
    action_labels = frame["action_label"].to_numpy()
    semantic_labels = frame["semantic_label"].to_numpy()
    if len(set(action_labels)) != 4 or len(set(semantic_labels)) != 4:
        raise AssertionError("Expected three base labels plus Mixed for both taxonomies")

    action_metrics, action_raw = label_diagnostics(
        embeddings, action_labels, args.seed, args.bootstrap
    )
    semantic_metrics, semantic_raw = label_diagnostics(
        embeddings, semantic_labels, args.seed + 1, args.bootstrap
    )
    differences = {
        "silhouette_action_minus_semantic": action_metrics["silhouette_cosine"]
        - semantic_metrics["silhouette_cosine"],
        "knn_macro_f1_action_minus_semantic": action_metrics["knn_macro_f1"]
        - semantic_metrics["knn_macro_f1"],
        "purity_action_minus_semantic": action_metrics["knn_neighborhood_purity"]
        - semantic_metrics["knn_neighborhood_purity"],
    }
    permutation = permutation_difference(
        embeddings, action_labels, semantic_labels,
        differences["silhouette_action_minus_semantic"],
        differences["purity_action_minus_semantic"],
        args.permutations, args.seed,
    )
    all_higher = all(value > 0 for value in differences.values())
    significant = (
        permutation["silhouette_difference_p_one_sided"] <= 0.05
        and permutation["purity_difference_p_one_sided"] <= 0.05
    )
    evidence = "STRONGER_ACTION_SEPARATION" if all_higher and significant else (
        "MIXED" if sum(value > 0 for value in differences.values()) >= 2 else "NO_ACTION_ADVANTAGE"
    )

    reducer = TSNE(
        n_components=2, perplexity=30.0, random_state=args.seed,
        init="pca", learning_rate="auto",
    )
    coordinates = reducer.fit_transform(embeddings)
    frame["tsne_x"] = coordinates[:, 0]
    frame["tsne_y"] = coordinates[:, 1]
    frame["action_silhouette"] = action_raw["silhouette"]
    frame["semantic_silhouette"] = semantic_raw["silhouette"]
    frame["action_knn_purity"] = action_raw["purity"]
    frame["semantic_knn_purity"] = semantic_raw["purity"]

    np.save(output / "v3_graph_embeddings.npy", embeddings)
    np.save(output / "tsne_coordinates.npy", coordinates)
    frame.to_csv(output / "plotting_data.csv", index=False)
    plot_embeddings(frame, output, args.seed)
    mapping_payload = {
        "action_tool_to_category": action_map,
        "semantic_groups": {key: list(value) for key, value in SEMANTIC_GROUPS.items()},
        "semantic_tool_to_category": semantic_map,
        "dag_rule": "unique plurality of node categories; exact ties are Mixed",
        "rule_deviation_from_preregistered_plan": (
            "Strict majority produced zero orchestration-labelled DAGs (95 retrieval, "
            "86 modification, 141 mixed, 0 orchestration). Unique plurality is used "
            "to preserve the requested equal observed category count."
        ),
        "matched_control": "Each semantic group has five tools: 2 retrieval, 2 modification, 1 orchestration.",
    }
    write_json(output / "category_mappings.json", mapping_payload)
    metrics = {
        "dataset": "Campaign V4 15-tool full gallery",
        "gallery_size": 322,
        "embedding_space": "frozen V3 graph encoder output (256d)",
        "tsne_used_for_metrics": False,
        "checkpoint": repo_relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "action": action_metrics,
        "semantic": semantic_metrics,
        "differences": differences,
        "permutation_test": permutation,
        "evidence": evidence,
        "environment": environment_snapshot(device),
    }
    write_json(output / "metrics.json", metrics)
    (output / "report.md").write_text(
        "# Campaign V4 V3 action-versus-semantic geometry\n\n"
        f"**Evidence classification:** {evidence}\n\n"
        "The same 322 frozen V3 DAG embeddings and the same t-SNE coordinates are used in both panels. "
        "All quantitative diagnostics are computed in the original 256-dimensional cosine space.\n\n"
        "```json\n" + json.dumps({"action": action_metrics, "semantic": semantic_metrics,
                                    "differences": differences, "permutation_test": permutation},
                                   indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (output / "reproduce.txt").write_text(
        f"{sys.executable} scripts/analyze_v3_action_semantic_geometry.py "
        f"--checkpoint {repo_relative(checkpoint_path)} --device {args.device} "
        f"--seed {args.seed} --bootstrap {args.bootstrap} --permutations {args.permutations} "
        f"--output {repo_relative(output)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"evidence": evidence, "action": action_metrics,
                      "semantic": semantic_metrics, "differences": differences,
                      "permutation_test": permutation}, indent=2))


if __name__ == "__main__":
    main()
