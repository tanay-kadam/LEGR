"""Deterministic functional-category audit for trained LEGR graph embeddings.

This module reads Campaign-v4 and existing checkpoints without modifying them.
All labels and diagnostics are derived transiently and written to a new artifact
directory by :func:`run_audit`.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import ModelConfig
from .data import (
    ResearchDataset,
    UniqueGraphDataset,
    make_collate,
    parse_edges,
    parse_tools,
)
from .model import LEGRResearchModel
from .structures import graph_key


ACTION_CATEGORY_MAP = {
    "DATA_RETRIEVAL": "read",
    "STATE_MODIFICATION": "edit",
    "ORCHESTRATION": "orchestrate",
}
PRIMARY_LABELS = ("read", "edit", "orchestrate")
ALL_LABELS = (*PRIMARY_LABELS, "mixed")
DEFAULT_SPLIT_FILES = (
    "train.csv",
    "dev.csv",
    "test_indomain.csv",
    "test_topology_heldout.csv",
    "candidate_corpus.csv",
)
EXPECTED_15_TOOL_COUNTS = {
    "read": 232,
    "edit": 223,
    "orchestrate": 32,
    "mixed": 164,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_snapshot(paths: Iterable[Path], root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    for path in sorted(set(files), key=lambda item: str(item).lower()):
        try:
            label = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            label = str(path.resolve()).replace("\\", "/")
        result[label] = sha256_file(path)
    return result


def load_action_registry(registry_path: Path) -> tuple[list[str], dict[str, str]]:
    frame = pd.read_csv(registry_path)
    required = {"tool_name", "category", "index_15"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Registry is missing columns: {sorted(missing)}")
    unknown = sorted(set(frame["category"]) - set(ACTION_CATEGORY_MAP))
    if unknown:
        raise ValueError(f"Unrecognized registry categories: {unknown}")
    label_by_tool = {
        str(row.tool_name): ACTION_CATEGORY_MAP[str(row.category)]
        for row in frame.itertuples(index=False)
    }
    tier = frame[frame["index_15"].notna()].copy()
    tier["index_15"] = tier["index_15"].astype(int)
    tier = tier.sort_values("index_15")
    vocabulary = tier["tool_name"].astype(str).tolist()
    if len(vocabulary) != 15 or len(set(vocabulary)) != 15:
        raise ValueError(f"Expected 15 unique tier-15 tools, found {len(vocabulary)}")
    return vocabulary, label_by_tool


def dominant_action_label(
    tools: Sequence[str], label_by_tool: dict[str, str]
) -> tuple[str, dict[str, int]]:
    """Return the unique plurality label, or ``mixed`` for an exact tie."""
    counts = {label: 0 for label in PRIMARY_LABELS}
    missing = sorted(set(tools) - set(label_by_tool))
    if missing:
        raise KeyError(f"Tools missing from action registry: {missing}")
    for tool in tools:
        counts[label_by_tool[tool]] += 1
    maximum = max(counts.values(), default=0)
    winners = [label for label in PRIMARY_LABELS if counts[label] == maximum]
    return (winners[0] if len(winners) == 1 else "mixed"), counts


def _raw_graph_metadata(csv_paths: Sequence[Path]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for path in csv_paths:
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            tools = parse_tools(row.get("tools"))
            edges = parse_edges(row.get("edges"))
            key = graph_key(tools, edges)
            canonical = str(row.get("canonical_dag_hash", ""))
            current = metadata.setdefault(
                key,
                {
                    "canonical_dag_hash": canonical,
                    "source_files": set(),
                    "declared_splits": set(),
                    "tool_edge_signature": (tuple(tools), tuple(sorted(edges))),
                },
            )
            if current["tool_edge_signature"] != (tuple(tools), tuple(sorted(edges))):
                raise ValueError(f"Inconsistent graph rows for graph key {key}")
            if canonical and current["canonical_dag_hash"] not in {"", canonical}:
                raise ValueError(f"Conflicting canonical hashes for graph key {key}")
            if canonical:
                current["canonical_dag_hash"] = canonical
            current["source_files"].add(path.name)
            current["declared_splits"].add(str(row.get("split", "")))
    return metadata


def build_analysis_dataset(
    data_root: Path,
    structure_kind: str,
) -> tuple[ResearchDataset, UniqueGraphDataset, pd.DataFrame, list[str], list[Path]]:
    registry_path = data_root / "tool_registry.csv"
    vocabulary, label_by_tool = load_action_registry(registry_path)
    tier_root = data_root / "campaign_v4_15tools"
    csv_paths = [tier_root / name for name in DEFAULT_SPLIT_FILES]
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = _raw_graph_metadata(csv_paths)
    dataset = ResearchDataset(csv_paths, vocabulary, structure_kind=structure_kind)
    unique = UniqueGraphDataset(dataset)
    rows = []
    canonical_seen: set[str] = set()
    for sample in unique.samples:
        signature = sample.signature
        info = metadata.get(signature.dag_key)
        if info is None:
            raise ValueError(f"No raw metadata for graph key {signature.dag_key}")
        canonical = info["canonical_dag_hash"]
        if not canonical:
            raise ValueError(f"Missing canonical hash for graph key {signature.dag_key}")
        if canonical in canonical_seen:
            raise ValueError(f"Canonical DAG hash was not unique after graph deduplication: {canonical}")
        canonical_seen.add(canonical)
        label, counts = dominant_action_label(signature.tools, label_by_tool)
        rows.append(
            {
                "canonical_dag_hash": canonical,
                "graph_key": signature.dag_key,
                "dag_text": sample.dag_text,
                "tools": ";".join(signature.tools),
                "edges": ";".join(f"{left}->{right}" for left, right in signature.edges),
                "tool_count": len(signature.tools),
                "edge_count": len(signature.edges),
                "read_count": counts["read"],
                "edit_count": counts["edit"],
                "orchestrate_count": counts["orchestrate"],
                "action_label": label,
                "included_primary": label in PRIMARY_LABELS,
                "source_files": ";".join(sorted(info["source_files"])),
                "declared_splits": ";".join(sorted(info["declared_splits"])),
            }
        )
    graph_frame = pd.DataFrame(rows)
    if graph_frame["canonical_dag_hash"].duplicated().any():
        raise ValueError("Duplicate canonical DAG hashes remain in the analysis frame")
    return dataset, unique, graph_frame, vocabulary, csv_paths


def load_research_model(
    checkpoint: Path,
    vocabulary: list[str],
    device: torch.device,
) -> tuple[LEGRResearchModel, ModelConfig, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_data = payload.get("config", {}).get("model")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint does not contain config.model")
    config = ModelConfig(**config_data)
    model = LEGRResearchModel(config, vocabulary)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    if model.semantic_expert is None:
        raise ValueError("The requested SBERT control is absent from this checkpoint")
    model.to(device)
    model.eval()
    return model, config, payload


@torch.no_grad()
def extract_embeddings(
    model: LEGRResearchModel,
    unique_dataset: UniqueGraphDataset,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int = 128,
) -> tuple[dict[str, np.ndarray], float]:
    loader = DataLoader(
        unique_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length=max_length),
        num_workers=0,
    )
    collected: dict[str, list[torch.Tensor]] = {
        "gps_adapter": [],
        "v3_graph": [],
        "sbert_document": [],
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        graph_x = batch["graph_x"].to(device)
        edge_index = batch["graph_edge_index"].to(device)
        graph_batch = batch["graph_batch"].to(device)
        structural = batch["graph_struct_x"].to(device)
        node_features = model.base_legr._node_features_from_tool_ids(graph_x)
        gps, _ = model.graph_adapter(node_features, structural, edge_index, graph_batch)
        v3 = model.base_legr.encode_graph(node_features, edge_index, graph_batch, topo_pos=None)
        sbert = model.semantic_expert.encode_document(
            batch["doc_input_ids"].to(device),
            batch["doc_attention_mask"].to(device),
        )
        collected["gps_adapter"].append(gps.cpu())
        collected["v3_graph"].append(v3.cpu())
        collected["sbert_document"].append(sbert.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    arrays = {
        name: torch.cat(values).numpy().astype(np.float32, copy=False)
        for name, values in collected.items()
    }
    expected = len(unique_dataset)
    for name, array in arrays.items():
        if array.shape != (expected, model.config.embed_dim):
            raise ValueError(f"Unexpected {name} shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {name} embeddings")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise ValueError(f"{name} embeddings are not L2-normalized")
    return arrays, elapsed


def _knn_predictions(
    distances: np.ndarray,
    encoded_labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    neighbors: int,
) -> np.ndarray:
    classes = int(encoded_labels.max()) + 1
    order = np.argsort(distances, axis=1)
    predictions = np.full(len(encoded_labels), -1, dtype=np.int64)
    for train, test in folds:
        train_mask = np.zeros(len(encoded_labels), dtype=bool)
        train_mask[train] = True
        for index in test:
            candidates = order[index][train_mask[order[index]]][:neighbors]
            weights = 1.0 / np.maximum(distances[index, candidates], 1e-6)
            votes = np.bincount(
                encoded_labels[candidates], weights=weights, minlength=classes
            )
            predictions[index] = int(np.argmax(votes))
    if (predictions < 0).any():
        raise RuntimeError("Cross-validation did not predict every sample")
    return predictions


def _distance_gap(distances: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    row, column = np.triu_indices(len(labels), k=1)
    same = labels[row] == labels[column]
    within = float(distances[row[same], column[same]].mean())
    between = float(distances[row[~same], column[~same]].mean())
    return within, between, between - within


def _stratified_bootstrap_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    purity_hits: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    f1_values = []
    purity_values = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        f1_values.append(f1_score(labels[sampled], predictions[sampled], average="macro"))
        purity_values.append(
            np.mean([purity_hits[sampled[labels[sampled] == value]].mean() for value in np.unique(labels)])
        )
    return {
        "knn_macro_f1_95ci": [float(v) for v in np.quantile(f1_values, [0.025, 0.975])],
        "balanced_neighborhood_purity_95ci": [
            float(v) for v in np.quantile(purity_values, [0.025, 0.975])
        ],
    }


def embedding_diagnostics(
    embeddings: np.ndarray,
    labels: Sequence[str],
    seed: int,
    permutations: int,
    bootstraps: int,
    neighbors: int = 5,
) -> dict:
    from sklearn.metrics import f1_score, silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.model_selection import StratifiedKFold

    label_to_id = {label: index for index, label in enumerate(PRIMARY_LABELS)}
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    counts = np.bincount(encoded, minlength=len(PRIMARY_LABELS))
    if (counts < 5).any():
        raise ValueError(f"Every primary class needs at least five graphs; counts={counts.tolist()}")
    distances = cosine_distances(embeddings).astype(np.float32)
    np.fill_diagonal(distances, 0.0)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(
            embeddings, encoded
        )
    )
    predictions = _knn_predictions(distances, encoded, folds, neighbors)
    macro_f1 = float(f1_score(encoded, predictions, average="macro"))
    neighbor_order = np.argsort(distances, axis=1)[:, 1 : neighbors + 1]
    purity_hits = (encoded[neighbor_order] == encoded[:, None]).mean(axis=1)
    class_purity = {
        label: float(purity_hits[encoded == index].mean())
        for index, label in enumerate(PRIMARY_LABELS)
    }
    balanced_purity = float(np.mean(list(class_purity.values())))
    silhouette = float(silhouette_score(distances, encoded, metric="precomputed"))
    within, between, gap = _distance_gap(distances, encoded)

    rng = np.random.default_rng(seed)
    null_f1 = np.empty(permutations, dtype=np.float64)
    null_silhouette = np.empty(permutations, dtype=np.float64)
    null_gap = np.empty(permutations, dtype=np.float64)
    for iteration in range(permutations):
        permuted = rng.permutation(encoded)
        permuted_predictions = _knn_predictions(distances, permuted, folds, neighbors)
        null_f1[iteration] = f1_score(permuted, permuted_predictions, average="macro")
        null_silhouette[iteration] = silhouette_score(
            distances, permuted, metric="precomputed"
        )
        null_gap[iteration] = _distance_gap(distances, permuted)[2]

    confidence = _stratified_bootstrap_ci(
        encoded, predictions, purity_hits, bootstraps, seed + 1
    )
    probability = lambda values, observed: float(
        (1 + np.count_nonzero(values >= observed)) / (len(values) + 1)
    )
    chance_macro_f1 = 1.0 / len(PRIMARY_LABELS)
    corroborated = bool(
        probability(null_f1, macro_f1) < 0.05
        and probability(null_gap, gap) < 0.05
        and confidence["knn_macro_f1_95ci"][0] > chance_macro_f1
    )
    return {
        "n": int(len(labels)),
        "label_counts": {
            label: int(counts[index]) for index, label in enumerate(PRIMARY_LABELS)
        },
        "cosine_silhouette": silhouette,
        "silhouette_permutation_p": probability(null_silhouette, silhouette),
        "knn_neighbors": neighbors,
        "knn_cv_macro_f1": macro_f1,
        "knn_macro_f1_chance": chance_macro_f1,
        "knn_macro_f1_permutation_mean": float(null_f1.mean()),
        "knn_macro_f1_permutation_95pct": float(np.quantile(null_f1, 0.95)),
        "knn_macro_f1_permutation_p": probability(null_f1, macro_f1),
        "neighborhood_purity_by_class": class_purity,
        "balanced_neighborhood_purity": balanced_purity,
        "mean_within_class_cosine_distance": within,
        "mean_between_class_cosine_distance": between,
        "between_minus_within_distance": gap,
        "distance_gap_permutation_mean": float(null_gap.mean()),
        "distance_gap_permutation_95pct": float(np.quantile(null_gap, 0.95)),
        "distance_gap_permutation_p": probability(null_gap, gap),
        "permutations": permutations,
        "bootstraps": bootstraps,
        **confidence,
        "corroborates_functional_clustering": corroborated,
    }


def compute_projections(
    embeddings: dict[str, np.ndarray], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    projections = {}
    for name, values in embeddings.items():
        pca = PCA(n_components=2, random_state=seed).fit_transform(values)
        perplexity = min(30.0, max(5.0, (len(values) - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(values)
        projections[name] = {
            "pca": pca.astype(np.float32),
            "tsne": tsne.astype(np.float32),
        }
    return projections


def save_projection_plot(
    projections: dict[str, dict[str, np.ndarray]], labels: Sequence[str], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"read": "#377eb8", "edit": "#e41a1c", "orchestrate": "#4daf4a"}
    display = {
        "gps_adapter": "Winning GPS adapter",
        "v3_graph": "Inherited V3 graph tower",
        "sbert_document": "SBERT document tower",
    }
    fig, axes = plt.subplots(len(projections), 2, figsize=(12, 14), dpi=180)
    labels_array = np.asarray(labels)
    for row, (name, methods) in enumerate(projections.items()):
        for column, method in enumerate(("pca", "tsne")):
            ax = axes[row, column]
            coordinates = methods[method]
            for label in PRIMARY_LABELS:
                mask = labels_array == label
                ax.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=18,
                    alpha=0.72,
                    color=colors[label],
                    label=label,
                    linewidths=0,
                )
            ax.set_title(f"{display[name]} — {method.upper()}")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and column == 1:
                ax.legend(frameon=False, loc="best")
    fig.suptitle("Campaign-v4 functional labels (seed 42; non-tied graphs)", y=0.995)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _format_number(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    output_dir: Path,
    metrics: dict[str, dict],
    counts: dict[str, int],
    run_metadata: dict,
    integrity: dict,
) -> None:
    winner = metrics["gps_adapter"]
    status = "CORROBORATED" if winner["corroborates_functional_clustering"] else "NOT CORROBORATED"
    rows = []
    names = {
        "gps_adapter": "Winning GPS adapter",
        "v3_graph": "Inherited V3 graph",
        "sbert_document": "SBERT document control",
    }
    for key in ("gps_adapter", "v3_graph", "sbert_document"):
        item = metrics[key]
        ci = item["knn_macro_f1_95ci"]
        rows.append(
            "| {name} | {sil} | {sp} | {f1} [{lo}, {hi}] | {fp} | {purity} | {gap} | {gp} |".format(
                name=names[key],
                sil=_format_number(item["cosine_silhouette"]),
                sp=_format_number(item["silhouette_permutation_p"]),
                f1=_format_number(item["knn_cv_macro_f1"]),
                lo=_format_number(ci[0]),
                hi=_format_number(ci[1]),
                fp=_format_number(item["knn_macro_f1_permutation_p"]),
                purity=_format_number(item["balanced_neighborhood_purity"]),
                gap=_format_number(item["between_minus_within_distance"]),
                gp=_format_number(item["distance_gap_permutation_p"]),
            )
        )
    conclusion = (
        "The seed-42 winning graph embedding passes the predeclared corroboration rule: "
        "its cross-validated macro-F1 and distance separation exceed shuffled-label "
        "performance, and the macro-F1 bootstrap lower bound is above 1/3 chance."
        if status == "CORROBORATED"
        else
        "The seed-42 winning graph embedding does not pass every part of the predeclared "
        "corroboration rule. The functional-categorization claim is therefore not supported "
        "by this audit."
    )
    lines = [
        "# LEGR Functional Clustering Audit — Seed 42",
        "",
        f"**Result: {status}**",
        "",
        conclusion,
        "",
        "## Population and labels",
        "",
        f"The audit deduplicated **{sum(counts.values())}** Campaign-v4 15-tool DAGs by canonical graph identity. "
        f"The primary three-class analysis contains **{sum(counts[label] for label in PRIMARY_LABELS)}** graphs; "
        f"**{counts['mixed']}** exact-plurality ties are retained in `graph_labels.csv` but excluded from metrics and plots.",
        "",
        "| Label | Graphs |",
        "|---|---:|",
        f"| read | {counts['read']} |",
        f"| edit | {counts['edit']} |",
        f"| orchestrate | {counts['orchestrate']} |",
        f"| mixed (excluded) | {counts['mixed']} |",
        "",
        "Labels were computed without an LLM: Campaign-v4 registry category `DATA_RETRIEVAL` maps to `read`, "
        "`STATE_MODIFICATION` to `edit`, and `ORCHESTRATION` to `orchestrate`. A graph receives the unique "
        "most frequent label among its tools; exact ties are `mixed`.",
        "",
        "## Original-space diagnostics",
        "",
        "| Representation | Cosine silhouette | Silhouette p | 5-NN macro-F1 (95% CI) | F1 p | Balanced purity | Distance gap | Gap p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Permutation p-values use shuffled action labels. The distance gap is mean between-class minus "
        "mean within-class cosine distance; positive values indicate separation. The predeclared criterion "
        "requires GPS-adapter 5-NN macro-F1 p < 0.05, distance-gap p < 0.05, and a macro-F1 bootstrap "
        "95% lower bound above the three-class chance value of 1/3.",
        "",
        "## Representation definitions",
        "",
        "- **Winning GPS adapter:** the trained degree-encoded, directed GPS graph adapter with dual-attention readout.",
        "- **Inherited V3 graph:** the V3 graph-tower output inside the same composite checkpoint.",
        "- **SBERT document control:** the frozen SBERT expert's embedding of the serialized DAG text. This is the semantic control, not a graph encoder.",
        "",
        "## Integrity and interpretation",
        "",
        f"Protected input integrity: **{'PASS' if integrity['unchanged'] else 'FAIL'}** "
        f"({integrity['file_count']} files checked before and after).",
        "",
        "PCA and t-SNE are visual summaries only. All evidence decisions use the original 256-dimensional "
        "cosine space. Because the population includes training graphs, this audit concerns learned representation "
        "geometry rather than zero-shot generalization. The orchestrate class is smaller than read/edit, so macro-F1, "
        "class-balanced purity, stratified folds, and stratified bootstrap intervals are reported. Finally, these labels "
        "are derived from the same tool identities available to the encoders; clustering supports functional organization "
        "but does not by itself prove causal or independently learned functional reasoning.",
        "",
        "## Run metadata",
        "",
        f"- Checkpoint: `{run_metadata['checkpoint']}`",
        f"- Device: `{run_metadata['device']}`",
        f"- Seed: {run_metadata['seed']}",
        f"- Embedding extraction: {run_metadata['embedding_seconds']:.2f} seconds",
        f"- Total runtime: {run_metadata['total_seconds']:.2f} seconds",
        f"- Permutations/bootstrap samples: {run_metadata['permutations']}/{run_metadata['bootstraps']}",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    root: Path,
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int = 64,
    seed: int = 42,
    permutations: int = 1000,
    bootstraps: int = 2000,
    verify_determinism: bool = True,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {output_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.perf_counter()
    seed_everything(seed)
    inherited_checkpoints = [
        root / "artifacts/campaign_v4/results/legr_setgnn_tied_no_ged_15t_s42/best_model.pt",
        root / "artifacts/campaign_v4/results/sbert_ft_ged_15t_s42/best_model.pt",
    ]
    protected = [data_root, checkpoint, *[p for p in inherited_checkpoints if p.is_file()]]
    hashes_before = hash_snapshot(protected, root)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_data = payload.get("config", {}).get("model")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint does not contain config.model")
    model_config = ModelConfig(**config_data)
    dataset, unique, frame, vocabulary, csv_paths = build_analysis_dataset(
        data_root, model_config.structure_kind
    )
    counts = {
        label: int((frame["action_label"] == label).sum()) for label in ALL_LABELS
    }
    if counts != EXPECTED_15_TOOL_COUNTS:
        raise ValueError(
            f"Campaign-v4 graph-label counts changed: expected {EXPECTED_15_TOOL_COUNTS}, got {counts}"
        )
    if len(unique) != len(frame) or len(frame) != sum(counts.values()):
        raise ValueError("Graph dataset and label frame are not aligned")

    model, model_config, _ = load_research_model(checkpoint, vocabulary, device)
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model, local_files_only=True)
    embeddings, embedding_seconds = extract_embeddings(
        model, unique, tokenizer, device, batch_size=batch_size
    )
    determinism = {"checked": verify_determinism, "passed": None, "max_abs_difference": None}
    if verify_determinism:
        repeated, repeated_seconds = extract_embeddings(
            model, unique, tokenizer, device, batch_size=batch_size
        )
        maximum = max(
            float(np.max(np.abs(embeddings[name] - repeated[name]))) for name in embeddings
        )
        determinism.update(
            {"passed": bool(maximum <= 1e-6), "max_abs_difference": maximum, "repeat_seconds": repeated_seconds}
        )
        if not determinism["passed"]:
            raise RuntimeError(f"Embedding determinism check failed; max difference={maximum}")

    primary_mask = frame["included_primary"].to_numpy(dtype=bool)
    primary_labels = frame.loc[primary_mask, "action_label"].astype(str).tolist()
    primary_embeddings = {name: values[primary_mask] for name, values in embeddings.items()}
    metrics = {
        name: embedding_diagnostics(
            values,
            primary_labels,
            seed=seed,
            permutations=permutations,
            bootstraps=bootstraps,
        )
        for name, values in primary_embeddings.items()
    }
    projections = compute_projections(primary_embeddings, seed)

    hashes_after = hash_snapshot(protected, root)
    changed = sorted(
        key for key in set(hashes_before) | set(hashes_after)
        if hashes_before.get(key) != hashes_after.get(key)
    )
    integrity = {
        "unchanged": not changed,
        "file_count": len(hashes_before),
        "changed_files": changed,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }
    if changed:
        raise RuntimeError(f"Protected input integrity failed: {changed}")

    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output_dir / "graph_labels.csv", index=False)
    np.savez_compressed(output_dir / "embeddings.npz", **embeddings)
    projection_columns = frame.loc[primary_mask, ["canonical_dag_hash", "action_label"]].reset_index(drop=True)
    projection_arrays = {}
    for name, methods in projections.items():
        for method, coordinates in methods.items():
            projection_columns[f"{name}_{method}_x"] = coordinates[:, 0]
            projection_columns[f"{name}_{method}_y"] = coordinates[:, 1]
            projection_arrays[f"{name}_{method}"] = coordinates
    projection_columns.to_csv(output_dir / "projection_coordinates.csv", index=False)
    np.savez_compressed(output_dir / "projection_coordinates.npz", **projection_arrays)
    save_projection_plot(projections, primary_labels, output_dir / "functional_clusters.png")

    total_seconds = time.perf_counter() - started
    metadata = {
        "checkpoint": str(checkpoint.resolve()).replace("\\", "/"),
        "data_root": str(data_root.resolve()).replace("\\", "/"),
        "dataset_csvs": [str(path.resolve()).replace("\\", "/") for path in csv_paths],
        "device": str(device),
        "seed": seed,
        "batch_size": batch_size,
        "permutations": permutations,
        "bootstraps": bootstraps,
        "embedding_seconds": embedding_seconds,
        "total_seconds": total_seconds,
        "model_config": asdict(model_config),
        "unique_graphs": len(frame),
        "primary_graphs": int(primary_mask.sum()),
        "label_counts": counts,
        "determinism": determinism,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(output_dir, metrics, counts, metadata, integrity)
    return {
        "status": "complete",
        "corroborated": metrics["gps_adapter"]["corroborates_functional_clustering"],
        "output": str(output_dir),
        "counts": counts,
        "total_seconds": total_seconds,
    }

