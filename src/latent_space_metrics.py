"""Quantitative cluster diagnostics in original embedding space (not t-SNE)."""

from __future__ import annotations

import numpy as np

from action_type_mapping import VALID_GROUPS

KNN_K = 5


def classify_support(
    silhouette: float | None,
    ami: float | None,
    purity: float,
    majority_frac: float,
) -> str:
    """Map diagnostics to STRONG / WEAK/PARTIAL / NO SUPPORT."""
    above_chance_purity = purity >= min(1.0, majority_frac + 0.10)
    sil_ok = silhouette is not None and silhouette >= 0.25
    ami_ok = ami is not None and ami >= 0.30
    sil_weak = silhouette is not None and silhouette >= 0.10
    ami_weak = ami is not None and ami >= 0.10
    if sil_ok and ami_ok and purity >= 0.70:
        return "STRONG SUPPORT"
    if sil_weak or ami_weak or above_chance_purity:
        return "WEAK/PARTIAL SUPPORT"
    return "NO SUPPORT"


def embedding_diagnostics(
    embeddings: np.ndarray,
    labels: list[str],
    knn_k: int = KNN_K,
) -> dict:
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors

    y = np.array(labels)
    n = len(y)
    unique = [g for g in VALID_GROUPS if (y == g).sum() >= 2]
    majority_frac = float(max((y == g).mean() for g in set(y))) if n else 0.0
    silhouette = None
    if len(unique) >= 2 and n >= 4:
        mask = np.isin(y, unique)
        if mask.sum() >= 4 and len(set(y[mask])) >= 2:
            silhouette = float(silhouette_score(embeddings[mask], y[mask]))

    k = max(1, min(knn_k, n - 1)) if n > 1 else 1
    purity = 0.0
    ami = None
    if n > 1 and k >= 1:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(embeddings)
        idx = nn.kneighbors(embeddings, return_distance=False)[:, 1:]
        hits = []
        knn_labels = []
        for i in range(n):
            neigh = y[idx[i]]
            hits.append(float((neigh == y[i]).mean()))
            knn_labels.append(neigh[0] if len(neigh) else y[i])
        purity = float(np.mean(hits)) if hits else 0.0
        ami = float(adjusted_mutual_info_score(y, knn_labels))
    return {
        "n": int(n),
        "label_counts": {g: int((y == g).sum()) for g in VALID_GROUPS},
        "majority_frac": majority_frac,
        "silhouette": silhouette,
        "neighborhood_purity_k": k,
        "neighborhood_purity": purity,
        "ami_vs_1nn": ami,
        "evidence": classify_support(silhouette, ami, purity, majority_frac),
        "embedding_space": "original",
        "tsne_not_used_for_metrics": True,
    }
