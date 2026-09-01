"""Prove SBERT-FT is a tool-bag retriever; compare same-toolset ranking to LEGR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _unique_dags(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [f.drop_duplicates("dag_id") for f in frames]
    return pd.concat(parts, ignore_index=True).drop_duplicates("dag_id")


def _recall_at_k(sim: torch.Tensor, gt: list[int], ks=(1, 3, 5)) -> dict:
    n = sim.size(0)
    out = {}
    for k in ks:
        pred = sim.topk(min(k, sim.size(1)), dim=1).indices.tolist()
        hits = sum(g in row for g, row in zip(gt, pred))
        out[f"recall@{k}"] = round(hits / n, 4)
    return out


def _reverse_dag_text(text: str) -> str:
    if "->" not in text:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    flipped = []
    for p in parts:
        if "->" not in p:
            flipped.append(p)
            continue
        a, b = p.split("->", 1)
        flipped.append(f"{b.strip()} -> {a.strip()}")
    return ", ".join(sorted(flipped))


def _pairwise_cos(emb: torch.Tensor, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    a = emb[[i for i, _ in pairs]]
    b = emb[[j for _, j in pairs]]
    return float(F.cosine_similarity(a, b).mean().item())


def _same_toolset_recall(sim: torch.Tensor, gt: list[int], groups: list[list[int]]) -> dict:
    """Rank gold only among gallery DAGs that share its toolset."""
    hits1 = 0
    n_chance = 0.0
    n = 0
    n_with_twins = 0
    for qi, g in enumerate(gt):
        pool = groups[g]
        if len(pool) < 2:
            continue
        n += 1
        n_with_twins += 1
        scores = sim[qi, pool]
        order = [pool[i] for i in scores.argsort(descending=True).tolist()]
        hits1 += int(order[0] == g)
        n_chance += 1.0 / len(pool)
    return {
        "n_queries_with_twins": n_with_twins,
        "same_toolset_r@1": round(hits1 / max(n, 1), 4),
        "chance_same_toolset_r@1": round(n_chance / max(n, 1), 4),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier_dir = ROOT / "data" / "campaign_v4" / "campaign_v4_15tools"
    test = pd.read_csv(tier_dir / "test_topology_heldout.csv")
    corpus = pd.read_csv(tier_dir / "candidate_corpus.csv")
    gallery = _unique_dags([test, corpus]).reset_index(drop=True)
    id_to_idx = {int(d): i for i, d in enumerate(gallery["dag_id"].tolist())}
    queries = test["query"].tolist()
    gt = [id_to_idx[int(d)] for d in test["dag_id"].tolist()]
    toolset = gallery["canonical_toolset_hash"].astype(str).tolist()
    groups_by_hash: dict[str, list[int]] = {}
    for i, h in enumerate(toolset):
        groups_by_hash.setdefault(h, []).append(i)
    groups = [groups_by_hash[toolset[i]] for i in range(len(gallery))]
    twin_pairs = []
    seen = set()
    for members in groups_by_hash.values():
        if len(members) < 2:
            continue
        for a in members:
            for b in members:
                if a < b and (a, b) not in seen:
                    seen.add((a, b))
                    twin_pairs.append((a, b))
    other_pairs = []
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(gallery), generator=rng).tolist()
    for i in range(min(len(twin_pairs), len(gallery))):
        a = i % len(gallery)
        b = perm[i]
        if toolset[a] != toolset[b]:
            other_pairs.append((a, b))

    from sbert_ft_baseline import SBERTFineTuneDualEncoder, encode_all_docs_sbert, encode_all_queries_sbert

    ckpt_path = ROOT / "artifacts" / "campaign_v4" / "results" / "sbert_ft_ged_15t_s42" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = SBERTFineTuneDualEncoder(
        embed_dim=cfg.get("embed_dim", 256),
        freeze_text_backbone=cfg.get("freeze_text", False),
        num_frozen_layers=cfg.get("num_frozen_layers", 4),
        tied=ckpt.get("tied", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    class _Q:
        def __init__(self, qs):
            self.samples = [{"query": q, "dag_id": 0} for q in qs]

        def __len__(self):
            return len(self.samples)

    qe = encode_all_queries_sbert(model, _Q(queries), tok, device)
    d_text = encode_all_docs_sbert(model, gallery["dag_text"].tolist(), tok, device)
    d_tools = encode_all_docs_sbert(
        model, gallery["tools"].str.replace(";", " ").tolist(), tok, device,
    )
    d_rev = encode_all_docs_sbert(
        model, [_reverse_dag_text(t) for t in gallery["dag_text"].tolist()], tok, device,
    )

    sim_text = qe @ d_text.t()
    sim_tools = qe @ d_tools.t()
    sim_rev = qe @ d_rev.t()

    def _same_toolset_tiebreak(sim: torch.Tensor, gt_ids: list[int], mode: str) -> float:
        hits = 0
        n = 0
        gen = torch.Generator().manual_seed(0)
        for qi, g in enumerate(gt_ids):
            pool = groups[g]
            if len(pool) < 2:
                continue
            n += 1
            scores = sim[qi, pool]
            m = float(scores.max())
            tied = [pool[i] for i, s in enumerate(scores.tolist()) if abs(s - m) < 1e-5]
            if mode == "lowest_index":
                pick = min(tied)
            elif mode == "highest_index":
                pick = max(tied)
            else:
                idx = int(torch.randint(len(tied), (1,), generator=gen).item())
                pick = tied[idx]
            hits += int(pick == g)
        return round(hits / max(n, 1), 4)

    gold_is_first = sum(
        int(g == min(groups[g])) for g in gt if len(groups[g]) >= 2
    ) / max(sum(len(groups[g]) >= 2 for g in gt), 1)

    report = {
        "tier": 15,
        "n_queries": len(queries),
        "gallery_size": len(gallery),
        "n_twin_pairs": len(twin_pairs),
        "n_toolsets_with_twins": sum(len(v) >= 2 for v in groups_by_hash.values()),
        "frac_queries_where_gold_has_lowest_gallery_index_among_twins": round(gold_is_first, 4),
        "sbert_ft_full_gallery": {
            "dag_text": _recall_at_k(sim_text, gt),
            "tools_only": _recall_at_k(sim_tools, gt),
            "reversed_edges_dag_text": _recall_at_k(sim_rev, gt),
        },
        "sbert_ft_embedding_collapse": {
            "mean_cosine_twin_dag_text": round(_pairwise_cos(d_text, twin_pairs), 4),
            "mean_cosine_twin_tools_only": round(_pairwise_cos(d_tools, twin_pairs), 4),
            "mean_cosine_twin_reversed": round(_pairwise_cos(d_rev, twin_pairs), 4),
            "mean_cosine_diff_toolset_dag_text": round(_pairwise_cos(d_text, other_pairs), 4),
            "mean_cosine_diff_toolset_tools_only": round(_pairwise_cos(d_tools, other_pairs), 4),
            "frac_twin_pairs_tools_cos_gt_0.99": round(
                float((F.cosine_similarity(d_tools[[a for a, _ in twin_pairs]],
                                           d_tools[[b for _, b in twin_pairs]]) > 0.99).float().mean()),
                4,
            ) if twin_pairs else None,
            "frac_twin_pairs_text_cos_gt_0.99": round(
                float((F.cosine_similarity(d_text[[a for a, _ in twin_pairs]],
                                           d_text[[b for _, b in twin_pairs]]) > 0.99).float().mean()),
                4,
            ) if twin_pairs else None,
        },
        "sbert_ft_same_toolset_ranking": {
            "dag_text": _same_toolset_recall(sim_text, gt, groups),
            "tools_only": _same_toolset_recall(sim_tools, gt, groups),
            "reversed_edges_dag_text": _same_toolset_recall(sim_rev, gt, groups),
        },
        "sbert_ft_tools_only_tiebreak": {
            "note": "Twins have essentially identical tools-only vectors. argmax on a tie returns the first gallery index. Test DAGs were concatenated first, so gold is often that first index.",
            "pick_lowest_gallery_index_on_tie": _same_toolset_tiebreak(sim_tools, gt, "lowest_index"),
            "pick_highest_gallery_index_on_tie": _same_toolset_tiebreak(sim_tools, gt, "highest_index"),
            "pick_random_tied_twin": _same_toolset_tiebreak(sim_tools, gt, "random"),
        },
    }

    from data_synth import build_dag, dag_canonical_hash
    from train import _parse_tools, _parse_edges
    from eval import _load_model_and_tokenizer, encode_all_queries, encode_all_dags
    from encoders import resolve_graph_encoder_settings

    class _TQ:
        def __init__(self, qs):
            self.samples = [{"query": q} for q in qs]

        def __len__(self):
            return len(self.samples)

    class _DagGallery:
        def __init__(self, dags):
            self._dags = dags
            self.num_unique_dags = len(dags)

        def get_unique_dag(self, i):
            return self._dags[i]

    dags = []
    h2i = {}
    for _, row in gallery.iterrows():
        G = build_dag(_parse_tools(row["tools"]), _parse_edges(row["edges"]))
        h = dag_canonical_hash(G)
        if h not in h2i:
            h2i[h] = len(dags)
            dags.append(G)
    ds = _DagGallery(dags)
    gt_legr = []
    for _, row in test.iterrows():
        G = build_dag(_parse_tools(row["tools"]), _parse_edges(row["edges"]))
        gt_legr.append(h2i[dag_canonical_hash(G)])

    gallery_csv = ROOT / "artifacts" / "campaign_v4" / "_gallery_15t.csv"
    gallery.to_csv(gallery_csv, index=False)

    legr_ckpts = {
        "toolname_ged": ROOT / "artifacts" / "campaign_v4" / "results" / "legr_directed_toolname_ged_15t_s42" / "best_model.pt",
        "setgnn_tied_no_ged": ROOT / "artifacts" / "campaign_v4" / "results" / "legr_setgnn_tied_no_ged_15t_s42" / "best_model.pt",
    }
    report["legr"] = {}
    for name, path in legr_ckpts.items():
        if not path.exists():
            continue
        model_l, cfg_l, tokenizer = _load_model_and_tokenizer(
            str(path), device, dataset_csv=str(gallery_csv),
        )
        _, _, bidirectional = resolve_graph_encoder_settings(cfg_l)
        qe_l = encode_all_queries(model_l, _TQ(queries), tokenizer, device)
        de_l = encode_all_dags(model_l, ds, device, bidirectional=bidirectional)
        sim_l = qe_l @ de_l.t()
        report["legr"][name] = {
            "full_gallery": _recall_at_k(sim_l, gt_legr),
            "same_toolset_ranking": _same_toolset_recall(sim_l, gt_legr, groups),
            "mean_cosine_twin": round(_pairwise_cos(de_l, twin_pairs), 4),
            "mean_cosine_diff_toolset": round(_pairwise_cos(de_l, other_pairs), 4),
        }
        del model_l
        torch.cuda.empty_cache()

    out = ROOT / "artifacts" / "campaign_v4" / "results" / "sbert_text_only_proof_15t.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
