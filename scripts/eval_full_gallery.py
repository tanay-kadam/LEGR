"""Rank queries against test DAGs UNION candidate-corpus twins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _unique_dags(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [f.drop_duplicates("dag_id") for f in frames]
    return pd.concat(parts, ignore_index=True).drop_duplicates("dag_id")


def _recall_at_k(sim: torch.Tensor, gt: list[int], ks=(1, 3, 5)) -> dict:
    n = sim.size(0)
    out = {}
    for k in ks:
        pred = sim.topk(k, dim=1).indices.tolist()
        hits = sum(g in row for g, row in zip(gt, pred))
        out[f"recall@{k}"] = round(hits / n, 4)
    top1 = sim.argmax(1).tolist()
    out["recall@1"] = round(sum(int(p == g) for p, g in zip(top1, gt)) / n, 4)
    return out


@torch.no_grad()
def _minilm_embed(texts: list[str], device: str, batch_size: int = 64) -> torch.Tensor:
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state[:, 0, :]
        chunks.append(F.normalize(h, p=2, dim=-1).cpu())
    return torch.cat(chunks, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tier_dir = ROOT / "data" / "campaign_v4" / f"campaign_v4_{args.tier}tools"
    test = pd.read_csv(tier_dir / "test_topology_heldout.csv")
    corpus = pd.read_csv(tier_dir / "candidate_corpus.csv")
    gallery = _unique_dags([test, corpus])
    id_to_idx = {int(d): i for i, d in enumerate(gallery["dag_id"].tolist())}
    queries = test["query"].tolist()
    gt = [id_to_idx[int(d)] for d in test["dag_id"].tolist()]
    device = args.device if torch.cuda.is_available() else "cpu"

    q_emb = _minilm_embed(queries, device)
    d_text = _minilm_embed(gallery["dag_text"].tolist(), device)
    d_tools = _minilm_embed(gallery["tools"].str.replace(";", " ").tolist(), device)

    report = {
        "tier": args.tier,
        "n_queries": len(queries),
        "gallery_size": len(gallery),
        "test_unique_dags": int(test["dag_id"].nunique()),
        "corpus_unique_dags": int(corpus["dag_id"].nunique()),
        "frozen_sbert_dag_text": _recall_at_k(q_emb @ d_text.t(), gt),
        "frozen_sbert_tools_only": _recall_at_k(q_emb @ d_tools.t(), gt),
    }

    from sbert_ft_baseline import SBERTFineTuneDualEncoder, encode_all_docs_sbert, encode_all_queries_sbert

    sbert_ckpt = ROOT / "artifacts" / "campaign_v4" / "results" / f"sbert_ft_ged_{args.tier}t_s42" / "best_model.pt"
    if sbert_ckpt.exists():
        ckpt = torch.load(sbert_ckpt, map_location=device, weights_only=False)
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

        qe = encode_all_queries_sbert(model, _Q(queries), tok, torch.device(device))
        de_text = encode_all_docs_sbert(model, gallery["dag_text"].tolist(), tok, torch.device(device))
        de_tools = encode_all_docs_sbert(
            model, gallery["tools"].str.replace(";", " ").tolist(), tok, torch.device(device),
        )
        report["sbert_ft_dag_text"] = _recall_at_k(qe @ de_text.t(), gt)
        report["sbert_ft_tools_only"] = _recall_at_k(qe @ de_tools.t(), gt)

    from eval import _load_model_and_tokenizer, encode_all_queries, encode_all_dags
    from encoders import resolve_graph_encoder_settings
    from data_synth import build_dag, dag_canonical_hash
    from train import _parse_tools, _parse_edges

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

    legr_path = ROOT / "artifacts" / "campaign_v4" / "results" / f"legr_directed_toolname_ged_{args.tier}t_s42" / "best_model.pt"
    if legr_path.exists():
        gallery_csv = ROOT / "artifacts" / "campaign_v4" / f"_gallery_{args.tier}t.csv"
        gallery.to_csv(gallery_csv, index=False)
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
        model, cfg, tokenizer = _load_model_and_tokenizer(
            str(legr_path), torch.device(device), dataset_csv=str(gallery_csv),
        )
        _, _, bidirectional = resolve_graph_encoder_settings(cfg)
        qe = encode_all_queries(model, _TQ(queries), tokenizer, torch.device(device))
        de = encode_all_dags(model, ds, torch.device(device), bidirectional=bidirectional)
        report["legr_toolname_ged"] = _recall_at_k(qe @ de.t(), gt_legr)
        report["legr_gallery_unique"] = ds.num_unique_dags

    out = ROOT / "artifacts" / "campaign_v4" / "results" / f"full_gallery_{args.tier}t.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
