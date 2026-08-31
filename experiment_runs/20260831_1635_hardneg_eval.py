"""Hard-negative eval: LEGR GCN vs SBERT FT on the same CSV.

Official protocol: cosine(query, negative_DAG) < 0.5 counts as a reject.
Also reports pairwise ranking: cosine(query, gold) > cosine(query, negative).

Invoke with --tool_count first.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\tkadam\LEGR")
sys.path.insert(0, str(ROOT / "src"))

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

bootstrap_tool_count_from_argv(sys.argv)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from data_synth import build_dag, dag_to_pyg, dag_to_text  # noqa: E402
from encoders import resolve_graph_encoder_settings  # noqa: E402
from eval import _load_model_and_tokenizer  # noqa: E402
from sbert_ft_baseline import SBERTFineTuneDualEncoder, get_tokenizer  # noqa: E402
from train import TrainConfig, _parse_edges, _parse_tools  # noqa: E402

CKPT = {
    15: {
        "legr": ROOT / "experiment_runs/20260831_1218/upgraded/TASK4_DIRECTION_ABLATION/LEGR_DEFAULT_GCN_15TOOL/best_model.pt",
        "sbert": ROOT / "experiment_runs/20260831_1536/upgraded/TASK1_SBERT_FINE_TUNE/SBERT_FT_GED030_15TOOL/best_model.pt",
        "hn": ROOT / "upgraded_data/graph_15tools/hard_negatives.csv",
        "splits": ROOT / "upgraded_data/graph_15tools",
    },
    30: {
        "legr": ROOT / "experiment_runs/20260831_1218/upgraded/TASK4_DIRECTION_ABLATION/LEGR_DEFAULT_GCN_30TOOL/best_model.pt",
        "sbert": ROOT / "experiment_runs/20260831_1536/upgraded/TASK1_SBERT_FINE_TUNE/SBERT_FT_GED030_30TOOL/best_model.pt",
        "hn": ROOT / "upgraded_data/graph_30tools/hard_negatives.csv",
        "splits": ROOT / "upgraded_data/graph_30tools",
    },
}


def gold_dags(split_dir: Path) -> dict:
    out = {}
    for name in ("train.csv", "dev.csv", "test_topology_heldout.csv"):
        path = split_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            did = row.get("dag_id")
            if did in out or pd.isna(did):
                continue
            tools = _parse_tools(row.get("tools", ""))
            edges = _parse_edges(row.get("edges", ""))
            if tools:
                out[int(did)] = (tools, edges)
    return out


@torch.no_grad()
def legr_sim_text_graph(model, tokenizer, device, query: str, tools, edges, bidirectional: bool):
    G = build_dag(tools, edges)
    pyg = dag_to_pyg(G, bidirectional=bidirectional)
    enc = tokenizer([str(query)], padding=True, truncation=True, max_length=128, return_tensors="pt")
    z_q = model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    batch = Batch.from_data_list([pyg])
    tp = getattr(batch, "topo_pos", None)
    if tp is not None:
        tp = tp.to(device)
    z_d = model.encode_graph(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), topo_pos=tp)
    return float(F.cosine_similarity(z_q, z_d).item())


@torch.no_grad()
def sbert_sim_text_doc(model, tokenizer, device, query: str, tools, edges):
    G = build_dag(tools, edges)
    enc_q = tokenizer([str(query)], padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc_d = tokenizer([dag_to_text(G)], padding=True, truncation=True, max_length=128, return_tensors="pt")
    z_q = model.encode_query(enc_q["input_ids"].to(device), enc_q["attention_mask"].to(device))
    z_d = model.encode_document(enc_d["input_ids"].to(device), enc_d["attention_mask"].to(device))
    return float(F.cosine_similarity(z_q, z_d).item())


def summarize(rows: list[dict]) -> dict:
    def pack(subset):
        n = len(subset)
        if n == 0:
            return {"n": 0}
        rej = sum(1 for r in subset if r["sim_neg"] < 0.5)
        ranked = [r for r in subset if r.get("sim_gold") is not None]
        return {
            "n": n,
            "threshold_reject_acc": round(rej / n, 4),
            "false_positive_rate": round(1.0 - (rej / n), 4),
            "mean_sim_neg": round(sum(r["sim_neg"] for r in subset) / n, 4),
            "pairwise_n": len(ranked),
            "pairwise_gold_beats_neg": (
                round(sum(1 for r in ranked if r["sim_gold"] > r["sim_neg"]) / len(ranked), 4)
                if ranked else None
            ),
        }

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["negative_type"]].append(r)
    return {"overall": pack(rows), "by_type": {k: pack(v) for k, v in sorted(by_type.items())}}


def load_sbert(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in config_dict.items() if k in fields})
    tied = bool(ckpt.get("tied", False))
    model = SBERTFineTuneDualEncoder(
        embed_dim=cfg.embed_dim,
        text_model_name=cfg.text_model,
        freeze_text_backbone=cfg.freeze_text,
        num_frozen_layers=cfg.num_frozen_layers,
        tied=tied,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, get_tokenizer(cfg.text_model), tied, cfg.lambda_ged, ckpt.get("epoch")


def main():
    p = argparse.ArgumentParser()
    add_tool_count_argument(p)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    tools = int(args.tool_count)
    paths = CKPT[tools]
    device = torch.device(args.device)
    hn = pd.read_csv(paths["hn"])
    gold = gold_dags(paths["splits"])

    print(f"tool_count={tools} device={device} n_hardneg={len(hn)} gold_dags={len(gold)}", flush=True)

    legr, lcfg, ltok = _load_model_and_tokenizer(str(paths["legr"]), device)
    _, _, bidirectional = resolve_graph_encoder_settings(lcfg)
    sbert, stok, tied, lam, epoch = load_sbert(paths["sbert"], device)

    legr_rows, sbert_rows = [], []
    skipped = 0
    for _, row in hn.iterrows():
        query = row.get("query", "")
        neg_tools = _parse_tools(row.get("neg_tools", ""))
        neg_edges = _parse_edges(row.get("neg_edges", ""))
        if not query or not neg_tools:
            skipped += 1
            continue
        ntype = str(row.get("negative_type", ""))
        try:
            ls = legr_sim_text_graph(legr, ltok, device, query, neg_tools, neg_edges, bidirectional)
            ss = sbert_sim_text_doc(sbert, stok, device, query, neg_tools, neg_edges)
        except Exception:
            skipped += 1
            continue
        rec_l = {"negative_type": ntype, "sim_neg": ls, "sim_gold": None}
        rec_s = {"negative_type": ntype, "sim_neg": ss, "sim_gold": None}
        did = row.get("positive_dag_id")
        if pd.notna(did) and int(did) in gold:
            gtools, gedges = gold[int(did)]
            try:
                rec_l["sim_gold"] = legr_sim_text_graph(legr, ltok, device, query, gtools, gedges, bidirectional)
                rec_s["sim_gold"] = sbert_sim_text_doc(sbert, stok, device, query, gtools, gedges)
            except Exception:
                pass
        legr_rows.append(rec_l)
        sbert_rows.append(rec_s)

    out = {
        "tool_count": tools,
        "hard_negative_csv": str(paths["hn"]),
        "legr_checkpoint": str(paths["legr"]),
        "sbert_checkpoint": str(paths["sbert"]),
        "sbert_tied": tied,
        "sbert_lambda_ged": lam,
        "sbert_epoch": epoch,
        "device": str(device),
        "skipped": skipped,
        "protocol": {
            "threshold_reject_acc": "fraction of (query, hard-neg DAG) pairs with cosine < 0.5",
            "pairwise_gold_beats_neg": "fraction where cosine(query, gold DAG) > cosine(query, hard-neg DAG)",
        },
        "LEGR": summarize(legr_rows),
        "SBERT_FT": summarize(sbert_rows),
    }
    dest = ROOT / "experiment_runs" / "20260831_1635_hardneg" / f"hardneg_{tools}tool.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
