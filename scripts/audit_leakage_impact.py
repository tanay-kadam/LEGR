"""Split the 30-tool held-out set into seen-DAG / unseen-DAG halves and report
LEGR recall on each, to isolate how much of the headline number depends on the
labelled DAGs that leaked into train."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.argv += ["--tool_count", "30"]

import networkx as nx
import pandas as pd
import torch

from legr_tool_count import bootstrap_tool_count_from_argv

bootstrap_tool_count_from_argv(sys.argv)

from data_synth import build_dag, dag_canonical_hash, dag_to_pyg, register_tools
from eval import _load_model_and_tokenizer, _parse_edges, _parse_tools
from torch_geometric.data import Batch


def labeled_hash(tools, edges):
    nodes = tuple(sorted(tools))
    e = tuple(sorted((tools[s], tools[d]) for s, d in edges
                     if s < len(tools) and d < len(tools)))
    return hashlib.sha256(f"{nodes}|{e}".encode()).hexdigest()[:16]


def load(path):
    df = pd.read_csv(path)
    all_tools = set()
    for cell in df["tools"]:
        if isinstance(cell, str):
            all_tools.update(t.strip() for t in cell.split(";") if t.strip())
    register_tools(sorted(all_tools))
    return df


train_df = load("upgraded/upgraded_30tools/train.csv")
train_hashes = set()
for _, row in train_df.iterrows():
    t, e = _parse_tools(row["tools"]), _parse_edges(row["edges"])
    if t:
        train_hashes.add(labeled_hash(t, e))

test_df = load("upgraded/upgraded_30tools/test_topology_heldout_1200.csv")

unique_dags, hash_to_id, samples = [], {}, []
for row_idx, row in test_df.iterrows():
    t, e = _parse_tools(row["tools"]), _parse_edges(row["edges"])
    if not t:
        continue
    try:
        G = build_dag(t, e)
    except (AssertionError, nx.NetworkXError):
        continue
    h = dag_canonical_hash(G)
    if h not in hash_to_id:
        hash_to_id[h] = len(unique_dags)
        unique_dags.append(G)
    samples.append({
        "dag_id": hash_to_id[h],
        "query": str(row["query"]),
        "seen_in_train": labeled_hash(t, e) in train_hashes,
        "topo_family": row["topo_family"],
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg, tok = _load_model_and_tokenizer("checkpoints_30tools/best_model.pt", device)

with torch.no_grad():
    q = []
    for i in range(0, len(samples), 64):
        chunk = [s["query"] for s in samples[i:i + 64]]
        enc = tok(chunk, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        q.append(model.encode_text(enc["input_ids"].to(device),
                                   enc["attention_mask"].to(device)).cpu())
    q = torch.cat(q)
    d = []
    for i in range(0, len(unique_dags), 64):
        b = Batch.from_data_list([dag_to_pyg(G) for G in unique_dags[i:i + 64]])
        tp = getattr(b, "topo_pos", None)
        d.append(model.encode_graph(b.x.to(device), b.edge_index.to(device),
                                    b.batch.to(device),
                                    topo_pos=tp.to(device) if tp is not None else None
                                    ).cpu())
    d = torch.cat(d)

top = torch.mm(q, d.t()).topk(k=5, dim=1).indices

for label, keep in [("ALL", lambda s: True),
                    ("SEEN labelled DAG in train", lambda s: s["seen_in_train"]),
                    ("UNSEEN labelled DAG", lambda s: not s["seen_in_train"])]:
    idx = [i for i, s in enumerate(samples) if keep(s)]
    if not idx:
        continue
    r1 = sum(1 for i in idx if top[i][0].item() == samples[i]["dag_id"]) / len(idx)
    r5 = sum(1 for i in idx
             if samples[i]["dag_id"] in top[i].tolist()) / len(idx)
    dags = len({samples[i]["dag_id"] for i in idx})
    print(f"  {label:<28} n={len(idx):5d}  unique DAGs={dags:3d}  "
          f"recall@1={r1:.4f}  recall@5={r5:.4f}")

print("\n  per-family unseen-only breakdown:")
fams = sorted({s["topo_family"] for s in samples})
for fam in fams:
    idx = [i for i, s in enumerate(samples)
           if s["topo_family"] == fam and not s["seen_in_train"]]
    tot = [i for i, s in enumerate(samples) if s["topo_family"] == fam]
    if not idx:
        print(f"    {fam:<14} 0/{len(tot)} rows unseen (entire family leaked)")
        continue
    r1 = sum(1 for i in idx if top[i][0].item() == samples[i]["dag_id"]) / len(idx)
    print(f"    {fam:<14} {len(idx):4d}/{len(tot):4d} rows unseen  recall@1={r1:.4f}")
