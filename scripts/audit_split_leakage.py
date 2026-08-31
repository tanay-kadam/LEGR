import hashlib
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_tools(cell):
    return [t.strip() for t in str(cell).split(";") if t.strip()]


def parse_edges(cell):
    out = []
    for part in str(cell).split(";"):
        part = part.strip()
        if "->" in part:
            s, d = part.split("->", 1)
            out.append((int(s), int(d)))
    return out


def labeled_hash(tools, edges):
    nodes = tuple(sorted(tools))
    e = tuple(sorted((tools[s], tools[d]) for s, d in edges
                     if s < len(tools) and d < len(tools)))
    return hashlib.sha256(f"{nodes}|{e}".encode()).hexdigest()[:16]


def topology_hash(tools, edges):
    """Unlabelled structure only: degree-sequence + edge pattern on node indices."""
    G = nx.DiGraph()
    G.add_nodes_from(range(len(tools)))
    G.add_edges_from([(s, d) for s, d in edges if s < len(tools) and d < len(tools)])
    return nx.weisfeiler_lehman_graph_hash(G, iterations=3)


def hashes(path):
    df = pd.read_csv(path)
    lab, top, fam = set(), set(), set()
    for _, row in df.iterrows():
        t = parse_tools(row["tools"])
        e = parse_edges(row["edges"])
        if not t:
            continue
        lab.add(labeled_hash(t, e))
        top.add(topology_hash(t, e))
        fam.add(row.get("topo_family", ""))
    return lab, top, fam


pairs = [
    (15, "upgraded/upgraded_15tools/train.csv",
     "upgraded/upgraded_15tools/test_topology_heldout.csv"),
    (30, "upgraded/upgraded_30tools/train.csv",
     "upgraded/upgraded_30tools/test_topology_heldout_1200.csv"),
    (45, "upgraded/upgraded_45tools/train.csv",
     "upgraded/upgraded_45tools/test_topology_heldout.csv"),
]

for tc, train_path, test_path in pairs:
    tr_lab, tr_top, tr_fam = hashes(train_path)
    te_lab, te_top, te_fam = hashes(test_path)
    print(f"=== {tc} tools")
    print(f"  labelled-DAG overlap : {len(tr_lab & te_lab)} "
          f"(train {len(tr_lab)}, test {len(te_lab)})")
    print(f"  unlabelled-topology overlap: {len(tr_top & te_top)} "
          f"(train {len(tr_top)}, test {len(te_top)})")
    print(f"  family overlap       : {sorted(tr_fam & te_fam)}")
