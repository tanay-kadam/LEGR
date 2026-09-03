from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

try:
    import data_synth as ds
except ImportError:  # pragma: no cover
    from src import data_synth as ds

from .structures import GraphSignature, build_signature, invariant_node_features


def parse_tools(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def parse_edges(value) -> list[tuple[int, int]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    result = []
    for part in str(value).split(";"):
        if "->" not in part:
            continue
        left, right = part.split("->", 1)
        result.append((int(left.strip()), int(right.strip())))
    return result


@dataclass
class ResearchSample:
    query: str
    dag_text: str
    dag_index: int
    group_index: int
    graph: Data
    signature: GraphSignature


class ResearchDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path | Sequence[str | Path],
        vocabulary: Sequence[str],
        structure_kind: str = "combined",
    ):
        paths = [Path(value) for value in csv_path] if isinstance(csv_path, (list, tuple)) else [Path(csv_path)]
        self.path = paths[0] if len(paths) == 1 else tuple(paths)
        self.vocabulary = list(vocabulary)
        ds.register_tools(self.vocabulary)
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        self.samples: list[ResearchSample] = []
        self.signatures: list[GraphSignature] = []
        self.graphs: list[Data] = []
        key_to_index: dict[str, int] = {}
        group_to_index: dict[str, int] = {}
        for row in frame.to_dict("records"):
            tools = parse_tools(row.get("tools"))
            edges = parse_edges(row.get("edges"))
            if not tools:
                continue
            group = str(row.get("structural_twin_group") or row.get("canonical_toolset_hash") or ";".join(sorted(tools)))
            signature = build_signature(tools, edges, self.vocabulary, group)
            if signature.dag_key not in key_to_index:
                dag_index = len(self.signatures)
                key_to_index[signature.dag_key] = dag_index
                group_index = group_to_index.setdefault(group, len(group_to_index))
                edge_index = (
                    torch.tensor(edges, dtype=torch.long).t().contiguous()
                    if edges else torch.zeros((2, 0), dtype=torch.long)
                )
                global_tool_ids = torch.tensor(
                    [ds.TOOL_TO_IDX[name] for name in tools], dtype=torch.long,
                ).unsqueeze(-1)
                struct_x = invariant_node_features(len(tools), edge_index, structure_kind)
                graph = Data(x=global_tool_ids, edge_index=edge_index, struct_x=struct_x)
                self.signatures.append(signature)
                self.graphs.append(graph)
            else:
                dag_index = key_to_index[signature.dag_key]
                signature = self.signatures[dag_index]
                group_index = group_to_index.setdefault(signature.group_id, len(group_to_index))
            self.samples.append(ResearchSample(
                query=str(row.get("query", "")),
                dag_text=str(row.get("dag_text", "")),
                dag_index=dag_index,
                group_index=group_index,
                graph=self.graphs[dag_index],
                signature=signature,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ResearchSample:
        return self.samples[index]


class UniqueGraphDataset(Dataset):
    """One existing row per unique DAG; does not synthesize candidate data."""

    def __init__(self, dataset: ResearchDataset):
        first = {}
        for sample in dataset.samples:
            first.setdefault(sample.dag_index, sample)
        self.samples = [first[index] for index in sorted(first)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_collate(tokenizer, max_length: int = 128):
    def collate(samples: list[ResearchSample]) -> dict:
        queries = tokenizer(
            [sample.query for sample in samples], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        docs = tokenizer(
            [sample.dag_text for sample in samples], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        graph_batch = Batch.from_data_list([sample.graph for sample in samples])
        return {
            "input_ids": queries["input_ids"],
            "attention_mask": queries["attention_mask"],
            "doc_input_ids": docs["input_ids"],
            "doc_attention_mask": docs["attention_mask"],
            "graph_x": graph_batch.x,
            "graph_edge_index": graph_batch.edge_index,
            "graph_batch": graph_batch.batch,
            "graph_struct_x": graph_batch.struct_x,
            "dag_ids": torch.tensor([sample.dag_index for sample in samples], dtype=torch.long),
            "group_ids": torch.tensor([sample.group_index for sample in samples], dtype=torch.long),
            "tool_targets": torch.stack([sample.signature.tool_target for sample in samples]),
            "relation_targets": torch.stack([sample.signature.relation_target for sample in samples]),
            "queries": [sample.query for sample in samples],
            "dag_texts": [sample.dag_text for sample in samples],
        }
    return collate


def campaign_paths(tier: int) -> dict[str, Path]:
    base = Path("data/campaign_v4") / f"campaign_v4_{tier}tools"
    return {
        "train": base / "train.csv",
        "dev": base / "dev.csv",
        "test": base / "test_topology_heldout.csv",
        "candidate": base / "candidate_corpus.csv",
    }
