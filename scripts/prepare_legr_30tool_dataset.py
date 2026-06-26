"""Build a finetuning-ready LEGR 30-tool dataset from the raw benchmark CSVs.

This script repackages ``upgraded_data/graph_30tools`` into
``upgraded/upgraded_30tools`` while preserving the raw benchmark outputs.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd


RAW_SPLITS: tuple[str, ...] = ("train", "dev", "test_topology_heldout")
OUTPUT_SPLITS: tuple[str, ...] = ("train", "dev", "test_topology_heldout")
TARGET_DAG_COUNTS: dict[str, int] = {
    "train": 138,
    "dev": 30,
    "test_topology_heldout": 30,
}
SPLIT_PRIORITY: dict[str, int] = {
    "train": 2,
    "dev": 1,
    "test_topology_heldout": 0,
}
OUTPUT_COLUMNS: list[str] = [
    "query",
    "dag_id",
    "dag_text",
    "tools",
    "edges",
    "topo_family",
    "source",
    "split",
    "strict_fix_applied",
    "had_duplicate_node_labels",
    "original_tools",
]
REQUIRED_COLUMNS: set[str] = {
    "query",
    "dag_id",
    "dag_text",
    "tools",
    "edges",
    "topo_family",
    "source",
    "split",
}


def _load_raw_pool(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    row_offset = 0

    for split_name in RAW_SPLITS:
        path = input_dir / f"{split_name}.csv"
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns {sorted(missing)}; "
                f"found {list(df.columns)}"
            )

        df = df.copy()
        df["_raw_split"] = split_name
        df["_row_order"] = range(row_offset, row_offset + len(df))
        row_offset += len(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["dag_id"] = combined["dag_id"].astype(int)
    return combined


def _has_duplicate_tool_labels(tools_cell: object) -> bool:
    tools = [token.strip() for token in str(tools_cell).split(";") if token.strip()]
    return len(tools) != len(set(tools))


def _allocate_counts(total_items: int, split_targets: Dict[str, int]) -> dict[str, int]:
    total_target = sum(split_targets.values())
    exact = {
        split_name: total_items * target / total_target
        for split_name, target in split_targets.items()
    }
    counts = {split_name: int(value) for split_name, value in exact.items()}
    remainder = total_items - sum(counts.values())

    if remainder > 0:
        order = sorted(
            split_targets,
            key=lambda split_name: (
                exact[split_name] - counts[split_name],
                split_targets[split_name],
                SPLIT_PRIORITY[split_name],
            ),
            reverse=True,
        )
        for split_name in order[:remainder]:
            counts[split_name] += 1

    return counts


def _build_dag_split_assignments(
    raw_df: pd.DataFrame,
    seed: int,
) -> dict[str, set[int]]:
    dag_meta = (
        raw_df.groupby("dag_id", sort=False)
        .agg(source=("source", "first"), topo_family=("topo_family", "first"))
    )

    for dag_id, group in raw_df.groupby("dag_id", sort=False):
        if group["source"].nunique() != 1:
            raise ValueError(f"dag_id={dag_id} mixes multiple source labels.")
        if group["topo_family"].nunique() != 1:
            raise ValueError(f"dag_id={dag_id} mixes multiple topology families.")

    source_targets = {
        source_name: _allocate_counts(len(source_meta), TARGET_DAG_COUNTS)
        for source_name, source_meta in dag_meta.groupby("source", sort=True)
    }

    rng = random.Random(seed)
    assignments: dict[str, list[int]] = {split_name: [] for split_name in OUTPUT_SPLITS}

    for source_name in sorted(source_targets):
        remaining = source_targets[source_name].copy()
        source_meta = dag_meta[dag_meta["source"] == source_name]
        family_to_dags: dict[str, list[int]] = defaultdict(list)

        for dag_id, row in source_meta.iterrows():
            family_to_dags[str(row["topo_family"])].append(int(dag_id))

        family_names = list(family_to_dags)
        rng.shuffle(family_names)

        for family_name in family_names:
            dag_ids = family_to_dags[family_name]
            rng.shuffle(dag_ids)
            for dag_id in dag_ids:
                candidate_splits = [
                    split_name
                    for split_name, remaining_count in remaining.items()
                    if remaining_count > 0
                ]
                if not candidate_splits:
                    raise RuntimeError(
                        f"No remaining split capacity while assigning dag_id={dag_id}."
                    )

                candidate_splits.sort(
                    key=lambda split_name: (
                        remaining[split_name],
                        SPLIT_PRIORITY[split_name],
                    ),
                    reverse=True,
                )
                chosen_split = candidate_splits[0]
                assignments[chosen_split].append(dag_id)
                remaining[chosen_split] -= 1

        if any(count != 0 for count in remaining.values()):
            raise RuntimeError(
                f"Unfilled source quotas for source={source_name}: {remaining}"
            )

    assigned_ids = [dag_id for dag_ids in assignments.values() for dag_id in dag_ids]
    dag_ids = raw_df["dag_id"].drop_duplicates().tolist()
    if sorted(assigned_ids) != sorted(dag_ids):
        raise RuntimeError("Assigned DAG ids do not match the raw DAG pool.")

    return {split_name: set(dag_ids) for split_name, dag_ids in assignments.items()}


def _build_output_split(
    raw_df: pd.DataFrame,
    dag_ids: set[int],
    split_name: str,
) -> pd.DataFrame:
    split_df = raw_df[raw_df["dag_id"].isin(dag_ids)].copy()
    split_df.sort_values("_row_order", inplace=True, kind="stable")
    split_df["split"] = split_name
    split_df["tools"] = split_df["tools"].fillna("").astype(str)
    split_df["original_tools"] = split_df["tools"]

    duplicate_mask = split_df["tools"].map(_has_duplicate_tool_labels)
    split_df["had_duplicate_node_labels"] = duplicate_mask.map(
        lambda flag: "TRUE" if flag else "FALSE"
    )
    split_df["strict_fix_applied"] = split_df["had_duplicate_node_labels"]

    return split_df[OUTPUT_COLUMNS]


def _validate_output_splits(split_frames: dict[str, pd.DataFrame]) -> None:
    dag_sets = {
        split_name: set(df["dag_id"].astype(int).unique().tolist())
        for split_name, df in split_frames.items()
    }
    for split_name, expected_count in TARGET_DAG_COUNTS.items():
        actual_count = len(dag_sets[split_name])
        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name} has {actual_count} unique DAGs; expected {expected_count}."
            )

    split_names = list(split_frames)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlap = dag_sets[left] & dag_sets[right]
            if overlap:
                raise RuntimeError(
                    f"DAG overlap detected between {left} and {right}: "
                    f"{sorted(list(overlap))[:5]}"
                )

    family_sets = {
        split_name: set(df["topo_family"].dropna().tolist())
        for split_name, df in split_frames.items()
    }
    if not family_sets["train"] & family_sets["dev"]:
        raise RuntimeError("train/dev topology-family overlap is empty.")
    if not family_sets["train"] & family_sets["test_topology_heldout"]:
        raise RuntimeError("train/test topology-family overlap is empty.")
    if not family_sets["dev"] & family_sets["test_topology_heldout"]:
        raise RuntimeError("dev/test topology-family overlap is empty.")

    for split_name, df in split_frames.items():
        sources = set(df["source"].dropna().tolist())
        if {"original", "generated"} - sources:
            raise RuntimeError(
                f"{split_name} does not contain both original and generated DAG rows."
            )
        if not (df["tools"] == df["original_tools"]).all():
            raise RuntimeError(f"{split_name} modified tools unexpectedly.")


def build_upgraded_30tool_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    raw_df = _load_raw_pool(input_dir)
    assignments = _build_dag_split_assignments(raw_df, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_frames: dict[str, pd.DataFrame] = {}

    for split_name in OUTPUT_SPLITS:
        split_df = _build_output_split(raw_df, assignments[split_name], split_name)
        split_frames[split_name] = split_df
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    shutil.copyfile(input_dir / "hard_negatives.csv", output_dir / "hard_negatives.csv")
    _validate_output_splits(split_frames)
    return split_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the finetuning-ready LEGR 30-tool CSV dataset.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("upgraded_data") / "graph_30tools",
        help="Raw benchmark graph directory to read.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("upgraded") / "upgraded_30tools",
        help="Output directory for the repackaged LEGR dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the DAG-level resplit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_frames = build_upgraded_30tool_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"Built upgraded LEGR 30-tool dataset at {args.output_dir}")
    for split_name, df in split_frames.items():
        dag_count = df["dag_id"].nunique()
        families = df["topo_family"].nunique()
        sources = sorted(df["source"].dropna().unique().tolist())
        print(
            f"  {split_name}: {len(df)} rows, {dag_count} DAGs, "
            f"{families} families, sources={sources}"
        )


if __name__ == "__main__":
    main()
