"""
upgrade_routing.py -- Part A: Upgrade the single-tool routing dataset
=====================================================================

Reads single_tool_1005.csv and produces three harder evaluation splits:

  1. lexical_cue_reduced.csv   -- direct action cues replaced with indirect phrasings
  2. confusable_intents.csv    -- surface-similar queries across confusable labels
  3. paraphrase_heldout_train.csv / paraphrase_heldout_test.csv
                               -- paraphrase families split to prevent leakage

Also produces base_cleaned.csv, report.json, and report.md.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_benchmark_specs import (
    cue_word_fraction_routing_30,
    detect_routing_30_cue_words,
    generate_routing_30_confusable_query,
    generate_routing_30_indirect_query,
    matches_routing_30_labels,
)
from utils.text_utils import (
    CONFUSABLE_LABEL_MAP,
    CUE_WORDS,
    cue_word_fraction,
    detect_cue_words,
    exact_duplicate_rate,
    fill_entities,
    generate_confusable_query,
    generate_indirect_query,
    generate_paraphrases,
    near_duplicate_rate,
    top_ngrams,
)


def _uses_routing_30_helpers(labels: List[str]) -> bool:
    return matches_routing_30_labels(labels)


def load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_routing_data(cfg: Dict) -> pd.DataFrame:
    """Load the routing CSV and auto-detect columns."""
    root = Path(__file__).resolve().parent.parent
    path = root / cfg["routing"]["input_path"]
    df = pd.read_csv(path)

    query_col = cfg["routing"]["query_col"]
    label_col = cfg["routing"]["label_col"]

    if query_col not in df.columns:
        for c in df.columns:
            if "query" in c.lower() or "text" in c.lower():
                query_col = c
                break
    if label_col not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "truth" in c.lower() or "tool" in c.lower():
                label_col = c
                break

    print(f"  Routing dataset: {len(df)} rows")
    print(f"  Query column  : {query_col}")
    print(f"  Label column  : {label_col}")
    print(f"  Labels ({df[label_col].nunique()}): {sorted(df[label_col].unique())}")
    return df, query_col, label_col


# ─────────────────────────────────────────────────────────────────────────────
#  Split generators
# ─────────────────────────────────────────────────────────────────────────────

def build_lexical_cue_reduced(
    df: pd.DataFrame,
    query_col: str,
    label_col: str,
    n_per_label: int,
    seed: int,
) -> pd.DataFrame:
    """Generate indirect-phrased queries that avoid direct lexical cues."""
    rng = random.Random(seed)
    rows = []
    labels = sorted(df[label_col].unique())
    use_routing_30 = _uses_routing_30_helpers(labels)

    for label in labels:
        subset = df[df[label_col] == label]
        used_rendered = set()

        for i in range(n_per_label):
            src_idx = i % len(subset)
            src_row = subset.iloc[src_idx]
            original = src_row[query_col]
            if use_routing_30:
                indirect = generate_routing_30_indirect_query(label, rng, used_rendered)
                cues_in_original = detect_routing_30_cue_words(original, label)
                cues_in_new = detect_routing_30_cue_words(indirect, label)
            else:
                indirect = generate_indirect_query(label, rng, used_rendered)
                cues_in_original = detect_cue_words(original, label)
                cues_in_new = detect_cue_words(indirect, label)
            if not indirect:
                continue

            removed = [c for c in cues_in_original if c not in cues_in_new]

            rows.append({
                "source_row_id": int(src_row.name) if hasattr(src_row, "name") else src_idx,
                "original_query": original,
                "transformed_query": indirect,
                "label": label,
                "generation_type": "lexical_cue_reduced",
                "lexical_cues_removed": ";".join(removed) if removed else "",
                "paraphrase_family": "",
                "confusable_with": "",
            })

    result = pd.DataFrame(rows)
    print(f"  lexical_cue_reduced: {len(result)} rows, "
          f"{result['label'].nunique()} labels")
    return result


def build_confusable_intents(
    df: pd.DataFrame,
    query_col: str,
    label_col: str,
    n_per_label: int,
    seed: int,
) -> pd.DataFrame:
    """Generate surface-similar queries across confusable label pairs."""
    rng = random.Random(seed)
    rows = []
    labels = sorted(df[label_col].unique())
    use_routing_30 = _uses_routing_30_helpers(labels)

    for label in labels:
        used_rendered = set()
        for _ in range(n_per_label):
            if use_routing_30:
                query, true_label, confusable = generate_routing_30_confusable_query(
                    label, rng, used_rendered,
                )
            else:
                query, true_label, confusable = generate_confusable_query(
                    label, rng, used_rendered,
                )
            if not query:
                continue
            rows.append({
                "source_row_id": -1,
                "original_query": "",
                "transformed_query": query,
                "label": true_label,
                "generation_type": "confusable_intent",
                "lexical_cues_removed": "",
                "paraphrase_family": "",
                "confusable_with": confusable,
            })

    result = pd.DataFrame(rows)
    print(f"  confusable_intents: {len(result)} rows, "
          f"{result['label'].nunique()} labels")
    return result


def build_paraphrase_heldout(
    df: pd.DataFrame,
    query_col: str,
    label_col: str,
    n_variants: int,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paraphrase families and split train/test by family."""
    rng = random.Random(seed)
    all_rows = []
    family_id = 0

    for idx, row in df.iterrows():
        query = row[query_col]
        label = row[label_col]

        paraphrases = generate_paraphrases(query, rng, n=n_variants)

        base_row = {
            "source_row_id": int(idx) if not isinstance(idx, int) else idx,
            "original_query": query,
            "transformed_query": query,
            "label": label,
            "generation_type": "paraphrase_original",
            "lexical_cues_removed": "",
            "paraphrase_family": family_id,
            "confusable_with": "",
        }
        all_rows.append(base_row)

        for p in paraphrases:
            all_rows.append({
                **base_row,
                "transformed_query": p,
                "generation_type": "paraphrase_variant",
            })

        family_id += 1

    all_df = pd.DataFrame(all_rows)

    # Split by family: each family goes entirely to train or test
    families = list(range(family_id))
    rng.shuffle(families)
    n_test_families = max(1, int(len(families) * test_frac))
    test_families = set(families[:n_test_families])
    train_families = set(families[n_test_families:])

    train_df = all_df[all_df["paraphrase_family"].isin(train_families)].reset_index(drop=True)
    test_df = all_df[all_df["paraphrase_family"].isin(test_families)].reset_index(drop=True)

    print(f"  paraphrase_heldout_train: {len(train_df)} rows, "
          f"{train_df['paraphrase_family'].nunique()} families")
    print(f"  paraphrase_heldout_test : {len(test_df)} rows, "
          f"{test_df['paraphrase_family'].nunique()} families")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
#  Validation report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    base_df: pd.DataFrame,
    lex_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    para_train: pd.DataFrame,
    para_test: pd.DataFrame,
    query_col: str,
    label_col: str,
    cfg: Dict,
) -> Dict:
    """Build a validation report comparing original and upgraded splits."""
    def split_stats(df: pd.DataFrame, q_col: str, l_col: str) -> Dict:
        queries = df[q_col].tolist()
        labels = df[l_col].tolist()
        class_counts = dict(Counter(labels))
        if _uses_routing_30_helpers(list(set(labels))):
            cue_fraction = cue_word_fraction_routing_30(queries, labels)
        else:
            cue_fraction = cue_word_fraction(queries, labels)
        return {
            "total_rows": len(df),
            "unique_labels": len(set(labels)),
            "class_counts": class_counts,
            "exact_duplicate_rate": round(exact_duplicate_rate(queries), 4),
            "near_duplicate_rate": round(
                near_duplicate_rate(queries, cfg["routing"]["near_dup_threshold"]), 4
            ),
            "avg_query_length": round(
                sum(len(q) for q in queries) / max(len(queries), 1), 1
            ),
            "cue_word_fraction": round(cue_fraction, 4),
            "top_3grams": top_ngrams(
                queries, n=cfg["routing"]["top_ngram_n"],
                top_k=cfg["routing"]["top_ngram_k"],
            ),
        }

    report = {
        "original": split_stats(base_df, query_col, label_col),
        "lexical_cue_reduced": split_stats(lex_df, "transformed_query", "label"),
        "confusable_intents": split_stats(conf_df, "transformed_query", "label"),
        "paraphrase_heldout_train": split_stats(para_train, "transformed_query", "label"),
        "paraphrase_heldout_test": split_stats(para_test, "transformed_query", "label"),
    }

    # Confusable coverage
    if len(conf_df) > 0 and "confusable_with" in conf_df.columns:
        pairs = conf_df[conf_df["confusable_with"] != ""][
            ["label", "confusable_with"]
        ].drop_duplicates()
        report["confusable_label_pairs"] = [
            {"true_label": r["label"], "confusable_with": r["confusable_with"]}
            for _, r in pairs.iterrows()
        ]

    # Paraphrase family overlap check
    train_fams = set(para_train["paraphrase_family"].unique())
    test_fams = set(para_test["paraphrase_family"].unique())
    report["paraphrase_family_overlap"] = len(train_fams & test_fams)

    return report


def write_report_md(report: Dict, path: Path) -> None:
    """Write a human-readable Markdown version of the report."""
    lines = ["# Routing Dataset Upgrade Report\n"]

    for split_name in ["original", "lexical_cue_reduced", "confusable_intents",
                        "paraphrase_heldout_train", "paraphrase_heldout_test"]:
        s = report.get(split_name, {})
        lines.append(f"## {split_name}\n")
        lines.append(f"- Total rows: {s.get('total_rows', 'N/A')}")
        lines.append(f"- Unique labels: {s.get('unique_labels', 'N/A')}")
        lines.append(f"- Exact duplicate rate: {s.get('exact_duplicate_rate', 'N/A')}")
        lines.append(f"- Near-duplicate rate: {s.get('near_duplicate_rate', 'N/A')}")
        lines.append(f"- Avg query length: {s.get('avg_query_length', 'N/A')}")
        lines.append(f"- Cue-word fraction: {s.get('cue_word_fraction', 'N/A')}")
        cc = s.get("class_counts", {})
        if cc:
            lines.append(f"- Class counts: min={min(cc.values())}, "
                          f"max={max(cc.values())}, "
                          f"mean={sum(cc.values()) / len(cc):.1f}")
        grams = s.get("top_3grams", [])
        if grams:
            lines.append("- Top 3-grams:")
            for gram, count in grams[:10]:
                lines.append(f"  - \"{gram}\" ({count})")
        lines.append("")

    overlap = report.get("paraphrase_family_overlap", 0)
    lines.append(f"## Paraphrase Family Overlap: {overlap}\n")

    pairs = report.get("confusable_label_pairs", [])
    if pairs:
        lines.append("## Confusable Label Pairs Covered\n")
        for p in pairs:
            lines.append(f"- {p['true_label']} <-> {p['confusable_with']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(output_dir_override: str | None = None) -> None:
    print("\n" + "=" * 60)
    print("  Part A: Upgrade Routing Dataset")
    print("=" * 60 + "\n")

    cfg = load_config()
    seed = cfg["seed"]
    rcfg = cfg["routing"]

    df, query_col, label_col = load_routing_data(cfg)

    routing_out = output_dir_override or rcfg["output_dir"]
    out_dir = Path(__file__).resolve().parent.parent / routing_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned base
    base_path = out_dir / "base_cleaned.csv"
    df.to_csv(base_path, index=False)
    print(f"  Saved base_cleaned.csv ({len(df)} rows)")

    # A) Lexical cue reduced
    lex_df = build_lexical_cue_reduced(
        df, query_col, label_col,
        n_per_label=rcfg["lexical_cue_reduced_per_label"],
        seed=seed,
    )
    lex_df.to_csv(out_dir / "lexical_cue_reduced.csv", index=False)

    # B) Confusable intents
    conf_df = build_confusable_intents(
        df, query_col, label_col,
        n_per_label=rcfg["confusable_per_label"],
        seed=seed + 1,
    )
    conf_df.to_csv(out_dir / "confusable_intents.csv", index=False)

    # C) Paraphrase held-out
    para_train, para_test = build_paraphrase_heldout(
        df, query_col, label_col,
        n_variants=rcfg["paraphrase_variants"],
        test_frac=rcfg["paraphrase_test_frac"],
        seed=seed + 2,
    )
    para_train.to_csv(out_dir / "paraphrase_heldout_train.csv", index=False)
    para_test.to_csv(out_dir / "paraphrase_heldout_test.csv", index=False)

    # Validation report
    report = build_report(
        df, lex_df, conf_df, para_train, para_test,
        query_col, label_col, cfg,
    )
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_report_md(report, out_dir / "report.md")
    print(f"\n  Reports saved to {out_dir}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
