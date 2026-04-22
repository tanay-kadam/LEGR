# Upgraded Benchmark Datasets

This directory contains strengthened versions of the single-tool routing
and multi-step DAG retrieval benchmarks, designed to resist shortcut learning
and provide harder, more realistic evaluation.

## What Changed and Why It Matters

### Routing Benchmark

The original single-tool routing dataset (`single_tool_1005.csv`) was class-balanced
and clean, but vulnerable to keyword matching -- most queries contained direct lexical
cues that trivially revealed the target tool (e.g., 'reset password' -> reset_password).

We produced three new evaluation splits that address distinct weaknesses:

1. **lexical_cue_reduced.csv** -- Queries rewritten to use indirect phrasings
   (symptoms, outcomes, context) instead of explicit tool/action keywords.
   Cue-word fraction dropped from 0.9756 to 0.2647.

2. **confusable_intents.csv** -- Queries intentionally surface-similar to confusable
   labels (e.g., a query that sounds like 'check_status' but actually requires
   'query_database'). Covers 60 label pairs.

3. **paraphrase_heldout_train/test.csv** -- Paraphrase families split across
   train/test so models cannot memorize phrasing patterns.
   Family overlap between splits: 0.

### DAG / Graph Retrieval Benchmark

The original multi-step DAG dataset (`test_corpus_for_all_models.csv`) had valid
DAGs but limited topology diversity, making it vulnerable to motif memorisation.

We strengthened it with:

1. **Topology diversity**: Increased from 12 to 132 unique topologies.

2. **New topology families**: Added diamond, asymmetric fork-join, deep asymmetric
   merge, multi-branch, repeated-tool patterns, long branched chains, double diamonds,
   W-shapes, hourglasses, and wide fan-out with depth.

3. **Hard negatives**: 312 hard-negative DAGs (same tools/different edges, same topology/different tools,
   single-edge perturbations, extra distractor nodes, missing dependencies).

4. **Topology-held-out splits**: Train/dev/test where some topology families only
   appear in test (5 families held out).

## File Structure

```
upgraded_data/
  routing/
    base_cleaned.csv              # Original data, validated
    lexical_cue_reduced.csv       # Indirect phrasings
    confusable_intents.csv        # Surface-similar cross-label queries
    paraphrase_heldout_train.csv  # Train split (family-separated)
    paraphrase_heldout_test.csv   # Test split (family-separated)
    report.json                   # Validation statistics
    report.md                     # Human-readable report
  graph/
    base_validated.csv            # Original data, validated
    generated_augmented.csv       # New diverse-topology DAGs
    hard_negatives.csv            # Hard-negative DAG variants
    train.csv                     # Training split
    dev.csv                       # Development split
    test_topology_heldout.csv     # Topology-held-out test
    topology_report.json          # Topology analysis
    topology_report.md            # Human-readable topology report
  combined_report.json            # Combined before/after analysis
  combined_report.md              # Human-readable combined report
  README.md                       # This file
```

## Reproducibility

All outputs are deterministic given the same random seed (default: 42).
Configuration is in `configs/pipeline_config.json`.

```powershell
python scripts/run_pipeline.py
```

## Dataset Card

| Property | Routing | Graph |
|----------|---------|-------|
| Source | Synthetic (rule-based generation) | Synthetic (template + rule-based) |
| Language | English | English + DAG notation |
| Domain | IT operations, 15-tool API routing | Multi-step workflow orchestration |
| License | Research use | Research use |
| Sensitive data | No PII (synthetic names only) | No PII |

## Benchmark Caveats

- All queries are synthetically generated. While phrasing diversity has been
  increased, the vocabulary and scenario space is bounded by the template library.
- Indirect phrasings in `lexical_cue_reduced` were handcrafted; an LLM-assisted
  expansion pass would further improve diversity (optional hook available).
- Hard negatives are structurally plausible but not semantically validated --
  some may be ambiguous edge cases.
- The topology-held-out split assumes that topology families transfer meaningfully;
  this is a reasonable but untested assumption for graph-structure-aware models.
- Entity pools are small (7 users, 6 servers, etc.); downstream models could
  overfit to specific entity names across splits.
