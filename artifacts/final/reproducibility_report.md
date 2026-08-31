# Reproducibility report

**Status:** commands documented; full train/eval **not** reproduced (code-first).

## Environment

- Python 3.11
- pytest: 138 passed, 1 xfailed
- Torch CPU installed for tests; GPU training not run

## Commands (repository-relative)

### A — SBERT-FT

See `artifacts/sbert_finetuned/repro.md`.

### B — Action latent

```
python scripts/analyze_action_latent_space.py --tool_count 30 --checkpoint path/to/best_model.pt --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv --output artifacts/action_latent_space
```

Dry path used here: `--synthetic` (random embeddings).

### C — Zero-shot atomic

```
python scripts/eval_zero_shot_atomic.py --tool_count 15 --checkpoint path/to/checkpoints_15tools/best_model.pt --output artifacts/zero_shot_atomic
```

Dry path used here: `--dry_run`.

### D — Direction

See `artifacts/direction_ablation/repro.md`.

## Paths

Configs use repository-relative paths only. No API keys in artifacts.

## What was not reproduced

- Cell 1/2 SBERT training
- DirGNN training
- Frozen LEGR embedding dumps
- Paper tables/figures from real metrics
