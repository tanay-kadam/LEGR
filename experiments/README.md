# Experiments: Paper-Ready Evaluation

This folder contains scripts to address common reviewer concerns (dataset scale, baselines, statistical significance, qualitative analysis).

## 1. Dataset scale and diversity

The main dataset now includes **scale-up templates** (see `data_synth._build_scale_up_templates`), adding programmatic 2-/3-/4-node linear chains and fan-out/fan-in DAGs so the corpus has **400+ unique DAGs** (configurable). Default scale-up is tuned for a reasonable build time; you can increase counts in `data_synth.py` for full runs.

- **Check stats:** `python -c "from data_synth import build_splits; t,v,te=build_splits(); print('Unique DAGs:', t.num_unique_dags, 'Test unique:', len(set(s.dag_id for s in te.samples)))"`

## 2. Stronger baselines (GCN vs GAT)

- **GAT encoder:** Use `--graph_encoder_type gat` when training to compare LEGR with a GAT-based graph encoder (same dual-encoder setup, different message passing).
- **Existing baselines:** S-BERT and BM25 are still run in `eval.py`. Add LEGR-GAT by training with `--graph_encoder_type gat` and evaluating that checkpoint.

## 3. Statistical significance (mean ± std)

Run multi-seed training and evaluation, then aggregate:

```bash
# Sequential (one GPU): train and eval 5 seeds, then report mean ± std
python experiments/run_multi_seed.py --seeds 42 43 44 45 46

# Parallel (e.g. 5 GPUs or eval-only): run 5 workers
python experiments/run_multi_seed.py --seeds 42 43 44 45 46 --n_workers 5

# Eval-only: aggregate existing checkpoints (checkpoints/seed_42/best_model.pt, ...)
python experiments/run_multi_seed.py --mode eval_only --checkpoint_dir checkpoints --seeds 42 43 44 45 46
```

Results are printed and saved to `experiments/multi_seed_results.json` (per-seed metrics and aggregated mean ± std for LEGR, S-BERT, BM25).

## 4. Qualitative case studies

Export examples where LEGR is correct but S-BERT or BM25 retrieve the wrong DAG (for the “why baselines fail” section):

```bash
python eval.py --export_case_studies case_studies --max_case_studies 10
```

This writes `case_studies/case_studies.json` and `case_studies/case_studies.md` with query, ground-truth DAG, baseline top-1 (wrong), and a short explanation.
