# Agentic Tool-Bound Taxonomies

Research code for two complementary experiments on **how AI agents should route a
user request to the right tool(s)**:

| # | Experiment | Entry point | Question |
|---|------------|-------------|----------|
| **1** | **Taxonomy Routing** — *Semantic* vs *Tool-Bound* hierarchies (LLM router) | `main.py` | Is it better to group tools by human topic or by the action they perform? |
| **2** | **LEGR** — *Latent Execution-Graph Routing* (trained dual-encoder) | `train.py` / `eval.py` | Can a learned text↔graph model retrieve the correct multi-step execution DAG better than S-BERT / BM25 / an LLM? |

The two experiments are independent. You can reproduce either one on its own.

---

## 1. Setup

```bash
# Clone, then from the repo root:
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### Configure an LLM (only needed for Experiment 1 and the LLM DAG baseline)

```bash
copy .env.example .env           # Windows  (cp on macOS/Linux)
```

Then edit `.env` and pick **one**:

- **Ollama (local, no API key, no rate limits — recommended):**
  `USE_OLLAMA=true` and optionally `OLLAMA_MODEL=llama3.2`
  (install [Ollama](https://ollama.com), then `ollama pull llama3.2`)
- **Gemini:** `GEMINI_API_KEY=...` (free key at [Google AI Studio](https://aistudio.google.com/app/apikey))

Experiment 2 (LEGR) needs no API key. W&B logging during training is optional
(`W&B_API_KEY=...`); training still runs without it.

---

## 2. Reproduce Experiment 1 — Taxonomy Routing

This routes every query through the **Semantic** taxonomy (baseline) and the
**Tool-Bound** taxonomy (proposed), then reports routing accuracy, branch
accuracy, hallucination rate, latency, and token cost side-by-side. The
benchmark CSVs are already in the repo under `upgraded_data/routing_<N>tools/`
(`N` = 15, 30, 45), so no data generation is needed.

**Step 1 — activate the environment** (from the repo root):

```bash
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
```

**Step 2 — run one split.** This loads `base_cleaned.csv` for the 30-tool
benchmark and prints the side-by-side report (run all commands from the repo
root):

```bash
python src/main.py --tool_count 30 --dataset_preset routing_base_cleaned
```

**Step 3 — run every split** to reproduce the full Experiment 1 table. Run these
five commands (swap `30` for `15` or `45` to reproduce the other tool tiers):

```bash
python src/main.py --tool_count 30 --dataset_preset routing_base_cleaned
python src/main.py --tool_count 30 --dataset_preset routing_lexical_cue_reduced
python src/main.py --tool_count 30 --dataset_preset routing_confusable_intents
python src/main.py --tool_count 30 --dataset_preset routing_paraphrase_train
python src/main.py --tool_count 30 --dataset_preset routing_paraphrase_test
```

**Step 4 — read the results.** Each run writes two CSVs to
`new_results/<model>_<N>tools/`:
- `experiment_log_<split>.csv` — per-query predictions
- `summary_metrics_<split>.csv` — aggregate metrics

What each split tests:

| Preset (`--dataset_preset`) | CSV loaded | Purpose |
|--------|----------------|---------|
| `routing_base_cleaned` | `base_cleaned.csv` | Clean baseline |
| `routing_lexical_cue_reduced` | `lexical_cue_reduced.csv` | Hard: direct action words removed |
| `routing_confusable_intents` | `confusable_intents.csv` | Hard: surface-similar confusable intents |
| `routing_paraphrase_train` | `paraphrase_heldout_train.csv` | Paraphrase generalization (train) |
| `routing_paraphrase_test` | `paraphrase_heldout_test.csv` | Paraphrase generalization (test) |

**Shortcut — run all splits, tool counts, and models in one command:**

```bash
python scripts/run_routing_experiments.py --tool_counts 30 45
```

To run on your own CSV instead of a preset:

```bash
python src/main.py --dataset_path path/to/your.csv
# columns auto-detected: query / transformed_query / text  and  ground_truth / label / tool
# force them if needed: --query_col my_query --label_col my_label
```

---

## 3. Reproduce Experiment 2 — LEGR (train + evaluate)

This trains the `LEGRDualEncoder` (text encoder + graph GCN) with the
Graph-Aware Contrastive Loss, then evaluates retrieval (Recall@k, MRR,
tool-set F1, GED error) against S-BERT and BM25 baselines. The training data is
already in the repo at `upgraded/upgraded_30tools/`
(`train.csv`, `dev.csv`, `test_topology_heldout.csv`).

**Step 1 — activate the environment:**

```bash
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
```

**Step 2 — train the main model (with GED loss).** Reads
`upgraded/upgraded_30tools/{train,dev}.csv` by default; saves `best_model.pt`
to `checkpoints_30tools/` (run from the repo root):

```bash
python src/train.py --tool_count 30 --checkpoint_dir checkpoints_30tools
```

**Step 3 — train the no-GED ablation model** (needed for the ablation table):

```bash
python src/train.py --tool_count 30 --lambda_ged 0 --checkpoint_dir checkpoints_30tools_no_ged
```

**Step 4 — evaluate both models and the baselines in one command.** This scores
the two checkpoints plus S-BERT and BM25 on the held-out test set and writes the
metrics CSV:

```bash
python src/eval.py --tool_count 30 \
  --checkpoint checkpoints_30tools/best_model.pt checkpoints_30tools_no_ged/best_model.pt \
  --checkpoint_labels LEGR_WITH_GED LEGR_NO_GED \
  --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv \
  --hard_negative_csv upgraded_data/graph_30tools/hard_negatives.csv \
  --save_results results/legr_ged_ablation.csv
```

To evaluate just the main model on its own:

```bash
python src/eval.py --tool_count 30 \
  --checkpoint checkpoints_30tools/best_model.pt \
  --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv \
  --hard_negative_csv upgraded_data/graph_30tools/hard_negatives.csv \
  --save_results results/legr_30tools_metrics.csv
```

**Step 5 — read the results.** Metrics are printed to the terminal and saved to
the `--save_results` CSV path you passed above.

### Optional steps

Multi-seed run for mean ± std (statistical significance):

```bash
python experiments/run_multi_seed.py --seeds 42 43 44 45 46
```

Extra baseline — let an LLM generate the DAG directly:

```bash
python src/llm_dag_baseline.py \
  --input upgraded_data/graph_30tools/test_topology_heldout.csv \
  --provider ollama --model llama3.2 --max_examples 200
```

GED-loss hyperparameter sweep:

```bash
python scripts/sweep_ged_hyperparams.py \
  --train_csv upgraded/upgraded_30tools/train.csv \
  --val_csv   upgraded/upgraded_30tools/dev.csv \
  --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv
```

Common training overrides: `--epochs`, `--lr`, `--batch_size`, and the GED-loss
knobs `--lambda_ged`, `--ged_scale`, `--ged_margin`.

---

## 4. Repository layout

All Python source lives in `src/`. Run the entry points from the repo root as
`python src/<file>.py`. Batch runners live in `scripts/` and `experiments/`.

```
src/
├── main.py                     # Experiment 1 entry point
├── evaluator.py                # runs both taxonomies, computes metrics
├── routers.py                  # two-step hierarchical LLM router
├── taxonomies.py               # Semantic + Tool-Bound trees
├── llm_backends.py             # Gemini / Ollama backends
├── dataset.py                  # canonical eval queries + dataset builders
├── routing_tiers.py            # per-tier routing tool vocabularies
├── routing_benchmark_specs.py  # 30-tool routing benchmark spec
│
├── train.py                    # Experiment 2 training loop
├── eval.py                     # evaluation + S-BERT / BM25 baselines
├── encoders.py                 # dual-encoder model (text + graph)
├── loss.py                     # Graph-Aware Contrastive Loss
├── legr_tool_count.py          # --tool_count CLI helper
├── llm_dag_baseline.py         # optional LLM-generates-DAG baseline
│
├── vocab_config.py             # shared: active tool-count switch (15/30/45)
├── data_synth.py               # shared: tool vocab + DAG/GED helpers
└── utils/                      # shared: read_datafile() CSV/Parquet loader

scripts/
├── run_routing_experiments.py  # batch runner for Experiment 1
├── sweep_ged_hyperparams.py    # optional GED-loss sweep
└── ...                         # dataset-build pipeline (not needed to reproduce)

experiments/
└── run_multi_seed.py           # optional multi-seed runner (Experiment 2)
```

### Datasets (committed, ready to use)

```
upgraded_data/routing_<N>tools/   # Experiment 1 routing splits (15/30/45 tools)
upgraded/upgraded_30tools/        # Experiment 2 LEGR train/dev/test
upgraded_data/graph_<N>tools/     # raw graph benchmark + hard negatives
```

### Tests

```bash
pytest -q          # structural regression tests (no LLM calls required)
```

---

## 5. Metrics

**Experiment 1:** routing accuracy, branch accuracy, branch-OK/tool-wrong rate,
hallucination rate, error propagation, latency (mean/median/P95), token usage, cost.

**Experiment 2:** Recall@{1,3,5}, MRR@{1,3,5}, tool-set F1, mean GED error,
hard-negative ranking accuracy, inference latency.

---

## 6. Requirements

- Python 3.10+
- Experiment 1: `google-genai` or `ollama`, plus `pandas`, `pydantic`, `python-dotenv`
- Experiment 2: `torch`, `torch_geometric`, `transformers`, `sentence-transformers`,
  `networkx`, `rank-bm25`, `scikit-learn`, `wandb` (optional)

See `requirements.txt` for pinned minimums.
