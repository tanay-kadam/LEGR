# Agentic Tool-Bound Taxonomies — Comparative Experiment

A research codebase that empirically compares two intent-routing strategies for AI agents:

| Strategy | Groups queries by… | Key question at each node |
|---|---|---|
| **Semantic Taxonomy** (baseline) | Human topic (*IT Support, Security, Billing*) | *"What is this query about?"* |
| **Tool-Bound Taxonomy** (proposed) | Downstream API operation (*Read, Mutate, Orchestrate*) | *"What kind of action does this query need?"* |

Both taxonomies contain the same 15 leaf-level API tools.  
The experiment routes 50 synthetic queries through each taxonomy via a two-step LLM-based hierarchical classifier, then compares **routing accuracy**, **latency**, and **token efficiency**.

## Quick Start

```bash
# 1. Clone / navigate to the project
cd "Agentic Tool-Bound Taxonomies"

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure LLM (choose one)
copy .env.example .env    # Windows
# cp .env.example .env   # macOS / Linux

# 5. Run the experiment
python main.py
```

### Using Ollama (recommended — no API key, no rate limits)

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`
2. In `.env` set: `USE_OLLAMA=true` and optionally `OLLAMA_MODEL=llama3.2`
3. Run: `python main.py`

### Using Gemini

1. In `.env` set: `GEMINI_API_KEY=...` (get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey))
2. Run: `python main.py` (free tier has rate limits; use Ollama for full 50-query runs)

### Scaled single-tool dataset (500–1000+ queries) for statistical power

For Section 4.1 / 5.1, the 50-query set is too small for statistical significance. Use the programmatic scale-up:

```bash
# 1,005 queries (67 per tool × 15 tools) — recommended for paper
python main.py --scaled --n_per_tool 67

# 510 queries (34 per tool)
python main.py --scaled --n_per_tool 34

# Save dataset to CSV, then run evaluator (e.g. overnight)
python main.py --scaled --n_per_tool 67 --save_dataset results/single_tool_1005.csv
# Later, re-run without regenerating:
python main.py --dataset_path results/single_tool_1005.csv
```

Generate the scaled dataset only (no evaluation):

```bash
python dataset.py scaled 67 results/single_tool_dataset_scaled.csv
```

### Using upgraded routing datasets (from `scripts/run_pipeline.py`)

`main.py` now supports upgraded routing splits directly via presets and
auto-maps upgraded columns (`transformed_query`, `label`) to
(`query`, `ground_truth`).

```bash
# Lexical cue reduced split
python main.py --dataset_preset routing_lexical_cue_reduced

# Confusable intents split
python main.py --dataset_preset routing_confusable_intents

# Paraphrase held-out train / test
python main.py --dataset_preset routing_paraphrase_train
python main.py --dataset_preset routing_paraphrase_test
```

Equivalent explicit paths:

```bash
python main.py --dataset_path upgraded_data/routing/lexical_cue_reduced.csv
python main.py --dataset_path upgraded_data/routing/confusable_intents.csv
python main.py --dataset_path upgraded_data/routing/paraphrase_heldout_test.csv
```

### Using upgraded graph CSVs with `llm_dag_baseline.py`

`llm_dag_baseline.py` now accepts either:
- `.jsonl` (original LEGR export format), or
- `.csv` with columns: `query`, `tools`, `edges` (plus optional `dag_id`, `dag_text`).

Example with topology-held-out graph test split:

```bash
python llm_dag_baseline.py \
  --input upgraded_data/graph/test_topology_heldout.csv \
  --provider ollama \
  --model llama3.2 \
  --max_examples 200
```

## Recommended paper datasets

For a NeurIPS-style benchmark section, report at least:

- **Routing (single-tool):**
  - Baseline: `results/single_tool_1005.csv`
  - Hard lexical generalization: `upgraded_data/routing/lexical_cue_reduced.csv`
  - Confusable intent stress-test: `upgraded_data/routing/confusable_intents.csv`
  - Paraphrase generalization: train on `paraphrase_heldout_train.csv`, evaluate on `paraphrase_heldout_test.csv`

- **Graph (multi-step / DAG):**
  - Topology OOD test: `upgraded_data/graph/test_topology_heldout.csv`
  - Hard negative retrieval pool: `upgraded_data/graph/hard_negatives.csv`
  - Optional in-domain references: `upgraded_data/graph/train.csv`, `upgraded_data/graph/dev.csv`

### Training/eval on upgraded graph splits

`train.py` supports CSV-backed train/val inputs:

```bash
python train.py \
  --train_csv upgraded_data/graph/train.csv \
  --val_csv upgraded_data/graph/dev.csv \
  --checkpoint_dir checkpoints_topology
```

`eval.py` supports CSV test override and hard-negative ranking metrics:

```bash
python eval.py \
  --checkpoint checkpoints_topology/best_model.pt \
  --dataset_csv upgraded_data/graph/test_topology_heldout.csv \
  --hard_negative_csv upgraded_data/graph/hard_negatives.csv \
  --save_results results/legr_topology_heldout_metrics.csv
```

## Project Structure

```
├── dataset.py        # 50 synthetic queries with ground-truth API labels
├── taxonomies.py     # Semantic tree (baseline) + Tool-Bound tree (proposed)
├── llm_backends.py   # Gemini + Ollama backends for structured LLM calls
├── routers.py        # Two-step LLM hierarchical router (Pydantic structured output)
├── evaluator.py      # Runs both taxonomies, computes accuracy / latency / tokens
├── main.py           # Entry point — runs experiment, prints report, exports CSV
├── requirements.txt  # Python dependencies
└── results/          # Created at runtime
    ├── experiment_log.csv      # Per-query telemetry for both taxonomies
    └── summary_metrics.csv     # Aggregate metrics side-by-side
```

## Metrics Collected

| Metric | Description |
|---|---|
| **Routing Accuracy** | % of queries routed to the correct final API tool |
| **Branch Accuracy** | % of queries routed to the correct top-level branch |
| **Error Propagation** | Count of hallucinated branch/tool names |
| **Latency** | Wall-clock time (mean, median, P95) per query |
| **Token Efficiency** | Total prompt + completion tokens consumed |
| **Estimated Cost** | USD cost (Gemini free tier = $0) |

## Extending the Experiment

- **Add queries**: Append tuples to `RAW_DATASET` in `dataset.py`.
- **Add APIs/tools**: Update `TOOL_DESCRIPTIONS` in `taxonomies.py` and place the new tool in both trees.
- **Try other models**: Pass a different `model` string in `main.py`.
- **Deeper trees**: Add a third level to the taxonomy dicts and extend `hierarchical_route()` in `routers.py`.

## Requirements

- Python 3.10+
- **LLM**: Either **Ollama** (local, recommended) or a **Gemini API key** (free tier at [Google AI Studio](https://aistudio.google.com/app/apikey))
