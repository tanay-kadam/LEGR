# LEGR Campaign v4 — Execution Plan

**Date:** 2026-08-31
**Role:** RESEARCH ARCHITECT
**Depends on:** architecture_audit.md

---

## Phase Summary

| Phase | Description | Depends On | Est. Time | Azure Cost |
|-------|-------------|------------|-----------|------------|
| 0 | Repository + architecture audit | — | COMPLETE | $0 |
| 1 | Tool registry design | Phase 0 | 30 min | $0 |
| 2 | Topology generator + split design | Phase 1 | 1 hr | $0 |
| 3 | Local DAG generation (no Azure) | Phase 2 | 1 hr | $0 |
| 4 | Dataset structural tests | Phase 3 | 30 min | $0 |
| 5 | Small Azure pilot (10 DAGs × 3 tiers) | Phase 4 | 30 min | ~$1-3 |
| 6 | Azure pilot validation + cost report | Phase 5 | 15 min | $0 |
| 7 | Full Azure query generation | Phase 6 | 2-4 hrs | ~$15-25 |
| 8 | Complete dataset validation | Phase 7 | 30 min | $0 |
| 9 | CUDA model smoke tests | Phase 8 | 15 min | $0 |
| 10 | 15-tool one-seed campaign | Phase 9 | 30-60 min | $0 |
| 11 | Manager review | Phase 10 | 15 min | $0 |
| 12 | 30-tool one-seed campaign | Phase 11 | 30-60 min | $0 |
| 13 | Manager review | Phase 12 | 15 min | $0 |
| 14 | 45-tool one-seed campaign | Phase 13 | 60-90 min | $0 |
| 15 | Manager review | Phase 14 | 15 min | $0 |
| 16 | Llama + GPT-OSS evaluation | Phase 15 | 1-3 hrs | $0 (Ollama) |
| 17 | Hard-topology evaluation | Phase 16 | 30 min | $0 |
| 18 | Additional seeds (42, 123, 2026) | Phase 17 | 2-4 hrs | $0 |
| 19 | Final tables + report | Phase 18 | 1 hr | $0 |

---

## Phase 1 — Tool Registry Design

### Actions
1. Create `data/campaign_v4/tool_registry.csv` with all 45 tools
2. Create `data/campaign_v4/tools_15.json`, `tools_30.json`, `tools_45.json`
3. Verify nested subset invariant: `set(15) ⊂ set(30) ⊂ set(45)`
4. Verify functional category balance: 6/6/3 → 12/12/6 → 18/18/9

### Tool Naming Convention
- ACTION_FIRST snake_case
- Tool name IS the description (no separate description field)
- Examples: `read_user_profile`, `edit_username`, `dispatch_message_to_usergroup`

### Tool Library (from prompt specification)

**TIER 1 (15 tools):**
- DATA_RETRIEVAL (6): read_user_profile, read_database_record, read_access_logs, check_service_status, read_subscription_status, scan_system_for_malware
- STATE_MODIFICATION (6): edit_username, write_database_record, reset_user_password, update_subscription_plan, restart_service, create_support_ticket
- ORCHESTRATION (3): dispatch_message_to_usergroup, escalate_case_to_human, route_task_by_condition

**TIER 2 (+15 = 30 tools):**
- DATA_RETRIEVAL (6): read_invoice, read_ticket_status, read_vm_status, read_security_policy, read_group_members, read_audit_events
- STATE_MODIFICATION (6): issue_refund, provision_virtual_machine, quarantine_system, update_security_policy, add_user_to_group, append_audit_event
- ORCHESTRATION (3): fanout_tasks_to_workers, merge_parallel_results, retry_failed_task

**TIER 3 (+15 = 45 tools):**
- DATA_RETRIEVAL (6): read_payment_status, read_deployment_status, read_backup_status, read_api_key_status, read_incident_timeline, read_usage_report
- STATE_MODIFICATION (6): cancel_payment, deploy_application, create_backup, rotate_api_key, resolve_incident, update_usage_quota
- ORCHESTRATION (3): schedule_dependent_tasks, wait_for_parallel_tasks, dispatch_approval_workflow

### Validation Assertions
```python
assert set(tools_15) < set(tools_30)
assert set(tools_30) < set(tools_45)
assert len(tools_15) == 15
assert len(tools_30) == 30
assert len(tools_45) == 45
# Category balance
assert count_retrieval(15) == 6
assert count_modification(15) == 6
assert count_orchestration(15) == 3
```

---

## Phase 2 — Topology Generator + Split Design

### Training Topology Families
1. single_node
2. chain_short (2 nodes)
3. chain_medium (3-4 nodes)
4. chain_long (5+ nodes)
5. fanout (1→2+)
6. fanin (2+→1)
7. hourglass (fan-in then fan-out)
8. y_shape (2→1→1)
9. inverted_y (1→1→2)
10. fork_join (generic fork+join)
11. w_shape (parallel chains merging)
12. multi_branch_independent
13. wide_fanout (1→3+)
14. wide_fanout_deep
15. long_chain_branched
16. deep_asymmetric_merge

### PRIMARY HELD-OUT Topology Families
- **diamond** (0→1, 0→2, 1→3, 2→3)
- **asymmetric_fork_join** (0→1, 0→2, 1→3, 2→3, 3→4)

### Optional Challenge
- **double_diamond** — only if Architect determines it is sufficiently different from diamond

### Split Structure
```
TRAIN:          only training topology families
VALIDATION:     training families, unseen DAG instances + unseen queries
TEST_INDOMAIN:  training families, unseen DAGs + unseen queries
TEST_TOPOLOGY_HELDOUT: ONLY diamond + asymmetric_fork_join
TEST_STRUCTURAL_HARD:  subset where same-toolset distractors exist
```

### Implementation
- Topology whitelist/blacklist enforced in code
- Hard validation: fail if any held-out topology DAG enters training
- Use `graph_utils.py` existing generators + add new ones

---

## Phase 3 — Local DAG Generation (No Azure)

### Structural Twin Generation Strategy
For each eligible tool multiset:
1. Select N tools (3-6) from the tier's vocabulary
2. Generate ALL valid topologies for that tool set from the template library
3. Ensure at least 2 (preferably 3+) different directed edge structures per tool set
4. Assign `structural_twin_group = hash(sorted(tool_multiset))`

### Target Counts (approximate)
| Tier | Unique DAGs | Queries/DAG | Total Rows |
|------|-------------|-------------|------------|
| 15 tools | 200-300 | 6-12 | 1,200-3,600 |
| 30 tools | 300-450 | 6-12 | 1,800-5,400 |
| 45 tools | 400-600 | 6-12 | 2,400-7,200 |

### Output Schema
```
query, dag_id, dag_text, tools, edges, topo_family, source, split,
strict_fix_applied, had_duplicate_node_labels, original_tools,
tool_count, functional_categories, canonical_dag_hash, canonical_toolset_hash,
num_nodes, num_edges, generation_prompt_hash, azure_model, generation_attempt,
query_condition, structural_twin_group, heldout_topology, dataset_version
```

---

## Phase 4 — Dataset Structural Tests

### 20-Point Test Suite
1. Schema: all required columns present
2. CSV parse: no corrupt rows
3. DAG acyclicity: all edges form valid DAGs
4. Node-index validity: edge indices within [0, N-1]
5. Edge validity: no self-loops, no duplicates
6. Topology inference: recomputed family matches stored family
7. Topology-label agreement: dag_text matches edges
8. Duplicate DAG: no identical labeled DAG across train/test
9. Split leakage: no train DAG hash in test
10. Query duplicate: no identical query strings across splits
11. Held-out family leakage: diamond/asymmetric_fork_join absent from train
12. Nested tool library: tools_15 ⊂ tools_30 ⊂ tools_45
13. Functional category balance: within 10% of target
14. Tool coverage: every tool appears in at least one DAG per tier
15. Structural-twin density: ≥80% of test queries have same-toolset distractor
16. Same-toolset/different-edge pairs exist
17. Query structural language: no forbidden graph words
18. Azure provenance: queries attributed to Azure model
19. Candidate corpus completeness: all unique DAGs present
20. Loader compatibility: existing eval.py can load the dataset

### Gate
**ALL critical tests must PASS before training proceeds.**

---

## Phase 5 — Azure Pilot

### Pilot Parameters
- 10 DAGs per tool tier (30 total)
- 4 query variations per DAG
- Total: ~120 Azure API calls

### Measurements
- Input tokens per call
- Output tokens per call
- Failed calls / retry rate
- Elapsed time
- Estimated cost (if pricing available)

### Azure Configuration
- Use existing `configs/llm_providers.json` → `azure_openai` profile
- Environment: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- Pre-flight: 1 connectivity test before batch generation

### Caching
```python
cache_key = hash(deployment + system_prompt + input_graph + query_condition + dataset_version)
```
Never pay for the same successful generation twice.

---

## Phase 6 — Azure Pilot Validation

### Create: `artifacts/campaign_v4/azure_budget_report.json`
```json
{
  "pilot_calls": 120,
  "total_input_tokens": ...,
  "total_output_tokens": ...,
  "avg_tokens_per_call": ...,
  "failed_calls": ...,
  "retry_rate": ...,
  "elapsed_seconds": ...,
  "estimated_full_cost_range": "...",
  "budget_recommendation": "proceed/reduce/abort"
}
```

---

## Phase 7 — Full Azure Query Generation

### Budget Strategy
- Soft target: ≤ $30 equivalent
- Hard safety margin: retain $10+ credit buffer
- Use exponential backoff for rate limits
- Bounded concurrency (max 5 parallel)
- Resume capability: crash at 80% doesn't restart from 0

### Query Conditions
For each DAG, generate queries in these conditions:
- `standard`: natural request with clear intent
- `paraphrase`: same intent, different syntax
- `structural_clear`: explicit dependency wording
- `structural_paraphrase`: same graph, less template-like dependency expression

### Azure Prompt Design
Do NOT list tools — describe the dependency logic:
```
The user request must require:
1. retrieving a user profile
2. after that result, scanning the relevant system
3. only after the scan, recording the audit event

Generate a natural user request. Do not mention: graph, DAG, node, edge, topology, chain.
```

---

## Phase 8 — Complete Dataset Validation

Rerun all 20 structural tests from Phase 4 on the full Azure-generated dataset.
Additional checks:
- No forbidden graph words in queries
- No excessive snake_case tool name copying
- Length range validation
- Semantic faithfulness sampling (manual spot check of 20 queries)

---

## Phase 9 — CUDA Model Smoke Tests

### Pre-flight Checks
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print(f"Allocated device: {torch.device('cuda:0')}")
```

### Smoke Test (per model variant)
- Tiny dataset: 50 samples, 5 unique DAGs
- 1 epoch
- Verify: forward pass succeeds, loss is finite, loss decreases, checkpoint saves/loads

### Model Variants to Smoke-Test
1. `legr_directed_toolname_no_ged` — New architecture, lambda_ged=0
2. `legr_directed_toolname_ged` — New architecture, lambda_ged=0.10/0.30
3. `sbert_ft_ged0` — Fine-tuned SBERT, lambda_ged=0
4. `sbert_ft_ged030` — Fine-tuned SBERT, lambda_ged=0.30

---

## Phase 10-15 — Training Campaigns

### Architecture: `legr_directed_toolname`
```
QUERY TOWER:
  query text → all-MiniLM-L6-v2 (layers 0-3 frozen)
  → mean pool → Linear(384, 256) → L2-normalize

GRAPH TOWER:
  tool_name text → all-MiniLM-L6-v2 (frozen/cached) → 384-dim
  → Linear(384, 64) → tool_text_emb
  topological rank → nn.Embedding(17, 64) → topo_emb
  [tool_text_emb; topo_emb] → DirectedGraphEncoder (W_self, W_in, W_out)
  → global_mean_pool → Linear(hidden, 256) → L2-normalize
```

### Experiment Matrix (per tool tier)

| Run ID | Architecture | lambda_ged | Seed |
|--------|-------------|------------|------|
| v4_tN_legr_directed_name_ged000_seed42 | legr_directed_toolname | 0 | 42 |
| v4_tN_legr_directed_name_gedXXX_seed42 | legr_directed_toolname | 0.10/0.30 | 42 |
| v4_tN_sbert_frozen_seed42 | sbert_frozen | — | 42 |
| v4_tN_sbert_ft_ged0_seed42 | sbert_ft | 0 | 42 |
| v4_tN_sbert_ft_ged030_seed42 | sbert_ft | 0.30 | 42 |

### Per-Tier Default Hyperparameters
| Tier | lambda_ged | epochs | batch_size |
|------|------------|--------|------------|
| 15 tools | 0.10 | 100 | 128 |
| 30 tools | 0.30 | 100 | 128 |
| 45 tools | 0.30 | 100 | 128 |

### Manager Review Checklist (after each tier)
- [ ] Training loss converges
- [ ] Validation loss tracked
- [ ] Checkpoint saved successfully
- [ ] CUDA device confirmed
- [ ] Correct dataset tier used
- [ ] No stale checkpoints reused
- [ ] Metrics are finite and non-trivial

---

## Phase 16 — LLM Baseline Evaluation

### Models
- **Llama 3.2 (3B):** via Ollama (`ollama_llama` profile)
- **GPT-OSS (120B):** via Ollama (`ollama_gpt_oss` profile)

### Evaluation Splits
- test_indomain (per tier)
- test_topology_heldout (diamond + asymmetric_fork_join)
- test_structural_hard (same-toolset distractors)

### Metrics
- Tool-set F1
- Mean GED
- Exact DAG match rate
- Parse failure rate
- Cyclic output rate
- Latency (mean, median, P95)

---

## Phase 17 — Hard-Topology Evaluation

### Per-model, per-topology breakdown:
- Diamond R@1, Exact DAG Match, Mean GED
- Asymmetric Fork-Join R@1, Exact DAG Match, Mean GED

### Structural Challenge Table
```
Model | Tool Count | Held-Out Topology | R@1 | Exact DAG Match | Mean GED | Twin Pairwise Acc
------|------------|-------------------|-----|-----------------|----------|-------------------
Frozen SBERT | 15 | Diamond | ... | ... | ... | ...
FT SBERT | 15 | Diamond | ... | ... | ... | ...
LEGR no GED | 15 | Diamond | ... | ... | ... | ...
LEGR with GED | 15 | Diamond | ... | ... | ... | ...
```

---

## Phase 18 — Additional Seeds

If Phase 17 passes:
- Rerun all neural models with seeds: 42, 123, 2026
- Report mean ± std for all metrics
- Skip if computationally impractical for full 45-tool tier

---

## Phase 19 — Final Report

### Deliverables
1. `artifacts/campaign_v4/architecture_audit.md` ✓
2. `artifacts/campaign_v4/execution_plan.md` ← this document
3. `artifacts/campaign_v4/azure_budget_report.json`
4. `artifacts/campaign_v4/dataset_test_report.md`
5. `artifacts/campaign_v4/training_summary.md`
6. `artifacts/campaign_v4/evaluation_summary.md`
7. `artifacts/campaign_v4/sbert_vs_legr_analysis.md`
8. `artifacts/campaign_v4/llm_baseline_summary.md`
9. `artifacts/campaign_v4/final_research_report.md`

### Research Questions to Answer
A. Does FT-SBERT still beat LEGR on general retrieval?
B. Does it beat LEGR on same-toolset topology discrimination?
C. Does it beat LEGR on Diamond OOD?
D. Does it beat LEGR on Asymmetric Fork-Join OOD?
E. Does GED improve LEGR?
F. Does the corrected (text-name, directed) encoder outperform legacy (integer-ID, undirected)?
G. Does performance degrade from 15→30→45 tools?
H. Does structural error increase with graph complexity?

---

## Proposed Commands for Phase 1

```bash
# Create tool registry
python -m src.data.tool_registry --validate

# Verify nested subsets
python -c "
from src.data.tool_registry import load_registry
r = load_registry()
assert set(r['tools_15']) < set(r['tools_30']) < set(r['tools_45'])
print('Nested subset invariant: PASS')
"
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Azure budget exhaustion | Medium | High | Pilot first, cache everything, token accounting |
| CUDA unavailable | Low | High | Explicit check at Phase 9, fail fast |
| Text-name features don't help LEGR | Medium | Medium | Report honestly, investigate causes |
| SBERT still wins structural-hard | Medium | Low | This is a valid research finding |
| Structural-twin density < 80% | Medium | High | Generate more twins, relax if necessary with justification |
| GED computation too slow for larger DAGs | Medium | Medium | Use fast surrogate (already in train.py) |
