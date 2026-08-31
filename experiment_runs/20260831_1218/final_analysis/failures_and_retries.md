# Failures and retries

Current suite status: **20/20 VERIFIED**. Failed attempts were archived, not deleted.

## 1. LEGR_30TOOL_ZERO_SHOT_ATOMIC attempt 1 (both datasets) — protocol rejection

- Run IDs:
  - `upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42`
  - `upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42`
- Attempt 1 status: `FAILED` / paper_state `NOT_SUPPORTED`
- Class: `implementation`
- Root cause: evaluator refused `--tool_count 30` because `routing_30tools` labels are a separate vocabulary and must not be registered into a frozen DAG embedding table.
- Archived:
  - `.../LEGR_30TOOL_ZERO_SHOT_ATOMIC__attempt1_not_supported/`
- Attempt 2 (VERIFIED): 30-tool frozen GCN + 30-tool compositional DAGs + **routing_15tools** queries and 15 one-node LEGR tools. No routing_30 aliases. No `register_tools` OOV.

## 2. Dataset B 30-tool default GCN — killed mid-train, then retrained

- Attempt 1: SIGKILL at epoch 62 (operator interrupt). Archived as `LEGR_DEFAULT_GCN_30TOOL__attempt1_killed`.
- Attempt 2: VERIFIED. SHA256 `576e98d7fe165a8d36471051048a5422cb6f6af237a1360ddc2ebb436cf927e3`.

## 3. Dataset B Task 2 — dependency failure, then rerun

- Attempt 1 blocked on (2). Archived as `LEGR_ACTION_LATENT__attempt1_dep_failed`.
- Attempt 2: VERIFIED, `REAL_CHECKPOINT_EMBEDDINGS`, evidence **NO SUPPORT**.

## 4. Checkpoint-manifest ID collision (fixed)

15-tool GCN overwrote the 30-tool key until IDs included `__tool{N}__`. Files on disk were never mixed.
