# Code changes made for this experiment campaign

These are integrity/logging patches, not hyperparameter or metric-definition changes.

1. `src/train.py`
   - Hard-fail if `--device cuda` but CUDA is unavailable.
   - Confirm parameters are on CUDA after `.to(device)`.
   - Print GPU name next to the existing "Training on" line.

2. `src/sbert_ft_baseline.py`
   - Same CUDA hard-fail / on-device check + GPU print.
   - Reload `best_model.pt` before the built-in test eval so reported numbers are from the best checkpoint, not the last epoch.

3. `src/eval.py`
   - Auto-attach packaged 30-tool hard negatives only when the eval CSV is the packaged default. An explicit `--dataset_csv` pointing at `upgraded_v3` no longer silently mixes `upgraded` hard negatives.

4. `scripts/analyze_action_latent_space.py`
   - Save `embeddings.npy` and `embedding_kind.txt` (`REAL_CHECKPOINT_EMBEDDINGS` vs `RANDOM_EMBEDDINGS_DRY_RUN`).

5. `scripts/eval_zero_shot_atomic.py`
   - `os._exit(0)` after `main()` because Windows CUDA teardown can hang.

6. `experiment_runs/20260831_1218/execute_master.py` `record_checkpoint`
   - Checkpoint IDs now include `__tool{N}__` so 15-tool and 30-tool GCN entries cannot overwrite each other.
   - `checkpoint_manifest.json` rebuilt from on-disk SHA256 after Dataset B retry.

7. `scripts/eval_zero_shot_atomic.py`
   - `--tool_count 30` is allowed for the frozen encoder / compositional DAG pool.
   - Queries and one-node candidates remain `routing_15tools` aliased onto LEGR DAG names.
   - `routing_30tools` labels are still OOV (no invented alias map, no `register_tools`).
   - CUDA hard-fail when `--device cuda` but CUDA is unavailable.
