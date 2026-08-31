# Reproduce direction ablation (not run in the code-first pass)

Directed:

```bash
python src/train.py --tool_count 30 --graph_direction directed --seed 42 ^
  --checkpoint_dir artifacts/direction_ablation/checkpoints/directed ^
  --train_csv upgraded/upgraded_30tools/train.csv ^
  --val_csv upgraded/upgraded_30tools/dev.csv
```

Tied W_in=W_out:

```bash
python src/train.py --tool_count 30 --graph_direction tied_in_out --seed 42 ^
  --checkpoint_dir artifacts/direction_ablation/checkpoints/tied_in_out ^
  --train_csv upgraded/upgraded_30tools/train.csv ^
  --val_csv upgraded/upgraded_30tools/dev.csv
```

Eval (after training):

```bash
python src/eval.py --tool_count 30 ^
  --checkpoint artifacts/direction_ablation/checkpoints/directed/best_model.pt ^
  --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv ^
  --save_results artifacts/direction_ablation/metrics_directed.csv
```

Default GCN training (`--graph_direction` omitted) is unchanged.
