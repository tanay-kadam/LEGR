# Reproduce fine-tuned Sentence-BERT (not run in the code-first pass)

From the repo root, after `pip install -r requirements.txt`.

Cell 1 (untied, InfoNCE):

```bash
python src/sbert_ft_baseline.py --tool_count 30 --lambda_ged 0 --seed 42 ^
  --checkpoint_dir artifacts/sbert_finetuned/checkpoints/cell1_untied_noged ^
  --wandb_run_name sbert-ft-cell1
```

Cell 2 (untied, GACL):

```bash
python src/sbert_ft_baseline.py --tool_count 30 --lambda_ged 0.30 --seed 42 ^
  --checkpoint_dir artifacts/sbert_finetuned/checkpoints/cell2_untied_ged ^
  --wandb_run_name sbert-ft-cell2
```

Tied InfoNCE:

```bash
python src/sbert_ft_baseline.py --tool_count 30 --lambda_ged 0 --tied --seed 42 ^
  --checkpoint_dir artifacts/sbert_finetuned/checkpoints/tied_infonce
```

Copy hyperparameters from an existing LEGR checkpoint instead of defaults:

```bash
python src/sbert_ft_baseline.py --tool_count 30 --lambda_ged 0 ^
  --legr_checkpoint checkpoints_30tools/best_model.pt ^
  --checkpoint_dir artifacts/sbert_finetuned/checkpoints/cell1_from_legr_cfg
```

Optional: pass `--legr_checkpoint` so `TrainConfig` fields are copied from the stored `config` dict.
