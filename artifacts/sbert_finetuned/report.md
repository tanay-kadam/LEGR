# Fine-tuned Sentence-BERT 2×2

**Status:** IMPLEMENTED, NOT EXECUTED. Metrics below are empty on purpose.

## Question

Does LEGR beat frozen Sentence-BERT because of the GNN, because of in-domain contrastive fine-tuning, and/or because of the GED term?

## Design

| | lambda_ged = 0 | lambda_ged = 0.30 |
|---|---|---|
| Text DAG encoder (MiniLM) | Cell 1 (this code) | Cell 2 (this code) |
| GNN DAG encoder (LEGR) | Cell 3 (existing no-GED ckpt) | Cell 4 (existing full LEGR) |

Secondary row: `--tied` shares one `TextEncoder` for query and document.

## Unavoidable differences vs LEGR

- Document tower is `TextEncoder` over `dag_to_text`, not a GNN.
- Collate tokenises DAG strings instead of batching PyG graphs.
- Parameter groups are text towers, not GCN.
- Hard-negative protocol uses the same 0.5 cosine threshold on the **text** document tower.

## What graph structure can be attributed (after a real run)

- Cell 1 vs Cell 3 isolates the GNN given plain InfoNCE.
- Cell 2 vs Cell 4 isolates the GNN given GACL.
- Cell 1 vs Cell 2 asks whether GED helps a purely textual model.
- Frozen SBERT vs Cell 1 isolates in-domain contrastive fine-tuning.

Do not fill Table 2 until `eval_metrics.json` exists for the trained run.

## Comparison stubs (pending)

- Vanilla Sentence-BERT: pending `src/eval.py --checkpoint <legr>` S-BERT row, or re-run `_sbert_baseline`.
- LEGR no-GED: pending existing ablation CSV.
- Full LEGR: pending existing eval CSV.
