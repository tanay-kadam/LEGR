# V3 routing tool-card adaptation

## Experiment

Both existing Campaign V4 15-tool V3 checkpoints were adapted using a clean,
small tool-card corpus and evaluated on all four complete routing datasets.
The original checkpoints and the earlier frozen-evaluation artifacts were not
modified.

The adaptation corpus contains 90 training and 30 validation sentences made
only from the 15 tool names, the two routing aliases, and registry descriptions.
No Standard, Lexical, Confusable, Paraphrase, or paraphrase-training query was
used for adaptation. This avoids source-family leakage while retaining all
1,005 Standard, 1,005 Lexical, 450 Confusable, and 1,255 Paraphrase queries for
evaluation.

Each candidate remains a one-node, zero-edge graph. Adaptation uses 15-way
cross-entropy over the complete candidate gallery. It does not use the GED
auxiliary; `GED` and `no-GED` below identify the source checkpoint initialization.

## Accuracy and macro-F1

| Initialization | Stage | Standard | Lexical | Confusable | Paraphrase | Micro accuracy |
|---|---|---:|---:|---:|---:|---:|
| V3 no-GED | Frozen | 52.44 / 49.53 | 28.66 / 26.06 | 40.22 / 33.75 | 51.24 / 49.32 | 44.12 |
| V3 no-GED | Adapted | **59.00 / 57.87** | **28.96 / 24.72** | **46.67 / 40.41** | **53.78 / 55.04** | **47.62** |
| V3 GED | Frozen | 53.13 / 50.20 | **27.36 / 24.87** | 40.00 / 33.98 | 53.07 / **51.03** | **44.55** |
| V3 GED | Adapted | **53.33 / 49.63** | 25.77 / 21.48 | **40.89 / 34.98** | **53.15 / 49.59** | 44.31 |

## Why the improvement is relatively small

The original frozen-transfer error was not primarily a vocabulary mismatch.
Only two routing labels differ from the Campaign V4 tool names:

- `query_database` maps to `db_read`.
- `update_database` maps to `db_write`.

The other 13 routing tools already have the same names seen during Campaign V4
training. Consequently, tool-name adaptation had limited room to improve the
model by repairing aliases alone.

The adaptation corpus is also intentionally small and conservative. It contains
90 generic tool-card sentences rather than realistic routing utterances. The
model reaches 100% tool-card validation accuracy, showing that it learns the
descriptions, but this validation distribution is much easier than the routing
benchmarks. In particular, tool descriptions do not teach the model how to
recover an intent after informative lexical cues have been removed.

The aggregate result also hides large improvements and regressions that partly
cancel. For the no-GED initialization on Standard queries:

- `generate_report` recall increases from 0.21 to 0.97.
- `provision_vm` recall increases from 0.10 to 0.67.
- `process_refund` recall increases from 0.00 to 0.46.
- `check_status` recall decreases from 0.39 to 0.00.
- `restart_service` recall decreases from 0.81 to 0.54.
- `db_read` recall decreases from 0.52 to 0.34.

After adaptation, many `check_status` requests are incorrectly assigned to
`generate_report`. This is consistent with mild catastrophic forgetting: a
small repetitive corpus moves selected class boundaries strongly but does not
represent the diversity needed to preserve every existing distinction. The
Lexical condition changes by only +0.30 accuracy points, and its macro-F1 falls,
because the adaptation does not supply cue-reduced or difficult sibling-tool
examples. The Lexical dataset itself also has a 19.7% exact-duplicate rate after
transformation, which can make some examples intrinsically difficult when
identifying cues have been removed.

Finally, atomic routing does not exercise most of V3's graph architecture. Every
candidate contains one node and no edges, so there are no predecessor/successor
messages and no meaningful topology. In this setting V3 behaves largely as a
text-to-tool retriever rather than a graph-structure retriever.

The +3.50-point aggregate improvement for the no-GED initialization is therefore
reasonable: adaptation repairs several tool concepts but does not provide enough
realistic intent supervision to learn robust routing.

## Recommended stronger follow-up

A stronger clean experiment should train on a new independent corpus of
realistic routing utterances, ideally 50--100 per tool, covering explicit,
cue-reduced, confusable, and paraphrased requests plus hard negatives such as
read versus write and status versus report. None of those new sentences should
appear in the four evaluation CSVs. This would test whether V3 can learn atomic
routing rather than only memorize tool-card descriptions.

Cells are accuracy percentage / macro-F1 percentage. Bold compares frozen and
adapted stages within the same initialization.

## Frozen-to-adapted accuracy change

| Initialization | Standard | Lexical | Confusable | Paraphrase | Micro aggregate |
|---|---:|---:|---:|---:|---:|
| V3 no-GED | +6.57 | +0.30 | +6.44 | +2.55 | +3.50 |
| V3 GED | +0.20 | -1.59 | +0.89 | +0.08 | -0.24 |

## Interpretation

The clean tool-card adaptation helps the no-GED initialization understand the
routing vocabulary: its aggregate accuracy rises from 44.12% to 47.62%, with
clear improvements on Standard, Confusable, and Paraphrase queries. It does not
solve lexical robustness; Lexical accuracy changes by only +0.30 points and its
macro-F1 falls slightly. The GED-initialized checkpoint receives no aggregate
benefit from the same adaptation.

This is stronger and cleaner than the frozen transfer result, but the adapted
no-GED model remains far below the two-stage Functional LLM routers. The result
supports a limited claim: lightweight routing-tool adaptation improves V3's
atomic transfer, especially when action cues remain available, but registry
descriptions alone are insufficient for cue-reduced intent routing.

## Artifacts

- `results.json`: configuration, input hashes, checkpoint hashes, and metrics.
- `routing_metrics.csv`: frozen and adapted metrics for every condition.
- `adaptation_train_toolcards.csv` and `adaptation_dev_toolcards.csv`: exact
  clean adaptation corpus.
- `v3_no_ged_15t/best_model.pt` and `v3_ged_15t/best_model.pt`: separate adapted
  checkpoints.
- Each checkpoint directory also contains per-query predictions, per-tool
  precision/recall/F1, confusion matrices, and training history.
- `reproduce.txt`: exact rerun command.
