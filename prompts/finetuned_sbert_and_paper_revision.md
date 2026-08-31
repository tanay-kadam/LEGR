# Task: Fine-tuned S-BERT baseline + paper revision

You are working in the repo `Agentic Tool-Bound Taxonomies` (LEGR: Latent Execution-Graph
Routing). There are three parts to this task. Do them in order. Part A produces new numbers;
Parts B and C consume them.

**Read before you start:** `src/train.py`, `src/eval.py`, `src/loss.py`, `src/encoders.py`,
`src/data_synth.py` (specifically `dag_to_text`, `dag_to_pyg`, `build_ged_matrix`), and
`src/utils/graph_utils.py`.

**Hard rules for the whole task:**

- Never invent a number. Every figure that ends up in the paper must be traceable to a file
  committed under `new_results/`. If you cannot reproduce a number, say so explicitly rather
  than carrying the old value forward.
- Do not modify, overwrite, or delete anything in `checkpoints_15tools/`,
  `checkpoints_30tools/`, `checkpoints_45tools/`, or any existing CSV in `new_results/`.
  New artifacts go in new directories.
- Keep a running changelog at `CHANGELOG_revision.md`: one line per change, with the file
  touched and the evidence (path + value) that justified it.

---

## Part A — Implement and run a fine-tuned S-BERT baseline

### A.1 Why this baseline exists

Today the paper compares LEGR against an **off-the-shelf** S-BERT retriever. Look at
`_sbert_baseline` in `src/eval.py`: it loads the *same* backbone as LEGR's text tower
(`sentence-transformers/all-MiniLM-L6-v2`, via `_text_model_name(cfg)`), encodes the query
with it, encodes the DAG as the canonical sorted edge string produced by `dag_to_text`
(e.g. `db_read -> create_ticket, db_read -> process_refund`), and ranks by cosine similarity.

So LEGR differs from that baseline along **three axes simultaneously**:

1. it was trained on in-domain data, the baseline was not;
2. it uses the graph-aware contrastive objective with the GED term, the baseline uses no
   objective at all;
3. its document side is a GNN over real topology, the baseline's is a transformer over a
   linearized string.

Because all three move together, the current results **cannot** attribute LEGR's gain to the
graph encoder. Any in-domain fine-tuning would improve the baseline, possibly by most of the
reported margin. The purpose of Part A is to hold axes 1 and 2 fixed so that the document-side
encoder is the only variable.

### A.2 The experimental design (a 2×2, not a three-way ladder)

| | Loss = plain InfoNCE (`lambda_ged = 0.0`) | Loss = GACL (`lambda_ged = 0.30`) |
|---|---|---|
| **DAG encoder = serialized text (MiniLM)** | Cell 1 — **new** | Cell 2 — **new** |
| **DAG encoder = GNN (LEGR)** | Cell 3 — already exists (the λ_GED ablation) | Cell 4 — already exists (main LEGR result) |

Only Cells 1 and 2 need to be trained. Cells 3 and 4 already exist in the repo — locate them
and reuse the saved numbers; do not retrain LEGR.

This decomposition matters: comparing Cell 2 to Cell 4 isolates the **graph encoder's**
contribution, and comparing Cell 1 to Cell 2 shows whether the **GED term** helps a purely
textual model too. If the GED term helps both, it is a property of the objective rather than
of the graph encoder, and the paper should say so.

Also keep the existing frozen S-BERT row so the paper reports the full progression:
frozen → fine-tuned (text DAG) → LEGR (GNN DAG).

### A.3 Implementation

Create `src/sbert_ft_baseline.py`. Do not fork `train.py`; import from it so the recipes cannot
drift apart.

Architecture — a two-tower bi-encoder:

- **Query tower:** `all-MiniLM-L6-v2` + mean pooling + linear projection to `embed_dim`,
  L2-normalized. This must be byte-for-byte the same construction as LEGR's `TextEncoder` in
  `src/encoders.py` — import and reuse the class, don't reimplement it.
- **Document tower:** a *second, untied* `TextEncoder` instance, fed the tokenized output of
  `dag_to_text(G)` instead of a graph. Untied is the fair analogue, because LEGR also has two
  independent encoders. Additionally run a **tied-weights variant** as a secondary row, since a
  reviewer may argue the untied version has an unfair parameter advantage — see A.6.
- **Loss:** reuse `GraphAwareContrastiveLoss` from `src/loss.py` unmodified, including the
  learnable `nn.Parameter` temperature. The GED matrix comes from the same
  `build_ged_matrix` / `train.py` surrogate path used for LEGR, so the supervision signal is
  identical.

Hyperparameters — **do not retype them from this document.** Load
`checkpoints_{15,30,45}tools/best_model.pt`, read the stored `config` dict, and copy it
programmatically. Log the resolved config so it appears in the run artifact. For reference,
these should come out as: MiniLM bottom-4-of-6-block freezing, AdamW with `lr=2e-4` on the head
and `text_backbone_lr=2e-5` on the backbone, `weight_decay=1e-4`, `max_grad_norm=1.0`, 3-epoch
linear warmup then cosine annealing to `text_backbone_lr * 0.01 = 2e-7`, `batch_size=128`,
`max_length=128`, 100-epoch budget with `patience=15`, `temperature_init=0.05`,
`ged_scale=2.5`, `ged_margin=0.05`. If what you read from the checkpoints disagrees with this
list, trust the checkpoints and flag the discrepancy in the changelog.

Apply the same freezing policy to *both* towers.

### A.4 Data — exact paths

Use only `upgraded/`. Do not regenerate data.

| Tier | Train | Val | Test |
|---|---|---|---|
| 15 tools | `upgraded/upgraded_15tools/train.csv` | `.../dev.csv` | `.../test_topology_heldout.csv` |
| 30 tools | `upgraded/upgraded_30tools/train.csv` | `.../dev.csv` | **both** `.../test_topology_heldout_1200.csv` and `.../test_topology_heldout.csv` |
| 45 tools | `upgraded/upgraded_45tools/train.csv` | `.../dev.csv` | `.../test_topology_heldout.csv` |

Two notes on the 30-tool tier:

- `test_topology_heldout_1200.csv` (1200 rows, 90 unique DAGs) is the split the paper's numbers
  were computed on. `test_topology_heldout.csv` (332 rows) is a *different, later* file sitting
  in the working tree. Evaluate on both and report both; the paper should use the 1200-row file
  for consistency with existing tables and must say which file it used.
- `upgraded/upgraded_30tools/hard_negatives.csv` exists only at this tier. Run the
  hard-negative condition here via `evaluate_hard_negatives` in `src/eval.py`.

### A.5 Evaluation — reuse, don't reimplement

Import `CSVEvalDataset`, `compute_metrics`, `evaluate_hard_negatives`, and `benchmark_latency`
from `src/eval.py` so the new baseline is scored by identical code. Report the same metric set
as the existing tables: Recall@1 / @3 / @5, MRR, Tool-Set F1, Exact Match, Mean GED.

Additional requirements:

- **Seeds.** Run seeds 42, 43, 44 and report mean ± std. This is cheap (MiniLM is ~22M
  parameters over a few thousand rows) and the existing LEGR numbers are single-seed at 42, so
  also report the seed-42 row alone for a like-for-like comparison.
- **Leakage-aware breakdown.** At 30 tools, 25 of the 90 test DAGs (346 of 1200 rows) also
  appear in training. Reuse `scripts/audit_leakage_impact.py` to split results into
  seen-in-train vs. unseen-labelled-DAG (854 rows) and report both for the new baseline, exactly
  as is being done for LEGR.
- **Parameter counts and latency.** Report trainable and total parameters per tower and per
  system, plus latency measured by the same protocol. This matters for the efficiency table: a
  second MiniLM document tower is far larger than LEGR's GNN, so if the fine-tuned text baseline
  matches LEGR on accuracy, LEGR's remaining advantage is size and document-side encoding cost.
  Measure that honestly rather than assuming it.
- **Latency protocol.** The current `benchmark_latency` times tokenization plus the text tower
  only, at batch size 1, over 100 queries, with no warmup iterations, no `cuda.synchronize()`,
  and the top-k search excluded. Fix it: add warmup, add `cuda.synchronize()`, and report two
  numbers — encode-only and end-to-end including top-k. Apply the corrected protocol to LEGR
  and both baselines so the comparison stays valid, and record the old and new values.

### A.6 Secondary rows

1. **Tied-weights variant** of the fine-tuned baseline (single shared MiniLM for both towers),
   to preempt the parameter-fairness objection.
2. **Verbalized frozen S-BERT.** The current frozen baseline is fed arrow notation
   (`db_read -> create_ticket`), which is out-of-distribution input for a sentence encoder
   trained on natural language — meaning the existing frozen baseline is probably *weaker than a
   fair frozen baseline*, and the reported gap flatters LEGR. Add a frozen row where the DAG is
   verbalized into a sentence (e.g. "First read the database, then create a ticket and process a
   refund"). Write the verbalizer as a deterministic function over the topological order. Report
   it alongside the arrow-notation row.

### A.7 Outputs

- Checkpoints: `checkpoints_sbert_ft_{15,30,45}tools/seed{N}/`
- Metrics: `new_results/sbert_ft/{tier}_{variant}_seed{N}.csv` plus an aggregated
  `new_results/sbert_ft/summary.csv` with one row per (tier, variant, seed).
- Full 2×2 table: `new_results/sbert_ft/apples_to_apples.csv`, with the LEGR cells filled from
  existing results and the source file for each cell named in a `source` column.
- A short `new_results/sbert_ft/README.md` recording the exact commands run, the resolved
  config, package versions, and hardware.
- Add `sentence-transformers` and anything else new to `requirements.txt` with versions.
  Note that `requirements.txt` currently has mojibake in its comment headers — fix those while
  you are in the file.

### A.8 Interpretation

Report what you find, not what would be convenient. In particular, there is a real chance the
fine-tuned text baseline lands close to LEGR on the standard held-out split: the DAG strings are
short and canonical, the tool names are semantically transparent, and matching a query against
roughly 90–200 short strings is not a hard retrieval problem. If that happens, do **not** bury
it. Check specifically whether LEGR separates on:

- the **hard-negative** condition, where distractors share the same tool set and differ only in
  wiring (two rewirings of the same tools produce nearly identical strings but genuinely
  different message passing and topological positions);
- **unseen topologies**, especially the 45-tool tier, which is the only fully clean split;
- **Mean GED / Exact Match** rather than Recall@1, since structural error is the claim.

Write 300–500 words in `new_results/sbert_ft/FINDINGS.md` stating plainly which of the three
confounded axes actually accounts for the gain, and what the paper is now entitled to claim.

---

## Part B — Paper corrections from the external review

The review is at `C:\Users\kadam\Downloads\Review_from_cursor.docx` (extracted text:
`temp/review_from_cursor.txt`). Read it in full and address every item.

### B.0 Two blockers to resolve first

1. **`paper.tex` in the repo is the old v1 draft.** It still contains Stage 0, the
   confidence-weighted extraction pipeline, and the "wrong topology in 43% of cases" audit. The
   review targets **v2**, which exists only as `C:\Users\kadam\Downloads\Functional_Taxonomy_v2.pdf`.
   Do not apply v2 section-numbered edits to the v1 file. Ask for the v2 `.tex` source before
   editing. If it cannot be supplied, reconstruct from the PDF and flag every place where the
   reconstruction is uncertain.
2. **The review document starts at item 2.** Item 1 is missing from the file. Ask what item 1
   was before declaring the review addressed.

### B.1 Claims that contradict the implementation (fix the prose, not the code)

- **Eq. (1) describes a directed GNN that was never implemented.** §3.3 argues that symmetric
  GCNs "would destroy the directed nature of execution graphs" and then presents separate
  `W_self`, `W_in`, `W_out` matrices. The code uses PyG `GCNConv` (symmetric), and `dag_to_pyg`
  deliberately bidirectionalizes every edge before message passing. The only direction signal is
  a topological-rank positional embedding concatenated to node features. Rewrite this honestly:
  a symmetric GCN over an undirected view of the DAG, with execution order injected
  positionally. That is a defensible design — three hops cover the whole graph while order is
  carried positionally — but the paper currently claims the opposite of the code, in an equation,
  after arguing why the code's approach would fail.
- **Node features do not come from the text backbone.** §3.3 claims initialization "from the
  same text backbone (tool-name and description embeddings)." The code uses
  `nn.Embedding(num_tools + 1, 64)`, randomly initialized and learned. A `use_text_node_features`
  flag exists but defaults to `False`, is never set by `TrainConfig`, and even when enabled feeds
  a zero tensor into the projection. Text node features were never used in any run.
- **WordNet was not used.** §4.1 claims "synonym substitution was enforced via WordNet [14]
  rather than manual author selection." Substitution comes from `TOOL_SYNONYMS`, a hand-written
  dictionary in `src/replace_templates.py` with four manually chosen phrases per tool. There is
  no `nltk` in `requirements.txt`. This is doubly bad because the sentence's stated purpose is to
  establish that the variants were *not* author-selected, when they were.
- **Edges were not populated by type matching.** The same paragraph claims "strict input-output
  type matching between tools." In `src/utils/graph_utils.py` every generator calls
  `_sample_tools(rng, n, vocab)` — a uniform random draw — and drops the tools into a hard-coded
  edge pattern (`gen_diamond` is always `[(0,1),(0,2),(1,3),(2,3)]`). There is no type system.
  The topology is controlled; the tool assignment is random.
- **"Pre-validated corpus" no longer has a mechanism.** Dropping Stage 0 removed the
  acyclicity-enforcement pipeline, but §3.3 and §5.3 still lean on "pre-validated execution
  graphs." Either restore a sentence describing actual validation, or drop the word.
- **§3.4's GED sentence is wrong in both directions.** It says exact GED is NP-hard so the matrix
  is precomputed "using standard approximation heuristics." In fact the *evaluation* metric is
  exact `networkx.graph_edit_distance` with unit costs (tractable because DAGs have ≤ 7 nodes),
  while the *training* prior is a bespoke hand-rolled surrogate in `train.py` (node/edge count
  deltas + tool-multiset difference + a degree-sum term). Neither is a standard approximation
  heuristic. Stating both explicitly makes the setup look more careful, not less.
- **τ is never described as learnable.** §3.4 says only "where τ is a temperature." It is an
  `nn.Parameter` initialized at 0.05 and converging to ≈0.030. Say so.

### B.2 The split-guarantee claim is false at two of three scales

§4.2 claims "The training split contains only Chain and Hourglass graphs, while the Diamond
family is reserved exclusively for evaluation… ensuring no identical (tool, edge) signature
appears in both train and test." Audited with the labelled-DAG hash:

- **45 tools:** clean. Zero labelled-DAG overlap, zero unlabelled-topology overlap, zero family
  overlap.
- **30 tools:** 25 of 90 test DAGs (346 of 1200 rows) also appear in train, and train contains
  all four test families including diamond. The dedup sentence is directly contradicted.
- **15 tools:** zero labelled-DAG overlap, but all 14 test topologies appear in train
  (unlabelled), and train spans 20 families including diamond.

Also, "exclusively contains Diamond topologies" (Table 2 caption, §5.2) is wrong at every tier:
15-tool held-out has 14 families, 30-tool has 4, 45-tool has 5.

Rewrite §4.2 to describe what the splits actually are, per tier. Then **report the mitigating
result, because it is genuinely good**: on the 854 rows whose labelled DAG never appears in
training, Recall@1 is 0.9930, *higher* than the 0.9798 on leaked rows. Per-family on unseen DAGs:
diamond 0.9970, hourglass 0.9848, inverted_y 1.0000, wide_fanout 1.0000. The conclusion survives.
Lead the structural-generalization argument with the 45-tool split, since that one is airtight.

### B.3 Equations that don't match `loss.py`

Eq. (4) shows exponentiated negatives, `e^{S_ij}(1 - w̃_ij)`. The code exponentiates only the
positive and sums **raw** `S_ij(1 - w̃_ij)` for negatives, so `L_ged` is not a log-softmax and its
denominator can go negative — which is why `clamp(min=1e-8)` guards it. Eq. (5)'s `clip(·, 0, 1)`
is really `clamp(min=0)` divided by the dataset-global max GED; the ranges coincide but the
mechanism differs. Rewrite as a GED-weighted margin penalty. The ablation still supports it.

### B.4 Numbers to fix

- **Llama 3.2 is not 8B.** Table 3 says "≈ 8.0 Billion." The model actually run is
  `llama3.2:latest`, which `ollama list` reports at 2.0 GB — the 3B variant. Llama 3.2 text
  models are 1B and 3B; 8B is Llama 3.1. Correct the parameter count and re-check any
  parameter-normalized claim that depends on it.
- **BM25 30-tool Tool-Set F1 is 0.158, not 0.150.** `new_results/legr_30tools.csv` gives 0.15774.
  Every other cell in Table 4 verified exact — all 15 LEGR/S-BERT/BM25 cells at the other scales
  and all six generative cells against the `llm_dag_*` CSVs. v1 had 0.158 correct, so this reads
  as an unexplained edit.
- **Table 1 is not traceable to any saved log.** The draft has GPT-OSS Standard 78.3 → 93.3 and
  Llama Standard 70.0 → 83.8. `new_results/gpt-oss_15tools/summary_metrics_base_cleaned.csv`
  gives 79.0 → 86.5 and the Llama equivalent gives 62.7 → 78.4. The topic-based numbers are
  close; the functional ones are not. The internal arithmetic is self-consistent (18.5 and 7.5
  point deltas both check out), so either these came from a run whose logs are not in the repo,
  or the table needs regenerating. Regenerate from the summaries that exist, or produce the
  missing logs. Do not ship unsourced numbers.
- **Latency compares local against cloud without saying so.** LEGR's 4.39 ms was measured on the
  local RTX 4050; `gpt-oss:120b-cloud` runs on Ollama Cloud, so its 2,742 ms includes network
  round-trip. Add a footnote, or the 625× headline is attackable. Also reconcile the reported
  GPU: earlier material says RTX 4060 and the review says 4050 — determine which is correct and
  use it consistently.

### B.5 Overclaims to soften

- **"Completely avoiding structural edge hallucinations" conflates two things.** LEGR always
  returns a structurally valid DAG because it retrieves from a fixed corpus — true and worth
  saying. But it does make edge-level errors: there are 13 misretrievals on the 30-tool split
  whose failure mode is precisely edge errors (a reversed `rotate_api_key` edge turning a fan-out
  into a fan-in; `enable_feature_flag` placed as a source instead of a sink). A Mean GED of 0.111
  rather than 0.000 is itself the proof. The claim must be "never produces a cyclic or
  non-executable plan," not "avoids edge errors."
- **"Strictly required" is too strong for the GED ablation.** Without the GED term, Recall@1 is
  still 0.944 and F1 0.984. It halves structural error; it is not load-bearing for basic
  function. "Substantially improves" is accurate and still makes the point.

### B.6 References — treat as an integrity issue

Several entries look machine-generated: [10] DSPy's author list ("Priyanjana Paranjape, Mikhail
Pu, Ashutosh Bhatia, Zaid Akbar, Sanneh Singh") does not match the real paper; [20] contains
"Yuqian facility Jiang"; [39] has "Joseph E Ba, Joseph E Gonzalez"; [14] reads "lexical
dtabase"; [40] is a Twitter URL with a 2026 date and an unverified status ID; [9] and [15] are
the same ReAct paper twice; [15]–[41] appear largely uncited. Fabricated author lists in a
NeurIPS submission are treated as misconduct, not typos. Verify **every** entry against the real
bibliography, remove uncited entries, and de-duplicate. Report anything you cannot verify rather
than guessing.

### B.7 Structural fixes

- Cross-references are broken: §5.3 points to "Table 3 and Figure 3… in Appendix A," but
  Appendix A contains Table 4 and Figure 4, and Table 3 is the efficiency table in §5.4.
  Figure 2's caption begins "Figure 1: Teaser." Fix all of them and verify by compiling.
- Citations `[?]` are unresolved in five places: SciToolAgent, GTool, OptGraph, TGR, and one
  in §1.
- The code link in the abstract is an unresolved placeholder.

### B.8 Rewrite §4.4 (highest value per unit effort)

§4.4 is eight lines and omits nearly everything needed for reproducibility. All of the following
is verified from the checkpoint configs — write it up properly: partial freezing (bottom 4 of 6
MiniLM blocks), AdamW with 2e-5 / 2e-4 differential learning rates, weight decay 1e-4, gradient
clipping 1.0, 3-epoch warmup then cosine to 2e-7, batch size 128, 100-epoch budget with patience
15 (stopping at epochs 56 and 73), learnable τ initialized at 0.05 converging to ≈0.030,
`ged_scale` 2.5, `ged_margin` 0.05, `lambda_ged` 0.30, seed 42, exact per-condition query counts,
split sizes, hardware, and the latency protocol (including the corrections from A.5).

### B.9 The opportunity the review flags — take it

Cutting Stage 0 also cut the structural validity audit, which was the paper's sharpest empirical
weapon and is exactly what would substantiate the new "edge hallucination" framing that §5.3
currently asserts with no numbers. The data exists:

- At 30 tools, Llama 3.2 produced cyclic graphs in 9.8% of parsed predictions (39/398) while
  reaching Tool-Set F1 of 0.738; GPT-OSS produced 0%.
- On the 13 queries where LEGR misretrieves, Llama emitted a cycle on 5 of 13 and failed to
  parse 2, while GPT-OSS exactly matched 8 of 13.

Restore this as a validity audit with those numbers. Then add to the limitations section the most
interesting result available: LEGR's residual errors are near-misses with the correct DAG at rank
2 in 10 of 13 cases, sometimes losing by as little as 0.002 cosine, and a 120B generator recovers
most of them. Since Recall@5 = 1.0, that directly motivates a concrete hybrid — retrieve by
default, escalate to generation only on low-margin queries. It turns the weakest result into a
well-supported future-work proposal and preempts the obvious reviewer question about what the
missing 3.7% looks like. The 13 cases are already formatted in
`case_studies/legr_failure_cases.md`.

---

## Part C — Additional corrections not in the external review

### C.1 Dataset composition facts the paper gets wrong or omits

Verified by counting nodes per row across every split:

- **Single-node graphs exist**, but only in training splits and in 30-tool dev; **zero** in every
  test split. Counts: 15T train 58 rows / 2 unique DAGs; 30T train 20 rows / 1 DAG; 30T dev 20
  rows / 1 DAG; 45T train 120 rows / 8 DAGs; 15T dev, 45T dev, and all four test files have none.
  They come from an explicit `single_node` topology family; at 30 tools the only such DAG is
  `db_write` with an empty edge list.
- **Node counts range from 1 to 7** across the dataset, and **no multi-node graph has an empty
  edge list** — every graph is either a lone node or fully connected by dependencies. Any claim
  that DAGs have "between 2 and 6 nodes" is wrong at both ends.
- **Test splits are node-count restricted**, which is not currently disclosed: 15T test has
  {2,3,4,5,7}; 30T-1200 has only {4,5}; 30T-332 has {2,3,4,5,6}; 45T test has {4,5,7}. So the
  held-out sets are narrower than the vocabulary in general. Soften any claim that the test set
  spans the full structural range, and state the per-tier node-count distribution in §4.2.
- Mechanically nothing breaks on single-node inputs: the empty `edge_index` still yields a valid
  representation because `GCNConv` adds self-loops by default. Worth one sentence if the paper
  discusses degenerate cases.

### C.2 Baseline framing

Rewrite the baselines paragraph so the comparison is described accurately. The current set
answers a *deployment* question (BM25 as lexical floor, off-the-shelf S-BERT as what a
practitioner reaches for, generative LLMs as the high-cost comparator) rather than an *ablation*
question. Once Part A lands, present the progression frozen → fine-tuned → LEGR and state which
axis each step isolates. If the paper attributes the gain to structure-aware encoding, that
sentence must now be backed by the Cell 2 vs. Cell 4 comparison or removed.

Add to limitations: the frozen S-BERT baseline is fed arrow notation, which is out-of-distribution
for a sentence encoder trained on natural language, so it is likely weaker than a fair frozen
baseline. Report the verbalized variant from A.6 and adjust any claim whose size depends on the
frozen gap.

---

## Deliverables

1. `src/sbert_ft_baseline.py` and any supporting script, with the exact commands documented.
2. All artifacts listed in A.7, including `FINDINGS.md`.
3. A revised paper source that compiles cleanly with zero unresolved references or citations.
4. `CHANGELOG_revision.md` mapping every review item (B.1–B.9, C.1–C.2) to the change made, or
   to an explicit statement of why it was not addressed.
5. A list of every number in the paper you could **not** trace to a file in `new_results/`.

## Order of work

Start with B.0, since both blockers need answers from me before the paper edits are safe. While
waiting, do Part A — it is self-contained and produces the numbers the revised baselines section
depends on. Then Part B, then Part C. Ask before making any change that would alter an existing
reported number.
