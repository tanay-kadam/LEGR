# NeurIPS Paper Revision Reference: Functional Taxonomy and LEGR-GPS

**Purpose.** This document is the evidence and rewrite dossier for revising `Functional_Taxonomy_v2.pdf` after the Campaign-v4 audit, the Cursor campaign recorded in `artifacts/campaign_v4/SESSION_LOG_20260831_20260901.md`, the 65-run model-only search, and the functional-clustering audit. It is intended for the person rewriting the NeurIPS paper, not as camera-ready prose.

**Cutoff.** Evidence available in the repository through 2 September 2026.

**Most important status statement.** The new LEGR-GPS model beats the frozen fine-tuned SBERT control on the **15-tool development protocol** across three seeds. It has **not** been evaluated on the final 322-DAG twin-filled test gallery or confirmed at 30/45 tools. The paper may report this as development/model-selection evidence, but must not yet claim final test-set or universal superiority.

---

## 1. Executive conclusions

The paper needs a substantive methodological rewrite rather than a numerical refresh.

1. The original PDF describes semantic tool-name node features and a directed graph encoder, but the audited legacy implementation used arbitrary integer tool embeddings and default bidirectional edges. The paper's described model and the model behind important earlier results were not the same system.
2. The old 50-DAG held-out protocol was overwhelmingly a tool-set retrieval task. The current CSV contains 50 DAGs but 49 unique tool sets—one same-toolset pair, not the “50/50 and zero twins” stated in the session log. This small discrepancy does not change the diagnosis: nearly every candidate can be identified by its tool set.
3. The 322-DAG gallery is the existing protocol that actually stresses same-toolset structural twins. On that gallery, earlier V2/V3 LEGR and SBERT-FT `dag_text` all had approximately 0.20–0.21 Recall@1; SBERT tools-only's apparent 0.6467 Recall@1 was an argmax/gallery-order artifact.
4. GED is not supported as the central contribution. It gave a small 15-tool gain for SBERT-FT, identical 30/45 SBERT-FT results, a small V2 gain, and worse V3 results. The winning model does not use the historical GED objective.
5. The new winning architecture is best named **LEGR-GPS** (or “multi-expert LEGR-GPS”), not `LEGRDualEncoderV4`: no formal V4 class exists. It is implemented as `LEGRResearchModel`, which composes unchanged V3 and SBERT components with a new directed GPS adapter, tool and relation heads, and query-conditioned score fusion.
6. The new search completed 65/65 runs without failures in 3.41 summed GPU-hours. Only MiniLM was locally available; MPNet, E5, and BGE were recorded as skipped.
7. On the 49-DAG/294-query development gallery, LEGR-GPS's three-seed mean was 0.9138 Recall@1, 0.9883 tool-set F1, and 0.9490 “same-toolset Recall@1,” versus 0.8741, 0.9809, and 0.9320 for the frozen SBERT control. The last metric currently includes singleton tool sets and must not be described as twin-only accuracy.
8. A post-hoc representation audit found significant read/edit/orchestrate organization in the winning graph embeddings: 5-NN macro-F1 0.9462 with 95% bootstrap CI [0.9102, 0.9735], permutation `p <= 0.001`. Because it combines training, development, candidate, and test graphs and labels are derived from tool identities visible to the model, this is an exploratory representation result, not held-out generalization evidence.
9. No Campaign-v4 data or splits were changed during the model search or clustering audit. The search integrity report found zero changed, missing, or added immutable files; the clustering audit separately rechecked 27 protected files with zero changes.

### Recommended paper-level thesis

The strongest defensible thesis is no longer “GED-regularized directed GNN retrieval obtains 0.963 held-out Recall@1.” It is:

> Execution-graph retrieval benchmarks must include same-toolset structural alternatives; otherwise dense text retrievers can succeed through tool-set matching. On a validated twin-aware benchmark, we develop a multi-expert directed GPS retriever that combines semantic retrieval with explicit tool and pair-relation supervision. Development results show improved exact-plan retrieval over the frozen SBERT expert while preserving tool-set accuracy, and controlled embedding analyses show that the graph representation retains broad functional organization.

The final sentence must be upgraded from “development results show” to a test claim only after the missing final evaluations in Section 15 are completed.

---

## 2. Claim disposition: what must change in the original PDF

| Original paper statement | Later evidence | Required revision |
|---|---|---|
| LEGR node features come from the same text backbone applied to tool names/descriptions. | Legacy `LEGRDualEncoder` used `nn.Embedding(tool_id)`; its text-feature flag fed zeros. Real text-name nodes first appear in V2. | Replace the old implementation description. Present V1 as a failed audited control, V2/V3 as corrections, and LEGR-GPS as the proposed model. |
| Default LEGR is a directed GNN. | Default graph conversion duplicated reverse edges and defaulted to GCN, making it effectively undirected. | Do not attribute directed results to the legacy default. Describe the exact new GPS direction handling. |
| Held-out Diamond Recall@1 is 0.963 and demonstrates structural zero-shot generalization. | The old/easy gallery was nearly unique by tool set; later SBERT-FT reached 0.9267 there. The fair 322-DAG gallery reduced prior LEGR/SBERT `dag_text` R@1 to about 0.20–0.21. The provenance of 0.963 is not compatible with Campaign-v4 artifacts. | Remove 0.963 from the abstract, contributions, Figure 1, Tables 2/4, conclusion, and appendix unless the exact old checkpoint/protocol can be independently reproduced and labeled legacy. It cannot support the new model. |
| GED weighting is what internalizes directed edges and halves mean GED. | GED effects are mixed or null: V3 no-GED beats GED; SBERT 30/45 GED files are identical to no-GED; the new winner replaces GED with explicit relation/distance objectives. | Remove GED/GACL as the headline novelty. Retain it as a historical ablation and negative/mixed finding. |
| LEGR latency is 4.39 ms and flat with plan size. | Original CUDA timing omitted synchronization and warm-up; nearest-neighbor search was excluded. New synchronized development p95 for the winner averages 16.85 ms. | Replace 4.39 ms and 401–625x claims. Report metric definition, device, gallery size, warm-up/synchronization, mean and p95. Do not mix retrieval p95 with LLM mean. |
| LEGR is 23.5M parameters. | LEGR V3 is about 23.5M, but the winning composite model is 74,540,563 parameters, 9,752,723 trainable. | Update all model-footprint tables and any “5,100x smaller” calculation. |
| Train contains Chain/Hourglass and held-out contains only Diamond. | Campaign-v4 has 20 training families and two held-out families: Diamond and asymmetric fork-join. | Replace dataset and split description with the Campaign-v4 construction. |
| Dataset sizes are 3,970/2,060/5,420 query-graph pairs and 193/197/348 DAGs. | Campaign-v4 has 3,906/5,496/7,878 rows and 651/916/1,313 unique DAGs across all split/candidate files. | Replace the old counts; include per-split counts and clarify candidate-only rows. |
| Sentence-BERT R@1 is 0.302 at 15 tools. | Frozen SBERT is 0.4333 on the 50-DAG gallery; SBERT-FT is 0.9267 there; SBERT-FT `dag_text` is 0.2133 on the 322-DAG twin gallery. | Separate frozen versus fine-tuned SBERT and identify the gallery in every table/caption. |
| GPT-OSS has 0.961 Recall@1 and 1.590 GED. | Corrected Campaign-v4 GPT-OSS v3 has tool F1 0.9450 and exact match 0.76 at 15 tools, mean GED 0.44. Generative systems do not have ranked Recall@k. | Use exact match and tool F1, not “Recall@1 = F1.” Report parsing and validity. |
| Functional Categorization improves routing by up to 18.5 points. | This is an original-paper single-tool result, not rerun in Campaign-v4. The new clustering audit is different evidence. | May remain only if its original raw outputs are recoverable and verified. Do not use clustering as a replacement routing experiment. |
| t-SNE demonstrates structural zero-shot generalization. | t-SNE is visualization, not statistical evidence; no corresponding validated topology-clustering artifact was found in the later campaign. | Remove causal/demonstrative wording. Use original-space metrics and held-out tests; label projections illustrative. |
| “Same-toolset Recall@1” measures twin discrimination in the new search. | Current implementation counts singleton tool sets. In the 49-DAG dev gallery, 39 tool-set groups are singletons and five groups contain two DAGs. | Rename it “toolset-restricted R@1 including singletons” or recompute a true twin-only metric before publication. |

### Claims that can remain with careful scope

- Retrieval returns a prevalidated plan from a finite corpus and therefore avoids generating malformed/cyclic graph syntax at inference.
- A graph encoder can represent same-toolset graphs differently; tools-only SBERT cannot.
- Candidate embeddings can be precomputed. Exact scoring remains linear in gallery size unless an ANN index is actually used and measured.
- Functional labels read/edit/orchestrate are operationally meaningful, but the original routing claim and the later graph-clustering claim are distinct experiments.
- The candidate-corpus coverage limitation remains fundamental and should be prominent.

---

## 3. Study chronology and architecture lineage

The revision should explain the lineage because it resolves the paper/implementation mismatch.

| Version | Node semantics | Graph processing | Readout | Score |
|---|---|---|---|---|
| V1 legacy | Learned integer tool IDs; optional text path was zeros | GCN/GAT or basic directed GNN; default data path bidirectional | Mean pooling | One query-graph cosine |
| V2 tool-name LEGR | Separate frozen MiniLM over tool names, projected 384→64 | Three directed layers with separate self/in/out transforms | Mean pooling | One cosine |
| V3 SetGNN tied | Query and node names share one MiniLM; 384→64 nodes | Three-layer directed GNN | Concatenate GNN mean pool with learned node-set attention, then project | One cosine |
| LEGR-GPS winner | V3 text-derived nodes plus invariant degree/source/sink features | Four GPS blocks: directed local residual propagation plus relation-biased global attention | Dual learned attention pools | Query-conditioned fusion of five expert scores |

Parameter counts recorded in the repository are approximately 22.90M (legacy V1), 45.73M (V2, due to two MiniLM copies), 23.48M (V3), and exactly 74,540,563 total / 9,752,723 trainable for the winner.

The paper should not call Campaign-v4 an architecture version. Campaign-v4 is the corrected dataset/experiment campaign. If a versioned model name is necessary, define “LEGR-GPS (V4)” explicitly once; the implementation class remains `LEGRResearchModel`.

### Terminology and checkpoint lineage

Use these names consistently; several old tables otherwise appear to compare different systems under the same label.

| Term | Exact meaning in this dossier |
|---|---|
| Frozen SBERT | The off-the-shelf `all-MiniLM-L6-v2` sentence encoder with no Campaign-v4 fine-tuning. |
| SBERT-FT | The existing Campaign-v4 dual-encoder checkpoint trained on serialized plans. |
| SBERT control / semantic control | That SBERT-FT checkpoint loaded read-only, frozen, and scored alone inside the new common evaluator. It is not the off-the-shelf frozen SBERT baseline. |
| Semantic expert | The same frozen SBERT-FT component inside LEGR-GPS. |
| V3 control | The separately trained `LEGRDualEncoderV3` checkpoint evaluated by itself. |
| Inherited V3 expert | A V3 instance initialized from the existing seed-42 V3 checkpoint inside LEGR-GPS. Its source implementation is unchanged, but its last two shared MiniLM layers are trainable in the winner; it must not be described as a frozen V3 expert. |
| GPS adapter | Only the new directed local/global graph branch and dual-attention readout. |
| LEGR-GPS winner | The complete five-expert `LEGRResearchModel`, including frozen SBERT-FT, initialized V3, GPS adapter, tool/relation heads, and gated fusion. |

All three confirmation seeds fall back to the existing seed-42 V3 and SBERT-FT initialization checkpoints when same-seed legacy checkpoints are absent. Thus seeds 42/123/2026 vary the new training initialization/order, not the inherited checkpoint provenance.

---

## 4. Campaign-v4 dataset and integrity

### 4.1 Registry and nested scales

The action-first vocabulary contains 45 tools, with nested tiers 15 ⊂ 30 ⊂ 45. Category counts are balanced in the ratio 2:2:1:

| Tier | Retrieval/read | Modification/edit | Orchestration | Total |
|---:|---:|---:|---:|---:|
| 15 | 6 | 6 | 3 | 15 |
| 30 | 12 | 12 | 6 | 30 |
| 45 | 18 | 18 | 9 | 45 |

### 4.2 Exact split sizes

Every DAG has six Azure GPT-4o query variants.

| Tier | Split | Rows | Unique DAGs | Unique tool sets |
|---:|---|---:|---:|---:|
| 15 | train | 1,488 | 248 | 159 |
| 15 | dev | 294 | 49 | 44 |
| 15 | test in-domain | 192 | 32 | 30 |
| 15 | topology held-out | 300 | 50 | 49 |
| 15 | candidate-only corpus | 1,632 | 272 | 49 |
| 30 | train | 2,076 | 346 | 229 |
| 30 | dev | 414 | 69 | 61 |
| 30 | test in-domain | 276 | 46 | 42 |
| 30 | topology held-out | 420 | 70 | 70 |
| 30 | candidate-only corpus | 2,310 | 385 | 70 |
| 45 | train | 2,988 | 498 | 326 |
| 45 | dev | 594 | 99 | 85 |
| 45 | test in-domain | 396 | 66 | 63 |
| 45 | topology held-out | 600 | 100 | 100 |
| 45 | candidate-only corpus | 3,300 | 550 | 100 |

The disjoint union across these files is 651, 916, and 1,313 unique DAGs, respectively. There is zero held-out DAG overlap with train/dev. The candidate-only corpus contains structural alternatives but does not contain the held-out gold DAG IDs; therefore ranking held-out queries against `candidate_corpus.csv` alone necessarily yields Recall@k = 0.

### 4.3 Topology and twin construction

- Twenty topology families appear in train/dev.
- Diamond and asymmetric fork-join are held out from train/dev.
- Every held-out test DAG has at least one same-toolset alternative in the candidate corpus.
- The combined 15-tool fair gallery has 322 DAGs: 50 held-out gold DAGs plus 272 candidate-only DAGs.
- That gallery contains 927 twin pairs across 49 tool sets; all 300 held-out queries have a twin-eligible gold graph.

### 4.4 Query generation

| Item | Value |
|---|---:|
| Azure GPT-4o calls | 2,781 |
| Failed calls | 0 |
| Prompt tokens | 1,447,171 |
| Completion tokens | 1,102,497 |
| Total generation latency | 12,970.76 s |
| Estimated cost | $5.0993 |
| Unique queries per DAG | 6.0 average |

The paper must disclose that GPT-4o generated the natural-language queries and describe the six query conditions present in every Campaign-v4 CSV: `standard`, `paraphrase`, `lexical`, `confusable`, `structural_clear`, and `structural_paraphrase`. Each condition occurs exactly once per DAG. Do not retain the PDF's WordNet-only generation story.

### 4.5 Validation

- Dataset suite: 34/34 tests passed.
- Complete validation: 30/30 passed, zero critical failures.
- Checks include schema, vocabulary membership, topology exclusion, canonical-DAG leakage, twin density, query diversity, Azure provenance, topology diversity, and acyclicity.
- Model-search immutable-path comparison: changed `[]`, missing `[]`, added `[]`.
- Functional-clustering audit: 27 protected files checked, zero changes.

---

## 5. Proposed LEGR-GPS method

### 5.1 Inputs and notation

Let a query be `q` and a candidate execution graph be `G=(V,E)`. Each node `v` has tool name `t_v`; each directed edge `u→v` indicates that `v` depends on `u`. Let `H_q` be MiniLM token states and `p_q` their masked mean.

The winner uses 256-dimensional query/graph embeddings, four GPS layers, 256 hidden units, eight graph-attention heads, dropout 0.1, and `all-MiniLM-L6-v2`. The last two query-backbone transformer layers are trainable.

### 5.2 Text-derived node features

Tool names are converted from snake case to readable text and encoded by the V3 MiniLM backbone. Mean-pooled tool states are projected to 64 dimensions:

\[
x_v = W_{node}\,\operatorname{MeanPool}(\operatorname{MiniLM}(t_v)).
\]

This corrects the legacy mismatch between pretrained query semantics and random graph-node IDs.

### 5.3 Permutation-equivariant structural features

The winning `degree` encoding appends four active values to each node feature: normalized in-degree, normalized out-degree, source indicator, and sink indicator. The implementation stores a six-dimensional structural vector but zeros source-depth and sink-depth for this variant:

\[
s_v = [0,0,d_{in}(v)/(n-1),d_{out}(v)/(n-1),\mathbb{1}[d_{in}=0],\mathbb{1}[d_{out}=0]].
\]

Unlike total topological-sort rank, these values do not arbitrarily order parallel nodes.

### 5.4 Directed local message passing

Initial hidden states are `h_v^0 = W_0[x_v;s_v]`. Each local residual block separately aggregates predecessor and successor messages using degree-normalized means:

\[
m^{in}_v=\operatorname{mean}_{u\to v}W_{in}h_u,\qquad
m^{out}_v=\operatorname{mean}_{v\to w}W_{out}h_w,
\]

\[
\tilde h_v=\operatorname{GELU}\!\left(W_m[W_s h_v;m^{in}_v+m^{out}_v]\right),\qquad
h'_v=\operatorname{LayerNorm}(h_v+\operatorname{Dropout}(\tilde h_v)).
\]

The separate `W_in` and `W_out` preserve direction.

### 5.5 Relation-biased global attention

Every GPS layer follows local propagation with global multi-head self-attention over nodes in the same graph. Each ordered node pair receives one of six attention-bias types:

1. self;
2. direct forward;
3. direct reverse;
4. ancestor/indirect forward;
5. descendant/indirect reverse;
6. unrelated/parallel.

The learned relation embedding supplies a head-specific additive attention bias. This provides a global receptive field while retaining directed-path information.

### 5.6 Dual-attention readout

Two independent scalar attention functions pool the final node states. Their outputs are concatenated and projected:

\[
r_a=\sum_v\operatorname{softmax}_v(a_a(h_v))h_v,\quad
r_b=\sum_v\operatorname{softmax}_v(a_b(h_v))h_v,
\]

\[
z_G=\operatorname{Normalize}(W_r[r_a;r_b]).
\]

The two pools are architecturally independent; although described as “tool” and “structure” attention, the implementation does not explicitly constrain one pool to either role.

### 5.7 Query outputs and auxiliary heads

The shared query state produces:

- structural query embedding `z_q = Normalize(W_q p_q)`;
- one tool-membership logit per vocabulary item, using learned tool queries attending to `H_q`;
- one five-class relation distribution for every ordered tool-pair representation built from `[r_i,r_j,r_i-r_j,r_i⊙r_j]`.

### 5.8 Five scoring experts

For every query/candidate pair, the model calculates:

1. `e_sem`: frozen SBERT query/document cosine;
2. `e_v3`: inherited V3 query/graph cosine;
3. `e_gps`: `z_q^T z_G`;
4. `e_tool`: cosine between sigmoid tool probabilities and the candidate tool-indicator vector;
5. `e_rel`: mean query log-probability of the candidate's active pairwise relations.

The final score is query-conditioned:

\[
g(q)=\operatorname{softmax}(W_g p_q),\qquad
s(q,G)=\sum_{k=1}^{5}g_k(q)\,a_k\,e_k(q,G).
\]

The gate weights `g_k` are nonnegative and sum to one. The learned expert scales `a_k` are unconstrained in the actual implementation; the paper must not call the complete product a convex/nonnegative mixture without this qualification.

### 5.9 Reranker status

A top-K query-to-candidate-node cross-attention reranker was implemented and tested, but it degraded accuracy. It is disabled in the winner. The implemented reranker does not directly add new edge/ancestor attention biases at reranking time; it attends to node states that already contain graph information. Do not describe the richer planned reranker as if it were the tested code.

---

## 6. Training objective and sampling

The winning objective is:

\[
L=L_{listwise}+0.5L_{twin}+0.25L_{tool}+0.5L_{relation}+0.1L_{multi}+0.1L_{distance}.
\]

### 6.1 Exact listwise retrieval

For fused scores `s_ij`, positives are candidates sharing the query row's DAG ID. The loss is the negative log probability of all positives under the eligible candidate set:

\[
L_{listwise}=-\frac{1}{B}\sum_i\log\frac{\sum_{j\in P(i)}e^{s_{ij}}}{\sum_j e^{s_{ij}}}.
\]

### 6.2 Same-toolset twin ranking

`L_twin` applies the same listwise form after masking eligibility to candidates in the same structural-twin/tool-set group. The alternative pairwise form tested was

\[
\operatorname{softplus}(m-s_i^++s_{ij}^-)
\]

for margins 0.1, 0.2, and 0.4.

### 6.3 Tool membership

`L_tool` is binary cross-entropy with batch-derived positive weighting equal to the number of negative labels divided by positive labels.

### 6.4 Five-class relation prediction

`L_relation` is cross-entropy over parallel, direct forward, indirect forward, direct reverse, and indirect reverse. Inactive tool pairs are ignored.

### 6.5 Multi-positive contrastive term

`L_multi` is symmetric InfoNCE over normalized structural query and GPS graph embeddings, treating identical DAG IDs as positives. However, the promoted group-aware sampler uses one paraphrase per DAG in a batch, so the winner normally has one positive per row. In this setting the term behaves as an additional symmetric diagonal contrastive loss; the sampler, not off-diagonal positives, is what prevents same-DAG paraphrases from becoming false negatives.

### 6.6 Structural-distance regularization

For graph pairs sharing active tool-pair positions, the target is normalized Hamming disagreement between pair-relation signatures. Smooth L1 aligns this with cosine distance between graph embeddings:

\[
L_{distance}=\operatorname{SmoothL1}(1-\cos(z_{G_i},z_{G_j}),d_{rel}(G_i,G_j)).
\]

This replaces the historical GED auxiliary in the winning model.

### 6.7 Group-aware sampler and curriculum

Each epoch selects one of the six existing paraphrases per DAG and rotates the choice across epochs. Same-toolset twin DAGs are colocated in batches. This changes sampling only; it creates no examples and modifies no CSV.

The winner linearly raises effective twin-loss pressure to full strength over the first half of training. It uses batch size 64, AdamW, learning rates `2e-5` for trainable backbone layers and `2e-4` for heads/adapters, weight decay `1e-4`, two warm-up epochs, cosine scheduling, gradient norm cap 1.0, 75-epoch cap, and patience 15. All three winning confirmations stopped after 23 completed epochs.

---

## 7. Evaluation protocols: never merge these columns

| Protocol | Queries | Gallery | What it primarily measures | Valid uses |
|---|---:|---:|---|---|
| Development search | 294 | 49 DAGs / 44 tool sets | Model selection with five two-DAG twin groups | Ablations and promotion only |
| Held-out compact gallery | 300 | 50 DAGs / 49 tool sets | Mostly tool-set retrieval on held-out topologies | Legacy comparison, not a strong twin test |
| Twin-filled full gallery | 300 | 322 DAGs / 49 tool sets | Exact graph ranking among same-toolset alternatives | Required final structural test |
| Candidate-only corpus | 300 held-out queries | 272 DAGs, gold absent | Invalid Recall@k setup | Do not report accuracy; use only as distractors when gold is added |
| Functional clustering | 487 non-tied graphs | Embedding analysis, no retrieval gallery | Train-inclusive representation geometry | Exploratory appendix only |

### Metric definitions and issues

- Recall@k: fraction of queries whose exact canonical DAG is ranked in the top k.
- MRR@k: reciprocal rank if the exact DAG is within top k, otherwise zero.
- Tool-set F1: F1 between the top-ranked candidate's tools and the gold tools, averaged over queries.
- Mean GED error: graph-edit distance between top-1 and gold under the specific evaluator; exact and surrogate GED must be distinguished.
- Current `same_toolset_recall@1`: restricts ranking to candidates matching the gold tool indicator, but includes singleton groups. It is not a pure twin metric.
- Required replacement: report `twin_only_same_toolset_R@1` only for gold DAGs whose same-toolset gallery contains at least two members, plus the mean random-tie chance `E[1/|C_t|]`.
- Candidate order must be deterministically randomized, and exact/numerical ties must be reported with tie-aware expected accuracy or randomized tie-breaking.
- Latency must use warm-up and CUDA synchronization, state whether candidate embeddings are cached, state gallery size and batch size, and report mean, median, synchronized p95, and number of trials.

### New-search latency implementation caveat

The new evaluator does call `torch.cuda.synchronize()` around the timed region, which corrects the old asynchronous-CUDA defect. It is still not a deployment-quality per-request benchmark:

- queries are evaluated in batches of 64;
- the complete candidate graph and SBERT document representations are recomputed inside `score_batches` for every query batch rather than served from a persistent cache;
- batch elapsed time is divided by the number of queries and copied to every query in that batch;
- p95 is therefore a percentile over repeated batch-average values, not independently timed single-query requests;
- the “semantic-only” accuracy control still executes the unused GPS, V3, tool, and relation branches before selecting the semantic score, so its latency is not standalone SBERT latency.

The 16.85 ms winner p95 is valid for comparing recorded search configurations under the same evaluator, but it should not replace the paper's deployment latency until a dedicated benchmark caches candidate embeddings and times batch size 1 as well as throughput batches.

---

## 8. Campaign-v4 experiments before the new search

### 8.1 Five-epoch CPU pilot

| Model | Validation loss | Validation R@1 |
|---|---:|---:|
| Legacy, no GED | 2.444 | 0.155 |
| Legacy, GED | 2.495 | 0.151 |
| Integer directed, no GED | 2.696 | 0.135 |
| Integer directed, GED | 2.740 | 0.135 |
| SBERT-FT, no GED | 2.671 | 0.142 |
| SBERT-FT, GED | 2.671 | 0.142 |

The short pilot is engineering evidence only. The corresponding SBERT held-out probe was R@1 0.66, R@3 0.85, R@5 0.89, tool F1 0.842.

### 8.2 Integer-ID LEGR, approximately 15 epochs

| Tier | Model | R@1 | R@3 | R@5 | Tool F1 | Mean GED |
|---:|---|---:|---:|---:|---:|---:|
| 15 | Legacy no GED | 0.0333 | 0.1233 | 0.1767 | 0.3873 | 4.443 |
| 15 | Legacy GED | 0.0267 | 0.1067 | 0.1567 | 0.3747 | 4.553 |
| 15 | Directed no GED | 0.0267 | 0.1233 | 0.1800 | 0.3601 | 4.520 |
| 15 | Directed GED | 0.0367 | 0.1033 | 0.1900 | 0.3626 | 4.507 |
| 30 | Legacy no GED | 0.0262 | 0.0833 | 0.1143 | 0.1696 | 4.869 |
| 30 | Legacy GED | 0.0262 | 0.0810 | 0.1167 | 0.1719 | 4.883 |
| 30 | Directed no GED | 0.0357 | 0.1048 | 0.1381 | 0.2041 | 4.650 |
| 30 | Directed GED | 0.0357 | 0.1000 | 0.1381 | 0.1980 | 4.683 |
| 45 | Legacy no GED | 0.0250 | 0.0850 | 0.1317 | 0.1724 | 4.847 |
| 45 | Legacy GED | 0.0200 | 0.0667 | 0.1267 | 0.1637 | 4.888 |
| 45 | Directed no GED | 0.0017 | 0.0117 | 0.0267 | 0.1061 | 4.900 |
| 45 | Directed GED | 0.0117 | 0.0433 | 0.0883 | 0.1341 | 4.852 |

### 8.3 Integer-ID LEGR, 75 epochs

| Tier | Model | R@1 | R@3 | R@5 | Tool F1 | Mean GED |
|---:|---|---:|---:|---:|---:|---:|
| 15 | Legacy no GED | 0.0033 | 0.0533 | 0.0967 | 0.2639 | 4.723 |
| 15 | Legacy GED | 0.0033 | 0.0567 | 0.0967 | 0.2619 | 4.733 |
| 15 | Directed no GED | 0.0533 | 0.1433 | 0.1833 | 0.3388 | 4.483 |
| 15 | Directed GED | 0.0800 | 0.1533 | 0.2267 | 0.3779 | 4.250 |
| 30 | Legacy no GED | 0.0071 | 0.0357 | 0.0833 | 0.1717 | 4.948 |
| 30 | Legacy GED | 0.0071 | 0.0357 | 0.0786 | 0.1722 | 4.940 |
| 30 | Directed no GED | 0.0333 | 0.0810 | 0.1214 | 0.1946 | 4.629 |
| 30 | Directed GED | 0.0429 | 0.0833 | 0.1357 | 0.2055 | 4.560 |
| 45 | Legacy no GED | 0.0267 | 0.0650 | 0.1183 | 0.1473 | 4.810 |
| 45 | Legacy GED | 0.0200 | 0.0800 | 0.1317 | 0.1479 | 4.917 |
| 45 | Directed no GED | 0.0133 | 0.0433 | 0.0767 | 0.1413 | 4.838 |
| 45 | Directed GED | 0.0250 | 0.0450 | 0.0650 | 0.1483 | 4.768 |

These results diagnose the implementation, not the value of graph networks: the graph tower lacked pretrained tool semantics, and longer training did not repair the mismatch.

### 8.4 Corrected V2/V3 on the compact 15-tool held-out gallery

| Model | Node/readout | R@1 | R@3 | R@5 | MRR@5 | Tool F1 | Mean GED |
|---|---|---:|---:|---:|---:|---:|---:|
| Integer directed + GED, 75ep | ID + mean | 0.0800 | 0.1533 | 0.2267 | 0.1268 | 0.3779 | 4.250 |
| Frozen SBERT | serialized text | 0.4333 | 0.7033 | 0.7567 | 0.5681 | 0.7116 | 2.557 |
| BM25 | lexical | 0.0200 | 0.0600 | 0.1000 | 0.0457 | 0.3351 | 4.820 |
| V2 tool-name, no GED | MiniLM nodes + mean | 0.6067 | 0.8600 | 0.9200 | 0.7353 | 0.8138 | 1.753 |
| V2 tool-name + GED | MiniLM nodes + mean | 0.6200 | 0.8300 | 0.9033 | 0.7309 | 0.8241 | 1.690 |
| V3 SetGNN tied, no GED | shared MiniLM + split pool | 0.8267 | 0.9700 | 0.9800 | 0.8990 | 0.9464 | 0.707 |
| V3 SetGNN tied + GED | shared MiniLM + split pool | 0.8133 | 0.9700 | 0.9800 | 0.8907 | 0.9402 | 0.763 |
| SBERT-FT, no GED | dual text | 0.9200 | 0.9933 | 0.9933 | 0.9561 | 0.9781 | 0.310 |
| SBERT-FT, GED | dual text | 0.9267 | 0.9933 | 0.9967 | 0.9596 | 0.9811 | 0.290 |

V3 no-GED beating V3 GED directly contradicts the original paper's universal claim that GED weighting is necessary for structure.

### 8.5 SBERT-FT scale ablation on compact held-out galleries

| Tier | Variant | R@1 | R@3 | R@5 | Tool F1 | Mean GED | Hard-neg accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 15 | no GED | 0.9200 | 0.9933 | 0.9933 | 0.9781 | 0.3100 | 1.0000 |
| 15 | GED | 0.9267 | 0.9933 | 0.9967 | 0.9811 | 0.2900 | 0.9936 |
| 30 | no GED | 0.9571 | 0.9929 | 0.9976 | 0.9724 | 0.2167 | 1.0000 |
| 30 | GED | 0.9571 | 0.9929 | 0.9976 | 0.9724 | 0.2167 | 1.0000 |
| 45 | no GED | 0.9467 | 0.9867 | 0.9933 | 0.9671 | 0.2433 | 1.0000 |
| 45 | GED | 0.9467 | 0.9867 | 0.9933 | 0.9671 | 0.2433 | 1.0000 |

The 30- and 45-tool GED/no-GED JSON files are numerically identical. The Cursor log contains an older conflicting 30/45 table; the current `eval_metrics.json` files above are the source of truth and should be cited by the rewriter.

### 8.6 Twin-filled 322-DAG gallery

| Representation/model | Full R@1 | R@3 | R@5 | Toolset-restricted R@1 | Mean twin cosine |
|---|---:|---:|---:|---:|---:|
| Frozen SBERT `dag_text` | 0.0167 | 0.1000 | 0.2367 | — | — |
| Frozen SBERT tools-only | 0.2867 | 0.2900 | 0.2967 | — | — |
| SBERT-FT `dag_text` | 0.2133 | 0.6100 | 0.8533 | 0.2333 | 0.9463 |
| SBERT-FT tools-only | 0.6467* | 0.6500 | 0.6567 | 0.6900* | 0.9998 |
| SBERT-FT reversed-edge text | 0.1500 | 0.5967 | 0.8667 | 0.1700 | 0.9432 |
| V2 tool-name + GED | 0.2000 | 0.4000 | 0.5033 | 0.2733 | 0.4317 |
| V3 SetGNN no GED | 0.2100 | 0.5667 | 0.8133 | 0.2367 | 0.6256 |

`*` The tools-only result is not structural accuracy. Test DAGs were concatenated before distractors; 98% of gold DAGs had the lowest gallery index among their equal-tool embeddings. Lowest-index tie-breaking gives 0.9733 toolset-restricted R@1, highest-index gives 0.0000, and random tied selection gives 0.1467, approximately the 0.1538 analytical chance level.

The key honest result is representational: all tools-only twin pairs have cosine above 0.99, while V2/V3 emit different graph vectors. Earlier LEGR nevertheless did not yet beat SBERT-FT `dag_text` on overall fair-gallery R@1.

### 8.7 Corrected generative baselines

Use only v3 runs, which supplied the dynamic Campaign-v4 vocabulary and `max_tokens=1024`.

| Model | Tier | n | Parse failures | Tool F1 | Exact match | Mean GED | Cyclic | Structurally valid | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS 120B | 15 | 50 | 0 | 0.9450 | 0.76 | 0.44 | 0 | 50 | 3.047 s | 4.468 s |
| GPT-OSS 120B | 30 | 50 | 0 | 0.9517 | 0.80 | 0.62 | 0 | 50 | 3.268 s | 4.777 s |
| GPT-OSS 120B | 45 | 50 | 0 | 0.9486 | 0.76 | 0.80 | 0 | 50 | 3.106 s | 4.609 s |
| Llama 3.2 3B | 15 | 50 | 0 | 0.7911 | 0.08 | 6.11 | 6 | 44 | 4.485 s | 15.301 s |
| Llama 3.2 3B | 30 | 50 | 2 | 0.7769 | 0.06 | 4.80 | 4 | 44 | 4.610 s | 15.038 s |
| Llama 3.2 3B | 45 | 50 | 5 | 0.6844 | 0.04 | 4.58 | 2 | 43 | 5.080 s | 15.685 s |

Generative exact match is a single-output metric, not Recall@1. Retrieval and generation have different output constraints: the retriever is guaranteed to choose a prevalidated in-corpus DAG, while the generator may create a novel DAG but can also emit invalid syntax or cycles.

---

## 9. The 65-run model-only search

### 9.1 Search accounting

| Stage | Runs | Summed elapsed time |
|---|---:|---:|
| Frozen semantic baseline | 1 | 16.4 s |
| Mathematical objectives | 12 | 773.2 s |
| Encoders/structure/readout | 20 | 2,421.0 s |
| Backbone/fusion | 5 | 886.2 s |
| Reranker/optimization | 15 | 4,374.2 s |
| Confirmation | 12 | 3,798.1 s |
| **Total** | **65** | **12,269.1 s = 3.41 h** |

All 65 runs completed; there were zero recorded failures. Screening used seed 42 and at most 15 epochs. Confirmation used seeds 42, 123, and 2026 with a 75-epoch cap and patience 15.

Promotion rejected p95 above 100 ms and, when possible, tool F1 more than 0.002 below the semantic baseline; remaining runs were ordered by development R@1, then the current toolset-restricted R@1, then tool F1. Development metrics were used repeatedly for architecture search, so final test evaluation is essential to control selection bias.

### 9.2 Semantic control

The control is the frozen existing fine-tuned SBERT expert used by itself inside the common scoring/evaluation framework.

| R@1 | R@3 | R@5 | Tool F1 | Toolset-restricted R@1 | Mean latency | p95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.874150 | 0.996599 | 1.000000 | 0.980895 | 0.931973 | 14.816 ms | 15.140 ms |

### 9.3 Mathematical ablations

All rows below use the V3-style/residual research adapter, gated five-score model where enabled, group-aware sampling except the last row, and the 49-DAG development gallery. `math_0` is listwise-only; rows 1–5 cumulatively add the named terms.

| Run | Active change | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| math_0 | listwise only | 0.897959 | 0.989796 | 0.993197 | 0.986584 | 0.938776 | 2.110 |
| math_1 | + multi-positive term | 0.891156 | 0.989796 | 0.996599 | 0.983655 | 0.938776 | 2.178 |
| math_2 | + twin listwise | 0.880952 | 0.989796 | 1.000000 | 0.976447 | 0.945578 | 2.156 |
| math_3 | + tool BCE | 0.897959 | 0.989796 | 1.000000 | 0.985720 | 0.945578 | 2.191 |
| math_4 | + relation CE | 0.891156 | 0.982993 | 0.996599 | 0.983711 | 0.945578 | 1.743 |
| math_5 | + relation-distance term | 0.891156 | 0.982993 | 0.996599 | 0.983711 | 0.945578 | 1.540 |
| pair m=0.1 | pairwise twin ranking | 0.901361 | 0.982993 | 1.000000 | 0.986206 | 0.945578 | 2.149 |
| pair m=0.2 | pairwise twin ranking | 0.901361 | 0.982993 | 1.000000 | 0.986206 | 0.945578 | 1.492 |
| pair m=0.4 | pairwise twin ranking | 0.897959 | 0.982993 | 1.000000 | 0.985720 | 0.945578 | 2.112 |
| weights ×0.5 | halve twin/tool/relation weights | **0.904762** | 0.989796 | 1.000000 | **0.986692** | 0.945578 | 1.513 |
| weights ×2 | double twin/tool/relation weights | 0.867347 | 0.979592 | 0.989796 | 0.976688 | 0.942177 | 2.200 |
| random batches | disable group-aware sampler | 0.887755 | 0.993197 | 0.993197 | 0.978015 | **0.948980** | 1.494 |

Interpretation must be conservative because the first six rows are cumulative, not one-factor ablations from the final model. Halving auxiliary weights was promoted. Strong auxiliary weights hurt. Pairwise margins 0.1/0.2 improved R@1 over the full-strength cumulative objective but were not the final loss. Random row batching processed redundant paraphrases, took 230.2 seconds versus roughly 14–81 seconds for group-aware rows, and reduced R@1/F1.

### 9.4 Graph encoder and structural-feature ablations

| Encoder / structural input | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| V3-style adapter / none | 0.904762 | 0.989796 | 1.000000 | 0.988083 | 0.945578 | 2.061 |
| V3-style / combined | 0.904762 | 0.989796 | 1.000000 | 0.986692 | 0.945578 | 1.391 |
| Residual / none | 0.904762 | 0.989796 | 1.000000 | 0.988083 | 0.945578 | 2.055 |
| Residual / combined | 0.904762 | 0.989796 | 1.000000 | 0.986692 | 0.945578 | 1.549 |
| Gated / none | 0.897959 | 0.989796 | 0.996599 | 0.979619 | 0.948980 | 1.762 |
| Gated / combined | 0.894558 | 0.989796 | 0.996599 | 0.979619 | 0.945578 | 1.384 |
| PNA-style / none | 0.894558 | 0.989796 | 0.996599 | 0.986449 | 0.945578 | 1.783 |
| PNA-style / combined | 0.897959 | 0.989796 | 0.996599 | 0.985234 | 0.948980 | 1.490 |
| Graphormer / none | 0.911565 | 0.989796 | 0.996599 | 0.987312 | 0.948980 | 13.628 |
| Graphormer / combined | 0.908163 | 0.989796 | 1.000000 | 0.987003 | 0.948980 | 15.535 |
| GPS / none | 0.908163 | 0.989796 | 0.996599 | 0.987327 | 0.948980 | 14.199 |
| GPS / combined | 0.911565 | 0.989796 | 1.000000 | 0.987825 | 0.948980 | 13.636 |
| GPS / source-sink depth | 0.911565 | 0.989796 | 1.000000 | 0.987825 | 0.948980 | 15.617 |
| **GPS / degree-source-sink** | **0.914966** | 0.989796 | 0.996599 | **0.989202** | 0.948980 | 15.090 |
| GPS / path mode | 0.908163 | 0.989796 | 0.996599 | 0.987327 | 0.948980 | 13.433 |

Implementation nuance: `graph_kind="v3"` and `graph_kind="residual"` select the same residual local-block implementation in `GraphAdapter`, explaining their identical metrics. They are not independent architectural variants. Likewise, the “PNA” option is PNA-style directional aggregation, not the full canonical PNA operator, and `path` supplies zero node features while GPS/Graphormer relation bias still encodes paths.

### 9.5 Readout ablations

These readout tests were run on the V3-style/residual adapter inherited from the mathematical stage, not directly on GPS-degree.

| Readout | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Mean/V3-style | 0.908163 | 0.986395 | 0.996599 | 0.987191 | 0.945578 | 2.018 |
| Dual attention | 0.904762 | 0.989796 | 1.000000 | 0.986692 | 0.945578 | 1.920 |
| Virtual-style attention | 0.908163 | 0.986395 | 0.996599 | 0.987947 | 0.945578 | 1.629 |
| Set2Set | 0.891156 | 0.989796 | 0.996599 | 0.982939 | 0.938776 | 1.453 |
| Concatenated mean/max/normalized-add | 0.908163 | 0.989796 | 0.996599 | 0.987894 | 0.945578 | 1.586 |

The winner uses dual attention because it was the inherited default in the best GPS-degree configuration, not because this table establishes it as the best GPS readout. A clean GPS-degree readout ablation remains desirable.

### 9.6 Backbone availability and fusion

Only MiniLM was locally cached. The search did not download replacements.

| Backbone | Status |
|---|---|
| all-MiniLM-L6-v2 | Tested |
| all-mpnet-base-v2 | Skipped: unavailable locally |
| e5-base-v2 | Skipped: unavailable locally |
| bge-base-en-v1.5 | Skipped: unavailable locally |

| Fusion | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| GPS graph only | 0.870748 | 0.976190 | 0.993197 | 0.970760 | 0.942177 | 14.304 |
| Frozen SBERT only | 0.874150 | 0.996599 | 1.000000 | 0.980895 | 0.931973 | 15.563 |
| Fixed mean | 0.721088 | 0.962585 | 0.986395 | 0.866407 | 0.945578 | 15.710 |
| Learned global scalar | 0.721088 | 0.962585 | 0.986395 | 0.866181 | 0.945578 | 16.798 |
| **Query-conditioned gate** | **0.914966** | 0.989796 | 0.996599 | **0.989202** | **0.948980** | 14.187 |

The fusion ablation is central: neither semantic nor graph expert alone wins. Naive averaging catastrophically mixes differently scaled score distributions. Query-conditioned gating provides the improvement.

### 9.7 Reranker ablations

| Top K / layers | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| 10 / 1 | 0.656463 | 0.931973 | 0.969388 | 0.837939 | 0.948980 | 25.444 |
| 10 / 2 | 0.727891 | 0.942177 | 0.979592 | 0.900253 | 0.942177 | 36.163 |
| 20 / 1 | 0.619048 | 0.867347 | 0.925170 | 0.837148 | **0.952381** | 35.330 |
| 20 / 2 | 0.704082 | 0.901361 | 0.959184 | 0.902539 | 0.938776 | 57.579 |
| 40 / 1 | 0.785714 | 0.965986 | 0.993197 | 0.941827 | 0.938776 | 58.335 |
| 40 / 2 | 0.588435 | 0.870748 | 0.938776 | 0.831826 | 0.928571 | 88.028 |

All rerankers remain below the 100 ms budget, but every one sharply hurts R@1 and tool F1. The winner therefore has no reranker.

### 9.8 Optimization ablations

| Setting | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Backbone frozen | 0.857143 | 0.986395 | 0.993197 | 0.967255 | 0.945578 | 13.919 |
| Last two layers trainable | **0.914966** | 0.989796 | 0.996599 | **0.989202** | 0.948980 | 15.880 |
| Full backbone trainable | **0.914966** | 0.989796 | 0.996599 | **0.989202** | 0.948980 | 24.293 |
| Cosine schedule | **0.914966** | 0.989796 | 0.996599 | **0.989202** | 0.948980 | 15.865 |
| Plateau schedule | 0.908163 | 0.989796 | 1.000000 | 0.987798 | 0.948980 | 13.838 |
| EMA | 0.908163 | 0.989796 | 1.000000 | 0.986692 | 0.948980 | 13.686 |
| SWA | 0.908163 | 0.989796 | 1.000000 | 0.986692 | 0.948980 | 17.249 |
| Online hard negatives | 0.911565 | 0.989796 | 0.996599 | 0.988744 | 0.948980 | 15.122 |
| Twin curriculum | **0.914966** | 0.989796 | 0.996599 | 0.988068 | **0.952381** | 13.626 |

Full unfreezing tied last-two accuracy but increased p95. Curriculum was promoted because it preserved top R@1 while producing the best toolset-restricted score. Implementation caution: the EMA run evaluates EMA weights but saves the restored raw state when it becomes “best,” so it is not a clean EMA-checkpoint test. SWA replaces the final state when available rather than selecting it on an independent validation criterion. These rows should be described as implementation-level screens, not definitive optimizer studies.

---

## 10. Confirmation and winning numbers

### 10.1 Three-seed winner

| Seed | R@1 | R@3 | R@5 | Tool F1 | Toolset-restricted R@1 | Mean latency | p95 latency | Epochs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.925170 | 0.989796 | 0.996599 | 0.989148 | 0.955782 | 10.468 ms | 14.804 ms | 23 |
| 123 | 0.908163 | 0.989796 | 1.000000 | 0.988176 | 0.945578 | 12.168 ms | 21.855 ms | 23 |
| 2026 | 0.908163 | 0.989796 | 0.996599 | 0.987636 | 0.945578 | 10.762 ms | 13.878 ms | 23 |
| **Mean** | **0.913832** | **0.989796** | **0.997732** | **0.988320** | **0.948980** | **11.133 ms** | **16.846 ms** | **23** |

Sample standard deviations are approximately 0.00982 for R@1, 0.00076 for tool F1, and 0.00589 for toolset-restricted R@1.

### 10.2 Comparison with the frozen semantic control

| Metric | SBERT control | LEGR-GPS mean | Absolute change |
|---|---:|---:|---:|
| R@1 | 0.874150 | **0.913832** | **+0.039682** |
| R@3 | **0.996599** | 0.989796 | -0.006803 |
| R@5 | **1.000000** | 0.997732 | -0.002268 |
| Tool F1 | 0.980895 | **0.988320** | **+0.007425** |
| Toolset-restricted R@1 | 0.931973 | **0.948980** | **+0.017007** |
| p95 latency | 15.140 ms | 16.846 ms | +1.706 ms |

This is the result that licenses “better development R@1 and tool F1,” not final test superiority. Per-query prediction files and paired-bootstrap deltas were not saved/calculated for this comparison, so the original campaign success criterion requiring confidence intervals excluding zero has not been met.

### 10.3 Confirmation duplication caveat

The manifest contains 12 confirmation runs named `confirm_r1` through `confirm_r4`, each across three seeds. However, `confirm_r2`, `confirm_r3`, and `confirm_r4` have identical resolved model/loss/training configurations and produce identical metrics for each seed. They are duplicate confirmations caused by the top-config collector promoting the same effective configuration from multiple earlier run names. There are only two unique confirmation configurations:

- `confirm_r1`: GPS-degree gated model with twin curriculum;
- `confirm_r2/r3/r4`: the same model without curriculum, repeated three times per seed.

Do not present these as four distinct finalist architectures. The valid winner evidence is the three distinct seeds of `confirm_r1`.

There is also a bookkeeping-only fingerprint mismatch: the three physical `confirm_r1_15t_*` directory suffixes do not match the internal `run_name` fingerprints stored in their `summary.json` files. The resolved configurations, seeds, checkpoint-load reports, and metrics agree, but artifact releases should normalize these names or provide a manifest mapping before publication.

### 10.4 Complete confirmation ledger

| Label | Curriculum | Seed | R@1 | R@3 | R@5 | Tool F1 | Toolset R@1 | p95 ms | Epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| confirm_r1 | yes | 42 | 0.925170 | 0.989796 | 0.996599 | 0.989148 | 0.955782 | 14.804 | 23 |
| confirm_r1 | yes | 123 | 0.908163 | 0.989796 | 1.000000 | 0.988176 | 0.945578 | 21.855 | 23 |
| confirm_r1 | yes | 2026 | 0.908163 | 0.989796 | 0.996599 | 0.987636 | 0.945578 | 13.878 | 23 |
| confirm_r2 | no | 42 | 0.908163 | 0.989796 | 1.000000 | 0.987798 | 0.948980 | 14.972 | 23 |
| confirm_r2 | no | 123 | 0.908163 | 0.989796 | 1.000000 | 0.988284 | 0.945578 | 24.659 | 23 |
| confirm_r2 | no | 2026 | 0.908163 | 0.989796 | 0.996599 | 0.988461 | 0.945578 | 15.324 | 22 |
| confirm_r3 | no, duplicate | 42 | 0.908163 | 0.989796 | 1.000000 | 0.987798 | 0.948980 | 26.111 | 23 |
| confirm_r3 | no, duplicate | 123 | 0.908163 | 0.989796 | 1.000000 | 0.988284 | 0.945578 | 15.808 | 23 |
| confirm_r3 | no, duplicate | 2026 | 0.908163 | 0.989796 | 0.996599 | 0.988461 | 0.945578 | 13.684 | 22 |
| confirm_r4 | no, duplicate | 42 | 0.908163 | 0.989796 | 1.000000 | 0.987798 | 0.948980 | 14.220 | 23 |
| confirm_r4 | no, duplicate | 123 | 0.908163 | 0.989796 | 1.000000 | 0.988284 | 0.945578 | 16.197 | 23 |
| confirm_r4 | no, duplicate | 2026 | 0.908163 | 0.989796 | 0.996599 | 0.988461 | 0.945578 | 13.685 | 22 |

The exact repetition of accuracy metrics for duplicate configuration/seed pairs is consistent with deterministic training. Their latency variation reflects runtime noise, not different architectures.

---

## 11. Functional clustering analysis

### 11.1 Setup

The audit deduplicated all 15-tool Campaign-v4 files into 651 unique graphs. Dominant categories were derived deterministically from the registry:

| Label | Graphs |
|---|---:|
| read | 232 |
| edit | 223 |
| orchestrate | 32 |
| exact tie / mixed | 164 |

The primary analysis excluded the 164 ties, leaving 487 graphs. It compared the winner's GPS graph embedding, the inherited V3 graph embedding, and the frozen SBERT document embedding.

### 11.2 Results in original 256-dimensional cosine space

| Representation | Silhouette | Silhouette p | 5-NN macro-F1 [95% CI] | F1 p | Balanced purity | Distance gap | Gap p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GPS adapter** | **0.1550** | 0.0010 | **0.9462 [0.9102, 0.9735]** | 0.0010 | **0.8931** | **0.2304** | 0.0010 |
| V3 graph | 0.0873 | 0.0010 | 0.9291 [0.8923, 0.9624] | 0.0010 | 0.8868 | 0.1888 | 0.0010 |
| SBERT document | 0.1163 | 0.0010 | 0.9359 [0.9006, 0.9658] | 0.0010 | 0.8763 | 0.1619 | 0.0010 |

GPS neighborhood purity is 0.9466 for read, 0.9390 for edit, and 0.7938 for orchestrate. With 1,000 permutations, 0.0010 is the smallest corrected p-value and should be written `p <= 0.001`.

### 11.3 Allowed interpretation

The analysis corroborates that broad action-function information is present in the winning graph representation and that the GPS space has the strongest observed separation of the three encoders. It does not prove that function emerged independently of tool names, that the model performs causal functional reasoning, or that the clusters generalize to unseen tools. It includes training and test graphs and is therefore not an independent held-out experiment. Use it in an appendix or analysis section, with PCA/t-SNE explicitly marked illustrative.

Recommended caption for `winning_gps_clusters_scatter.png`:

> PCA and t-SNE projections of 487 non-tied Campaign-v4 graph embeddings from the seed-42 LEGR-GPS checkpoint, colored by the uniquely dominant registry action type. Crosses mark class centroids. Projections are illustrative; all hypothesis tests are computed in the original 256-dimensional cosine space. The population includes training and evaluation graphs and should not be interpreted as zero-shot evidence.

---

## 12. Original Functional Categorization results

The original PDF reports the following single-tool routing table:

| Router | Categorization | Standard | Lexical | Confusable | Paraphrase |
|---|---|---:|---:|---:|---:|
| GPT-OSS | Topic-based | 78.3 | 68.1 | 62.7 | 78.0 |
| GPT-OSS | Functional | 93.3 | 75.6 | 68.0 | 90.6 |
| Llama 3.2 | Topic-based | 70.0 | 43.3 | 43.3 | 64.9 |
| Llama 3.2 | Functional | 83.8 | 61.8 | 55.3 | 79.8 |

The maximum absolute gain is 18.5 percentage points for Llama under lexical stress. These runs are not part of the Campaign-v4 model-search manifest, and their raw artifact provenance was not established during this revision audit. Before retaining the table, locate the exact query set, prompts, model identifiers, raw responses, parser, and seed/sampling settings. If those cannot be recovered, move the result out of the main empirical claims rather than treating the graph-clustering audit as a substitute.

The original branching-factor caveat remains important: functional routing has three top-level branches while the topic taxonomy has more than three. A matched-branch or hierarchical control is necessary for a strong claim that label semantics, rather than coarser branching, causes the gain.

---

## 13. Section-by-section rewrite instructions

### Title

The original title can remain only if Functional Categorization retains a fully reproducible experiment. Otherwise use a title centered on twin-aware structural retrieval, for example:

> Beyond Tool Sets: Multi-Expert Directed Graph Retrieval for Agentic Planning

or

> LEGR-GPS: Twin-Aware Execution-Graph Retrieval with Semantic and Structural Experts

### Abstract

Delete 0.963 held-out R@1, 4.39 ms, 400–625x, and the claim that GED drives structural alignment. The abstract should contain:

1. the tool-set-shortcut problem;
2. the structural-twin benchmark design;
3. LEGR-GPS architecture in one sentence;
4. three-seed development result clearly labeled development;
5. a final-test result only after Section 15 is completed;
6. synchronized latency under the exact reported gallery.

Safe current numerical sentence:

> On a 15-tool development gallery with 294 queries and 49 candidate DAGs, LEGR-GPS improves mean Recall@1 from 0.874 for its frozen SBERT expert to 0.914 across three seeds while increasing tool-set F1 from 0.981 to 0.988; final twin-gallery evaluation remains pending.

That final clause is unsuitable for a finished submission, which is why final evaluation is mandatory.

### Introduction and contributions

Replace the old four contributions with:

1. an audit showing that unique-toolset galleries permit semantic shortcutting and that gallery-order tie behavior can create false structural wins;
2. Campaign-v4, with action-first nested vocabularies, two held-out topology families, Azure-generated query variants, and structural twins;
3. LEGR-GPS, combining local directed propagation, relation-biased global attention, explicit tool/relation objectives, and query-conditioned semantic/structural fusion;
4. controlled model search and ablations;
5. final held-out/twin results, once available.

If the single-tool Functional Categorization study remains, frame it as a separate contribution and do not imply the atomic router and LEGR-GPS are jointly trained.

### Related work

Retain dense retrieval, tool learning, graph representation, and generative planning categories, but repair the bibliography. The PDF contains unresolved `[?]` references for graph planners, a duplicated ReAct citation, malformed characters, and unsupported wording such as “remains, to our knowledge, is underexplored.” Verify GTool, OptGraph, TGR, GPT-OSS, BFCL, and any 2026 citations against primary sources before submission.

### Method

Replace the old three-layer mean-pooled directed-GNN/GACL section with Sections 5–6 of this dossier. Include an architecture figure showing:

```text
query -> shared MiniLM -> structural query / tool head / relation head / gate
                      \-> inherited V3 query representation
query -> frozen SBERT query tower ------------------------------\
                                                                    five scores -> query gate -> ranking
candidate text -> frozen SBERT document tower ------------------/
candidate graph -> tool-name MiniLM -> [degree/source/sink]
                -> 4 x (directed local block + relation-biased global attention)
                -> dual attention readout -> GPS graph embedding
                -> inherited V3 graph representation
```

State that SBERT is frozen, reranking is disabled, V3 is retained as an expert, and total topological rank is not used by the new adapter.

### Dataset

Replace all old sizes and split definitions with Section 4. Explain candidate-only versus gold files and why the union is required for evaluation. Include one concrete same-toolset example with different edges. Report 100% held-out twin availability relative to the combined gallery, not “100% twins inside the held-out-only file.”

### Experimental setup

Separate three eras:

- legacy Campaign-v4 diagnostics and V1–V3 baselines;
- the 65-run development search on RTX 6000 Ada;
- final evaluation, still pending.

Give seeds, batch size, early stopping, optimizer, learning rates, loss weights, model parameters, trainable parameters, gallery composition, deterministic candidate shuffle, and synchronized latency procedure. Report the three unavailable backbones as skipped, not tested.

### Main results

The main table should ultimately be the final 322-DAG twin gallery, not the development table. Until it is run, use the confirmation table only in a “Development model selection” subsection. Include SBERT-FT `dag_text`, V3, LEGR-GPS, GPT-OSS, Llama, and BM25 where the protocol is comparable; do not put generative F1 in a retrieval Recall@1 column.

### Ablations

Use the fusion table, graph/structure table, objective table, reranker table, and optimization table. Explicitly state that many screens are sequential/staged rather than factorial. Put the full 65-run ledger in the appendix or supplement.

### Analysis

Include:

- SBERT twin-embedding collapse and tie-order intervention;
- reversed-edge sensitivity;
- true twin-only ranking after recomputation;
- functional clustering as exploratory appendix evidence;
- failure cases with tool-correct/edge-wrong versus tool-wrong categories.

### Efficiency

Replace the old latency figure. Plot synchronized mean/p95 against both gallery size and graph node count. Include candidate encoding/index build time separately from query-time scoring. If claiming ANN complexity, actually build and benchmark an ANN index.

### Conclusion and limitations

Lead with the benchmark lesson and complementary semantic/structural experts. Retain the finite-corpus limitation. Add selection on one dev set, only MiniLM tested, 15-tool-only winner, synthetic/programmatic topology families, GPT-generated queries, orchestrate class imbalance, semantic visibility of function labels, and lack of deployment/load testing.

### LLM-use declaration

Update the original declaration to include all actual uses: Azure GPT-4o for dataset query generation, the named Claude/Gemini systems from the original PDF if accurate, the Cursor coding/research session, and OpenAI Codex for the later model search, audit code, and documentation. Follow the current NeurIPS disclosure requirements and distinguish scientific decisions from editing/code assistance.

---

## 14. Proposed main-paper tables and figures

### Main paper

1. **Figure 1:** Structural-twin motivation: identical tools, different valid dependency graphs; show why tool-set matching cannot decide.
2. **Figure 2:** LEGR-GPS architecture with the five experts and query-conditioned gate.
3. **Table 1:** Campaign-v4 split/gallery statistics, including DAGs, tool sets, topology families, and twin eligibility.
4. **Table 2:** Final 322-DAG results with exact R@1/3/5, MRR, tool F1, true twin-only R@1, reversed-edge sensitivity, GED, and synchronized p95.
5. **Table 3:** Core ablation: semantic-only, GPS-only, V3-only, naive fusion, gated fusion, plus no-tool/no-relation/no-distance losses retrained on the final architecture.
6. **Figure 3:** Accuracy/latency versus gallery size with correctly synchronized timing.
7. **Figure 4:** SBERT versus graph-encoder twin cosine and tie-aware ranking.

### Appendix/supplement

- Full 65-run tables from Section 9.
- V1/V2/V3 diagnostic history.
- All LLM parsing/validity results.
- Functional cluster PCA/t-SNE plus original-space statistics.
- Dataset validation and immutable-file checks.
- Per-topology and per-query-condition breakdowns.
- Failure taxonomy and representative examples.
- Hyperparameters and checkpoint-selection details.

Do not reuse the original topology t-SNE figure as evidence unless its embeddings, labels, source checkpoint, and original-space statistics are recovered.

---

## 15. Experiments required before a NeurIPS submission

### Priority A: blockers for the central claim

1. **Final winner evaluation.** Evaluate each `confirm_r1` seed on the unchanged 322-DAG gallery and compact 50-DAG held-out gallery.
2. **Identical baselines.** Evaluate SBERT-FT, V3, and LEGR-GPS with identical queries, candidate order, tie handling, and cached-candidate policy.
3. **True twin-only metric.** Exclude singleton tool sets and report group-size distribution and chance.
4. **Per-query outputs.** Save exact ranks, predicted DAG/tool set, gold DAG, twin group, topology, condition, latency, and scores.
5. **Paired uncertainty.** Compute paired-bootstrap 95% CIs for LEGR-GPS minus SBERT in exact R@1 and per-query tool F1. The predefined win criterion requires intervals excluding zero.
6. **Tie-aware evaluation.** Detect scores equal within a declared tolerance and report expected/randomized tie metrics.
7. **Synchronized latency.** Warm up CUDA, synchronize around every timed region, and separately measure query encoding, scoring, and full end-to-end retrieval.

### Priority B: needed for scaling/generalization claims

8. Train/evaluate LEGR-GPS at 30 and 45 tools with seeds 42/123/2026. No such winner runs exist now.
9. Report per-held-out-family results for Diamond and asymmetric fork-join.
10. Report per-query-condition results to separate semantic and structural language effects.
11. Run a clean final-architecture loss ablation. The current cumulative math screen used a residual adapter, not GPS-degree.
12. Run a GPS-degree readout ablation; the existing readout screen used the residual adapter.
13. Evaluate at least one additional cached language backbone if the paper claims backbone generality.

### Priority C: strengthens reviewer confidence

14. Run held-out-only or cross-fitted functional clustering; control for exact tool-set identity and tool-count/topology confounds.
15. Add matched-capacity controls: V3 + extra MLP parameters and SBERT + comparably sized head.
16. Calibrate and inspect gate weights by query type; verify that claimed expert specialization is actually observed.
17. Add edge perturbation tests for deletion, insertion, reversal, transitive-edge changes, and parallel-node permutations.
18. Benchmark corpus coverage and an abstention/fallback policy for queries whose correct DAG is absent.
19. Compare against retrieval-augmented generation, where both methods receive the same candidate corpus.
20. Reproduce the single-tool Functional Categorization study with matched branching factors if it remains a central contribution.

### Decision rule for the revised paper

- If LEGR-GPS beats SBERT-FT on final exact R@1 and tool F1 with paired CIs excluding zero, headline the multi-expert architecture.
- If exact R@1 wins but F1 does not, claim improved structural selection at a semantic cost and report the tradeoff.
- If F1 wins but exact R@1 does not, do not claim improved planning; it remains better tool-set matching.
- If neither wins, center the paper on benchmark diagnosis and representational capacity rather than model superiority.
- Under no outcome should the dataset, gallery, or tie order be changed after seeing test results.

---

## 16. NeurIPS reproducibility and reporting checklist

- Define every train/dev/test/candidate file and canonical deduplication key.
- Publish the 45-tool registry and nested tier memberships.
- Publish topology-family membership and generation templates.
- Document Azure query prompts, deployment identifier, sampling parameters, retry/cache logic, and cost.
- Provide all seeds and resolved configs, not only command-line defaults.
- State that screening used the same development set 65 times and reserve test strictly for final evaluation.
- Publish all attempted runs, failures, skipped backbones, and promotion criteria.
- Distinguish total from trainable parameters and frozen semantic-expert parameters.
- Specify whether candidate embeddings are precomputed and whether timing includes tokenization/index search.
- Report mean, standard deviation, and paired confidence intervals.
- Report metric aggregation unit: query, unique DAG, twin group, or topology family.
- State tie tolerance and tie-breaking behavior.
- Separate exact graph match, tool-set F1, graph edit distance, validity, and generation parse rate.
- Include compute: RTX 6000 Ada 49,140 MiB; 3.41 summed hours for the 65 recorded search runs, excluding earlier campaign work and clustering.
- Include environmental and financial cost where relevant: Azure generation $5.0993 estimated; local/remote LLM serving details.
- Release immutable-input hashes and explain that derived relation/tool targets were computed in memory.
- Discuss synthetic benchmark limits, finite candidate coverage, generated-query artifacts, and operational risk of wrong edges.
- Repair all unresolved citations and malformed PDF characters before submission.

---

## 17. Historical LLM-run ledger and parser lessons

Earlier v1/v2 LLM results must not be used as model-quality comparisons because the parser used a small hard-coded vocabulary and outputs were often truncated. They remain useful as an engineering ablation showing how evaluation infrastructure can dominate apparent LLM quality.

### GPT-OSS 120B

| Tier/version | Parse failures | Tool F1 | Exact match | Mean GED | Cyclic | Structurally valid | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 v1 | 37 | 0.0364 | 0.00 | 4.00 | 0 | 13 | 1.338 s |
| 15 v2 | 30 | 0.3950 | 0.38 | 0.20 | 0 | 20 | 0.961 s |
| 15 v3 | 0 | 0.9450 | 0.76 | 0.44 | 0 | 50 | 3.047 s |
| 30 v1 | 46 | 0.0000 | 0.00 | 4.00 | 0 | 4 | 2.414 s |
| 30 v2 | 38 | 0.2400 | 0.24 | 0.00 | 0 | 12 | 2.167 s |
| 30 v3 | 0 | 0.9517 | 0.80 | 0.62 | 0 | 50 | 3.268 s |
| 45 v1 | 49 | 0.0000 | 0.00 | 4.00 | 0 | 1 | 2.489 s |
| 45 v2 | 43 | 0.1400 | 0.14 | 0.00 | 0 | 7 | 2.272 s |
| 45 v3 | 0 | 0.9486 | 0.76 | 0.80 | 0 | 50 | 3.106 s |

### Llama 3.2 3B

| Tier/version | Parse failures | Tool F1 | Exact match | Mean GED | Cyclic | Structurally valid | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 v1 | 5 | 0.0513 | 0.00 | 8.42 | 9 | 36 | 0.911 s |
| 15 v2 | 1 | 0.7826 | 0.08 | 5.65 | 6 | 43 | 1.681 s |
| 15 v3 | 0 | 0.7911 | 0.08 | 6.11 | 6 | 44 | 4.485 s |
| 30 v1 | 2 | 0.0263 | 0.00 | 7.16 | 5 | 43 | 0.949 s |
| 30 v2 | 2 | 0.7769 | 0.06 | 4.80 | 4 | 44 | 1.435 s |
| 30 v3 | 2 | 0.7769 | 0.06 | 4.80 | 4 | 44 | 4.610 s |
| 45 v1 | 4 | 0.0908 | 0.00 | 7.22 | 1 | 45 | 0.944 s |
| 45 v2 | 5 | 0.6844 | 0.04 | 4.58 | 2 | 43 | 1.527 s |
| 45 v3 | 5 | 0.6844 | 0.04 | 4.58 | 2 | 43 | 5.080 s |

The longer v3 latency is partly caused by allowing complete outputs (`max_tokens=1024`) instead of truncating responses. This is not a hardware-controlled latency ablation.

### Fair-gallery execution note

The first full-gallery attempt spent about 19 minutes constructing an unnecessary 322×322 GED matrix and then failed because a query wrapper lacked `__len__`; no metrics were saved. The corrected evaluator skipped that matrix and completed in 47 seconds. Do not count the failed attempt as an experimental run or average its wall time into inference latency. It is relevant only to reproducibility and implementation history.

---

## 18. Original-PDF figure and table replacement map

| Original item | Original role | Revision action |
|---|---|---|
| Figure 1 | Claimed flat 4.39 ms latency and 0.963 held-out R@1 | Replace completely with synchronized, cache-aware final evaluation. |
| Figure 2 | Generic autoregressive-versus-retrieval teaser | Retain concept only; redraw to emphasize same-toolset graph alternatives and finite-corpus retrieval. |
| Figure 3 | Three-layer directed LEGR + GED/GACL | Replace with five-expert LEGR-GPS architecture. |
| Table 1 | Topic versus Functional single-tool routing | Retain only after raw-run provenance is verified; add matched-branch caveat/control. |
| Table 2 | Structural retrieval and latency at 15 tools | Replace with final 322-DAG twin-gallery table. Never mix F1 with generative Recall@1. |
| Table 3 | Parameters, GED, latency | Replace 23.5M LEGR with 74.54M winner and use valid latency protocol. |
| Table 4 | Claimed 15/30/45 scaling | Remove until LEGR-GPS is trained/evaluated at 30/45 tools. Legacy SBERT/LLM scaling may appear separately. |
| Figure 4 | Old scalability curves | Remove or regenerate from future LEGR-GPS 30/45 results. |
| Figure 5 | t-SNE claimed to demonstrate zero-shot topology generalization | Do not use as proof. Replace with the functional plot only as exploratory analysis or regenerate topology results with original-space tests. |
| Appendix B.3 | Thirteen old LEGR failures | Recompute using final LEGR-GPS predictions; old cases belong to a different checkpoint/protocol. |
| Appendix C | GED necessity claim | Replace with mixed GED findings and the new composite-objective ablation. |

The original PDF also needs typographic cleanup: several arrows/multiplication symbols were corrupted by PDF encoding, multiple references are rendered as `[?]`, and the sentence claiming query-to-DAG retrieval “remains, to our knowledge, is underexplored” is grammatically malformed.

---

## 19. Source-of-truth artifact map

| Evidence | Repository source |
|---|---|
| Original claims and layout | `Functional_Taxonomy_v2.pdf` |
| Campaign chronology and diagnostic tables | `artifacts/campaign_v4/SESSION_LOG_20260831_20260901.md` |
| Architecture defects | `artifacts/campaign_v4/architecture_audit.md` |
| Dataset tests | `artifacts/campaign_v4/dataset_test_report.md` |
| Full validation | `artifacts/campaign_v4/complete_validation_report.json` |
| SBERT GED/no-GED exact files | `artifacts/campaign_v4/results/sbert_ft_{no_ged,ged}_{15,30,45}t_s42/eval_metrics.json` |
| Compact-gallery V1/V2/V3 CSVs | `artifacts/campaign_v4/results/legr_*_eval*.csv` |
| Twin-filled gallery | `artifacts/campaign_v4/results/full_gallery_15t.json` |
| SBERT edge/tie proof | `artifacts/campaign_v4/results/sbert_text_only_proof_15t.json` |
| Corrected LLM runs | `artifacts/campaign_v4/results/{gptoss,llama}_{15,30,45}t_heldout_v3.json` |
| New model code | `src/legr_experiments/` |
| Search definitions | `src/legr_experiments/search.py` |
| Exact 65-run record | `artifacts/legr_model_search/search_manifest.json` |
| Search integrity | `artifacts/legr_model_search/integrity_report.json` |
| Winner configs/checkpoints | `artifacts/legr_model_search/confirm_r1_15t_s{42,123,2026}_*/` |
| Winner-focused summary | `artifacts/legr_model_search/LEGR_WINNER_ANALYSIS.md` |
| Model-search implementation log | `artifacts/legr_model_search/SESSION_IMPLEMENTATION_LOG.md` |
| Functional audit | `artifacts/legr_model_search/action_cluster_seed42/FUNCTIONAL_CLUSTERING_ANALYSIS.md` |
| Functional plot | `artifacts/legr_model_search/action_cluster_seed42/winning_gps_clusters_scatter.png` |

When this dossier conflicts with a prose session summary, prefer the current raw CSV/JSON artifact and resolved run configuration. Known examples are the compact 15-tool gallery's 49—not 50—unique tool sets, and the current 30/45 SBERT-FT JSON values.

---

## 20. Final handoff summary

The paper editor should treat the original PDF as a narrative scaffold, not an authoritative result source. Preserve the high-level problem—semantic tool choice versus structural plan choice—but rebuild the LEGR method, dataset section, results, efficiency claims, and limitations around Campaign-v4 and LEGR-GPS.

The new architecture is scientifically more credible because it fixes semantic node initialization, preserves direction, removes arbitrary topological rank, explicitly supervises tools and relations, and learns when to trust semantic versus graph experts. The development improvement is real across three seeds, and the functional audit supports organized graph representations. The central NeurIPS claim nevertheless remains incomplete until the confirmed winner is tested on the unchanged twin-filled gallery with true twin-only metrics, deterministic tie handling, synchronized latency, and paired confidence intervals.
