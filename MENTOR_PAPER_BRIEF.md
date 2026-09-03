# Mentor Brief: LEGR Paper

## Paper at a glance

**Working title:** *Beyond the Next Token: Latent Execution-Graph Retrieval for Agentic Planning*

**Author:** Tanay Kadam, North Carolina State University / Lenovo

The paper studies **intent mapping**: converting a natural-language request into either a single tool call or a multi-step executable workflow. It separates this into two related but distinct problems:

1. **Atomic routing:** selecting one tool for a request.
2. **Compositional planning:** selecting an entire directed acyclic graph (DAG) of tools, including the dependency edges between them.

The paper proposes a different solution for each problem. Atomic requests use a zero-shot LLM router with tools grouped by **functional action type**. Multi-step requests use **Latent Execution-Graph Retrieval (LEGR)**, which retrieves a prevalidated workflow from a candidate gallery rather than generating a plan token by token.

The two methods are complementary, not one end-to-end trained system. The paper conceptually assumes an upstream atomic-versus-compositional gate, but that gate is not evaluated.

## Central motivation

Tool-using systems have two recurring failure modes.

### Semantic drift in single-tool routing

Tools are often grouped by topic, such as accounts, databases, or support. Topic words can be misleading when two tools concern the same entity but perform different actions. For example, reading a profile and editing a profile share topical vocabulary but have different side effects. A router that relies heavily on nouns can therefore select the wrong operational action when the request is paraphrased or important keywords are removed.

The paper tests whether grouping tools by **Data Retrieval**, **State Modification**, and **Orchestration** gives an LLM a more stable decision boundary than conventional topic grouping.

### Edge hallucination in multi-step planning

A multi-tool plan must identify both the correct tools and the correct dependencies. Two workflows can use exactly the same tools while connecting them differently. A flattened text representation can identify the tool set without resolving this structural ambiguity, while an autoregressive LLM can hallucinate or omit edges.

LEGR addresses this by embedding the request and each candidate DAG in the same vector space. The system retrieves an already validated DAG, so it cannot return a malformed edge structure as long as the indexed corpus itself is valid.

## Claimed contributions

The paper includes the following contributions:

1. A training-free, two-stage single-tool router based on functional rather than topical categorization.
2. LEGR, which formulates multi-step orchestration as closed-corpus execution-graph retrieval rather than autoregressive graph generation.
3. LEGR V3, a parameter-efficient graph retriever using one shared MiniLM backbone, set attention, and a directed GNN.
4. V2/V3 and InfoNCE/GED ablations at 15, 30, and 45 tools.
5. A single-model fine-tuned Sentence-BERT baseline that does not contain a second encoder or GNN.
6. Campaign V4, a synthetic, multi-scale graph-retrieval benchmark with structural twins and topology-held-out splits.
7. Synchronized latency measurements for all 12 V2/V3 checkpoint variants.
8. Diagnostic studies of action-versus-semantic embedding geometry, frozen V3 transfer to atomic routing, and a frozen-encoder routing adapter trained on independent routing utterances.

## System formulation

### Atomic path: functional categorization

The single-tool router uses two zero-shot LLM decisions:

1. Select a category.
2. Select a tool only from that category.

The experiment compares two label spaces under otherwise identical prompting:

- **Topic-based categories:** tools grouped by subject or domain.
- **Functional categories:** tools grouped by action—retrieval, modification, or orchestration.

No router fine-tuning is performed. The purpose is to isolate whether changing the category organization improves robustness.

### Compositional path: LEGR

For a request `q` and candidate graph `G`, LEGR learns normalized embeddings `z_q` and `z_g`. Retrieval selects the graph with the highest cosine similarity. Candidate graph embeddings are computed once and cached; online inference therefore consists primarily of encoding one query and scoring it against the cached gallery.

This is a **closed-corpus retrieval** formulation. LEGR cannot invent a graph absent from the gallery. Its intended deployment advantage is that every returned plan can be validated before indexing.

## LEGR V3 architecture

V3 has one shared `all-MiniLM-L6-v2` backbone.

### Query representation

The shared MiniLM encodes the natural-language request. A learned projection produces a normalized 256-dimensional retrieval vector.

### Node representation

Every graph node is a tool invocation. The same MiniLM encodes the readable tool name, after which a learned projection produces a 64-dimensional node feature. A learned topological-position embedding supplies order information.

### Directed GNN branch

A three-layer directed message-passing network uses separate transformations for:

- the current node,
- predecessor messages, and
- successor messages.

This distinction is important because dependency direction changes the meaning of a workflow. Mean pooling over the final node states produces the structural summary.

### Set-attention branch

A learned scalar attention score pools the pre-message-passing node features. This branch preserves information about which tools are present even when message passing makes structurally similar node states difficult to distinguish.

### Fusion

The directed-GNN summary and attention-pooled tool-set summary are concatenated and projected to a normalized 256-dimensional graph embedding.

V3 has **23.48 million parameters**. V2 uses a separate frozen MiniLM for node names and only the mean-pooled directed-GNN representation, resulting in **46.11 million parameters**. The single-SBERT baseline has **22.71 million parameters**.

## Training objectives

### No-GED models

The no-GED V2 and V3 models use standard symmetric InfoNCE. For a batch of paired query and graph embeddings, cross-entropy is computed query-to-graph and graph-to-query and the two directions are averaged.

### GED variants

The GED checkpoints use the same symmetric InfoNCE objective plus a weighted structural auxiliary term. Pairwise graph edit distances are precomputed, and structurally close negatives receive more weight.

An important implementation caveat is disclosed in the paper: the evaluated auxiliary denominator combines an exponentiated positive logit with weighted **raw negative logits**. This is not a conventional probability-normalized contrastive loss. Consequently:

- the existing GED results remain valid measurements of the implemented checkpoints;
- they should be described as an implementation-specific ablation;
- they do not establish a general result about correctly formulated GED-aware contrastive learning; and
- a corrected exponentiated-negative version would require retraining.

The paper therefore does not select GED as the default V3 objective. The primary V3 result uses standard InfoNCE without GED.

## Campaign V4 dataset

Campaign V4 uses nested libraries of 15, 30, and 45 tools and 22 programmatic DAG families. Six fixed natural-language requests are generated for every DAG using Azure GPT-4o.

| Tools | Unique DAGs | Train | Development | In-domain test | Topology-held-out | Candidate-only | Total queries |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 651 | 248 | 49 | 32 | 50 | 272 | 3,906 |
| 30 | 916 | 346 | 69 | 46 | 70 | 385 | 5,496 |
| 45 | 1,313 | 498 | 99 | 66 | 100 | 550 | 7,878 |

### Held-out protocol

Diamond and Asymmetric Fork-Join graphs are excluded from training and development and used for topology-held-out testing. Canonical hashes over labelled nodes and directed edges prevent exact DAG leakage.

The complete retrieval galleries are the union of candidate-only graphs and the relevant test graphs:

- 15 tools: 322 candidates, 300 held-out requests, 50 gold DAGs.
- 30 tools: 455 candidates, 420 held-out requests, 70 gold DAGs.
- 45 tools: 650 candidates, 600 held-out requests, 100 gold DAGs.

Every gold graph is present exactly once in its gallery.

### Structural twins

The dataset deliberately contains graphs with the same tool multiset but different edges. These cases test whether a model can rank the correct topology rather than merely recover the correct tool names.

## Baselines

### Fine-tuned single Sentence-BERT

One `all-MiniLM-L6-v2` model encodes both requests and flattened DAG dependency strings. It is trained using Multiple Negatives Ranking Loss. It has no separate query and graph models, no GNN, and no LEGR-specific projection. This is the principal dense text-retrieval baseline.

### LEGR V2

V2 uses a query MiniLM, a separate frozen node-name MiniLM, and a directed GNN. It is retained as an architectural ablation and is evaluated both with and without the GED auxiliary.

### Generative models

Llama 3.2 3B and GPT-OSS 120B generate one JSON workflow containing tools and edges. These models are evaluated on fixed 50-query subsets and are not directly equivalent to closed-corpus retrieval.

## Evaluation metrics

For graph retrieval, the paper reports:

- **Tool-Set F1:** overlap between tools in the predicted and gold graphs.
- **Recall@1:** exact gold-DAG retrieval accuracy.
- **Recall@3 and Recall@5:** whether the exact gold DAG appears in the first 3 or 5 results.
- **MRR@5:** reciprocal-rank quality within the first five results.
- **Mean GED:** structural distance between the top-ranked and gold graphs; lower is better.

For atomic routing, the primary metric is end-to-end tool-selection accuracy. The frozen V3 routing diagnostic additionally reports macro-F1, ranked recall, per-tool metrics, and confusion matrices in its saved artifacts.

## Experiment 1: functional versus topic routing

Four linguistic conditions are evaluated:

1. **Standard:** original explicit requests.
2. **Lexical:** keyword masking or synonym substitution reduces noun matching.
3. **Confusable:** sibling tools share descriptions but have different actions.
4. **Paraphrase:** syntax changes while intent is preserved.

Dataset sizes are 1,005 Standard, 1,005 Lexical, 450 Confusable, and 1,255 Paraphrase requests.

### Routing accuracy

| Router | Categorization | Standard | Lexical | Confusable | Paraphrase |
|---|---|---:|---:|---:|---:|
| GPT-OSS | Topic | 78.3% | 68.1% | 62.7% | 78.0% |
| GPT-OSS | Functional | **93.3%** | **75.6%** | **68.0%** | **90.6%** |
| Llama 3.2 | Topic | 70.0% | 43.3% | 43.3% | 64.9% |
| Llama 3.2 | Functional | **83.8%** | **61.8%** | **55.3%** | **79.8%** |

Functional categorization wins for both LLMs under every condition. The largest improvement is **18.5 absolute points** for Llama under lexical cue reduction.

This supports the scoped claim that action-oriented categories improve zero-shot routing robustness at the 15-tool scale. It does not prove that action semantics alone cause the gain because functional routing has three top-level branches while topic routing has more than three. A matched-branching-factor semantic control remains needed.

## Experiment 2: Campaign V4 graph retrieval

### V3 versus the single-SBERT baseline

| Tools | Model | Tool F1 | R@1 | R@3 | R@5 | MRR@5 | Mean GED |
|---:|---|---:|---:|---:|---:|---:|---:|
| 15 | Fine-tuned SBERT | **0.982** | 0.203 | **0.753** | **0.907** | **0.476** | **2.617** |
| 15 | V3, InfoNCE | 0.951 | **0.210** | 0.567 | 0.813 | 0.424 | 3.097 |
| 15 | V3, InfoNCE+GED | 0.953 | 0.163 | 0.537 | 0.770 | 0.378 | 3.213 |
| 30 | Fine-tuned SBERT | **0.985** | **0.250** | **0.840** | **0.948** | **0.538** | **2.390** |
| 30 | V3, InfoNCE | 0.974 | 0.231 | 0.752 | 0.898 | 0.497 | not recorded |
| 30 | V3, InfoNCE+GED | 0.966 | 0.229 | 0.731 | 0.905 | 0.491 | 3.098 |
| 45 | Fine-tuned SBERT | **0.978** | 0.255 | **0.853** | **0.968** | 0.542 | **2.385** |
| 45 | V3, InfoNCE | 0.971 | **0.418** | 0.817 | 0.940 | **0.624** | not recorded |
| 45 | V3, InfoNCE+GED | 0.961 | 0.378 | 0.820 | 0.942 | 0.605 | 2.450 |

### Interpretation

- At 15 tools, V3 no-GED improves exact retrieval over SBERT by only 0.7 points. SBERT is substantially better at R@3, R@5, MRR, Tool F1, and GED.
- At 30 tools, SBERT remains stronger than V3.
- At 45 tools, V3 no-GED reaches **0.418 exact R@1**, compared with **0.255** for SBERT: a 16.3-point advantage.
- SBERT remains stronger at R@3 and R@5 even at 45 tools. It usually places the correct graph somewhere near the top, but V3 is better at placing the exact graph first at the largest scale.
- No model exceeds 0.418 exact accuracy, so structural-twin ranking remains an open problem.

The strongest LEGR claim is therefore narrow but meaningful: **V3 improves exact structural ranking at the 45-tool scale while retaining a model footprint close to single-SBERT.** It is not uniformly superior across every metric or scale.

## Experiment 3: V2/V3 and GED ablation

All 12 combinations are evaluated: V2/V3, InfoNCE/InfoNCE+GED, and 15/30/45 tools.

Important findings:

- V3 consistently produces higher Tool-Set F1 than V2.
- V3 is approximately half the size of V2 because it shares MiniLM.
- V3 no-GED is the best exact retriever at 15 and 45 tools.
- V2+GED is best at 30 tools, reaching R@1 of 0.295.
- GED lowers V3 R@1 at every scale.
- GED improves V2 R@1 by 7.1 points at 30 tools and approximately 1.1 points at 45 tools.

These results show that GED is architecture- and scale-dependent. They do not support the claim that GED is the main reason V3 learns directed structure.

## Experiment 4: measured latency

Latency is measured on one NVIDIA RTX 6000 Ada GPU for all 12 checkpoints.

### Protocol

- Candidate embeddings are cached for online evaluation.
- Online batch size is one.
- Online time includes tokenization, host-to-device transfer, query encoding, exact cosine scoring, and top-1 selection.
- Model loading, file I/O, and graph indexing are excluded from online latency.
- Twenty warm-up requests are followed by three complete passes over the held-out query set.
- CUDA is synchronized around every measurement.
- Indexing uses one warm-up and five timed repetitions at batch size 64.

### Headline measurements

Across all variants, online mean latency ranges from **4.58 to 5.71 ms**, while p95 ranges from **5.10 to 8.16 ms**.

For the primary V3 no-GED model:

| Tools | Gallery | Online mean | Online p95 | One-time indexing |
|---:|---:|---:|---:|---:|
| 15 | 322 | 5.49 ms | 8.16 ms | 103.1 ms |
| 30 | 455 | 5.28 ms | 7.93 ms | 145.0 ms |
| 45 | 650 | 5.39 ms | 7.91 ms | 212.3 ms |

Online latency is nearly flat at these gallery sizes because MiniLM query encoding dominates the relatively small exact similarity calculation. V3 indexing is slower than V2 because V3 computes the additional set-attention fusion. GED does not change the inference architecture, so small GED/no-GED latency differences are measurement variation rather than algorithmic speedups.

## Experiment 5: action-versus-semantic latent geometry

This diagnostic asks whether frozen 15-tool V3 no-GED graph embeddings organize more clearly by action class than by topic.

The same 322 graph embeddings are assigned two taxonomies:

- **Action:** Data Retrieval, State Modification, Orchestration, or Mixed.
- **Matched semantic:** Account & Subscription, Data & Access Management, Service & Incident Operations, or Mixed.

Each semantic category contains five tools: exactly two retrieval, two modification, and one orchestration tool. A strict-majority assignment produced no orchestration-labelled graphs, so the analysis uses a disclosed unique-plurality rule and labels exact ties Mixed.

The paper includes separate PCA and deterministic t-SNE visualizations. Mixed graphs are omitted from the figures for legibility:

- Action plot: 131 retrieval, 116 modification, and 6 orchestration graphs.
- Semantic plot: 58 account/subscription, 51 data/access, and 87 service/incident graphs.

The statistical comparison uses all 322 embeddings in their original 256-dimensional cosine space:

| Diagnostic | Action labels | Semantic labels |
|---|---:|---:|
| Cosine silhouette | 0.014 | **0.030** |
| 5-NN macro-F1 | 0.821 | **0.885** |
| 5-NN purity | **0.786** | 0.717 |

One-sided permutation tests for the action-minus-semantic difference are not significant: silhouette `p=0.354` and purity `p=0.306`.

The correct conclusion is mixed. Action labels have stronger local-neighborhood purity, but semantic labels have better global silhouette and 5-NN macro-F1. The analysis does **not** establish a globally action-dominant V3 representation. The plots are descriptive, and their visual axes cannot be directly compared because each non-mixed subset is projected separately.

## Experiment 6: frozen V3 atomic-routing transfer

The two existing 15-tool V3 checkpoints are frozen and evaluated on all four routing conditions without retraining. Every routing candidate is represented as a one-node, zero-edge graph, and the gallery contains exactly 15 tools. Dataset labels are mapped through explicit aliases; unmapped labels are rejected.

| Model | Standard accuracy/F1 | Lexical accuracy/F1 | Confusable accuracy/F1 | Paraphrase accuracy/F1 |
|---|---:|---:|---:|---:|
| V3 no-GED | 0.524/0.495 | **0.287/0.261** | **0.402/0.338** | 0.512/0.493 |
| V3 with GED | **0.531/0.502** | 0.274/0.249 | 0.400/**0.340** | **0.531/0.510** |

Across all 3,715 requests, GED changes aggregate accuracy only from 44.1% to 44.5%.

This is framed as **cross-dataset atomic-routing transfer**, not unseen-topology generalization. The results are above random chance but much weaker than the two-stage functional routers, especially under lexical cue removal. They show that V3 can technically score atomic graphs but do not show that V3 learned or implements the functional taxonomy. Campaign V4 and routing tool vocabularies also differ, making this partly a vocabulary-transfer test.

## Experiment 7: supervised routing adapter on frozen V3

To test whether atomic-routing accuracy improves when V3 receives realistic routing supervision without rewriting the Campaign checkpoint, a residual adapter was trained on top of frozen 15-tool V3 no-GED embeddings.

### Protocol

- Start from the existing V3 no-GED 15-tool checkpoint and keep it byte-identical.
- Freeze MiniLM, set attention, directed GNN, and fusion weights; no V3 parameter receives gradients.
- Train only a residual query MLP (`256 → 128 → 256`) plus 15 residual tool prototypes in the shared 256-dimensional space.
- Use an independent corpus of 900 training and 150 validation utterances (60 / 10 per tool), covering explicit, cue-reduced, confusable, paraphrase, and hard-negative styles.
- Assert zero exact normalized overlap with Standard, Lexical, Confusable, and Paraphrase evaluation files.
- Keep one-node, zero-edge candidates and evaluate all four complete datasets.
- Report mean ± standard deviation over seeds 42, 123, and 2026.

### Accuracy (%)

| Stage | Standard | Lexical | Confusable | Paraphrase |
|---|---:|---:|---:|---:|
| Frozen V3 no-GED | 52.44 | 28.66 | 40.22 | 51.24 |
| Frozen V3 + routing adapter | **75.12 ± 0.53** | **48.72 ± 0.75** | **61.56 ± 3.20** | **73.49 ± 0.80** |

Corresponding macro-F1 means are 0.746 / 0.472 / 0.591 / 0.736. The original checkpoint SHA256 is unchanged before and after training (`6027a4df…7f727a`).

### Interpretation

Realistic routing supervision substantially improves atomic transfer over frozen V3, including a ~20-point Lexical gain. The experiment therefore supports the claim that **Campaign-pretrained V3 can be adapted to atomic routing with routing-specific supervision while leaving the original model untouched**.

It does **not** show that Campaign graph training alone transfers to robust atomic routing. Because every candidate still has one node and zero edges, the directed GNN remains inactive and the adapter mainly reshapes residual boundaries in the shared text-embedding space. The adapter also remains below the strongest GPT-OSS Functional router on Standard, Lexical, and Paraphrase, so Functional Categorization stays the better training-free atomic solution.

## Generative comparison

The paper reports 50 held-out requests per tool scale for each generator.

| Tools | Model | Exact | Tool F1 | GED | Valid | Mean latency |
|---:|---|---:|---:|---:|---:|---:|
| 15 | Llama 3.2 3B | 0.080 | 0.791 | 6.114 | 0.880 | 4,485 ms |
| 15 | GPT-OSS 120B | **0.760** | **0.945** | **0.440** | **1.000** | **3,047 ms** |
| 30 | Llama 3.2 3B | 0.060 | 0.777 | 4.796 | 0.917 | 4,610 ms |
| 30 | GPT-OSS 120B | **0.800** | **0.952** | **0.620** | **1.000** | **3,268 ms** |
| 45 | Llama 3.2 3B | 0.040 | 0.684 | 4.581 | 0.956 | 5,080 ms |
| 45 | GPT-OSS 120B | **0.760** | **0.949** | **0.800** | **1.000** | **3,106 ms** |

GPT-OSS is much more accurate than the retrievers on its sampled requests, while retrieval operates in milliseconds. However, this is not a controlled head-to-head comparison:

- generators use only 50 requests per scale;
- retrievers use all 300/420/600 held-out requests;
- generators may synthesize a graph absent from the corpus;
- retrieval is guaranteed that the gold graph exists in the gallery; and
- GPT-OSS latency includes cloud network time, while retrieval runs locally on a GPU.

The appropriate interpretation is that the two approaches have complementary advantages and motivate a hybrid: retrieval supplies a fast validated shortlist, while generation handles novel workflows or difficult reranking cases.

## Training configuration

For LEGR:

- backbone: `all-MiniLM-L6-v2`;
- lower four of six transformer blocks frozen;
- upper two blocks, embeddings, projections, GNN, set attention, and temperature trained;
- backbone learning rate: `2e-5`;
- remaining-module learning rate: `2e-4`;
- AdamW weight decay: `1e-4`;
- gradient clipping: `1.0`;
- three warm-up epochs;
- batch size: `128`;
- maximum sequence length: `128`;
- maximum 75 epochs with patience-15 early stopping;
- learnable temperature initialized to `0.05`;
- GED scale: `2.5`;
- GED margin: `0.05`;
- GED coefficient: `0.10` at 15 tools and `0.30` at 30/45 tools;
- random seed: `42`.

The single-SBERT model uses AdamW at `2e-5`, batch size 128, maximum sequence length 128, a 100-epoch cap, and patience-15 early stopping.

## What the paper can responsibly claim

1. Functional grouping improves the tested zero-shot routers under all four linguistic conditions at 15 tools.
2. V3 provides a parameter-efficient way to jointly encode tool identity and directed graph structure.
3. On the 45-tool Campaign V4 gallery, V3 no-GED substantially improves exact R@1 over the single-SBERT baseline.
4. Cached LEGR retrieval operates at approximately five milliseconds on the measured hardware and galleries.
5. GED is not uniformly beneficial and should remain an ablation.
6. The geometry and frozen-routing studies provide mixed evidence; neither proves that V3 implements an action taxonomy.
7. A frozen-encoder routing adapter trained on independent routing utterances substantially improves atomic accuracy over zero-shot V3 transfer, while leaving the original Campaign checkpoint unchanged.

## What the paper should not claim

1. LEGR is not an unrestricted graph generator or a solution for workflows absent from its corpus.
2. V3 is not uniformly better than SBERT across scales or top-k metrics.
3. The GED auxiliary has not been validated as a generally correct GED-aware contrastive objective.
4. The functional-routing gains are not fully disentangled from the number of routing branches.
5. The frozen routing results do not demonstrate unseen-topology generalization.
6. The routing-adapter gains demonstrate supervised atomic adaptation of a frozen encoder, not automatic transfer from multi-step graph retrieval.
7. The t-SNE figures alone do not demonstrate action separation.
8. The generative and retrieval results are not directly controlled comparisons.

## Main limitations and next experiments

### Highest-priority issues

1. **Correct and retrain the GED objective.** Use exponentiated negatives or another mathematically justified weighting formulation, then rerun all GED variants.
2. **Run multiple random seeds.** Current learned-model results use only seed 42. At least three to five seeds should be reported for V3 no-GED, V2 no-GED, and fine-tuned SBERT.
3. **Control routing branch count.** Compare three functional branches against exactly three semantic branches with comparable tool counts.

### External-validity issues

4. Add human-authored or naturally occurring requests and workflows. Campaign V4 is synthetic and tool names contain strong action cues.
5. Test more held-out topology families and richer control flow, including conditions, cycles, and data-dependent branches.
6. Measure candidate-corpus coverage and abstention when the requested workflow is absent.

### Systems and comparison issues

7. Evaluate approximate nearest-neighbor indexing, larger galleries, concurrent load, CPU serving, memory usage, and complete end-to-end service latency.
8. Run paired retrieval/generation comparisons on identical requests and, where possible, comparable hardware.
9. Evaluate a hybrid retrieval-plus-generation or retrieval-plus-reranking pipeline.
10. Consolidate experiment manifests and generate LaTeX tables directly from recorded result files to reduce reporting drift.

## Recommended mentor discussion points

The most useful decisions to discuss are:

1. Should the paper lead with the 45-tool exact-ranking result or present routing and LEGR as equally weighted contributions?
2. Is the current title too broad given that LEGR retrieves from a closed validated corpus?
3. Should the existing GED experiment remain in the main paper, move entirely to an appendix, or be replaced after retraining?
4. Which multi-seed runs are essential before submission?
5. Is the functional-routing experiment publishable without a matched three-branch semantic control?
6. Should the geometry result remain as an honest negative diagnostic, or is it distracting from the central retrieval contribution?
7. Would a hybrid retrieval/generation experiment make the systems contribution more compelling?

## One-paragraph summary for discussion

The paper separates agentic intent mapping into atomic tool routing and multi-step execution-graph retrieval. For atomic routing, reorganizing tools by action class improves zero-shot GPT-OSS and Llama routing under standard, lexical, confusable, and paraphrased requests, with a maximum gain of 18.5 points. Frozen V3 transfer to atomic routing is only moderate (about 44% aggregate), but a residual adapter over the unchanged V3 no-GED checkpoint raises accuracy to 75.1 / 48.7 / 61.6 / 73.5 across the four conditions when trained on independent routing utterances. For multi-step tasks, LEGR V3 embeds requests and prevalidated DAGs using one shared MiniLM, a directed GNN, and set attention. On Campaign V4, V3's clearest advantage occurs at 45 tools, where exact R@1 is 0.418 versus 0.255 for a one-model fine-tuned SBERT baseline, although SBERT remains stronger at top-k coverage. Cached V3 retrieval takes about 5.4 ms, compared with multi-second generative inference. GED ablations are mixed and affected by an implementation-specific loss formulation. Geometry and frozen-routing diagnostics do not establish a globally action-oriented latent space, supporting a cautious conclusion: functional categorization and LEGR are complementary solutions with different strengths rather than one unified learned mechanism.
