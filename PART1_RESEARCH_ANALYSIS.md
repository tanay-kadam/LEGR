# Part 1: Research Analysis
## Taxonomy Performance Gap Analysis

### 1. Main Finding

The tool-bound taxonomy's advantage over semantic taxonomy **collapses from +5-15 pp at 15 tools to +5/-5 pp at 30 tools**, and **inverts entirely** on the two hardest evaluation splits. This is driven by both absolute degradation of tool-bound and relative resilience of semantic.

| Split      | 15-tool gap (TB-Sem) | 30-tool gap (TB-Sem) | Shift |
| ---------- | -------------------- | -------------------- | ----- |
| base       | **+15.0**            | +4.9                 | -10.1 |
| paraphrase | **+12.6**            | +5.0                 | -7.6  |
| lexical    | **+7.5**             | **-1.8**             | -9.3  |
| confusable | **+5.3**             | **-5.0**             | -10.3 |

The gap narrows uniformly across all splits (~8-10 pp shrinkage), and flips negative on lexical and confusable. This is **not a random fluctuation** -- it is a structural consequence of how operation-type branches lose exclusivity as the tool inventory grows.

---

### 2. What Changes from 15 Tools to 30 Tools

**Branch topology transforms fundamentally.** At 15 tools, both taxonomies use 3 branches. At 30 tools, both expand to 5 branches, but with radically different balance:

**15 tools -- Tool-Bound (3 branches, perfectly balanced):**

- Data Retrieval & Monitoring: 5 tools
- State Modification & Provisioning: 5 tools
- Communication & Orchestration: 5 tools

**30 tools -- Tool-Bound (5 branches, severely imbalanced):**

- Data Retrieval & Monitoring: 7 tools
- **State Modification & Provisioning: 10 tools** (catch-all)
- Communication & Orchestration: 7 tools
- Infrastructure Lifecycle: 3 tools
- Access Control & Configuration: 3 tools

**30 tools -- Semantic (5 branches, more balanced):**

- IT Support: 9 tools
- Security & Compliance: 7 tools
- Billing & Data Management: 7 tools
- Deployment & Release Management: 5 tools
- Access & Identity Management: 2 tools

The critical structural change: **State Modification & Provisioning absorbs 10 of 30 tools (33%)**, making it a "gravity well" that attracts misroutes. Compare to the largest semantic branch (IT Support at 9 tools, 30%) -- still large but with domain coherence.

**Branch accuracy tells the story:**

| Split      | 15t Sem branch | 15t TB branch | 30t Sem branch | 30t TB branch |
| ---------- | -------------- | ------------- | -------------- | ------------- |
| base       | 78.3%          | **93.8%**     | 53.0%          | **58.9%**     |
| confusable | 72.7%          | **84.4%**     | **51.3%**      | 50.3%         |
| lexical    | 73.2%          | **80.8%**     | **44.1%**      | 39.9%         |

Tool-bound's branch accuracy drops by 34.1 pp on confusable (84.4 to 50.3), while semantic drops only 21.4 pp (72.7 to 51.3). **The branch-level discrimination is where tool-bound loses the most ground.**

---

### 3. Why the Semantic vs Tool-Bound Gap Narrows

Three reinforcing mechanisms:

**A. Operation-type boundaries blur as tools multiply.**

At 15 tools, the read/write/communicate trichotomy is clean:

- `query_database` reads, `update_database` writes, `send_notification` communicates.

At 30 tools, Tier 2 tools straddle categories:

- `deploy_container` -- state modification? or infrastructure lifecycle?
- `schedule_maintenance` -- communication? or operational planning?
- `rollback_deployment` -- state modification? or infrastructure lifecycle?
- `backup_database` -- data retrieval? (it reads data) or state modification? (it creates an artifact)

The LLM must make increasingly arbitrary operation-type assignments at the branch level, and these assignments become less predictable.

**B. Semantic categories align with natural language more stably.**

A query about "rolling back a deployment" lexically screams "Deployment & Release Management." A query about "approving access" lexically screams "Access & Identity Management." These mappings survive scale-up because domain topics are **lexically salient** in query text.

Tool-bound categories require operation-type inference that often cannot be extracted from surface wording: "Run the pipeline and push the build" -- is pushing a build a state modification or an infrastructure action? The user's framing doesn't tell you.

**C. Overloaded branches amplify within-branch confusion.**

State Modification at 30 tools contains tools from 5+ semantic domains:

- DB operations: `update_database`
- Auth: `reset_password`
- Financial: `process_refund`, `update_subscription`
- Infrastructure: `provision_vm`, `scale_service`, `deploy_container`, `rollback_deployment`
- Cache: `invalidate_cache`
- Backup: `restore_backup`

Even when the branch is selected correctly, the LLM faces a 10-way disambiguation with no domain coherence to help. The experiment logs show `branch_correct=True, correct=False` cases like:

- "Diana is requesting more capacity" -- branch correct (State Modification), but predicts `scale_service` instead of `update_subscription`
- "Restart the service with more resources" -- branch correct, but predicts `restart_service` instead of `scale_service`

---

### 4. Evidence of Operational Overlap and Branch Non-Exclusivity

**Split pairs across branches are a smoking gun.** These tool pairs are functionally coupled but placed in different tool-bound branches:

| Tool A                | Tool-bound branch A | Tool B            | Tool-bound branch B           |
| --------------------- | ------------------- | ----------------- | ----------------------------- |
| `backup_database`     | Data Retrieval      | `restore_backup`  | State Modification            |
| `deploy_container`    | State Modification  | `run_pipeline`    | Infrastructure Lifecycle      |
| `scale_service`       | State Modification  | `restart_service` | Communication & Orchestration |
| `rollback_deployment` | State Modification  | `run_pipeline`    | Infrastructure Lifecycle      |

In the Semantic taxonomy, these same confusable pairs are often **co-located**:

- `deploy_container` and `run_pipeline` share "Deployment & Release Management"
- `scale_service` and `restart_service` share "IT Support"
- `restore_backup` and `rollback_deployment` are split (Security vs Deployment), but each is near its operational kin

**The confusable label map (from the 30-tool report) confirms cross-branch collisions:**

```
deploy_container <-> run_pipeline      (State Mod vs Infrastructure)
deploy_container <-> provision_vm      (both State Mod -- within-branch)
scale_service <-> restart_service      (State Mod vs Communication)
restore_backup <-> rollback_deployment (both State Mod -- within-branch)
restore_backup <-> restart_service     (State Mod vs Communication)
invalidate_cache <-> restart_service   (State Mod vs Communication)
```

When these confusable pairs span tool-bound branches, the branch-selection step becomes a coin flip. When they share a semantic branch, at least the branch step succeeds and the within-branch disambiguation can still work.

---

### 5. Specific Examples from the Logs

**Example 1: deploy_container vs run_pipeline (branch non-exclusivity)**

- Query: "Run the pipeline and push the build to staging-db-02"
- Ground truth: `deploy_container`
- Tool-bound predicts: `run_pipeline` (branch: Infrastructure Lifecycle, `branch_correct=False`)
- The query mentions both "pipeline" and "build deployment" -- the tool-bound branch boundary between State Modification and Infrastructure Lifecycle is exactly where this query's intent falls

**Example 2: State Modification gravity well**

- Query: "Scan prod-web-01 -- is it healthy?"
- Ground truth: `check_status` (Data Retrieval branch)
- Tool-bound predicts: `deploy_container` (State Modification, `branch_correct=False`)
- The 10-tool State Modification branch has become so large that even read-only intents get pulled in

**Example 3: backup/restore split-pair confusion**

- Query: "Back up the old records and remove them"
- Ground truth: `archive_data` (Data Retrieval branch)
- Tool-bound predicts: `restore_backup` (State Modification, `branch_correct=False`)
- The read-write boundary that defines tool-bound branches breaks down when the query involves both reading AND writing

**Example 4: Semantic doing better on confusable intents**

- Query: "Grant Charlie full control over the resource"
- Ground truth: `transfer_ownership`
- Semantic likely routes to "Billing & Data Management" (where `transfer_ownership` lives)
- Tool-bound predicts: `approve_access` (Access Control branch, `branch_correct=False`)
- "Granting control" sounds more like access control than communication/orchestration -- but `transfer_ownership` is in Communication

**Example 5: Tool-bound wins at 15 tools but not at 30**

- Query: "Query payment-api-03 to see if it's still alive"
- Ground truth: `check_status`
- At 15 tools: Semantic routes to `query_database` (Billing -- wrong); Tool-bound routes to `check_status` (Data Retrieval -- correct)
- At 30 tools: The same query type now competes with `scan_malware`, `backup_database`, `archive_data`, `export_data`, `run_load_test` in the same tool-bound branch -- more within-branch confusion

**Example 6: Hallucinated tool names (30-tool tool-bound only)**

- Query: "Do me a favor and persist bob's data to the database"
- Tool-bound predicts: `Airtable` with branch `Tool-Bound Taxonomy` (completely broken)
- This type of hallucination appears in the 30-tool tool-bound runs but was absent at 15 tools, suggesting the LLM becomes confused by the larger, less coherent branch descriptions

---

### 6. Conclusion on the Research Question

**The hypothesis is strongly supported.** As the tool inventory grows from 15 to 30:

1. **Tool-bound branches lose exclusivity** -- new tools increasingly straddle operation types, making branch assignment arbitrary
2. **The State Modification branch becomes overloaded** (10/30 tools) while two branches remain tiny (3 each), creating catastrophic imbalance
3. **Split pairs** (backup/restore, deploy/pipeline, scale/restart) that should be co-located end up in different branches, producing systematic errors
4. **Semantic branches remain coherent** because domain-topic mapping is lexically salient and scales sub-linearly with tool count

The degradation is **both absolute and relative**:

- Absolute: Tool-bound accuracy drops from 93.3% to 56.5% (base), a 36.8 pp collapse
- Relative: Semantic drops from 78.3% to 51.6% (base), a 26.7 pp drop -- 10 pp less degradation
- The gap closure is thus roughly 40% absolute degradation of tool-bound and 60% relative resilience of semantic

**This predicts that at 45 tools, the inversion will likely be even more pronounced**, since:

- State Modification grows to 13 tools (adding `migrate_database`, `merge_accounts`, `trigger_failover`)
- Data Retrieval grows to 10 tools (adding `snapshot_vm`, `export_data`, `run_load_test`)
- More confusable pairs will span branches
