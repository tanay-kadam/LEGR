# Campaign v4 Dataset Test Report

**Date:** 2026-08-31T21:00:27.052990

**Result:** 34/34 tests passed

| Test | Status | Detail |
|------|--------|--------|
| T01_registry_invariants | PASS |  |
| T02_json_15 | PASS |  |
| T02_json_30 | PASS |  |
| T02_json_45 | PASS |  |
| T03_csvs_exist_15 | PASS | found={'train.csv', 'test_indomain.csv', 'dev.csv', 'candidate_corpus.csv', 'test_topology_heldout.csv'} |
| T03_csvs_exist_30 | PASS | found={'train.csv', 'test_indomain.csv', 'dev.csv', 'candidate_corpus.csv', 'test_topology_heldout.csv'} |
| T03_csvs_exist_45 | PASS | found={'train.csv', 'test_indomain.csv', 'dev.csv', 'candidate_corpus.csv', 'test_topology_heldout.csv'} |
| T04_schema_compat_15 | PASS | found=['canonical_dag_hash', 'canonical_toolset_hash', 'dag_id', 'dag_text', 'dataset_version', 'edges', 'had_duplicate_node_labels', 'heldout_topology']... |
| T04_schema_compat_30 | PASS | found=['canonical_dag_hash', 'canonical_toolset_hash', 'dag_id', 'dag_text', 'dataset_version', 'edges', 'had_duplicate_node_labels', 'heldout_topology']... |
| T04_schema_compat_45 | PASS | found=['canonical_dag_hash', 'canonical_toolset_hash', 'dag_id', 'dag_text', 'dataset_version', 'edges', 'had_duplicate_node_labels', 'heldout_topology']... |
| T05_tools_within_vocab_15 | PASS | all in vocab |
| T05_tools_within_vocab_30 | PASS | all in vocab |
| T05_tools_within_vocab_45 | PASS | all in vocab |
| T06_no_heldout_leak_15_dev.csv | PASS |  |
| T06_no_heldout_leak_15_train.csv | PASS |  |
| T06_no_heldout_leak_30_dev.csv | PASS |  |
| T06_no_heldout_leak_30_train.csv | PASS |  |
| T06_no_heldout_leak_45_dev.csv | PASS |  |
| T06_no_heldout_leak_45_train.csv | PASS |  |
| T07_heldout_only_heldout_15 | PASS | found={'diamond', 'asymmetric_fork_join'} |
| T07_heldout_only_heldout_30 | PASS | found={'diamond', 'asymmetric_fork_join'} |
| T07_heldout_only_heldout_45 | PASS | found={'diamond', 'asymmetric_fork_join'} |
| T08_no_dag_leakage_15 | PASS |  |
| T08_no_dag_leakage_30 | PASS |  |
| T08_no_dag_leakage_45 | PASS |  |
| T09_twin_density_15 | PASS | 100.0% (target >= 80%) |
| T09_twin_density_30 | PASS | 100.0% (target >= 80%) |
| T09_twin_density_45 | PASS | 100.0% (target >= 80%) |
| T10_acyclicity | PASS | 0/2880 cyclic DAGs found |
| T11_backward_compat_load | PASS | 0 load errors |
| T12_category_balance_15 | PASS | got={'DATA_RETRIEVAL': 6, 'STATE_MODIFICATION': 6, 'ORCHESTRATION': 3} |
| T12_category_balance_30 | PASS | got={'DATA_RETRIEVAL': 12, 'STATE_MODIFICATION': 12, 'ORCHESTRATION': 6} |
| T12_category_balance_45 | PASS | got={'DATA_RETRIEVAL': 18, 'STATE_MODIFICATION': 18, 'ORCHESTRATION': 9} |
| T13_manifest | PASS |  |

All tests passed. Safe to proceed to training.
