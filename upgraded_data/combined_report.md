# Combined Benchmark Strengthening Report

## Routing Benchmark

### Lexical Shortcut Pressure

- Before: 0.9756
- After (lexical_cue_reduced): 0.2647
- Improvement: 0.7109

### Class Balance

- **original**: min=67, max=67, std=0.0
- **lexical_cue_reduced**: min=67, max=67, std=0.0
- **confusable_intents**: min=30, max=30, std=0.0
- **paraphrase_heldout_train**: min=215, max=290, std=17.37
- **paraphrase_heldout_test**: min=45, max=120, std=17.37

### Paraphrase Separation: family overlap = 0

### Confusable Label Coverage: 60 pairs

- approve_access <-> transfer_ownership
- approve_access <-> reset_password
- archive_data <-> backup_database
- archive_data <-> update_database
- backup_database <-> archive_data
- backup_database <-> query_database
- check_status <-> scan_malware
- check_status <-> query_database
- create_ticket <-> send_notification
- create_ticket <-> log_audit_event
- deploy_container <-> run_pipeline
- deploy_container <-> provision_vm
- disable_feature_flag <-> rollback_deployment
- disable_feature_flag <-> enable_feature_flag
- enable_feature_flag <-> deploy_container
- enable_feature_flag <-> disable_feature_flag
- escalate_to_human <-> send_notification
- escalate_to_human <-> create_ticket
- generate_report <-> query_database
- generate_report <-> log_audit_event
- invalidate_cache <-> update_database
- invalidate_cache <-> restart_service
- log_audit_event <-> create_ticket
- log_audit_event <-> generate_report
- process_refund <-> update_subscription
- process_refund <-> update_database
- provision_vm <-> restart_service
- provision_vm <-> update_database
- quarantine_system <-> scan_malware
- quarantine_system <-> restart_service
- query_database <-> check_status
- query_database <-> generate_report
- reset_password <-> update_database
- reset_password <-> quarantine_system
- restart_service <-> provision_vm
- restart_service <-> quarantine_system
- restore_backup <-> restart_service
- restore_backup <-> rollback_deployment
- revoke_access <-> quarantine_system
- revoke_access <-> rotate_api_key
- rollback_deployment <-> restore_backup
- rollback_deployment <-> restart_service
- rotate_api_key <-> revoke_access
- rotate_api_key <-> reset_password
- run_pipeline <-> deploy_container
- run_pipeline <-> restart_service
- scale_service <-> provision_vm
- scale_service <-> restart_service
- scan_malware <-> check_status
- scan_malware <-> quarantine_system
- schedule_maintenance <-> create_ticket
- schedule_maintenance <-> send_notification
- send_notification <-> escalate_to_human
- send_notification <-> create_ticket
- transfer_ownership <-> approve_access
- transfer_ownership <-> update_database
- update_database <-> log_audit_event
- update_database <-> reset_password
- update_subscription <-> process_refund
- update_subscription <-> update_database

## Graph Benchmark

### Topology Diversity

- Before: 12 unique topologies
- After: 132 unique topologies

### Motif Concentration (Shannon Entropy)

- Before: 2.26 (9 families)
- After: 3.303 (20 families)

### Hard Negative Availability

- Total: 312
- add_edge: 52
- extra_node: 78
- remove_edge: 52
- swap_edges: 52
- swap_tools: 78

### Topology-Held-Out Evaluation

- Train families: 14
- Test families: 5
- Overlap: 0
- Is nontrivial: True
- Held-out test-only families: asymmetric_fork_join, diamond, wide_fanin, wide_fanout, wide_fanout_deep
