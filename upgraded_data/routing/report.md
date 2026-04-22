# Routing Dataset Upgrade Report

## original

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.1204
- Near-duplicate rate: 0.0006
- Avg query length: 57.0
- Cue-word fraction: 0.9756
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "nothing else needed" (404)
  - "all i need" (402)
  - "i need is" (402)
  - "need is for" (402)
  - "is for someone" (402)
  - "for someone to" (402)
  - "and we're good" (398)
  - "the feature flag" (45)
  - "the flag for" (40)
  - "i need you" (31)

## lexical_cue_reduced

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.4756
- Near-duplicate rate: 0.0028
- Avg query length: 45.0
- Cue-word fraction: 0.2647
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "we need to" (139)
  - "needs to be" (37)
  - "can we get" (36)
  - "shouldn't have access" (35)
  - "have access to" (35)
  - "someone needs to" (31)
  - "we need a" (27)
  - "the new version" (26)
  - "policy says we" (25)
  - "to get the" (23)

## confusable_intents

- Total rows: 900
- Unique labels: 30
- Exact duplicate rate: 0.4822
- Near-duplicate rate: 0.0056
- Avg query length: 46.8
- Cue-word fraction: 0.6789
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "in the system" (28)
  - "the new feature" (24)
  - "so they can" (23)
  - "they can get" (23)
  - "roll back the" (22)
  - "as the new" (22)
  - "the new owner" (22)
  - "new owner of" (22)
  - "-- isolate and" (19)
  - "from the database" (18)

## paraphrase_heldout_train

- Total rows: 7540
- Unique labels: 30
- Exact duplicate rate: 0.0405
- Near-duplicate rate: 0.0003
- Avg query length: 70.1
- Cue-word fraction: 0.9768
- Class counts: min=215, max=290, mean=251.3
- Top 3-grams:
  - "need is for" (1895)
  - "all i need" (1540)
  - "i need is" (1540)
  - "is for someone" (1540)
  - "for someone to" (1540)
  - "nothing else needed" (1520)
  - "and we're good" (1490)
  - "need you to" (519)
  - "the situation requires" (403)
  - "would anyone be" (383)

## paraphrase_heldout_test

- Total rows: 2510
- Unique labels: 30
- Exact duplicate rate: 0.0116
- Near-duplicate rate: 0.0004
- Avg query length: 69.5
- Cue-word fraction: 0.9721
- Class counts: min=45, max=120, mean=83.7
- Top 3-grams:
  - "need is for" (617)
  - "and we're good" (500)
  - "nothing else needed" (500)
  - "all i need" (470)
  - "i need is" (470)
  - "is for someone" (470)
  - "for someone to" (470)
  - "need you to" (168)
  - "what we need" (147)
  - "we need is" (147)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

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
