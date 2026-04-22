# Routing Dataset Upgrade Report

## original

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.1363
- Near-duplicate rate: 0.0006
- Avg query length: 59.9
- Cue-word fraction: 0.9761
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "and we're good" (410)
  - "nothing else needed" (402)
  - "all i need" (393)
  - "i need is" (393)
  - "need is for" (393)
  - "is for someone" (393)
  - "for someone to" (393)
  - "the reviewer role" (39)
  - "i need you" (32)
  - "need you to" (32)

## lexical_cue_reduced

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.6075
- Near-duplicate rate: 0.0031
- Avg query length: 47.1
- Cue-word fraction: 0.3687
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "we need to" (227)
  - "needs the ability" (34)
  - "the ability to" (34)
  - "ability to make" (34)
  - "to make changes" (34)
  - "make changes in" (34)
  - "someone needs to" (31)
  - "can we get" (29)
  - "need to clean" (26)
  - "to clean up" (26)

## confusable_intents

- Total rows: 900
- Unique labels: 30
- Exact duplicate rate: 0.5856
- Near-duplicate rate: 0.0067
- Avg query length: 45.1
- Cue-word fraction: 0.7078
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "update the database" (60)
  - "save a copy" (22)
  - "a copy of" (22)
  - "to the new" (22)
  - "as the new" (22)
  - "the new owner" (22)
  - "new owner of" (22)
  - "in the system" (21)
  - "the new feature" (21)
  - "restart the service" (21)

## paraphrase_heldout_train

- Total rows: 7540
- Unique labels: 30
- Exact duplicate rate: 0.0411
- Near-duplicate rate: 0.0003
- Avg query length: 72.9
- Cue-word fraction: 0.9788
- Class counts: min=215, max=290, mean=251.3
- Top 3-grams:
  - "need is for" (1860)
  - "and we're good" (1525)
  - "nothing else needed" (1510)
  - "all i need" (1505)
  - "i need is" (1505)
  - "is for someone" (1505)
  - "for someone to" (1505)
  - "need you to" (520)
  - "the situation requires" (403)
  - "would anyone be" (383)

## paraphrase_heldout_test

- Total rows: 2510
- Unique labels: 30
- Exact duplicate rate: 0.0131
- Near-duplicate rate: 0.0004
- Avg query length: 72.8
- Cue-word fraction: 0.9681
- Class counts: min=45, max=120, mean=83.7
- Top 3-grams:
  - "need is for" (607)
  - "and we're good" (525)
  - "nothing else needed" (500)
  - "all i need" (460)
  - "i need is" (460)
  - "is for someone" (460)
  - "for someone to" (460)
  - "need you to" (168)
  - "what we need" (147)
  - "we need is" (147)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

- acknowledge_alert <-> send_notification
- acknowledge_alert <-> log_audit_event
- archive_data <-> update_database
- archive_data <-> backup_database
- assign_role <-> approve_access
- assign_role <-> transfer_ownership
- block_ip_address <-> revoke_access
- block_ip_address <-> quarantine_system
- check_status <-> query_database
- check_status <-> scan_malware
- create_alert_rule <-> check_status
- create_alert_rule <-> send_notification
- create_dns_record <-> update_database
- create_dns_record <-> deploy_container
- enable_feature_flag <-> deploy_container
- enable_feature_flag <-> disable_feature_flag
- escalate_to_human <-> send_notification
- escalate_to_human <-> create_ticket
- export_data <-> query_database
- export_data <-> generate_report
- generate_report <-> query_database
- generate_report <-> log_audit_event
- invalidate_cache <-> update_database
- invalidate_cache <-> restart_service
- merge_accounts <-> update_database
- merge_accounts <-> transfer_ownership
- migrate_database <-> update_database
- migrate_database <-> restore_backup
- process_refund <-> update_subscription
- process_refund <-> update_database
- provision_vm <-> update_database
- provision_vm <-> restart_service
- quarantine_system <-> scan_malware
- quarantine_system <-> restart_service
- remove_role <-> assign_role
- remove_role <-> revoke_access
- renew_certificate <-> rotate_api_key
- renew_certificate <-> schedule_maintenance
- reset_password <-> quarantine_system
- reset_password <-> update_database
- rollback_deployment <-> restore_backup
- rollback_deployment <-> restart_service
- run_load_test <-> check_status
- run_load_test <-> scan_malware
- run_pipeline <-> restart_service
- run_pipeline <-> deploy_container
- scale_service <-> provision_vm
- scale_service <-> restart_service
- schedule_maintenance <-> send_notification
- schedule_maintenance <-> create_ticket
- snapshot_vm <-> backup_database
- snapshot_vm <-> archive_data
- tag_resource <-> create_ticket
- tag_resource <-> update_database
- transfer_ownership <-> update_database
- transfer_ownership <-> approve_access
- trigger_failover <-> restart_service
- trigger_failover <-> rollback_deployment
- unblock_ip_address <-> approve_access
- unblock_ip_address <-> block_ip_address
