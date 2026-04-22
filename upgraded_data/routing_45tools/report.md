# Routing Dataset Upgrade Report

## original

- Total rows: 3015
- Unique labels: 45
- Exact duplicate rate: 0.1264
- Near-duplicate rate: 0.0004
- Avg query length: 58.8
- Cue-word fraction: 0.9765
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "all i need" (619)
  - "i need is" (619)
  - "need is for" (619)
  - "is for someone" (619)
  - "for someone to" (619)
  - "nothing else needed" (617)
  - "and we're good" (593)
  - "cdn-edge-06 nothing else" (48)
  - "the feature flag" (45)
  - "i need you" (44)

## lexical_cue_reduced

- Total rows: 3015
- Unique labels: 45
- Exact duplicate rate: 0.5473
- Near-duplicate rate: 0.0021
- Avg query length: 46.2
- Cue-word fraction: 0.3257
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "we need to" (256)
  - "we need a" (48)
  - "needs to be" (47)
  - "the new version" (47)
  - "can we get" (38)
  - "needs the ability" (34)
  - "the ability to" (34)
  - "ability to make" (34)
  - "to make changes" (34)
  - "make changes in" (34)

## confusable_intents

- Total rows: 1350
- Unique labels: 45
- Exact duplicate rate: 0.5407
- Near-duplicate rate: 0.0038
- Avg query length: 45.2
- Cue-word fraction: 0.6911
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "in the system" (46)
  - "update the database" (40)
  - "restart the service" (31)
  - "save a copy" (30)
  - "a copy of" (30)
  - "notify the team" (23)
  - "the service using" (23)
  - "service using the" (23)
  - "as the new" (22)
  - "the new owner" (22)

## paraphrase_heldout_train

- Total rows: 11310
- Unique labels: 45
- Exact duplicate rate: 0.0377
- Near-duplicate rate: 0.0001
- Avg query length: 71.6
- Cue-word fraction: 0.9761
- Class counts: min=205, max=285, mean=251.3
- Top 3-grams:
  - "need is for" (2875)
  - "nothing else needed" (2360)
  - "all i need" (2305)
  - "i need is" (2305)
  - "is for someone" (2305)
  - "for someone to" (2305)
  - "and we're good" (2135)
  - "need you to" (780)
  - "is it possible" (587)
  - "it possible to" (587)

## paraphrase_heldout_test

- Total rows: 3765
- Unique labels: 45
- Exact duplicate rate: 0.0154
- Near-duplicate rate: 0.0003
- Avg query length: 72.2
- Cue-word fraction: 0.9774
- Class counts: min=50, max=130, mean=83.7
- Top 3-grams:
  - "need is for" (982)
  - "and we're good" (830)
  - "all i need" (790)
  - "i need is" (790)
  - "is for someone" (790)
  - "for someone to" (790)
  - "nothing else needed" (725)
  - "need you to" (260)
  - "it would help" (207)
  - "would help if" (207)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

- acknowledge_alert <-> send_notification
- acknowledge_alert <-> log_audit_event
- approve_access <-> reset_password
- approve_access <-> transfer_ownership
- archive_data <-> backup_database
- archive_data <-> update_database
- assign_role <-> approve_access
- assign_role <-> transfer_ownership
- backup_database <-> archive_data
- backup_database <-> query_database
- block_ip_address <-> quarantine_system
- block_ip_address <-> revoke_access
- check_status <-> query_database
- check_status <-> scan_malware
- create_alert_rule <-> send_notification
- create_alert_rule <-> check_status
- create_dns_record <-> deploy_container
- create_dns_record <-> update_database
- create_ticket <-> send_notification
- create_ticket <-> log_audit_event
- deploy_container <-> provision_vm
- deploy_container <-> run_pipeline
- disable_feature_flag <-> rollback_deployment
- disable_feature_flag <-> enable_feature_flag
- enable_feature_flag <-> deploy_container
- enable_feature_flag <-> disable_feature_flag
- escalate_to_human <-> create_ticket
- escalate_to_human <-> send_notification
- export_data <-> generate_report
- export_data <-> query_database
- generate_report <-> query_database
- generate_report <-> log_audit_event
- invalidate_cache <-> update_database
- invalidate_cache <-> restart_service
- log_audit_event <-> create_ticket
- log_audit_event <-> generate_report
- merge_accounts <-> update_database
- merge_accounts <-> transfer_ownership
- migrate_database <-> restore_backup
- migrate_database <-> update_database
- process_refund <-> update_subscription
- process_refund <-> update_database
- provision_vm <-> restart_service
- provision_vm <-> update_database
- quarantine_system <-> scan_malware
- quarantine_system <-> restart_service
- query_database <-> generate_report
- query_database <-> check_status
- remove_role <-> assign_role
- remove_role <-> revoke_access
- renew_certificate <-> rotate_api_key
- renew_certificate <-> schedule_maintenance
- reset_password <-> quarantine_system
- reset_password <-> update_database
- restart_service <-> provision_vm
- restart_service <-> quarantine_system
- restore_backup <-> rollback_deployment
- restore_backup <-> restart_service
- revoke_access <-> quarantine_system
- revoke_access <-> rotate_api_key
- rollback_deployment <-> restore_backup
- rollback_deployment <-> restart_service
- rotate_api_key <-> reset_password
- rotate_api_key <-> revoke_access
- run_load_test <-> check_status
- run_load_test <-> scan_malware
- run_pipeline <-> deploy_container
- run_pipeline <-> restart_service
- scale_service <-> provision_vm
- scale_service <-> restart_service
- scan_malware <-> check_status
- scan_malware <-> quarantine_system
- schedule_maintenance <-> send_notification
- schedule_maintenance <-> create_ticket
- send_notification <-> escalate_to_human
- send_notification <-> create_ticket
- snapshot_vm <-> backup_database
- snapshot_vm <-> archive_data
- tag_resource <-> update_database
- tag_resource <-> create_ticket
- transfer_ownership <-> approve_access
- transfer_ownership <-> update_database
- trigger_failover <-> restart_service
- trigger_failover <-> rollback_deployment
- unblock_ip_address <-> block_ip_address
- unblock_ip_address <-> approve_access
- update_database <-> log_audit_event
- update_database <-> reset_password
- update_subscription <-> update_database
- update_subscription <-> process_refund
