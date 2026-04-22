# Routing Dataset Upgrade Report

## original

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.1303
- Near-duplicate rate: 0.0007
- Avg query length: 57.0
- Cue-word fraction: 0.9726
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "and we're good" (410)
  - "nothing else needed" (395)
  - "all i need" (388)
  - "i need is" (388)
  - "need is for" (388)
  - "is for someone" (388)
  - "for someone to" (388)
  - "i need you" (32)
  - "need you to" (32)
  - "ml-infer-05 nothing else" (30)

## lexical_cue_reduced

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.4254
- Near-duplicate rate: 0.0025
- Avg query length: 45.9
- Cue-word fraction: 0.3124
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "we need to" (141)
  - "needs to be" (43)
  - "needs the ability" (34)
  - "the ability to" (34)
  - "ability to make" (34)
  - "to make changes" (34)
  - "make changes in" (34)
  - "we need a" (34)
  - "can we get" (34)
  - "shouldn't have access" (32)

## confusable_intents

- Total rows: 900
- Unique labels: 30
- Exact duplicate rate: 0.4656
- Near-duplicate rate: 0.0047
- Avg query length: 45.5
- Cue-word fraction: 0.61
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "in the system" (39)
  - "-- isolate and" (23)
  - "so they can" (21)
  - "they can get" (21)
  - "a record of" (19)
  - "from the system" (17)
  - "run the pipeline" (16)
  - "the pipeline and" (16)
  - "pipeline and push" (16)
  - "and push the" (16)

## paraphrase_heldout_train

- Total rows: 7540
- Unique labels: 30
- Exact duplicate rate: 0.0432
- Near-duplicate rate: 0.0003
- Avg query length: 69.9
- Cue-word fraction: 0.9702
- Class counts: min=215, max=290, mean=251.3
- Top 3-grams:
  - "need is for" (1825)
  - "and we're good" (1540)
  - "nothing else needed" (1480)
  - "all i need" (1470)
  - "i need is" (1470)
  - "is for someone" (1470)
  - "for someone to" (1470)
  - "need you to" (519)
  - "the situation requires" (403)
  - "would anyone be" (383)

## paraphrase_heldout_test

- Total rows: 2510
- Unique labels: 30
- Exact duplicate rate: 0.0183
- Near-duplicate rate: 0.0004
- Avg query length: 70.1
- Cue-word fraction: 0.9801
- Class counts: min=45, max=120, mean=83.7
- Top 3-grams:
  - "need is for" (617)
  - "and we're good" (510)
  - "nothing else needed" (495)
  - "all i need" (470)
  - "i need is" (470)
  - "is for someone" (470)
  - "for someone to" (470)
  - "need you to" (169)
  - "what we need" (147)
  - "we need is" (147)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

- approve_access <-> transfer_ownership
- approve_access <-> reset_password
- assign_role <-> approve_access
- assign_role <-> transfer_ownership
- backup_database <-> query_database
- backup_database <-> archive_data
- check_status <-> query_database
- check_status <-> scan_malware
- create_dns_record <-> update_database
- create_dns_record <-> deploy_container
- create_ticket <-> send_notification
- create_ticket <-> log_audit_event
- deploy_container <-> provision_vm
- deploy_container <-> run_pipeline
- disable_feature_flag <-> rollback_deployment
- disable_feature_flag <-> enable_feature_flag
- escalate_to_human <-> create_ticket
- escalate_to_human <-> send_notification
- generate_report <-> query_database
- generate_report <-> log_audit_event
- log_audit_event <-> create_ticket
- log_audit_event <-> generate_report
- process_refund <-> update_subscription
- process_refund <-> update_database
- provision_vm <-> restart_service
- provision_vm <-> update_database
- quarantine_system <-> restart_service
- quarantine_system <-> scan_malware
- query_database <-> generate_report
- query_database <-> check_status
- remove_role <-> revoke_access
- remove_role <-> assign_role
- renew_certificate <-> rotate_api_key
- renew_certificate <-> schedule_maintenance
- reset_password <-> update_database
- reset_password <-> quarantine_system
- restart_service <-> provision_vm
- restart_service <-> quarantine_system
- restore_backup <-> restart_service
- restore_backup <-> rollback_deployment
- revoke_access <-> quarantine_system
- revoke_access <-> rotate_api_key
- run_load_test <-> check_status
- run_load_test <-> scan_malware
- run_pipeline <-> deploy_container
- run_pipeline <-> restart_service
- scan_malware <-> check_status
- scan_malware <-> quarantine_system
- schedule_maintenance <-> send_notification
- schedule_maintenance <-> create_ticket
- send_notification <-> escalate_to_human
- send_notification <-> create_ticket
- snapshot_vm <-> archive_data
- snapshot_vm <-> backup_database
- trigger_failover <-> restart_service
- trigger_failover <-> rollback_deployment
- update_database <-> log_audit_event
- update_database <-> reset_password
- update_subscription <-> process_refund
- update_subscription <-> update_database
