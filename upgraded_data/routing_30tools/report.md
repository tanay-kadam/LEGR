# Routing Dataset Upgrade Report

## original

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.0
- Near-duplicate rate: 0.0012
- Avg query length: 59.1
- Cue-word fraction: 0.9831
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "finance for request" (51)
  - "the current state" (51)
  - "the security incident" (45)
  - "service to the" (45)
  - "the customer data" (45)
  - "about the latest" (44)
  - "engineering for request" (43)
  - "reverse the charge" (43)
  - "the charge on" (43)
  - "review the threat" (42)

## lexical_cue_reduced

- Total rows: 2010
- Unique labels: 30
- Exact duplicate rate: 0.5925
- Near-duplicate rate: 0.008
- Avg query length: 56.8
- Cue-word fraction: 0.7368
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "out of the" (92)
  - "needs to be" (79)
  - "there should be" (74)
  - "is waiting for" (71)
  - "tied to order" (61)
  - "we need to" (57)
  - "should be an" (52)
  - "be an official" (52)
  - "the latest release" (50)
  - "cannot proceed until" (49)

## confusable_intents

- Total rows: 900
- Unique labels: 30
- Exact duplicate rate: 0.6967
- Near-duplicate rate: 0.0155
- Avg query length: 72.5
- Cue-word fraction: 0.9178
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "this is not" (238)
  - "is not a" (128)
  - "do not just" (115)
  - "if you want" (56)
  - "you want but" (56)
  - "is not an" (49)
  - "later if you" (47)
  - "if needed but" (47)
  - "the service back" (42)
  - "can wait first" (41)

## paraphrase_heldout_train

- Total rows: 7540
- Unique labels: 30
- Exact duplicate rate: 0.0
- Near-duplicate rate: 0.0002
- Avg query length: 73.1
- Cue-word fraction: 0.9828
- Class counts: min=215, max=290, mean=251.3
- Top 3-grams:
  - "need you to" (499)
  - "the situation requires" (403)
  - "would anyone be" (383)
  - "anyone be able" (383)
  - "be able to" (383)
  - "is it possible" (382)
  - "it possible to" (382)
  - "-- that's what" (376)
  - "that's what needs" (376)
  - "what needs to" (376)

## paraphrase_heldout_test

- Total rows: 2510
- Unique labels: 30
- Exact duplicate rate: 0.0
- Near-duplicate rate: 0.0004
- Avg query length: 71.9
- Cue-word fraction: 0.9841
- Class counts: min=45, max=120, mean=83.7
- Top 3-grams:
  - "need you to" (157)
  - "what we need" (147)
  - "we need is" (147)
  - "need is for" (147)
  - "-- that's what" (146)
  - "that's what needs" (146)
  - "what needs to" (146)
  - "needs to happen" (146)
  - "is it possible" (134)
  - "it possible to" (134)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

- approve_access_request <-> notify_access_change
- approve_access_request <-> reset_user_password
- archive_customer_data <-> snapshot_system_state
- archive_customer_data <-> authorize_data_export
- assign_access_role <-> provision_workspace
- assign_access_role <-> approve_access_request
- authorize_data_export <-> archive_customer_data
- authorize_data_export <-> enable_feature_flag
- check_service_status <-> restart_service
- check_service_status <-> inspect_security_alerts
- create_support_ticket <-> provision_workspace
- create_support_ticket <-> escalate_security_incident
- deploy_service_release <-> provision_workspace
- deploy_service_release <-> rollback_service_release
- enable_feature_flag <-> authorize_data_export
- enable_feature_flag <-> record_release_note
- escalate_security_incident <-> log_compliance_event
- escalate_security_incident <-> create_support_ticket
- generate_access_report <-> update_identity_record
- generate_access_report <-> validate_release_readiness
- inspect_security_alerts <-> quarantine_endpoint
- inspect_security_alerts <-> check_service_status
- log_compliance_event <-> update_identity_record
- log_compliance_event <-> inspect_security_alerts
- notify_access_change <-> send_customer_notification
- notify_access_change <-> update_identity_record
- process_refund <-> update_customer_record
- process_refund <-> rollback_service_release
- provision_workspace <-> deploy_service_release
- provision_workspace <-> create_support_ticket
- quarantine_endpoint <-> inspect_security_alerts
- quarantine_endpoint <-> restart_service
- record_release_note <-> log_compliance_event
- record_release_note <-> validate_release_readiness
- reset_user_password <-> create_support_ticket
- reset_user_password <-> approve_access_request
- restart_service <-> check_service_status
- restart_service <-> quarantine_endpoint
- revoke_system_access <-> approve_access_request
- revoke_system_access <-> quarantine_endpoint
- rollback_service_release <-> deploy_service_release
- rollback_service_release <-> trigger_failover
- run_load_test <-> scale_service_capacity
- run_load_test <-> validate_release_readiness
- scale_service_capacity <-> schedule_maintenance_window
- scale_service_capacity <-> update_customer_record
- schedule_maintenance_window <-> notify_access_change
- schedule_maintenance_window <-> run_load_test
- send_customer_notification <-> notify_access_change
- send_customer_notification <-> archive_customer_data
- snapshot_system_state <-> trigger_failover
- snapshot_system_state <-> record_release_note
- trigger_failover <-> snapshot_system_state
- trigger_failover <-> rollback_service_release
- update_customer_record <-> process_refund
- update_customer_record <-> provision_workspace
- update_identity_record <-> log_compliance_event
- update_identity_record <-> generate_access_report
- validate_release_readiness <-> deploy_service_release
- validate_release_readiness <-> generate_access_report
