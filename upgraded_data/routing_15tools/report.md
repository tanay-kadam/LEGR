# Routing Dataset Upgrade Report

## original

- Total rows: 1005
- Unique labels: 15
- Exact duplicate rate: 0.0
- Near-duplicate rate: 0.0
- Avg query length: 44.3
- Cue-word fraction: 0.9264
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "i need you" (120)
  - "need you to" (120)
  - "when you have" (45)
  - "you have a" (45)
  - "have a moment" (45)
  - "to the database" (23)
  - "a new vm" (17)
  - "virtual machine for" (17)
  - "the service on" (17)
  - "the status of" (17)

## lexical_cue_reduced

- Total rows: 1005
- Unique labels: 15
- Exact duplicate rate: 0.197
- Near-duplicate rate: 0.0023
- Avg query length: 43.0
- Cue-word fraction: 0.1751
- Class counts: min=67, max=67, mean=67.0
- Top 3-grams:
  - "we need to" (39)
  - "can we get" (26)
  - "needs to be" (25)
  - "someone needs to" (22)
  - "isn't going to" (20)
  - "we get a" (19)
  - "want to see" (19)
  - "is asking for" (17)
  - "in a while" (15)
  - "need to make" (15)

## confusable_intents

- Total rows: 450
- Unique labels: 15
- Exact duplicate rate: 0.3822
- Near-duplicate rate: 0.0097
- Avg query length: 48.1
- Cue-word fraction: 0.5378
- Class counts: min=30, max=30, mean=30.0
- Top 3-grams:
  - "in the system" (28)
  - "a record of" (22)
  - "-- isolate and" (20)
  - "file a record" (13)
  - "record of what" (13)
  - "of what happened" (13)
  - "what happened with" (13)
  - "access needs to" (13)
  - "needs to be" (13)
  - "to be fixed" (13)

## paraphrase_heldout_train

- Total rows: 3770
- Unique labels: 15
- Exact duplicate rate: 0.0212
- Near-duplicate rate: 0.0001
- Avg query length: 53.4
- Cue-word fraction: 0.9324
- Class counts: min=220, max=285, mean=251.3
- Top 3-grams:
  - "need you to" (338)
  - "-- that's what" (209)
  - "that's what needs" (209)
  - "what needs to" (209)
  - "needs to happen" (209)
  - "what we need" (195)
  - "we need is" (195)
  - "need is for" (195)
  - "would anyone be" (195)
  - "anyone be able" (195)

## paraphrase_heldout_test

- Total rows: 1255
- Unique labels: 15
- Exact duplicate rate: 0.008
- Near-duplicate rate: 0.0002
- Avg query length: 54.4
- Cue-word fraction: 0.9084
- Class counts: min=50, max=115, mean=83.7
- Top 3-grams:
  - "need you to" (117)
  - "is it possible" (73)
  - "it possible to" (73)
  - "it would help" (70)
  - "would help if" (70)
  - "would anyone be" (66)
  - "anyone be able" (66)
  - "be able to" (66)
  - "-- that's what" (64)
  - "that's what needs" (64)

## Paraphrase Family Overlap: 0

## Confusable Label Pairs Covered

- check_status <-> query_database
- check_status <-> scan_malware
- create_ticket <-> send_notification
- create_ticket <-> log_audit_event
- escalate_to_human <-> send_notification
- escalate_to_human <-> create_ticket
- generate_report <-> log_audit_event
- generate_report <-> query_database
- log_audit_event <-> create_ticket
- log_audit_event <-> generate_report
- process_refund <-> update_subscription
- process_refund <-> update_database
- provision_vm <-> update_database
- provision_vm <-> restart_service
- quarantine_system <-> restart_service
- quarantine_system <-> scan_malware
- query_database <-> generate_report
- query_database <-> check_status
- reset_password <-> update_database
- reset_password <-> quarantine_system
- restart_service <-> provision_vm
- restart_service <-> quarantine_system
- scan_malware <-> check_status
- scan_malware <-> quarantine_system
- send_notification <-> create_ticket
- send_notification <-> escalate_to_human
- update_database <-> reset_password
- update_database <-> log_audit_event
- update_subscription <-> update_database
- update_subscription <-> process_refund
