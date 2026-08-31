# LEGR Failure Cases vs. Generative Baselines

Every query on the held-out split where LEGR's top-1 retrieved DAG is not the ground-truth DAG, paired with what each generative baseline produced for the identical query and the same system prompt.

## Case 1 — row 6 (wide_fanout)

**Query:** Stand up the machine and kick off DB write, notification, and audit logging all at once.

**Ground truth:** `provision_vm -> db_write, provision_vm -> log_audit_event, provision_vm -> send_notification`

**LEGR top-1 (wrong):** `backup_database -> log_audit_event, schedule_maintenance -> backup_database, schedule_maintenance -> send_notification, send_notification -> log_audit_event`  
Ground truth recovered at rank 2 · GED 5.0

**llama3.2:** `provision_vm -> send_notification, db_write -> send_notification, send_notification -> log_audit_event, log_audit_event -> db_write`  
Tool-Set F1 1.0 · GED undefined (cyclic) · 70.47s · **cyclic — not executable**

**gpt-oss:120b-cloud:** `provision_vm -> db_write, provision_vm -> send_notification, provision_vm -> log_audit_event`  
Tool-Set F1 1.0 · GED 0.0 · 3.009s · **exact match**

---

## Case 2 — row 22 (diamond)

**Query:** Check the outage, restart the app and ping a human, then log it all in a ticket.

**Ground truth:** `check_status -> escalate_to_human, check_status -> restart_service, escalate_to_human -> create_ticket, restart_service -> create_ticket`

**LEGR top-1 (wrong):** `restart_service -> log_audit_event, restart_service -> quarantine_system, restart_service -> send_notification`  
Ground truth recovered at rank 4 · GED 7.0

**llama3.2:** `check_status -> restart_service, restart_service -> escalate_to_human, escalate_to_human -> create_ticket`  
Tool-Set F1 1.0 · GED 3.0 · 1.121s

**gpt-oss:120b-cloud:** `check_status -> restart_service, restart_service -> send_notification, send_notification -> create_ticket`  
Tool-Set F1 0.75 · GED 4.0 · 3.524s

---

## Case 3 — row 66 (hourglass)

**Query:** Shelve Finance's outdated data and pass cdn-edge-06 to Alice, then converge and persist Diana's data to the database while also whip up a report for Engineering and set up downtime for ml-infer-05.

**Ground truth:** `archive_data -> db_write, db_write -> generate_report, db_write -> schedule_maintenance, transfer_ownership -> db_write`

**LEGR top-1 (wrong):** `archive_data -> schedule_maintenance, run_pipeline -> schedule_maintenance, schedule_maintenance -> db_write, schedule_maintenance -> generate_report`  
Ground truth recovered at rank 2 · GED 3.0

**llama3.2:** `disable_feature_flag -> archive_data, rollback_deployment -> schedule_maintenance, enable_feature_flag -> approve_access, transfer_ownership -> deploy_container, approve_access -> scale_service, scale_service -> run_pipeline, run_pipeline -> send_notification, send_notification -> process_refund, process_refund -> check_status, check_status -> escalate_to_human, escalate_to_human -> log_audit_event, log_audit_event -> db_read, transfer_ownership -> disable_feature_flag, disable_feature_flag -> rollback_deployment, rollback_deployment -> scale_service, scale_service -> check_status, check_status -> log_audit_event, log_audit_event -> db_read, db_read -> db_write, approve_access -> disable_feature_flag, disable_feature_flag -> scale_service, scale_service -> check_status, check_status -> log_audit_event, log_audit_event -> db_read, db_read -> db_write, archive_data -> disable_feature_flag, disable_feature_flag -> rollback_deployment, rollback_deployment -> run_pipeline, run_pipeline -> process_refund, process_refund -> escalate_to_human, escalate_to_human -> db_read, transfer_ownership -> schedule_maintenance, schedule_maintenance -> enable_feature_flag, enable_feature_flag -> deploy_container, deploy_container -> scale_service, scale_service -> send_notification, send_notification -> check_status, check_status -> log_audit_event, log_audit_event -> db_write`  
Tool-Set F1 0.3636 · GED undefined (cyclic) · 7.095s · **cyclic — not executable**

**gpt-oss:120b-cloud:** `archive_data -> db_write, transfer_ownership -> db_write, archive_data -> generate_report, transfer_ownership -> generate_report, archive_data -> schedule_maintenance, transfer_ownership -> schedule_maintenance`  
Tool-Set F1 1.0 · GED 6.0 · 5.113s

---

## Case 4 — row 67 (hourglass)

**Query:** We gotta archive Charlie's old records and hand payment-api-03 over to Finance. When all of that's complete, store the Engineering config in the database and at the same time build a compliance doc for Finance and set up downtime for ml-infer-05.

**Ground truth:** `archive_data -> db_write, db_write -> generate_report, db_write -> schedule_maintenance, transfer_ownership -> db_write`

**LEGR top-1 (wrong):** `archive_data -> db_read, create_ticket -> archive_data, create_ticket -> schedule_maintenance, schedule_maintenance -> db_read`  
Ground truth recovered at rank 3 · GED 8.0

**llama3.2:** `archive_data -> disable_feature_flag, disable_feature_flag -> invalidate_cache, invalidate_cache -> deploy_container, deploy_container -> run_pipeline, run_pipeline -> send_notification, send_notification -> generate_report, generate_report -> process_refund, process_refund -> update_subscription, update_subscription -> provision_vm, provision_vm -> restart_service, restart_service -> check_status, check_status -> log_audit_event, log_audit_event -> run_pipeline, run_pipeline -> approve_access, approve_access -> revoke_access, revoke_access -> schedule_maintenance`  
Tool-Set F1 0.3636 · GED 31.0 · 14.09s · structurally invalid (disconnected)

**gpt-oss:120b-cloud:** `archive_data -> db_write, archive_data -> generate_report, archive_data -> schedule_maintenance, transfer_ownership -> db_write, transfer_ownership -> generate_report, transfer_ownership -> schedule_maintenance`  
Tool-Set F1 1.0 · GED 6.0 · 4.957s

---

## Case 5 — row 138 (inverted_y)

**Query:** Hey, can you switch off the flag for HR, followed by notify the Marketing team and concurrently generate a new API key for ml-infer-05 and store the Engineering config in the database.

**Ground truth:** `disable_feature_flag -> send_notification, send_notification -> db_write, send_notification -> rotate_api_key`

**LEGR top-1 (wrong):** `disable_feature_flag -> send_notification, rotate_api_key -> send_notification, send_notification -> db_write, send_notification -> escalate_to_human`  
Ground truth recovered at rank 2 · GED 3.0

**llama3.2:** `rotate_api_key -> disable_feature_flag, disable_feature_flag -> send_notification, disable_feature_flag -> rotate_api_key, rotate_api_key -> db_write, db_write -> db_read`  
Tool-Set F1 0.8889 · GED undefined (cyclic) · 3.436s · **cyclic — not executable**

**gpt-oss:120b-cloud:** `disable_feature_flag -> send_notification, send_notification -> rotate_api_key, send_notification -> db_write`  
Tool-Set F1 1.0 · GED 0.0 · 4.109s · **exact match**

---

## Case 6 — row 162 (diamond)

**Query:** The HR team needs us to change Eve's plan, and while you're at it, scan staging-db-02 for threats and hand Alice's issue off to an engineer. Once both are done, schedule maintenance for cdn-edge-06.

**Ground truth:** `escalate_to_human -> schedule_maintenance, scan_malware -> schedule_maintenance, update_subscription -> escalate_to_human, update_subscription -> scan_malware`

**LEGR top-1 (wrong):** `schedule_maintenance -> schedule_maintenance, send_notification -> schedule_maintenance, update_subscription -> schedule_maintenance, update_subscription -> send_notification`  
Ground truth recovered at rank 2 · GED 2.0

**llama3.2:** `process_refund -> scan_malware, schedule_maintenance -> db_write, scan_malware -> escalate_to_human, escalate_to_human -> process_refund, process_refund -> schedule_maintenance`  
Tool-Set F1 0.6667 · GED undefined (cyclic) · 1.537s · **cyclic — not executable**

**gpt-oss:120b-cloud:** `update_subscription -> schedule_maintenance, scan_malware -> schedule_maintenance, escalate_to_human -> schedule_maintenance`  
Tool-Set F1 1.0 · GED 3.0 · 5.141s

---

## Case 7 — row 301 (inverted_y)

**Query:** Here's the execution plan: shut down the feature on prod-web-01. Next, notify Eve. Wrap the sequence by doing issue a new secret for payment-api-03 and update Alice's record in the DB at the same time.

**Ground truth:** `disable_feature_flag -> send_notification, send_notification -> db_write, send_notification -> rotate_api_key`

**LEGR top-1 (wrong):** `disable_feature_flag -> send_notification, rotate_api_key -> send_notification, send_notification -> db_write, send_notification -> escalate_to_human`  
Ground truth recovered at rank 2 · GED 3.0

**llama3.2:** `send_notification -> db_write`  
Tool-Set F1 0.6667 · GED 4.0 · 4.093s

**gpt-oss:120b-cloud:** `disable_feature_flag -> send_notification`  
Tool-Set F1 1.0 · GED 2.0 · 12.695s · structurally invalid (disconnected)

---

## Case 8 — row 304 (inverted_y)

**Query:** Here's the execution plan: disable the release flag for Engineering. Next, shoot Eve an email. Wrap the sequence by doing refresh the API token for Legal and push the changes for Frank to the DB at the same time.

**Ground truth:** `disable_feature_flag -> send_notification, send_notification -> db_write, send_notification -> rotate_api_key`

**LEGR top-1 (wrong):** `disable_feature_flag -> send_notification, rotate_api_key -> send_notification, send_notification -> db_write, send_notification -> escalate_to_human`  
Ground truth recovered at rank 2 · GED 3.0

**llama3.2:** `rotate_api_key -> disable_feature_flag, disable_feature_flag -> send_notification, send_notification -> rotate_api_key, rotate_api_key -> provision_vm`  
Tool-Set F1 0.75 · GED undefined (cyclic) · 1.278s · **cyclic — not executable**

**gpt-oss:120b-cloud:** `disable_feature_flag -> send_notification, send_notification -> rotate_api_key, send_notification -> db_write`  
Tool-Set F1 1.0 · GED 0.0 · 5.82s · **exact match**

---

## Case 9 — row 425 (hourglass)

**Query:** Here's the execution plan: start by doing execute the build pipeline for payment-api-03 and refresh the API token for Marketing in parallel. After that branch settles, create a support case for Finance. End the flow with enable the rollout flag for Engineering and disable the release flag for HR together.

**Ground truth:** `create_ticket -> disable_feature_flag, create_ticket -> enable_feature_flag, rotate_api_key -> create_ticket, run_pipeline -> create_ticket`

**LEGR top-1 (wrong):** `create_ticket -> disable_feature_flag, create_ticket -> revoke_access, enable_feature_flag -> create_ticket, run_pipeline -> create_ticket`  
Ground truth recovered at rank 2 · GED 2.0

**llama3.2:** `run_pipeline -> rotate_api_key, run_pipeline -> enable_feature_flag, rotate_api_key -> disable_feature_flag, disable_feature_flag -> create_ticket, create_ticket -> escalate_to_human, escalate_to_human -> log_audit_event, log_audit_event -> provision_vm, provision_vm -> scale_service, scale_service -> send_notification`  
Tool-Set F1 0.6667 · GED 16.0 · 2.57s

**gpt-oss:120b-cloud:** `run_pipeline -> create_ticket, rotate_api_key -> create_ticket, create_ticket -> enable_feature_flag, create_ticket -> disable_feature_flag`  
Tool-Set F1 1.0 · GED 0.0 · 3.937s · **exact match**

---

## Case 10 — row 432 (hourglass)

**Query:** Use this sequence: open by handling start the CI job for cdn-edge-06 and swap out the API secret together. With those done, open a ticket for Alice. Finish with turn the feature flag back on for Engineering and disable the release flag for Marketing in parallel.

**Ground truth:** `create_ticket -> disable_feature_flag, create_ticket -> enable_feature_flag, rotate_api_key -> create_ticket, run_pipeline -> create_ticket`

**LEGR top-1 (wrong):** `create_ticket -> disable_feature_flag, create_ticket -> revoke_access, enable_feature_flag -> create_ticket, run_pipeline -> create_ticket`  
Ground truth recovered at rank 2 · GED 2.0

**llama3.2:** `db_write -> db_read, db_read -> db_read, db_write -> db_read, db_read -> db_read, db_write -> db_write, create_ticket -> db_read, send_notification -> db_write, send_notification -> db_write, send_notification -> db_read, send_notification -> db_write, send_notification -> create_ticket, send_notification -> send_notification, enable_feature_flag -> send_notification, enable_feature_flag -> send_notification, disable_feature_flag -> send_notification, disable_feature_flag -> db_read`  
Tool-Set F1 0.5455 · GED 30.0 · 4.019s

**gpt-oss:120b-cloud:** `run_pipeline -> create_ticket, rotate_api_key -> create_ticket, create_ticket -> enable_feature_flag, create_ticket -> disable_feature_flag`  
Tool-Set F1 1.0 · GED 0.0 · 4.468s · **exact match**

---

## Case 11 — row 433 (hourglass)

**Query:** Run this flow: open by handling execute the build pipeline for cdn-edge-06 and swap out the API secret together. With those done, open a ticket for Frank. Finish with enable the rollout flag for HR and shut down the feature on ml-infer-05 in parallel.

**Ground truth:** `create_ticket -> disable_feature_flag, create_ticket -> enable_feature_flag, rotate_api_key -> create_ticket, run_pipeline -> create_ticket`

**LEGR top-1 (wrong):** `create_ticket -> disable_feature_flag, create_ticket -> revoke_access, enable_feature_flag -> create_ticket, run_pipeline -> create_ticket`  
Ground truth recovered at rank 2 · GED 2.0

**llama3.2:** `deploy_container -> send_notification, run_pipeline -> run_pipeline, disable_feature_flag -> create_ticket, send_notification -> invalidate_cache, enable_feature_flag -> send_notification, send_notification -> enable_feature_flag, send_notification -> send_notification`  
Tool-Set F1 0.6667 · GED 15.0 · 3.809s · structurally invalid (disconnected)

**gpt-oss:120b-cloud:** `run_pipeline -> create_ticket, rotate_api_key -> create_ticket, create_ticket -> enable_feature_flag, create_ticket -> disable_feature_flag`  
Tool-Set F1 1.0 · GED 0.0 · 4.786s · **exact match**

---

## Case 12 — row 434 (hourglass)

**Query:** Operationally, begin with execute the build pipeline for cdn-edge-06 and refresh the API token for Finance at the same time. After that branch settles, open a ticket for Diana. Wrap the sequence by doing reactivate the feature on auth-svc-04 and disable the release flag for Legal at the same time.

**Ground truth:** `create_ticket -> disable_feature_flag, create_ticket -> enable_feature_flag, rotate_api_key -> create_ticket, run_pipeline -> create_ticket`

**LEGR top-1 (wrong):** `create_ticket -> disable_feature_flag, create_ticket -> revoke_access, enable_feature_flag -> create_ticket, run_pipeline -> create_ticket`  
Ground truth recovered at rank 2 · GED 2.0

**llama3.2:** parse failure — no usable JSON plan · 135.183s

**gpt-oss:120b-cloud:** `run_pipeline -> create_ticket, rotate_api_key -> create_ticket, create_ticket -> enable_feature_flag, create_ticket -> disable_feature_flag`  
Tool-Set F1 1.0 · GED 0.0 · 5.66s · **exact match**

---

## Case 13 — row 1121 (hourglass)

**Query:** Here's the execution plan: begin with investigate auth-svc-04 for infections and trigger a build for Finance at the same time. With those done, plan maintenance on staging-db-02. Wrap the sequence by doing enable the rollout flag for Marketing and refund Frank at the same time.

**Ground truth:** `run_pipeline -> schedule_maintenance, scan_malware -> schedule_maintenance, schedule_maintenance -> enable_feature_flag, schedule_maintenance -> process_refund`

**LEGR top-1 (wrong):** `process_refund -> schedule_maintenance, run_pipeline -> schedule_maintenance, schedule_maintenance -> disable_feature_flag, schedule_maintenance -> update_subscription`  
Ground truth recovered at rank 2 · GED 3.0

**llama3.2:** parse failure — no usable JSON plan · 2.324s

**gpt-oss:120b-cloud:** `scan_malware -> schedule_maintenance, run_pipeline -> schedule_maintenance, schedule_maintenance -> enable_feature_flag, schedule_maintenance -> process_refund`  
Tool-Set F1 1.0 · GED 0.0 · 5.005s · **exact match**

---

**llama3.2 on these 13 LEGR-failure queries:** 0 exact match, 5 cyclic, 2 parse failure.

**gpt-oss:120b-cloud on these 13 LEGR-failure queries:** 8 exact match, 0 cyclic, 0 parse failure.
