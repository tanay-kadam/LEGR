# Qualitative Case Studies: Why LEGR Outperforms Text-Only Baselines

Cases where LEGR retrieves the correct execution DAG but S-BERT or BM25 fail.

## Case 1
**Query:** Produce the Marketing performance summary, followed by ping Bob about it.

**Ground-truth DAG:** generate_report -> send_notification

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). BM25 retrieves a keyword-similar but wrong DAG.

## Case 2
**Query:** Frank is asking us to generate the Engineering analytics. Next, blast an alert to Legal.

**Ground-truth DAG:** generate_report -> send_notification

**S-BERT top-1 (wrong):** escalate_to_human -> send_notification, send_notification -> generate_report

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 3
**Query:** We gotta yank payment-api-03 off the network — then record the event in Legal's log.

**Ground-truth DAG:** quarantine_system -> log_audit_event

**S-BERT top-1 (wrong):** log_audit_event -> create_ticket

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 4
**Query:** When you get a chance, loop in a Finance manager — then restart the service on staging-db-02, and lastly deploy a server for Finance.

**Ground-truth DAG:** escalate_to_human -> restart_service, restart_service -> provision_vm

**S-BERT top-1 (wrong):** db_read -> reset_password, db_read -> restart_service

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 5
**Query:** From Engineering: store the Legal config in the database, followed by reset Alice's password. Once that's done, process the payout for Diana. Last step: fire off a notification to Frank.

**Ground-truth DAG:** db_write -> reset_password, process_refund -> send_notification, reset_password -> process_refund

**S-BERT top-1 (wrong):** check_status -> db_write, db_write -> reset_password, escalate_to_human -> check_status

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 6
**Query:** ASAP — verify auth-svc-04's health, and then notify the HR team.

**Ground-truth DAG:** check_status -> send_notification

**S-BERT top-1 (wrong):** check_status -> reset_password, escalate_to_human -> reset_password

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 7
**Query:** From Legal: whip up a report for Legal. Once that's done, sweep payment-api-03 for IOCs. End with credit Alice's account.

**Ground-truth DAG:** generate_report -> scan_malware, scan_malware -> process_refund

**S-BERT top-1 (wrong):** generate_report -> update_subscription, process_refund -> update_subscription

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 8
**Query:** Time to give ml-infer-05 a kick and open a ticket for Frank, merge everything and escalate Diana's case to the on-call.

**Ground-truth DAG:** create_ticket -> escalate_to_human, restart_service -> escalate_to_human

**S-BERT top-1 (wrong):** process_refund -> escalate_to_human

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 9
**Query:** I need you to check the DB for Bob. Next, put together a report on ml-infer-05. After that, push the changes for Frank to the DB. Last step: do a security check on payment-api-03.

**Ground-truth DAG:** db_read -> generate_report, db_write -> scan_malware, generate_report -> db_write

**S-BERT top-1 (wrong):** db_read -> process_refund, update_subscription -> db_read

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.

## Case 10
**Query:** Per Eve's request, compile a summary for Engineering, and then notify the Legal team.

**Ground-truth DAG:** generate_report -> send_notification

**S-BERT top-1 (wrong):** check_status -> provision_vm, escalate_to_human -> check_status, provision_vm -> create_ticket

**BM25 top-1 (wrong):** log_audit_event -> update_subscription, process_refund -> update_subscription

**Why LEGR succeeds:** LEGR retrieves the correct DAG (structure + tools). S-BERT retrieves a structurally/semantically wrong DAG. BM25 retrieves a keyword-similar but wrong DAG.
