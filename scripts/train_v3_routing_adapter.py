"""Train a routing-specific adapter on top of a byte-identical frozen V3 model.

The training corpus is generated from independent, hand-authored intent templates
defined in this file. Every exact normalized query is checked against all four
routing evaluation datasets. Only adapter weights are optimized and saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Batch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepaper_common import (
    ROOT, checkpoint_manifest, create_output_dir, environment_snapshot,
    repo_relative, sha256_file, validate_checkpoint_metadata, write_json,
)

sys.path.insert(0, str(ROOT / "src"))


ROUTING_FILES = {
    "Standard": ROOT / "upgraded_data/routing_15tools/base_cleaned.csv",
    "Lexical": ROOT / "upgraded_data/routing_15tools/lexical_cue_reduced.csv",
    "Confusable": ROOT / "upgraded_data/routing_15tools/confusable_intents.csv",
    "Paraphrase": ROOT / "upgraded_data/routing_15tools/paraphrase_heldout_test.csv",
}
EXPECTED_ROWS = {"Standard": 1005, "Lexical": 1005, "Confusable": 450, "Paraphrase": 1255}

TRAIN_WRAPPERS = (
    "{intent}", "Please handle this: {intent}", "Can someone take care of this: {intent}",
    "This needs attention: {intent}", "The requested operation is: {intent}",
    "For the current case, {intent}", "Could the system do the following: {intent}",
    "The user needs us to {intent}", "Operational request: {intent}",
    "Complete this task safely: {intent}",
)
DEV_WRAPPERS = (
    "New request: {intent}", "Please act on this instruction: {intent}",
    "What must happen is: {intent}", "Handle the following user intent: {intent}",
    "Required next action: {intent}",
)

# Keys are LEGR 15-tool names (db_read/db_write), not routing aliases.
TRAIN_INTENTS = {
    "db_read": (
        "obtain the current account information for {user}",
        "return what is presently stored under record {record}",
        "inspect transaction {order} without changing its data",
        "show the saved profile details associated with {user}",
        "access the existing entry for reference {record}",
        "find the information already on file for {user}",
    ),
    "db_write": (
        "replace the stored email address for {user}",
        "correct the existing database record {record}",
        "persist the revised profile fields for {user}",
        "change what is saved under reference {record}",
        "write the approved value into {user}'s record",
        "apply new account details rather than merely reading them",
    ),
    "reset_password": (
        "issue fresh login credentials to {user}",
        "restore account access by replacing the old secret",
        "invalidate {user}'s current password and create another",
        "help {user} regain access with a new credential",
        "force a credential replacement for account {record}",
        "change the authentication secret, not the profile data",
    ),
    "create_ticket": (
        "open a trackable support case for incident {incident}",
        "create a new issue record about {service}",
        "file a case so the problem has a ticket number",
        "start an incident record for the failure on {service}",
        "log a new support request for {user}",
        "create a case rather than only notifying someone",
    ),
    "send_notification": (
        "deliver an alert to the {team} team",
        "inform {user} about the latest change",
        "send the incident update to the intended recipients",
        "notify the on-call group about {service}",
        "communicate the result to {user} without opening a case",
        "distribute a warning message to {team}",
    ),
    "quarantine_system": (
        "remove {service} from network access immediately",
        "isolate the affected host while investigation continues",
        "contain {service} so it cannot communicate with other systems",
        "place the suspicious machine in an isolated state",
        "disconnect host {host} rather than merely scanning it",
        "prevent the compromised endpoint from reaching the network",
    ),
    "scan_malware": (
        "inspect {host} for malicious software and indicators",
        "perform a security sweep of {service}",
        "look for threats on the endpoint without isolating it",
        "analyze host {host} for signs of infection",
        "run a malware assessment against {service}",
        "check the machine for suspicious files and vulnerabilities",
    ),
    "generate_report": (
        "compile the quarterly figures into a reviewable document",
        "produce an analytics summary for the {team} team",
        "assemble the collected results into a formal report",
        "create a compliance summary from the available records",
        "prepare a performance document about {service}",
        "turn the measurements into a report rather than checking live status",
    ),
    "process_refund": (
        "return the payment for order {order} to the customer",
        "reverse the completed charge associated with {record}",
        "credit {user} for the disputed transaction",
        "send the purchase amount back for order {order}",
        "complete the monetary reversal, not a subscription change",
        "reimburse {user} for transaction {record}",
    ),
    "update_subscription": (
        "move {user} to a different service plan",
        "change the subscription tier on account {record}",
        "modify the recurring plan without editing profile fields",
        "switch {user}'s membership to the annual option",
        "apply the requested subscription upgrade",
        "revise the customer's billing plan rather than refunding a charge",
    ),
    "provision_vm": (
        "bring a new virtual compute instance online for {team}",
        "allocate and initialize another machine for the workload",
        "create a fresh VM to host {service}",
        "supply the {team} team with a new server instance",
        "launch additional compute rather than restarting an existing service",
        "prepare a new virtual host named {host}",
    ),
    "restart_service": (
        "stop and start the running process on {service}",
        "cycle the existing application service",
        "bring {service} back by restarting it",
        "relaunch the process without provisioning another machine",
        "perform a controlled restart on host {host}",
        "bounce the currently deployed service",
    ),
    "check_status": (
        "determine whether {service} is currently healthy",
        "return the present availability state of host {host}",
        "inspect live service health without restarting anything",
        "tell us whether {service} is responding normally",
        "obtain the current operational state, not a historical report",
        "verify whether the deployed endpoint is up or down",
    ),
    "escalate_to_human": (
        "transfer incident {incident} to a human specialist",
        "hand the unresolved case to an on-call engineer",
        "request human intervention for {user}'s problem",
        "move the issue out of automation and to a person",
        "involve a senior responder rather than sending a notification",
        "assign the difficult case to human support",
    ),
    "log_audit_event": (
        "record the completed action in the compliance history",
        "append an immutable audit entry for incident {incident}",
        "preserve evidence of the operation in the audit trail",
        "write a compliance event rather than opening a support ticket",
        "add the security activity to the official system log",
        "create an auditable record of what happened on {service}",
    ),
}

DEV_INTENTS = {
    "db_read": ("show the facts currently retained for {user}", "read back entry {record} without editing it"),
    "db_write": ("save corrected account values for {user}", "alter the persisted entry identified by {record}"),
    "reset_password": ("replace {user}'s sign-in secret", "provide new credentials after access was lost"),
    "create_ticket": ("establish a support case for {incident}", "give the {service} problem a new tracked issue"),
    "send_notification": ("message the {team} recipients about the event", "deliver an informational alert to {user}"),
    "quarantine_system": ("contain host {host} away from the network", "cut off communications from the affected machine"),
    "scan_malware": ("search host {host} for indicators of compromise", "assess {service} for malicious content"),
    "generate_report": ("prepare a summarized document for {team}", "aggregate the measurements into a review artifact"),
    "process_refund": ("reimburse the buyer for {order}", "undo the payment recorded as {record}"),
    "update_subscription": ("revise the recurring membership for {user}", "place account {record} on another plan"),
    "provision_vm": ("initialize a brand-new compute instance for {team}", "make another virtual host available"),
    "restart_service": ("recycle the running {service} process", "stop then relaunch the existing application"),
    "check_status": ("report whether host {host} is presently available", "inspect the live health of {service}"),
    "escalate_to_human": ("send {incident} to a person for resolution", "bring a human responder into the unresolved case"),
    "log_audit_event": ("place the action in the compliance trail", "preserve an official audit entry for {incident}"),
}

SLOTS_TRAIN = {
    "user": ("Zara", "Mateo", "Noor", "Inez", "Kofi"),
    "record": ("REC-731", "REC-884", "REC-926", "REC-447", "REC-518"),
    "order": ("ORD-711", "ORD-842", "ORD-935", "ORD-406", "ORD-529"),
    "service": ("catalog-core", "identity-edge", "ledger-worker", "search-node", "media-api"),
    "host": ("host-k17", "host-m28", "host-q42", "host-r63", "host-v91"),
    "team": ("Platform", "Risk", "Operations", "Reliability", "Customer Care"),
    "incident": ("CASE-671", "CASE-728", "CASE-835", "CASE-914", "CASE-562"),
}
SLOTS_DEV = {
    "user": ("Amina", "Luca"), "record": ("REC-263", "REC-694"),
    "order": ("ORD-257", "ORD-683"), "service": ("billing-core", "auth-gateway"),
    "host": ("host-b36", "host-x74"), "team": ("Governance", "Site Operations"),
    "incident": ("CASE-239", "CASE-486"),
}


class RoutingAdapter(nn.Module):
    """Residual query adapter plus routing-specific residual tool prototypes."""

    def __init__(self, dim: int = 256, bottleneck: int = 128, num_tools: int = 15):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.tool_delta = nn.Parameter(torch.zeros(num_tools, dim))
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def adapted(self, query: torch.Tensor, base_tools: torch.Tensor):
        query_residual = self.up(F.gelu(self.down(query)))
        return F.normalize(query + query_residual, dim=-1), F.normalize(base_tools + self.tool_delta, dim=-1)

    def scores(self, query: torch.Tensor, base_tools: torch.Tensor, temperature: float = 0.05):
        q, tools = self.adapted(query, base_tools)
        return (q @ tools.T) / temperature


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(value).lower())).strip()


def fill_slots(text: str, slots: dict[str, tuple[str, ...]], index: int) -> str:
    result = text
    for offset, (name, values) in enumerate(slots.items()):
        result = result.replace("{" + name + "}", values[(index + offset) % len(values)])
    return result


def build_corpus(tool_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, dev = [], []
    if set(TRAIN_INTENTS) != set(tool_names) or set(DEV_INTENTS) != set(tool_names):
        raise AssertionError("Intent dictionaries must cover exactly the LEGR 15-tool gallery")
    if list(TRAIN_INTENTS) != tool_names:
        # Preserve gallery order for class_id alignment with frozen candidates.
        missing = [name for name in tool_names if name not in TRAIN_INTENTS]
        if missing:
            raise AssertionError(f"Missing training intents for: {missing}")
    for class_id, label in enumerate(tool_names):
        for intent_id, intent in enumerate(TRAIN_INTENTS[label]):
            for wrapper_id, wrapper in enumerate(TRAIN_WRAPPERS):
                index = intent_id * len(TRAIN_WRAPPERS) + wrapper_id
                query = wrapper.format(intent=fill_slots(intent, SLOTS_TRAIN, index))
                train.append({"query": query, "ground_truth": label, "class_id": class_id,
                              "intent_family": intent_id, "wrapper_family": wrapper_id})
        for intent_id, intent in enumerate(DEV_INTENTS[label]):
            for wrapper_id, wrapper in enumerate(DEV_WRAPPERS):
                index = intent_id * len(DEV_WRAPPERS) + wrapper_id
                query = wrapper.format(intent=fill_slots(intent, SLOTS_DEV, index))
                dev.append({"query": query, "ground_truth": label, "class_id": class_id,
                            "intent_family": intent_id, "wrapper_family": wrapper_id})
    return pd.DataFrame(train), pd.DataFrame(dev)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_queries(model, tokenizer, queries: list[str], device: torch.device, batch_size: int):
    model.eval(); result = []
    for start in range(0, len(queries), batch_size):
        tokens = tokenizer(queries[start:start + batch_size], padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        result.append(model.encode_text(tokens["input_ids"].to(device),
                                        tokens["attention_mask"].to(device)).cpu())
    return torch.cat(result)


def metrics_from_scores(scores: torch.Tensor, truth: np.ndarray, tool_names: list[str]):
    order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
    pred = order[:, 0]
    ranks = np.asarray([int(np.where(order[i] == truth[i])[0][0]) + 1 for i in range(len(truth))])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(15), zero_division=0)
    metrics = {
        "n": int(len(truth)), "correct": int((pred == truth).sum()),
        "accuracy": float((pred == truth).mean()), "accuracy_pct": float(100 * (pred == truth).mean()),
        "recall@1": float(np.mean(ranks <= 1)), "recall@3": float(np.mean(ranks <= 3)),
        "recall@5": float(np.mean(ranks <= 5)),
        "mrr@5": float(np.mean(np.where(ranks <= 5, 1 / ranks, 0.0))),
        "macro_f1": float(f1.mean()),
    }
    per_tool = pd.DataFrame({"tool": tool_names, "precision": precision, "recall": recall,
                             "f1": f1, "support": support.astype(int)})
    matrix = confusion_matrix(truth, pred, labels=np.arange(15))
    return metrics, per_tool, matrix, pred, ranks, order


def train_adapter(train_embeddings, train_labels, dev_embeddings, dev_labels,
                  base_tools, seed: int, device: torch.device, epochs: int,
                  batch_size: int, lr: float, patience: int):
    seed_all(seed)
    adapter = RoutingAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    train_embeddings, base_tools = train_embeddings.to(device), base_tools.to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    dev_embeddings = dev_embeddings.to(device)
    dev_labels_t = torch.tensor(dev_labels, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_accuracy, best_epoch, stale, history = None, -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        adapter.train(); order = torch.randperm(len(train_embeddings), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size].to(device)
            logits = adapter.scores(train_embeddings[ids], base_tools)
            classification = F.cross_entropy(logits, train_labels[ids])
            regularizer = 1e-3 * adapter.tool_delta.square().mean()
            loss = classification + regularizer
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        adapter.eval()
        with torch.no_grad():
            dev_pred = adapter.scores(dev_embeddings, base_tools).argmax(dim=1)
            dev_accuracy = float((dev_pred == dev_labels_t).float().mean())
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_embeddings),
                        "dev_accuracy": dev_accuracy})
        if dev_accuracy > best_accuracy:
            best_accuracy, best_epoch, stale = dev_accuracy, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(json.dumps(history[-1]), flush=True)
    adapter.load_state_dict(best_state); adapter.to(device); adapter.eval()
    return adapter, history, best_epoch, best_accuracy


def write_report(path: Path, result: dict) -> None:
    frozen = result["frozen"]
    aggregate = result["aggregate"]
    lines = [
        "# Frozen V3 + supervised routing adapter",
        "",
        "## Experiment",
        "",
        "A residual routing adapter was trained on top of the byte-identical Campaign V4",
        "V3 no-GED 15-tool checkpoint. V3 MiniLM, set-branch, directed GNN, and fusion",
        "weights received no gradients and were never rewritten. Only the adapter",
        "(256→128→256 residual query MLP plus 15 residual tool prototypes) was optimized.",
        "",
        "The training corpus contains 900 independent routing utterances (60 per tool)",
        "and 150 held-out validation utterances (10 per tool). Exact normalized queries",
        "were asserted to have zero overlap with Standard, Lexical, Confusable, and",
        "Paraphrase evaluation files. Candidates remain one-node, zero-edge graphs.",
        "",
        f"Source checkpoint: `{result['source_checkpoint']}`",
        f"Source SHA256 before/after: `{result['source_sha256_before']}` / "
        f"`{result['source_sha256_after']}`",
        f"Original model modified: `{result['original_model_modified']}`",
        "",
        "## Accuracy mean ± std across seeds",
        "",
        "| Stage | Standard | Lexical | Confusable | Paraphrase |",
        "|---|---:|---:|---:|---:|",
        (
            "| Frozen V3 | "
            f"{100 * frozen['Standard']['accuracy']:.2f} | "
            f"{100 * frozen['Lexical']['accuracy']:.2f} | "
            f"{100 * frozen['Confusable']['accuracy']:.2f} | "
            f"{100 * frozen['Paraphrase']['accuracy']:.2f} |"
        ),
        (
            "| Adapter | "
            f"{100 * aggregate['Standard']['accuracy_mean']:.2f} ± "
            f"{100 * aggregate['Standard']['accuracy_std']:.2f} | "
            f"{100 * aggregate['Lexical']['accuracy_mean']:.2f} ± "
            f"{100 * aggregate['Lexical']['accuracy_std']:.2f} | "
            f"{100 * aggregate['Confusable']['accuracy_mean']:.2f} ± "
            f"{100 * aggregate['Confusable']['accuracy_std']:.2f} | "
            f"{100 * aggregate['Paraphrase']['accuracy_mean']:.2f} ± "
            f"{100 * aggregate['Paraphrase']['accuracy_std']:.2f} |"
        ),
        "",
        "## Interpretation",
        "",
        "This measures supervised atomic-routing adaptation of a frozen Campaign-pretrained",
        "encoder, not zero-shot graph transfer. Because every candidate has one node and",
        "zero edges, the directed GNN remains inactive; the adapter learns residual",
        "routing boundaries in the shared 256-d embedding space.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/v3_frozen_routing_adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    output = create_output_dir(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from legr_tool_count import apply_tool_count_override
    apply_tool_count_override(15)
    from atomic_zero_shot import LEGR_15_TOOLS, alias_routing_tool, canonicalise_routing_columns, one_node_candidates
    from data_synth import dag_to_pyg
    from encoders import resolve_graph_encoder_settings
    from eval import _load_model_and_tokenizer

    entry = checkpoint_manifest()[15][2]
    source = Path(entry["checkpoint"])
    source_hash_before = sha256_file(source)
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    source_metadata = validate_checkpoint_metadata(entry, source_payload, 15)
    model, cfg, tokenizer = _load_model_and_tokenizer(str(source), device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    _, _, bidirectional = resolve_graph_encoder_settings(cfg)
    tool_names = list(LEGR_15_TOOLS)
    graph_batch = Batch.from_data_list([dag_to_pyg(g, bidirectional=bidirectional)
                                        for g in one_node_candidates(tool_names)]).to(device)
    with torch.no_grad():
        base_tools = model.encode_graph(graph_batch.x, graph_batch.edge_index, graph_batch.batch,
                                        topo_pos=getattr(graph_batch, "topo_pos", None)).cpu()

    train, dev = build_corpus(tool_names)
    if len(train) != 900 or len(dev) != 150:
        raise AssertionError(f"Unexpected corpus sizes: {len(train)}, {len(dev)}")
    if set(train["ground_truth"]) != set(tool_names) or set(dev["ground_truth"]) != set(tool_names):
        raise AssertionError("Generated corpus labels do not match LEGR gallery")
    evaluation, dataset_manifest, eval_normalized = {}, {}, set()
    for condition, path in ROUTING_FILES.items():
        frame = canonicalise_routing_columns(pd.read_csv(path), str(path))
        if len(frame) != EXPECTED_ROWS[condition]:
            raise AssertionError(f"{condition}: wrong row count")
        frame["ground_truth_tool"] = [alias_routing_tool(v) for v in frame["ground_truth"].astype(str)]
        frame["class_id"] = [tool_names.index(v) for v in frame["ground_truth_tool"]]
        evaluation[condition] = frame
        eval_normalized.update(normalize_text(q) for q in frame["query"])
        dataset_manifest[condition] = {"path": repo_relative(path), "sha256": sha256_file(path), "rows": len(frame)}
    train_normalized = set(normalize_text(q) for q in train["query"])
    dev_normalized = set(normalize_text(q) for q in dev["query"])
    if train_normalized & eval_normalized or dev_normalized & eval_normalized:
        raise AssertionError("Generated routing corpus overlaps evaluation queries")
    if train_normalized & dev_normalized:
        raise AssertionError("Training and validation queries overlap")
    train.to_csv(output / "routing_adapter_train.csv", index=False)
    dev.to_csv(output / "routing_adapter_dev.csv", index=False)

    train_embeddings = encode_queries(model, tokenizer, train["query"].tolist(), device, args.batch_size)
    dev_embeddings = encode_queries(model, tokenizer, dev["query"].tolist(), device, args.batch_size)
    eval_embeddings = {name: encode_queries(model, tokenizer, frame["query"].tolist(), device, args.batch_size)
                       for name, frame in evaluation.items()}

    frozen_results = {}
    for condition, frame in evaluation.items():
        metrics, _, _, _, _, _ = metrics_from_scores(eval_embeddings[condition] @ base_tools.T,
                                                       frame["class_id"].to_numpy(), tool_names)
        frozen_results[condition] = metrics

    all_results, summary_rows = {}, []
    for seed in args.seeds:
        print(json.dumps({"event": "train_seed", "seed": seed}), flush=True)
        adapter, history, best_epoch, best_dev = train_adapter(
            train_embeddings, train["class_id"].to_numpy(), dev_embeddings,
            dev["class_id"].to_numpy(), base_tools, seed, device, args.epochs,
            args.batch_size, args.lr, args.patience)
        seed_dir = output / f"seed_{seed}"; seed_dir.mkdir()
        checkpoint = seed_dir / "routing_adapter.pt"
        torch.save({"adapter_state": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    "source_checkpoint": repo_relative(source), "source_sha256": source_hash_before,
                    "seed": seed, "best_epoch": best_epoch, "best_dev_accuracy": best_dev,
                    "architecture": "frozen V3 + residual query adapter + residual tool prototypes"}, checkpoint)
        write_json(seed_dir / "training_history.json", history)
        seed_results = {}
        for condition, frame in evaluation.items():
            with torch.no_grad():
                scores = adapter.scores(eval_embeddings[condition].to(device), base_tools.to(device)).cpu()
            metrics, per_tool, matrix, pred, ranks, order = metrics_from_scores(
                scores, frame["class_id"].to_numpy(), tool_names)
            seed_results[condition] = metrics
            per_tool.to_csv(seed_dir / f"per_tool_{condition.lower()}.csv", index=False)
            prediction = frame.copy(); prediction["predicted_tool"] = [tool_names[i] for i in pred]
            prediction["correct"] = pred == frame["class_id"].to_numpy(); prediction["rank"] = ranks
            prediction["top5_tools"] = [";".join(tool_names[j] for j in row[:5]) for row in order]
            prediction.to_csv(seed_dir / f"per_query_{condition.lower()}.csv", index=False)
            pd.DataFrame(matrix, index=tool_names, columns=tool_names).to_csv(
                seed_dir / f"confusion_{condition.lower()}.csv")
            summary_rows.append({"seed": seed, "condition": condition, **metrics})
        all_results[str(seed)] = {"best_epoch": best_epoch, "best_dev_accuracy": best_dev,
                                  "adapter_checkpoint": repo_relative(checkpoint),
                                  "adapter_sha256": sha256_file(checkpoint), "conditions": seed_results}

    summary = pd.DataFrame(summary_rows); summary.to_csv(output / "routing_adapter_metrics.csv", index=False)
    aggregate = {}
    for condition in ROUTING_FILES:
        rows = summary[summary["condition"] == condition]
        aggregate[condition] = {
            "accuracy_mean": float(rows["accuracy"].mean()), "accuracy_std": float(rows["accuracy"].std(ddof=1)),
            "macro_f1_mean": float(rows["macro_f1"].mean()), "macro_f1_std": float(rows["macro_f1"].std(ddof=1)),
        }
    source_hash_after = sha256_file(source)
    if source_hash_before != source_hash_after:
        raise AssertionError("Original V3 checkpoint changed during adapter training")
    result = {
        "experiment": "supervised routing adapter over frozen Campaign V4 V3 no-GED",
        "original_model_modified": False, "source_checkpoint": repo_relative(source),
        "source_sha256_before": source_hash_before, "source_sha256_after": source_hash_after,
        "source_metadata": source_metadata,
        "adapter_architecture": "residual 256-128-256 query MLP plus 15x256 residual tool prototypes",
        "training_corpus": {"train_rows": len(train), "dev_rows": len(dev),
                            "train_per_tool": 60, "dev_per_tool": 10,
                            "normalized_exact_overlap_with_evaluation": 0},
        "datasets": dataset_manifest, "frozen": frozen_results, "seeds": all_results,
        "aggregate": aggregate, "arguments": vars(args), "environment": environment_snapshot(device),
    }
    write_json(output / "results.json", result)
    write_report(output / "report.md", result)
    (output / "reproduce.txt").write_text(
        f"{sys.executable} scripts/train_v3_routing_adapter.py --output {repo_relative(output)} "
        f"--device {args.device} --seeds {' '.join(map(str,args.seeds))} --epochs {args.epochs} "
        f"--patience {args.patience} --batch-size {args.batch_size} --lr {args.lr}\n", encoding="utf-8")
    print(json.dumps({"frozen": frozen_results, "aggregate": aggregate,
                      "source_unchanged": source_hash_before == source_hash_after}, indent=2))


if __name__ == "__main__":
    main()
