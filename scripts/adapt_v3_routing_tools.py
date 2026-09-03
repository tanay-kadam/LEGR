"""Adapt Campaign V4 V3 to routing tool cards, then evaluate all four datasets.

The adaptation corpus contains only tool names, aliases, and registry descriptions.
It intentionally excludes every query from the Standard, Lexical, Confusable, and
Paraphrase evaluation CSVs. Original checkpoints are loaded read-only and adapted
checkpoints are written beneath a new no-clobber output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Batch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepaper_common import (
    ROOT,
    checkpoint_manifest,
    create_output_dir,
    environment_snapshot,
    repo_relative,
    sha256_file,
    validate_checkpoint_metadata,
    write_json,
)

sys.path.insert(0, str(ROOT / "src"))


ROUTING_FILES = {
    "Standard": ROOT / "upgraded_data/routing_15tools/base_cleaned.csv",
    "Lexical": ROOT / "upgraded_data/routing_15tools/lexical_cue_reduced.csv",
    "Confusable": ROOT / "upgraded_data/routing_15tools/confusable_intents.csv",
    "Paraphrase": ROOT / "upgraded_data/routing_15tools/paraphrase_heldout_test.csv",
}
EXPECTED_ROWS = {"Standard": 1005, "Lexical": 1005, "Confusable": 450, "Paraphrase": 1255}

# Generic tool-card templates, not templates used to construct routing queries.
TRAIN_TEMPLATES = (
    "Tool name: {routing_name}. Purpose: {description}.",
    "Select {routing_name} when the requested operation is to {description_lower}.",
    "API operation {routing_name}: {description}.",
    "The {routing_name} capability is used to {description_lower}.",
    "Intent label {routing_name} corresponds to: {description}.",
    "Route requests about this operation to {routing_name}: {description}.",
)
DEV_TEMPLATES = (
    "Which tool handles the following intent? {description}.",
    "Appropriate API: {routing_name}; supported action: {description_lower}.",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_batch(graphs, device: torch.device, bidirectional: bool) -> Batch:
    from data_synth import dag_to_pyg

    batch = Batch.from_data_list([dag_to_pyg(g, bidirectional=bidirectional) for g in graphs])
    return batch.to(device)


def encode_graph_batch(model, batch: Batch) -> torch.Tensor:
    return model.encode_graph(
        batch.x,
        batch.edge_index,
        batch.batch,
        topo_pos=getattr(batch, "topo_pos", None),
    )


def build_tool_card_corpus(candidate_tools: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from atomic_zero_shot import ROUTING_TO_LEGR_15
    from data_synth import TOOL_DESCRIPTIONS

    legr_to_routing = {value: key for key, value in ROUTING_TO_LEGR_15.items()}
    rows_train, rows_dev = [], []
    for class_id, legr_name in enumerate(candidate_tools):
        routing_name = legr_to_routing.get(legr_name, legr_name)
        description = TOOL_DESCRIPTIONS[legr_name]
        fields = {
            "routing_name": routing_name.replace("_", " "),
            "description": description,
            "description_lower": description[0].lower() + description[1:],
        }
        for template_id, template in enumerate(TRAIN_TEMPLATES):
            rows_train.append({
                "query": template.format(**fields), "class_id": class_id,
                "routing_tool": routing_name, "legr_tool": legr_name,
                "template_id": f"train_{template_id}",
            })
        for template_id, template in enumerate(DEV_TEMPLATES):
            rows_dev.append({
                "query": template.format(**fields), "class_id": class_id,
                "routing_tool": routing_name, "legr_tool": legr_name,
                "template_id": f"dev_{template_id}",
            })
    return pd.DataFrame(rows_train), pd.DataFrame(rows_dev)


def score_frame(model, tokenizer, graph_batch: Batch, frame: pd.DataFrame,
                candidate_tools: list[str], device: torch.device, batch_size: int):
    model.eval()
    with torch.no_grad():
        candidates = encode_graph_batch(model, graph_batch)
        chunks = []
        queries = frame["query"].astype(str).tolist()
        for start in range(0, len(queries), batch_size):
            tokens = tokenizer(
                queries[start:start + batch_size], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            zq = model.encode_text(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            )
            chunks.append((zq @ candidates.T).cpu())
    scores = torch.cat(chunks)
    order = torch.argsort(scores, dim=1, descending=True).numpy()
    truth = frame["class_id"].to_numpy(dtype=np.int64)
    pred = order[:, 0]
    ranks = np.asarray([int(np.where(order[i] == truth[i])[0][0]) + 1 for i in range(len(frame))])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(15), zero_division=0
    )
    metrics = {
        "n": int(len(frame)),
        "correct": int((pred == truth).sum()),
        "accuracy": float((pred == truth).mean()),
        "accuracy_pct": float(100 * (pred == truth).mean()),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@3": float(np.mean(ranks <= 3)),
        "recall@5": float(np.mean(ranks <= 5)),
        "mrr@5": float(np.mean(np.where(ranks <= 5, 1.0 / ranks, 0.0))),
        "macro_f1": float(f1.mean()),
    }
    per_tool = pd.DataFrame({
        "tool": candidate_tools, "precision": precision, "recall": recall,
        "f1": f1, "support": support.astype(int),
    })
    per_query = frame.copy()
    per_query["predicted_tool"] = [candidate_tools[i] for i in pred]
    per_query["correct"] = pred == truth
    per_query["rank"] = ranks
    per_query["top5_tools"] = [";".join(candidate_tools[j] for j in row[:5]) for row in order]
    matrix = confusion_matrix(truth, pred, labels=np.arange(15))
    return metrics, per_tool, per_query, matrix


def adapt_model(model, tokenizer, graph_batch: Batch, train: pd.DataFrame,
                dev: pd.DataFrame, device: torch.device, epochs: int,
                batch_size: int, head_lr: float, backbone_lr: float,
                weight_decay: float, seed: int, initial_temperature: float):
    temperature = torch.nn.Parameter(torch.tensor(float(initial_temperature), device=device))
    backbone_params, head_params = [], [temperature]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_params if "text_encoder.backbone" in name else head_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    best_state, best_dev, history = None, -1.0, []
    indices = np.arange(len(train))
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(indices)
        running_loss = 0.0
        for start in range(0, len(indices), batch_size):
            rows = train.iloc[indices[start:start + batch_size]]
            tokens = tokenizer(
                rows["query"].tolist(), padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            zq = model.encode_text(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            )
            zg = encode_graph_batch(model, graph_batch)
            logits = (zq @ zg.T) / temperature.clamp(min=0.01, max=0.20)
            labels = torch.tensor(rows["class_id"].to_numpy(), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params + backbone_params, 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(rows)
        dev_metrics, _, _, _ = score_frame(
            model, tokenizer, graph_batch, dev,
            [str(index) for index in range(15)], device, batch_size
        )
        record = {
            "epoch": epoch, "train_loss": running_loss / len(train),
            "dev_accuracy": dev_metrics["accuracy"],
            "temperature": float(temperature.detach()),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if dev_metrics["accuracy"] > best_dev:
            best_dev = dev_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Adaptation did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return history, best_dev, float(temperature.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/v3_routing_toolcard_adaptation_s42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=45)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output = create_output_dir(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from legr_tool_count import apply_tool_count_override
    apply_tool_count_override(15)
    from atomic_zero_shot import (
        LEGR_15_TOOLS, alias_routing_tool, canonicalise_routing_columns,
        is_one_node, one_node_candidates,
    )
    from encoders import resolve_graph_encoder_settings
    from eval import _load_model_and_tokenizer

    candidate_tools = list(LEGR_15_TOOLS)
    graphs = one_node_candidates(candidate_tools)
    if len(graphs) != 15 or len(set(candidate_tools)) != 15 or not all(is_one_node(g) for g in graphs):
        raise AssertionError("Expected exactly 15 distinct one-node, zero-edge graphs")

    train_cards, dev_cards = build_tool_card_corpus(candidate_tools)
    train_cards.to_csv(output / "adaptation_train_toolcards.csv", index=False)
    dev_cards.to_csv(output / "adaptation_dev_toolcards.csv", index=False)

    datasets, dataset_manifest = {}, {}
    for condition, path in ROUTING_FILES.items():
        frame = canonicalise_routing_columns(pd.read_csv(path), str(path))
        if len(frame) != EXPECTED_ROWS[condition]:
            raise AssertionError(f"{condition}: expected {EXPECTED_ROWS[condition]}, got {len(frame)}")
        truth_tools = [alias_routing_tool(x) for x in frame["ground_truth"].astype(str)]
        if set(truth_tools) != set(candidate_tools):
            raise AssertionError(f"{condition}: candidate labels do not match all 15 tools")
        frame["ground_truth_tool"] = truth_tools
        frame["class_id"] = [candidate_tools.index(x) for x in truth_tools]
        datasets[condition] = frame
        dataset_manifest[condition] = {
            "path": repo_relative(path), "sha256": sha256_file(path), "rows": len(frame),
        }

    entries = [checkpoint_manifest()[15][2], checkpoint_manifest()[15][3]]
    results, checkpoint_audit = {}, {}
    for entry in entries:
        seed_everything(args.seed)
        source = Path(entry["checkpoint"])
        payload = torch.load(source, map_location="cpu", weights_only=False)
        metadata = validate_checkpoint_metadata(entry, payload, 15)
        model, cfg, tokenizer = _load_model_and_tokenizer(str(source), device)
        _, _, bidirectional = resolve_graph_encoder_settings(cfg)
        graphs_on_device = candidate_batch(graphs, device, bidirectional)
        model_id = entry["model_id"]
        checkpoint_audit[model_id] = {
            "source": repo_relative(source), "source_sha256": sha256_file(source), **metadata,
        }

        model_results = {"source_objective": entry["objective"], "frozen": {}, "adapted": {}}
        for condition, frame in datasets.items():
            metrics, _, _, _ = score_frame(
                model, tokenizer, graphs_on_device, frame, candidate_tools, device, args.batch_size
            )
            model_results["frozen"][condition] = metrics

        initial_temperature = float(payload.get("criterion_state", {}).get("temperature", 0.05))
        history, best_dev, learned_temperature = adapt_model(
            model, tokenizer, graphs_on_device, train_cards, dev_cards, device,
            args.epochs, args.batch_size, args.head_lr, args.backbone_lr,
            args.weight_decay, args.seed, initial_temperature,
        )

        adapted_dir = output / model_id
        adapted_dir.mkdir()
        adapted_checkpoint = adapted_dir / "best_model.pt"
        adapted_config = deepcopy(payload.get("config", {}))
        adapted_config.update({
            "checkpoint_dir": str(adapted_dir),
            "routing_toolcard_adaptation": True,
            "routing_adaptation_seed": args.seed,
            "routing_adaptation_epochs": args.epochs,
            "routing_adaptation_head_lr": args.head_lr,
            "routing_adaptation_backbone_lr": args.backbone_lr,
        })
        torch.save({
            "epoch": args.epochs,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "criterion_state": {"temperature": torch.tensor(learned_temperature)},
            "config": adapted_config,
            "tool_count": 15,
            "adaptation": {
                "type": "routing tool-card only", "source_checkpoint": repo_relative(source),
                "source_sha256": sha256_file(source), "best_dev_accuracy": best_dev,
                "evaluation_queries_used_for_training": False,
            },
        }, adapted_checkpoint)
        write_json(adapted_dir / "training_history.json", history)

        for condition, frame in datasets.items():
            metrics, per_tool, per_query, matrix = score_frame(
                model, tokenizer, graphs_on_device, frame, candidate_tools, device, args.batch_size
            )
            model_results["adapted"][condition] = metrics
            slug = condition.lower()
            per_tool.to_csv(adapted_dir / f"per_tool_{slug}.csv", index=False)
            per_query.to_csv(adapted_dir / f"per_query_{slug}.csv", index=False)
            pd.DataFrame(matrix, index=candidate_tools, columns=candidate_tools).to_csv(
                adapted_dir / f"confusion_{slug}.csv"
            )
        model_results["adapted_checkpoint"] = repo_relative(adapted_checkpoint)
        model_results["adapted_checkpoint_sha256"] = sha256_file(adapted_checkpoint)
        model_results["best_toolcard_dev_accuracy"] = best_dev
        results[model_id] = model_results
        del model, graphs_on_device
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for model_id, model_result in results.items():
        for stage in ("frozen", "adapted"):
            for condition, metrics in model_result[stage].items():
                rows.append({"model_id": model_id, "stage": stage, "condition": condition, **metrics})
    pd.DataFrame(rows).to_csv(output / "routing_metrics.csv", index=False)
    final = {
        "experiment": "V3 routing tool-card adaptation and full four-condition evaluation",
        "training_data": "tool names, aliases, and registry descriptions only",
        "evaluation_queries_used_for_training": False,
        "candidate_protocol": "15 one-node, zero-edge LEGR graphs",
        "adaptation_rows": {"train": len(train_cards), "dev": len(dev_cards)},
        "datasets": dataset_manifest,
        "source_checkpoints": checkpoint_audit,
        "models": results,
        "environment": environment_snapshot(device),
        "arguments": vars(args),
    }
    write_json(output / "results.json", final)
    (output / "reproduce.txt").write_text(
        f"{sys.executable} scripts/adapt_v3_routing_tools.py --output {repo_relative(output)} "
        f"--device {args.device} --epochs {args.epochs} --batch-size {args.batch_size} "
        f"--head-lr {args.head_lr} --backbone-lr {args.backbone_lr} --seed {args.seed}\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v["adapted"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
