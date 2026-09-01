"""
collate_results.py — Campaign v4 Final Results Collation
========================================================

Collects all experiment results and produces a unified comparison table.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("artifacts/campaign_v4/results")
OUTPUT_DIR = Path("artifacts/campaign_v4/final_analysis")


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _extract_eval_metrics(manifest_path: Path) -> dict:
    """Extract eval metrics from training manifests (best val loss/metrics)."""
    data = _load_json(manifest_path)
    if not data:
        return {}

    results = {}
    for run in data.get("runs", []):
        label = run.get("label", "unknown")
        success = run.get("success", False)
        elapsed = run.get("elapsed_s", 0)

        stdout = run.get("stdout_tail", "")
        val_loss = None
        for line in stdout.split("\n"):
            if "best_val_loss" in line.lower() or "val_loss" in line.lower():
                try:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if "loss" in p.lower() and i + 1 < len(parts):
                            val_loss = float(parts[i + 1].strip(","))
                except (ValueError, IndexError):
                    pass

        results[label] = {
            "success": success,
            "elapsed_s": elapsed,
            "val_loss": val_loss,
        }
    return results


def collate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "campaign": "campaign_v4",
        "generated_at": datetime.now().isoformat(),
        "tiers": {},
    }

    for tier in [15, 30, 45]:
        tier_results = {"tier": tier, "models": {}}

        manifest = _load_json(RESULTS_DIR / f"manifest_{tier}tools_s42.json")
        if manifest:
            for run in manifest.get("runs", []):
                label = run.get("label", "")
                tier_results["models"][label] = {
                    "type": "training",
                    "success": run.get("success", False),
                    "elapsed_s": run.get("elapsed_s", 0),
                    "checkpoint_dir": str(RESULTS_DIR / f"{label}_s42"),
                }

        bm25 = _load_json(RESULTS_DIR / f"bm25_{tier}t.json")
        if bm25:
            tier_results["models"]["BM25"] = {
                "type": "baseline",
                "test_indomain": bm25.get("test_indomain", {}),
                "test_topology_heldout": bm25.get("test_topology_heldout", {}),
            }

        frozen = _load_json(RESULTS_DIR / f"frozen_sbert_{tier}t.json")
        if frozen:
            tier_results["models"]["Frozen_SBERT"] = {
                "type": "baseline",
                "test_indomain": frozen.get("test_indomain", {}),
                "test_topology_heldout": frozen.get("test_topology_heldout", {}),
            }

        llama = _load_json(RESULTS_DIR / f"llama_{tier}t_heldout.json")
        if llama:
            tier_results["models"]["Llama_3.2_3B"] = {
                "type": "generative_baseline",
                "test_topology_heldout": {
                    "tool_set_f1": llama.get("tool_set_f1"),
                    "mean_ged_error": llama.get("mean_ged_error"),
                    "exact_match_rate": llama.get("exact_match_rate"),
                    "parse_failures": llama.get("parse_failures"),
                    "structural_validity_rate": llama.get("structural_validity_rate"),
                    "cyclic_rate": llama.get("cyclic_rate"),
                    "mean_latency_s": llama.get("mean_latency_s"),
                    "n_examples": llama.get("completed_examples"),
                },
            }

        gptoss = _load_json(RESULTS_DIR / f"gptoss_{tier}t_heldout.json")
        if gptoss:
            tier_results["models"]["GPT_OSS_120B"] = {
                "type": "generative_baseline",
                "test_topology_heldout": {
                    "tool_set_f1": gptoss.get("tool_set_f1"),
                    "mean_ged_error": gptoss.get("mean_ged_error"),
                    "exact_match_rate": gptoss.get("exact_match_rate"),
                    "parse_failures": gptoss.get("parse_failures"),
                    "structural_validity_rate": gptoss.get("structural_validity_rate"),
                    "cyclic_rate": gptoss.get("cyclic_rate"),
                    "mean_latency_s": gptoss.get("mean_latency_s"),
                    "n_examples": gptoss.get("completed_examples"),
                },
            }

        report["tiers"][str(tier)] = tier_results

    sbert_eval_metrics = {}
    for tier in [15, 30, 45]:
        for variant in ["no_ged", "ged"]:
            ckpt_dir = RESULTS_DIR / f"sbert_ft_{variant}_{tier}t_s42"
            eval_path = ckpt_dir / "eval_metrics.json"
            if eval_path.exists():
                data = json.loads(eval_path.read_text(encoding="utf-8"))
                label = f"sbert_ft_{variant}_{tier}t"
                sbert_eval_metrics[label] = data

    if sbert_eval_metrics:
        report["sbert_eval_details"] = sbert_eval_metrics

    legr_eval_metrics = {}
    for tier in [15, 30, 45]:
        for arch in ["legacy", "directed"]:
            for variant in ["no_ged", "ged"]:
                label = f"legr_{arch}_{variant}_{tier}t"
                ckpt_dir = RESULTS_DIR / f"{label}_s42"
                best_pt = ckpt_dir / "best_model.pt"
                if best_pt.exists():
                    legr_eval_metrics[label] = {
                        "checkpoint_exists": True,
                        "checkpoint_path": str(best_pt),
                    }

    if legr_eval_metrics:
        report["legr_checkpoints"] = legr_eval_metrics

    out_path = OUTPUT_DIR / "experiment_comparison.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Results collated to {out_path}")

    print_summary(report)
    return report


def print_summary(report: dict):
    print("\n" + "=" * 80)
    print("  CAMPAIGN v4 — EXPERIMENT SUMMARY")
    print("=" * 80)

    for tier_key in ["15", "30", "45"]:
        tier_data = report.get("tiers", {}).get(tier_key, {})
        models = tier_data.get("models", {})
        print(f"\n--- Tier {tier_key} tools ---")

        for name, info in models.items():
            mtype = info.get("type", "unknown")
            if mtype == "training":
                status = "OK" if info.get("success") else "FAIL"
                elapsed = info.get("elapsed_s", 0)
                print(f"  {name:40s}  {status:6s}  {elapsed:7.1f}s")
            elif mtype in ("baseline",):
                heldout = info.get("test_topology_heldout", {})
                indomain = info.get("test_indomain", {})
                r1_h = heldout.get("recall@1", "N/A")
                r1_i = indomain.get("recall@1", "N/A")
                print(f"  {name:40s}  R@1(heldout)={r1_h}  R@1(indomain)={r1_i}")
            elif mtype == "generative_baseline":
                heldout = info.get("test_topology_heldout", {})
                f1 = heldout.get("tool_set_f1", "N/A")
                ged = heldout.get("mean_ged_error", "N/A")
                em = heldout.get("exact_match_rate", "N/A")
                pf = heldout.get("parse_failures", "N/A")
                print(f"  {name:40s}  F1={f1}  GED={ged}  EM={em}  ParseFail={pf}")

    sbert_details = report.get("sbert_eval_details", {})
    if sbert_details:
        print("\n--- SBERT Fine-tuned Eval Metrics ---")
        for name, metrics in sbert_details.items():
            if isinstance(metrics, dict):
                r1 = metrics.get("recall@1", "N/A")
                mrr = metrics.get("mrr@5", "N/A")
                print(f"  {name:40s}  R@1={r1}  MRR@5={mrr}")

    legr_ckpts = report.get("legr_checkpoints", {})
    if legr_ckpts:
        print("\n--- LEGR Checkpoints ---")
        for name, info in legr_ckpts.items():
            exists = info.get("checkpoint_exists", False)
            print(f"  {name:40s}  {'EXISTS' if exists else 'MISSING'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    collate_all()
