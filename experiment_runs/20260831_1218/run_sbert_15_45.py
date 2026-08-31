"""Fine-tune SBERT on 15-tool and 45-tool splits; train 45-tool LEGR for comparison."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_master import (  # noqa: E402
    RUN_ROOT,
    SEED,
    execute_run,
    journal,
    load_manifest,
    read_json,
    save_manifest,
    write_final_artifacts,
    write_json,
)


def make_sbert(dataset: str, tools: int, variant: str) -> dict:
    folder = {
        "ged_0": "SBERT_FT_GED0",
        "ged_030": "SBERT_FT_GED030",
        "tied_weights": "SBERT_FT_TIED",
    }[variant]
    out = RUN_ROOT / dataset / "TASK1_SBERT_FINE_TUNE" / f"{folder}_{tools}TOOL"
    rid = f"{dataset}__task1_sbert__sbert_ft__{variant}__tool{tools}__seed_{SEED}"
    return {
        "run_id": rid,
        "run_name": rid,
        "dataset": dataset,
        "task": "task1_sbert",
        "model": "sbert_ft",
        "variant": variant,
        "tool_count": tools,
        "kind": "sbert_train",
        "seed": SEED,
        "device_requirement": "cuda",
        "out_dir": str(out),
        "status": "PENDING",
        "dependencies": [],
        "expected_checkpoint": str(out / "best_model.pt"),
        "expected_outputs": [],
        "verified": False,
        "added_for": "user requested SBERT FT vs 15-tool and 45-tool, not only 30-tool",
    }


def make_legr45(dataset: str) -> dict:
    out = RUN_ROOT / dataset / "TASK4_DIRECTION_ABLATION" / "LEGR_DEFAULT_GCN_45TOOL"
    rid = f"{dataset}__dep_legr__legr_gcn__gcn_undirected__tool45__seed_{SEED}"
    return {
        "run_id": rid,
        "run_name": rid,
        "dataset": dataset,
        "task": "dep_legr",
        "model": "legr_gcn",
        "variant": "gcn_undirected",
        "tool_count": 45,
        "kind": "legr_train",
        "seed": SEED,
        "device_requirement": "cuda",
        "out_dir": str(out),
        "status": "PENDING",
        "dependencies": [],
        "expected_checkpoint": str(out / "best_model.pt"),
        "expected_outputs": [],
        "verified": False,
        "added_for": "fair LEGR comparison against 45-tool SBERT FT",
    }


def planned() -> list[dict]:
    runs = []
    for dataset in ("upgraded", "upgraded_v3"):
        for variant in ("ged_0", "ged_030", "tied_weights"):
            runs.append(make_sbert(dataset, 15, variant))
        runs.append(make_legr45(dataset))
        for variant in ("ged_0", "ged_030", "tied_weights"):
            runs.append(make_sbert(dataset, 45, variant))
    return runs


def main() -> None:
    man = load_manifest()
    existing = {r["run_id"] for r in man["runs"]}
    for rec in planned():
        if rec["run_id"] not in existing:
            man["runs"].append(rec)
            existing.add(rec["run_id"])
        else:
            old = next(r for r in man["runs"] if r["run_id"] == rec["run_id"])
            if old.get("status") != "VERIFIED":
                old["status"] = "PENDING"
                old["verified"] = False
                old.pop("error", None)
                old.pop("metrics", None)
    save_manifest(man)
    ckpts = read_json(RUN_ROOT / "checkpoint_manifest.json", {})
    env = read_json(RUN_ROOT / "environment.json", {})
    ids = [r["run_id"] for r in planned()]
    for rid in ids:
        rec = next(r for r in man["runs"] if r["run_id"] == rid)
        journal("START " + rid)
        execute_run(rec, man, ckpts)
        save_manifest(man)
        write_json(RUN_ROOT / "checkpoint_manifest.json", ckpts)
    write_final_artifacts(man, env)
    journal("SBERT 15/45 + LEGR 45 COMPLETE")


if __name__ == "__main__":
    main()
