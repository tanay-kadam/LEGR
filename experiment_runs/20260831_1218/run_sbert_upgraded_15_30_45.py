"""SBERT fine-tune only on Dataset A `upgraded` for 15, 30, and 45 tools. No LEGR training."""
from __future__ import annotations

import shutil
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
    write_json,
)

DATASET = "upgraded"
TOOLS = (15, 30, 45)
VARIANTS = ("ged_0", "ged_030", "tied_weights")
FOLDERS = {
    "ged_0": "SBERT_FT_GED0",
    "ged_030": "SBERT_FT_GED030",
    "tied_weights": "SBERT_FT_TIED",
}


def make_run(tools: int, variant: str) -> dict:
    folder = f"{FOLDERS[variant]}_{tools}TOOL"
    out = RUN_ROOT / DATASET / "TASK1_SBERT_FINE_TUNE" / folder
    rid = f"{DATASET}__task1_sbert__sbert_ft__{variant}__tool{tools}__seed_{SEED}"
    return {
        "run_id": rid,
        "run_name": rid,
        "dataset": DATASET,
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
        "added_for": "SBERT FT on upgraded 15/30/45 only; reuse existing LEGR checkpoints for comparison",
    }


def planned() -> list[dict]:
    return [make_run(t, v) for t in TOOLS for v in VARIANTS]


def archive_partial(out: Path) -> None:
    if not out.exists():
        return
    dest = out.parent / (out.name + "__attempt1_aborted")
    if dest.exists():
        return
    shutil.move(str(out), str(dest))


def write_comparison(man: dict) -> None:
    rows = []
    for tools in TOOLS:
        legr = None
        for r in man["runs"]:
            if (
                r["dataset"] == DATASET
                and r["task"] == "dep_legr"
                and r["tool_count"] == tools
                and r.get("status") == "VERIFIED"
            ):
                legr = r.get("metrics") or {}
                break
        block = {"tool_count": tools, "legr": legr, "sbert": {}}
        for rec in planned():
            if rec["tool_count"] != tools:
                continue
            live = next(r for r in man["runs"] if r["run_id"] == rec["run_id"])
            block["sbert"][live["variant"]] = {
                "status": live.get("status"),
                "metrics": live.get("metrics"),
            }
        rows.append(block)
    out = RUN_ROOT / "final_analysis" / "sbert_vs_legr_upgraded_15_30_45.json"
    write_json(out, rows)
    lines = [
        "# SBERT FT vs existing LEGR — Dataset A `upgraded` only",
        "",
        "No new LEGR training. LEGR rows are the already-verified GCN checkpoints.",
        "45-tool LEGR was not trained in this campaign, so that cell is empty.",
        "",
        "| Tools | Model | Recall@1 | Tool-Set F1 | Mean GED | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def cell(m, k):
        if not m or k not in m:
            return ""
        return f"{m[k]:.4f}"

    for block in rows:
        tools = block["tool_count"]
        lm = block["legr"]
        if lm:
            lines.append(
                f"| {tools} | LEGR GCN (existing) | {cell(lm, 'recall@1')} | "
                f"{cell(lm, 'tool_set_f1')} | {cell(lm, 'mean_ged_error')} | VERIFIED |"
            )
        else:
            lines.append(f"| {tools} | LEGR GCN |  |  |  | NOT_RUN |")
        for variant, payload in block["sbert"].items():
            m = payload.get("metrics") or {}
            st = payload.get("status")
            label = {
                "ged_0": "SBERT FT λ_GED=0",
                "ged_030": "SBERT FT λ_GED=0.30",
                "tied_weights": "SBERT FT tied",
            }[variant]
            lines.append(
                f"| {tools} | {label} | {cell(m, 'recall@1')} | "
                f"{cell(m, 'tool_set_f1')} | {cell(m, 'mean_ged_error')} | {st} |"
            )
        lines.append("| | | | | | |")
    (RUN_ROOT / "final_analysis" / "sbert_vs_legr_upgraded_15_30_45.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    man = load_manifest()
    existing = {r["run_id"]: r for r in man["runs"]}
    for rec in planned():
        out = Path(rec["out_dir"])
        if rec["run_id"] == "upgraded__task1_sbert__sbert_ft__ged_0__tool15__seed_42":
            ckpt = out / "best_model.pt"
            if out.exists() and not (ckpt.exists() and ckpt.stat().st_size > 1000):
                archive_partial(out)
        if rec["run_id"] not in existing:
            man["runs"].append(rec)
        else:
            old = existing[rec["run_id"]]
            if old.get("status") != "VERIFIED":
                old.update({k: rec[k] for k in rec if k != "run_id"})
                old["status"] = "PENDING"
                old["verified"] = False
                old.pop("error", None)
                old.pop("metrics", None)
    save_manifest(man)
    ckpts = read_json(RUN_ROOT / "checkpoint_manifest.json", {})
    for rec in planned():
        live = next(r for r in man["runs"] if r["run_id"] == rec["run_id"])
        journal("START " + live["run_id"])
        execute_run(live, man, ckpts)
        save_manifest(man)
        write_json(RUN_ROOT / "checkpoint_manifest.json", ckpts)
    write_comparison(man)
    journal("SBERT upgraded 15/30/45 COMPLETE")


if __name__ == "__main__":
    main()
