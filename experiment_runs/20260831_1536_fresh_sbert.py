"""Fresh SBERT FT on Dataset A `upgraded` only: train, val, held-out test.

Does not train LEGR. Does not use upgraded_v3.
Does not reuse prior SBERT checkpoints.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\tkadam\LEGR")
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
RUN = ROOT / "experiment_runs" / "20260831_1536"
SEED = 42
DATASET = ROOT / "upgraded"

JOBS = [
    (15, "ged_0", []),
    (15, "ged_030", ["--lambda_ged", "0.30"]),
    (15, "tied_weights", ["--lambda_ged", "0", "--tied"]),
    (30, "ged_0", []),
    (30, "ged_030", ["--lambda_ged", "0.30"]),
    (30, "tied_weights", ["--lambda_ged", "0", "--tied"]),
    (45, "ged_0", []),
    (45, "ged_030", ["--lambda_ged", "0.30"]),
    (45, "tied_weights", ["--lambda_ged", "0", "--tied"]),
]
FOLDERS = {
    "ged_0": "SBERT_FT_GED0",
    "ged_030": "SBERT_FT_GED030",
    "tied_weights": "SBERT_FT_TIED",
}


def env() -> dict:
    e = os.environ.copy()
    e["PYTHONUNBUFFERED"] = "1"
    e["PYTHONPATH"] = str(ROOT / "src")
    e["WANDB_MODE"] = "offline"
    e["WANDB_SILENT"] = "true"
    e["WANDB_START_METHOD"] = "thread"
    e["HF_HUB_DISABLE_SYMLINKS"] = "1"
    return e


def log(msg: str) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with (RUN / "journal.md").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(tools: int, variant: str, extra: list[str]) -> dict:
    split = DATASET / f"upgraded_{tools}tools"
    train_csv = split / "train.csv"
    val_csv = split / "dev.csv"
    test_csv = split / "test_topology_heldout.csv"
    out = RUN / "upgraded" / "TASK1_SBERT_FINE_TUNE" / f"{FOLDERS[variant]}_{tools}TOOL"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), str(ROOT / "src" / "sbert_ft_baseline.py"),
        "--tool_count", str(tools),
        "--seed", str(SEED),
        "--device", "cuda",
        "--train_csv", str(train_csv),
        "--val_csv", str(val_csv),
        "--checkpoint_dir", str(out),
        "--wandb_run_name", f"fresh_{tools}_{variant}"[:80],
        "--lambda_ged", "0",
        *extra,
    ]
    # extra may repeat --lambda_ged; last wins in argparse? actually argparse typically first or last depending. Better not duplicate.
    if extra:
        cmd = [
            str(PYTHON), str(ROOT / "src" / "sbert_ft_baseline.py"),
            "--tool_count", str(tools),
            "--seed", str(SEED),
            "--device", "cuda",
            "--train_csv", str(train_csv),
            "--val_csv", str(val_csv),
            "--checkpoint_dir", str(out),
            "--wandb_run_name", f"fresh_{tools}_{variant}"[:80],
            *extra,
        ]
        if "--lambda_ged" not in extra:
            cmd += ["--lambda_ged", "0"]
    (out / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    log(f"TRAIN {tools} {variant}")
    with (out / "stdout.log").open("w", encoding="utf-8") as so, (out / "stderr.log").open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env(), stdout=so, stderr=se, text=True)
    rec = {
        "tools": tools,
        "variant": variant,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "out_dir": str(out),
        "train_returncode": proc.returncode,
        "checkpoint": str(out / "best_model.pt"),
    }
    if proc.returncode != 0:
        rec["status"] = "FAILED"
        rec["error"] = (out / "stderr.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        log(f"FAIL train {tools} {variant} rc={proc.returncode}")
        write_json(out / "run_metadata.json", rec)
        return rec

    vdir = out / "independent_test"
    vdir.mkdir(parents=True, exist_ok=True)
    vcmd = [
        str(PYTHON), str(ROOT / "experiment_runs" / "20260831_1218" / "verify_one.py"),
        "--tool_count", str(tools),
        "--arch", "sbert",
        "--checkpoint", str(out / "best_model.pt"),
        "--dataset_csv", str(test_csv),
        "--output_dir", str(vdir),
    ]
    log(f"TEST {tools} {variant}")
    vr = subprocess.run(vcmd, cwd=str(ROOT), env=env(), capture_output=True, text=True)
    (vdir / "verify_stdout.log").write_text(vr.stdout or "", encoding="utf-8")
    (vdir / "verify_stderr.log").write_text(vr.stderr or "", encoding="utf-8")
    rec["verify_returncode"] = vr.returncode
    payload = {}
    vp = vdir / "verification.json"
    if vp.exists():
        payload = json.loads(vp.read_text(encoding="utf-8"))
    rec["verification"] = payload
    rec["metrics"] = payload.get("metrics")
    rec["n_eval"] = payload.get("n_eval")
    rec["n_unique_dags"] = payload.get("n_unique_dags")
    rec["dataset_csv"] = payload.get("dataset_csv")
    rec["status"] = "VERIFIED" if vr.returncode == 0 and payload.get("verified_load") else "FAILED"
    write_json(out / "run_metadata.json", rec)
    log(f"END {tools} {variant} status={rec['status']}")
    return rec


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    log("FRESH SBERT start dataset=" + str(DATASET))
    results = []
    for tools, variant, extra in JOBS:
        results.append(run_one(tools, variant, extra))
        write_json(RUN / "results.json", results)

    lines = [
        "# Fresh SBERT FT — `upgraded` only",
        "",
        "Train: `upgraded/upgraded_{N}tools/train.csv`",
        "Val: `dev.csv` (early stopping)",
        "Test: `test_topology_heldout.csv` (independent reload of best_model.pt)",
        "",
        "| Tools | Variant | Recall@1 | Tool-Set F1 | Mean GED | n_eval | unique DAGs | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r.get("metrics") or {}
        def c(k):
            v = m.get(k)
            return "" if v is None else f"{float(v):.4f}"
        lines.append(
            f"| {r['tools']} | {r['variant']} | {c('recall@1')} | {c('tool_set_f1')} | "
            f"{c('mean_ged_error')} | {r.get('n_eval','')} | {r.get('n_unique_dags','')} | {r.get('status')} |"
        )
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("FRESH SBERT COMPLETE")


if __name__ == "__main__":
    main()
