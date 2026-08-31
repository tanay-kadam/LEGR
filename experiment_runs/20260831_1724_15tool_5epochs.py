"""15-tool SBERT FT, 5 epochs. `upgraded` only. Does not train LEGR."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\tkadam\LEGR")
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
RUN = ROOT / "experiment_runs" / "20260831_1724"
SEED = 42
EPOCHS = 5
DATASET = ROOT / "upgraded"

JOBS = [
    (15, "ged_0", []),
    (15, "ged_030", ["--lambda_ged", "0.30"]),
    (15, "tied_weights", ["--lambda_ged", "0", "--tied"]),
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


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def run_one(tools: int, variant: str, extra: list[str]) -> dict:
    split = DATASET / f"upgraded_{tools}tools"
    train_csv = split / "train.csv"
    val_csv = split / "dev.csv"
    test_csv = split / "test_topology_heldout.csv"
    out = RUN / "upgraded" / "TASK1_SBERT_FINE_TUNE" / f"{FOLDERS[variant]}_{tools}TOOL"
    out.mkdir(parents=True, exist_ok=True)
    extra_ged = extra if extra else ["--lambda_ged", "0"]
    if extra and "--lambda_ged" not in extra:
        extra_ged = extra + ["--lambda_ged", "0"]
    cmd = [
        str(PYTHON), str(ROOT / "src" / "sbert_ft_baseline.py"),
        "--tool_count", str(tools),
        "--seed", str(SEED),
        "--device", "cuda",
        "--epochs", str(EPOCHS),
        "--train_csv", str(train_csv),
        "--val_csv", str(val_csv),
        "--checkpoint_dir", str(out),
        "--wandb_run_name", f"ep5_{tools}_{variant}"[:80],
        *extra_ged,
    ]
    (out / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    log(f"TRAIN {tools} {variant} epochs={EPOCHS}")
    with (out / "stdout.log").open("w", encoding="utf-8") as so, (out / "stderr.log").open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env(), stdout=so, stderr=se, text=True)
    rec = {
        "tools": tools,
        "variant": variant,
        "epochs": EPOCHS,
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
    rec["metrics"] = payload.get("metrics")
    rec["n_eval"] = payload.get("n_eval")
    rec["best_epoch"] = payload.get("epoch")
    rec["status"] = "VERIFIED" if vr.returncode == 0 and payload.get("verified_load") else "FAILED"
    write_json(out / "run_metadata.json", rec)
    log(f"END {tools} {variant} status={rec['status']}")
    return rec


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    log(f"5-EPOCH 15-tool SBERT start")
    results = []
    for tools, variant, extra in JOBS:
        results.append(run_one(tools, variant, extra))
        write_json(RUN / "results.json", results)

    lines = [
        "# 15-tool SBERT FT — 5 epochs",
        "",
        "| Variant | Recall@1 | Tool-Set F1 | Mean GED | best epoch | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r.get("metrics") or {}

        def c(k):
            v = m.get(k)
            return "" if v is None else f"{float(v):.4f}"

        lines.append(
            f"| {r['variant']} | {c('recall@1')} | {c('tool_set_f1')} | "
            f"{c('mean_ged_error')} | {r.get('best_epoch', '')} | {r.get('status')} |"
        )
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("5-EPOCH 15-tool SBERT COMPLETE")


if __name__ == "__main__":
    main()
