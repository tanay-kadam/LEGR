"""Rebuild checkpoint_manifest.json with tool_count-disambiguated IDs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parent
GIT_COMMIT = "fae6f498512f442218366b8fb264ff35c3834f1c"
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    man = json.loads((RUN / "experiment_manifest.json").read_text(encoding="utf-8"))
    ckpts: dict = {}
    skip_tasks = {"task2_latent", "task3_atomic"}
    for rec in man["runs"]:
        if rec.get("task") in skip_tasks:
            continue
        ckpt = rec.get("checkpoint") or rec.get("expected_checkpoint")
        if not ckpt:
            continue
        path = Path(ckpt)
        if not path.exists():
            print("missing", rec["run_id"], path)
            continue
        tc = rec.get("tool_count")
        cid = f"{rec['dataset']}__{rec['model']}__{rec['variant']}__tool{tc}__seed_{SEED}__best"
        ckpts[cid] = {
            "checkpoint_id": cid,
            "absolute_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "dataset": rec["dataset"],
            "task": rec["task"],
            "model": rec["model"],
            "variant": rec["variant"],
            "tool_count": tc,
            "seed": SEED,
            "source_run": rec["run_id"],
            "creation_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "bytes": path.stat().st_size,
            "cuda_device": "cuda:0",
            "git_commit": GIT_COMMIT,
            "verified_load": bool((rec.get("verification") or {}).get("verified_load")),
            "verified_eval": rec.get("status") == "VERIFIED",
        }

    killed = (
        RUN
        / "upgraded_v3"
        / "TASK4_DIRECTION_ABLATION"
        / "LEGR_DEFAULT_GCN_30TOOL__attempt1_killed"
        / "best_model.pt"
    )
    if killed.exists():
        cid = "upgraded_v3__legr_gcn__gcn_undirected__tool30__seed_42__attempt1_killed"
        ckpts[cid] = {
            "checkpoint_id": cid,
            "absolute_path": str(killed.resolve()),
            "sha256": sha256_file(killed),
            "dataset": "upgraded_v3",
            "task": "dep_legr",
            "model": "legr_gcn",
            "variant": "gcn_undirected",
            "tool_count": 30,
            "seed": SEED,
            "source_run": "upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42",
            "creation_time": datetime.fromtimestamp(killed.stat().st_mtime, timezone.utc).isoformat(),
            "bytes": killed.stat().st_size,
            "cuda_device": "cuda:0",
            "git_commit": GIT_COMMIT,
            "verified_load": False,
            "verified_eval": False,
            "note": (
                "Incomplete training; process killed at epoch 62. "
                "Do not use for paper metrics. Successful retry is "
                "upgraded_v3__legr_gcn__gcn_undirected__tool30__seed_42__best"
            ),
        }

    out = RUN / "checkpoint_manifest.json"
    out.write_text(json.dumps(ckpts, indent=2), encoding="utf-8")
    print(f"wrote {len(ckpts)} entries to {out}")
    for cid, rec in ckpts.items():
        print(rec["sha256"][:16], cid, "eval" if rec["verified_eval"] else "NOEVAL")


if __name__ == "__main__":
    main()
