"""Retry Dataset-B 30-tool GCN (killed mid-train) then Task 2 latent analysis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_master import (  # noqa: E402
    RUN_ROOT,
    execute_run,
    find_run,
    journal,
    load_manifest,
    read_json,
    save_manifest,
    write_final_artifacts,
    capture_environment,
    write_json,
)

def main():
    man = load_manifest()
    ckpts = read_json(RUN_ROOT / "checkpoint_manifest.json", {})
    train_id = "upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42"
    latent_id = "upgraded_v3__task2_latent__legr__action_type_analysis__seed_42"
    for rid in (train_id, latent_id):
        rec = find_run(man, rid)
        rec["status"] = "PENDING"
        rec["verified"] = False
        rec.pop("metrics", None)
        rec.pop("error", None)
        rec.pop("checkpoint", None)
        rec.pop("checkpoint_sha256", None)
        rec.pop("returncode", None)
        rec["retry_reason"] = (
            "attempt1 30-tool GCN was SIGKILL'd during epoch 62 after a "
            "mistaken hung-process cleanup; failed outputs archived as "
            "LEGR_DEFAULT_GCN_30TOOL__attempt1_killed"
        )
    save_manifest(man)
    journal("RETRY START " + train_id)
    execute_run(find_run(man, train_id), man, ckpts)
    save_manifest(man)
    write_json(RUN_ROOT / "checkpoint_manifest.json", ckpts)
    journal("RETRY START " + latent_id)
    execute_run(find_run(man, latent_id), man, ckpts)
    save_manifest(man)
    write_json(RUN_ROOT / "checkpoint_manifest.json", ckpts)
    env = read_json(RUN_ROOT / "environment.json", {})
    write_final_artifacts(man, env)
    journal("RETRY COMPLETE")


if __name__ == "__main__":
    main()
