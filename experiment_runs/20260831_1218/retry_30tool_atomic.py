"""Run 30-tool zero-shot atomic eval on both datasets after protocol fix."""
from __future__ import annotations

import shutil
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
    write_json,
)

RUN_IDS = [
    "upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42",
    "upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42",
]


def archive_failed(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    dest = out_dir.parent / (out_dir.name + "__attempt1_not_supported")
    if dest.exists():
        return
    shutil.move(str(out_dir), str(dest))


def main() -> None:
    man = load_manifest()
    ckpts = read_json(RUN_ROOT / "checkpoint_manifest.json", {})
    env = read_json(RUN_ROOT / "environment.json", {})
    for rid in RUN_IDS:
        rec = find_run(man, rid)
        archive_failed(Path(rec["out_dir"]))
        rec["kind"] = "atomic_eval"
        rec["status"] = "PENDING"
        rec["verified"] = False
        rec["dependencies"] = [
            f"{rec['dataset']}__dep_legr__legr_gcn__gcn_undirected__seed_42"
        ]
        rec["retry_reason"] = (
            "attempt1 captured script rejection of --tool_count 30. "
            "Retry uses 30-tool frozen GCN + routing_15tools queries "
            "(no routing_30 OOV aliases)."
        )
        rec.pop("metrics", None)
        rec.pop("error", None)
        rec.pop("paper_state", None)
        rec.pop("returncode", None)
        rec.pop("independent_rerun", None)
        rec.pop("metric_diffs", None)
        Path(rec["out_dir"]).mkdir(parents=True, exist_ok=True)
        journal("RETRY START " + rid)
        execute_run(rec, man, ckpts)
        save_manifest(man)
        write_json(RUN_ROOT / "checkpoint_manifest.json", ckpts)
        journal("RETRY END " + rid + " status=" + rec["status"])
    write_final_artifacts(man, env)
    journal("RETRY 30TOOL ATOMIC COMPLETE")


if __name__ == "__main__":
    main()
