from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable


DEFAULT_IMMUTABLE = (
    Path("data/campaign_v4"),
    Path("src/encoders.py"),
    Path("src/encoders_v2.py"),
    Path("src/sbert_ft_baseline.py"),
    Path("src/train.py"),
    Path("src/eval.py"),
    Path("artifacts/campaign_v4/results"),
)


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths: Iterable[Path] = DEFAULT_IMMUTABLE) -> dict[str, str]:
    result = {}
    for root in paths:
        candidates = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            result[path.as_posix()] = _hash_file(path)
    return result


def write_snapshot(path: str | Path, values: dict[str, str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")


def read_snapshot(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "changed": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
        "missing": sorted(before.keys() - after.keys()),
        "added": sorted(after.keys() - before.keys()),
    }
