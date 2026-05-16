"""One-shot migration: rename legacy 'run'-noun keys to Repetition vocabulary.

Walks the given root, rewriting any .json/.yaml/.yml file in-place so that
keys produced by older orchestrator versions match the current schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

KEY_RENAMES: dict[str, str] = {
    "num_runs": "num_repetitions",
    "inter_run_delay": "inter_repetition_delay",
    "mapped_runs": "mapped_repetitions",
    "run_timestamp_utc": "repetition_timestamp_utc",
}


def _rewrite(obj: object) -> tuple[object, int]:
    changed = 0
    if isinstance(obj, dict):
        new: dict[object, object] = {}
        for k, v in obj.items():
            new_k: object = KEY_RENAMES.get(k, k) if isinstance(k, str) else k
            if new_k != k:
                changed += 1
            sub, sub_changed = _rewrite(v)
            changed += sub_changed
            new[new_k] = sub
        return new, changed
    if isinstance(obj, list):
        out: list[object] = []
        for item in obj:
            sub, sub_changed = _rewrite(item)
            changed += sub_changed
            out.append(sub)
        return out, changed
    return obj, 0


def _migrate_file(path: Path, dry_run: bool) -> int:
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        new, changed = _rewrite(data)
        dumped = json.dumps(new, indent=2)
    else:
        data = yaml.safe_load(text)
        new, changed = _rewrite(data)
        dumped = yaml.safe_dump(new, sort_keys=False)
    if changed and not dry_run:
        path.write_text(dumped)
    if changed:
        print(f"  {'would rewrite' if dry_run else 'rewrote'} {path} ({changed} keys)")
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="Artifact directory to migrate")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    total = 0
    for path in args.root.rglob("*"):
        if path.suffix in {".json", ".yaml", ".yml"} and path.is_file():
            total += _migrate_file(path, args.dry_run)
    print(f"Total key renames: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
