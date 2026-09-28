"""Create a symlink-only unified view of canonical dense-route sequences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    index = json.loads(args.sequence_index.read_text())
    rows = index["canonical_sequences"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    linked = []
    for row in rows:
        source = Path(row["path"]).resolve()
        target = args.output_root / row["sequence_id"]
        if target.exists() or target.is_symlink():
            if not args.force:
                if target.is_symlink() and target.resolve() == source:
                    linked.append({"sequence_id": row["sequence_id"], "source": str(source), "target": str(target), "reused": True})
                    continue
                raise FileExistsError(f"refusing to replace existing path: {target}")
            if target.is_dir() and not target.is_symlink():
                raise IsADirectoryError(f"refusing to remove directory: {target}")
            target.unlink()
        os.symlink(source, target, target_is_directory=True)
        linked.append({"sequence_id": row["sequence_id"], "source": str(source), "target": str(target), "reused": False})
    report = {
        "output_root": str(args.output_root.resolve()),
        "sequence_count": len(linked),
        "source_bytes_copied": 0,
        "links": linked,
    }
    (args.output_root / "view_contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_root": str(args.output_root), "sequence_count": len(linked), "source_bytes_copied": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
