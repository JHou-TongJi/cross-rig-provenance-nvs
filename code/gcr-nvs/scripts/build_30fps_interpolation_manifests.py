"""Build timestamp-aware interpolation manifests for all captured sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_30fps_interpolation_manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing sequence folders")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sequence", action="append", default=None)
    args = parser.parse_args()

    sequence_names = args.sequence
    if sequence_names:
        sequences = [args.root / name for name in sequence_names]
    else:
        sequences = sorted(
            path for path in args.root.iterdir()
            if path.is_dir() and (path / "Key_frames").is_dir()
        )

    reports: list[dict] = []
    failures: list[dict] = []
    for sequence in sequences:
        try:
            rows, summary = build_manifest(sequence, args.fps)
            destination = args.output_root / f"{sequence.name}_30fps.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps({"summary": summary, "frames": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reports.append(summary)
        except Exception as exc:
            failures.append({"sequence_id": sequence.name, "error": str(exc)})

    summary = {
        "requested_fps": args.fps,
        "sequences_seen": len(sequences),
        "sequences_written": len(reports),
        "failures": failures,
        "real_frames": sum(row["real_frames"] for row in reports),
        "synthetic_records": sum(row["synthetic_records"] for row in reports),
        "output_records": sum(row["output_records"] for row in reports),
        "contract": (
            "real frames retain RGB/camera/LiDAR metadata; interpolated records carry only anchor IDs and alpha; "
            "no synthetic LiDAR metric ground truth"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
